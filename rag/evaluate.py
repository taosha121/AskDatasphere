"""
Score run_eval.py output with RAGAS.

Overview:
  eval_runs.jsonl (produced by run_eval.py)
      ↓ each record contains question / ground_truth / answer / retrieved_contexts
  RAGAS framework
      ↓ Claude as judge LLM, local bge for embeddings
  eval_scores.csv     ← per-question scores for all four metrics
  eval_summary.json   ← aggregate means across all questions

How each metric works:
- context_recall
    Judge LLM decomposes ground_truth into individual claims, checks each against
    retrieved_contexts for supporting content. Measures "did retrieval find the
    information needed to answer this question?"

- context_precision
    Judge LLM decides whether each retrieved chunk is genuinely useful for answering
    ground_truth, then computes a rank-weighted ratio of useful chunks.
    Measures "are useful chunks ranked near the top?" (ranking quality).

- faithfulness
    Judge LLM extracts all factual claims from answer, checks each against
    retrieved_contexts for support, reports the supported ratio.
    Measures "does the answer fabricate anything not in the retrieved context?"

- answer_relevancy
    Embedding model reverse-generates candidate questions from answer, computes
    cosine similarity between those candidates and the original question.
    Measures "is the answer actually answering this question, not something else?"
    (Does not require ground_truth — only question and answer.)

Judge LLM uses Claude (via base_url from .env); embeddings use local bge
(same model as retrieval, avoiding extra variance). No OpenAI dependency.

Usage:
    python -m rag.evaluate
    python -m rag.evaluate --input eval_runs.jsonl --limit 20
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

CLAUDE_MODEL = "claude-sonnet-4-5"


class BGEEmbeddings(Embeddings):
    """Wrap the local bge SentenceTransformer as a langchain Embeddings interface.

    RAGAS calls embeddings via the langchain Embeddings protocol; this reuses the
    already-initialized model instance from backend to avoid loading it twice.
    """

    def __init__(self):
        from backend.main import _get_embed_model
        self._model = _get_embed_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def _make_judge() -> LangchainLLMWrapper:
    """Create the RAGAS judge LLM.

    RAGAS calls the LLM via the langchain LLM protocol, so a LangchainLLMWrapper
    is required. context_recall, context_precision, and faithfulness all make
    multiple judge calls per sample.
    """
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        max_tokens=4096,
        timeout=60,
    )
    return LangchainLLMWrapper(llm)


def build(inp: Path, out: Path, limit: int | None) -> None:
    rows = [json.loads(line) for line in inp.open(encoding="utf-8") if line.strip()]
    if limit:
        rows = rows[:limit]

    # RAGAS requires fixed field names (user_input / response / retrieved_contexts / reference);
    # map from our internal names (question / answer / retrieved_contexts / ground_truth).
    samples = [
        {
            "user_input":         r["question"],           # the user's question
            "response":           r["answer"],             # the RAG system's generated answer
            "retrieved_contexts": r["retrieved_contexts"], # chunks actually returned by the retriever
            "reference":          r["ground_truth"],       # ground-truth answer from gen_eval_dataset, used as scoring reference
        }
        for r in rows
    ]
    dataset = EvaluationDataset.from_list(samples)

    print(f"Evaluating {len(samples)} samples with Claude judge + bge embeddings…\n")

    judge = _make_judge()
    # answer_relevancy needs embeddings to compute question similarity; the other three metrics use the judge LLM
    embeddings = LangchainEmbeddingsWrapper(BGEEmbeddings())

    # evaluate() runs judge LLM and embeddings concurrently; each sample triggers multiple LLM calls,
    # so large sample counts take significant time and tokens.
    result = evaluate(
        dataset=dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
    )

    print("\n=== Aggregate scores ===")
    print(result)

    # Write per-question scores to CSV for post-hoc analysis of low-scoring cases
    df = result.to_pandas()
    df.to_csv(out, index=False)

    # Write aggregate means to JSON for cross-run comparison (run after each tuning step)
    metric_cols = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    summary = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "input":           inp.name,
        "n_samples":       len(df),
        "judge_model":     CLAUDE_MODEL,
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "scores":          {c: round(df[c].mean(), 4) for c in metric_cols if c in df},
    }
    summary_path = out.with_name("eval_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nPer-sample results written to: {out}")
    print(f"Aggregate summary written to:  {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score RAG eval runs with RAGAS")
    p.add_argument("--input",  type=Path, default=ROOT / "eval_runs.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "eval_scores.csv")
    p.add_argument("--limit",  type=int, default=None, help="Score only the first N rows")
    args = p.parse_args()
    build(args.input, args.output, args.limit)
