"""
用 RAGAS 对 run_eval.py 产出的结果算分。

judge LLM 用 Claude（走 .env 里的 base_url），embedding 用本地 bge
（与检索一致），全程不依赖 OpenAI。

四个指标：
- context_recall     检索是否召回了回答所需的信息（retrieved_contexts vs ground_truth）
- context_precision  召回内容的相关性/排序（retrieved_contexts vs ground_truth）
- faithfulness       答案是否忠于召回内容、有无编造（answer vs retrieved_contexts）
- answer_relevancy   答案是否切题（answer vs question，需 embedding）

运行方式：
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
    """把本地 bge SentenceTransformer 包成 langchain Embeddings 接口。"""

    def __init__(self):
        from backend.main import _get_embed_model
        self._model = _get_embed_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def _make_judge() -> LangchainLLMWrapper:
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

    samples = [
        {
            "user_input":         r["question"],
            "response":           r["answer"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference":          r["ground_truth"],
        }
        for r in rows
    ]
    dataset = EvaluationDataset.from_list(samples)

    print(f"Evaluating {len(samples)} samples with Claude judge + bge embeddings…\n")

    judge = _make_judge()
    embeddings = LangchainEmbeddingsWrapper(BGEEmbeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
    )

    print("\n=== Aggregate scores ===")
    print(result)

    df = result.to_pandas()
    df.to_csv(out, index=False)

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
    p.add_argument("--limit",  type=int, default=None, help="只评前 N 条")
    args = p.parse_args()
    build(args.input, args.output, args.limit)
