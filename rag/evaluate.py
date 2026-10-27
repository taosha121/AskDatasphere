"""
用 RAGAS 对 run_eval.py 产出的结果算分。

整体思路：
  eval_runs.jsonl（run_eval.py 产出）
      ↓ 每条记录含 question / ground_truth / answer / retrieved_contexts
  RAGAS 框架
      ↓ 用 Claude 作为 judge LLM，用本地 bge 做 embedding
  eval_scores.csv     ← 每条问题的逐项得分
  eval_summary.json   ← 所有问题的指标均值汇总

四个指标的评估逻辑：
- context_recall
    judge LLM 把 ground_truth 拆成若干命题，逐一检查 retrieved_contexts 里
    有没有支撑该命题的内容。衡量"检索有没有找到答对这题所需的信息"。

- context_precision
    judge LLM 判断每个召回 chunk 对回答 ground_truth 是否真正有用，
    然后加权计算有用 chunk 在排名靠前位置的比例。
    衡量"有用的 chunk 排在前面了吗"（排序质量）。

- faithfulness
    judge LLM 从 answer 中提取所有事实性陈述，再逐一检查 retrieved_contexts
    里有没有支撑，统计比例。衡量"答案有没有捏造 context 里没有的内容"。

- answer_relevancy
    embedding 模型把 answer 反向生成多个假想问题，计算这些假想问题与原始
    question 的余弦相似度均值。衡量"答案是否在答这道题，而不是答非所问"。
    （这个指标不需要 ground_truth，只需要 question 和 answer）

judge LLM 用 Claude（走 .env 里的 base_url），embedding 用本地 bge
（与检索时的 embedding 保持一致，避免引入额外误差），全程不依赖 OpenAI。

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
    """把本地 bge SentenceTransformer 包成 langchain Embeddings 接口。

    RAGAS 内部通过 langchain Embeddings 协议调用 embedding，
    复用 backend 里已经初始化好的同一个模型实例，避免重复加载。
    """

    def __init__(self):
        from backend.main import _get_embed_model
        self._model = _get_embed_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def _make_judge() -> LangchainLLMWrapper:
    """创建 RAGAS 用的 judge LLM。

    RAGAS 内部通过 langchain LLM 协议调用大模型，所以需要用
    LangchainLLMWrapper 包一层。context_recall / context_precision /
    faithfulness 三个指标都会多次调用这个 judge 来做逐条判断。
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

    # RAGAS 要求固定的字段名（user_input / response / retrieved_contexts / reference），
    # 与我们内部命名（question / answer / retrieved_contexts / ground_truth）不完全一致，
    # 这里做一次映射。
    samples = [
        {
            "user_input":         r["question"],        # 用户提的问题
            "response":           r["answer"],           # RAG 系统生成的答案
            "retrieved_contexts": r["retrieved_contexts"],  # 检索器实际召回的 chunk 列表
            "reference":          r["ground_truth"],    # gen_eval_dataset 生成的标准答案，作为评分基准
        }
        for r in rows
    ]
    dataset = EvaluationDataset.from_list(samples)

    print(f"Evaluating {len(samples)} samples with Claude judge + bge embeddings…\n")

    judge = _make_judge()
    # answer_relevancy 指标需要 embedding 来计算问题相似度，其余三个指标用 judge LLM
    embeddings = LangchainEmbeddingsWrapper(BGEEmbeddings())

    # evaluate() 会并发调用 judge LLM 和 embedding，每条样本会产生多次 LLM 调用，
    # 样本量大时耗时和 token 消耗不低。
    result = evaluate(
        dataset=dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
    )

    print("\n=== Aggregate scores ===")
    print(result)

    # 逐条得分写入 CSV，方便后续分析哪些问题分低、哪类 deliverable 表现差
    df = result.to_pandas()
    df.to_csv(out, index=False)

    # 聚合均值写入 JSON，方便跨次运行对比（每次调参后跑一遍，对比这个文件）
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
