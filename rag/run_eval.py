"""
运行真实 RAG 管线，为评估集补全 answer 和 retrieved_contexts。

读取 gen_eval_dataset.py 产出的 eval_dataset.jsonl（只含 question + ground_truth），
对每条 question 调用真实管线 backend.main.ask（检索 + Claude 生成），
补上系统实际生成的 answer 和实际召回的 retrieved_contexts，
输出供 RAGAS 评估使用的 jsonl。

与 eval_dataset.jsonl 的区别：
- contexts（原来的"正确来源 chunk"）不进入这里；RAGAS 要评的是检索器实际召回的内容
- 新增 answer（系统答案）、retrieved_contexts（实际召回）

运行方式：
    python -m rag.run_eval                 # 全量 400 条，烧 token
    python -m rag.run_eval --limit 20      # 先跑通链路
    python -m rag.run_eval --limit 20 --top-k 5
"""
import argparse
import json
from pathlib import Path

from backend.main import AskRequest, ask

ROOT = Path(__file__).parent.parent


def build(inp: Path, out: Path, limit: int | None, top_k: int) -> None:
    rows = [json.loads(line) for line in inp.open(encoding="utf-8") if line.strip()]
    if limit:
        rows = rows[:limit]

    print(f"Loaded {len(rows)} eval rows. Running RAG (top_k={top_k})…\n")

    written = errors = 0
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            question = row["question"]
            try:
                res = ask(AskRequest(question=question, top_k=top_k))
                record = {
                    "question":           question,
                    "ground_truth":       row.get("ground_truth", ""),
                    "answer":             res["answer"],
                    "retrieved_contexts": [c["text"] for c in res["chunks"]],
                    "retrieved_scores":   [c["score"] for c in res["chunks"]],
                    "source_chunk_id":    row.get("chunk_id", ""),
                    "deliverable":        row.get("deliverable", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                print(f"  [{i:>3}/{len(rows)}] {question[:60]}")
            except Exception as e:
                errors += 1
                print(f"  [{i:>3}/{len(rows)}] ERROR — {e}")

    print(f"\nDone. Written: {written}  Errors: {errors}")
    print(f"Output: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run RAG pipeline over eval questions")
    p.add_argument("--input",  type=Path, default=ROOT / "eval_dataset.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "eval_runs.jsonl")
    p.add_argument("--limit",  type=int, default=None, help="只处理前 N 条")
    p.add_argument("--top-k",  type=int, default=3)
    args = p.parse_args()
    build(args.input, args.output, args.limit, args.top_k)
