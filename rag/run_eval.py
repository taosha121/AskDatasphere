"""
Run the real RAG pipeline to fill in answers and retrieved contexts for the eval set.

Reads eval_dataset.jsonl (produced by gen_eval_dataset.py, containing only question +
ground_truth), calls the live pipeline backend.main.ask (retrieval + Claude generation)
for each question, appends the system's actual answer and retrieved contexts, and writes
an output JSONL file ready for RAGAS scoring.

Difference from eval_dataset.jsonl:
- contexts (the "correct source chunk") are not included; RAGAS evaluates what the
  retriever actually returned, not what it should have
- adds answer (system-generated) and retrieved_contexts (actually retrieved)

Usage:
    python -m rag.run_eval                 # full 400-question run (token-heavy)
    python -m rag.run_eval --limit 20      # smoke test
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
    p.add_argument("--limit",  type=int, default=None, help="Process only the first N rows")
    p.add_argument("--top-k",  type=int, default=5)
    args = p.parse_args()
    build(args.input, args.output, args.limit, args.top_k)
