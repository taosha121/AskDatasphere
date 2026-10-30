"""
Evaluation dataset generation script.

Randomly samples chunks from ChromaDB proportional to each deliverable's share,
uses Claude to generate 2 Q&A pairs per chunk, and writes a JSONL file for RAGAS.

Each record format:
{
    "question":     str,
    "ground_truth": str,
    "contexts":     [str],   # original chunk text
    "chunk_id":     str,
    "deliverable":  str,
    "source_file":  str
}

Usage:
    python -m rag.gen_eval_dataset
    python -m rag.gen_eval_dataset --sample 100 --output my_eval.jsonl
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
CLAUDE_MODEL = "claude-sonnet-4-5"
DEFAULT_SAMPLE = 200
QA_PER_CHUNK = 2

_PROMPT = """\
You are building an evaluation dataset for a RAG system on SAP Datasphere documentation.

Given the documentation excerpt below, generate exactly {n} question-answer pairs.

Rules:
- Each question must be answerable SOLELY from the excerpt — no outside knowledge
- Answers must be accurate and directly grounded in the excerpt text
- Questions should be diverse and realistic (what an SAP Datasphere user or admin would ask)
- Answers should be concise (1-4 sentences)

Return ONLY a JSON array, no extra text or markdown fences:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Excerpt:
{text}"""


def _sample_chunks(col, n: int) -> list[dict]:
    all_data = col.get(include=["documents", "metadatas"])
    total = len(all_data["ids"])

    rows = [
        {"id": cid, "text": doc, **meta}
        for cid, doc, meta in zip(
            all_data["ids"], all_data["documents"], all_data["metadatas"]
        )
    ]

    if n >= total:
        return rows

    # Group by deliverable, then sample proportionally to ensure coverage across all docs
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row.get("deliverable", "unknown"), []).append(row)

    sampled: list[dict] = []
    remaining = n
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (_, group) in enumerate(sorted_groups):
        total_left = sum(len(g) for _, g in sorted_groups[i:])
        quota = round(remaining * len(group) / total_left)
        quota = max(1, min(quota, len(group)))
        sampled.extend(random.sample(group, quota))
        remaining -= quota
        if remaining <= 0:
            break

    return sampled[:n]


def _generate_qa(client: anthropic.Anthropic, text: str) -> list[dict]:
    prompt = _PROMPT.format(n=QA_PER_CHUNK, text=text[:3000])
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def build(sample: int, output: Path) -> None:
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("datasphere")

    print(f"Total chunks in DB: {col.count()}")
    print(f"Sampling {sample} chunks…")
    chunks = _sample_chunks(col, sample)
    print(f"Sampled {len(chunks)} chunks. Generating Q&A…\n")

    written = errors = 0
    with output.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            try:
                pairs = _generate_qa(client, chunk["text"])
                for pair in pairs:
                    record = {
                        "question":     pair["question"],
                        "ground_truth": pair["answer"],
                        "contexts":     [chunk["text"]],
                        "chunk_id":     chunk["id"],
                        "deliverable":  chunk.get("deliverable", ""),
                        "source_file":  chunk.get("source_file", ""),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

                label = chunk.get("deliverable", "")[:45]
                print(f"  [{i:>3}/{len(chunks)}] {label:<45} +{len(pairs)}")
                time.sleep(0.3)

            except Exception as e:
                errors += 1
                print(f"  [{i:>3}/{len(chunks)}] ERROR — {e}")

    print(f"\nDone.  Written: {written}  Errors: {errors}")
    print(f"Output: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG evaluation dataset from ChromaDB chunks")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="Number of chunks to sample (default: 200)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "eval_dataset.jsonl", help="Output JSONL file path")
    args = parser.parse_args()
    build(args.sample, args.output)
