# RAG Optimization

This document records completed optimizations and future directions for RAG quality in AskDatasphere. Evaluation results are in [EVAL_LOG.md](EVAL_LOG.md); the project overview is in [README.md](README.md).

---

## Current Status

| Optimization | Status | Impact |
|---|---|---|
| RAGAS evaluation pipeline | Done | Established a quantitative baseline to drive decisions |
| Reduce chunk size (1500 / 200) | Done | Fixed 512-token truncation bug, reduced semantic dilution |
| BM25 hybrid retrieval + RRF fusion | Done | Context Recall 0.793 → 0.858 (+8.2%) |
| CrossEncoder reranker | Done | Context Precision 0.631 → 0.793 (+25.7%) |
| Semantic chunking (paragraph / heading boundaries) | Planned | Expected to improve recall on fine-grained procedural questions |
| PDF noise cleaning | Planned | Expected to improve Context Precision |
| Agentic RAG (iterative retrieval) | Planned | Expected to improve recall on constraint / negative-answer questions |
| Graph RAG | Planned | Expected to improve multi-hop / global relationship questions |

---

## Completed Optimizations

### Evaluation-Driven Development

Optimization requires measurement. This project uses [RAGAS](https://docs.ragas.io) for offline evaluation, scoring retrieval and generation separately across four metrics: Context Recall, Context Precision, Faithfulness, and Answer Relevancy. Three-step pipeline:

1. `gen_eval_dataset.py` samples chunks from the vector store by deliverable proportion and uses Claude to generate Q&A pairs
2. `run_eval.py` runs the live retrieval + generation pipeline on each question, capturing the system's actual answers and retrieved contexts
3. `evaluate.py` scores with RAGAS, using Claude as the judge and local bge for embeddings

### Chunk Size (1500 / 200)

`BAAI/bge-base-en-v1.5` has `max_seq_length` = 512 tokens. At 3200 characters (~800 tokens for English), SentenceTransformer silently truncates everything past the limit — roughly the last third of every chunk was never encoded. Reducing to 1500 characters (~375 tokens) fixed two problems simultaneously: the truncation bug and semantic dilution (large chunks bury precise facts inside a dominant surrounding context).

### Hybrid Retrieval (BM25 + Dense → RRF)

Pure vector search is weak on exact token matches (version numbers, numeric ranges, rare terms); BM25 literal matching fills that gap. Implementation: at startup, all documents are pulled from Chroma and a `BM25Okapi` inverted index (`rank_bm25` library) is built in memory. Both paths retrieve 20 candidates, which are merged with RRF (k=60) to produce a Top 10.

### CrossEncoder Reranker

A Bi-Encoder encodes query and document independently into vectors and computes distance — fast, but limited in precision, since cosine similarity can't capture fine-grained match patterns. A CrossEncoder (`BAAI/bge-reranker-base`) concatenates the question and candidate document and passes both through the model together; the attention mechanism operates across both texts and produces more accurate relevance scores. Applied to the RRF Top 10 to rerank before returning Top K. Context Precision improved by 25.7%.

---

## Future Directions

### 1. Semantic Chunking

Fixed-character chunking ignores natural boundaries — a complete procedural step can be split across two chunks. Cutting at paragraph or heading boundaries ensures each chunk is a semantically self-contained unit. This should reduce the recall=0 cases that remain after Run #3, where roughly 65% are fine-grained UI behavior or procedural description failures.

Implementation: use `fitz` (PyMuPDF) to extract block-level structure from PDFs (each block typically maps to a paragraph or heading), merge by block boundary, then split blocks exceeding 1500 characters.

### 2. PDF Noise Cleaning

PDF text extraction flattens tables into isolated per-row cells, and mixes page headers/footers (page numbers, chapter titles, "PUBLIC" watermarks) into the main body. This noise dilutes chunk semantics and is a likely drag on Context Precision. Can be filtered in the ingest stage: lines shorter than ~20 characters, lines containing only digits (page numbers), lines matching known watermark keywords ("PUBLIC", "Document Version").

### 3. Agentic RAG (Iterative Retrieval)

The current pipeline does a single retrieval pass. Constraint/negative-answer questions ("Can I do X?" → "No") frequently fail because one pass rarely surfaces that single prohibitive sentence. Agentic RAG adds an LLM-driven evaluate-decide loop between retrieval and generation:

| | Standard RAG | Agentic RAG |
|---|---|---|
| Retrieval passes | Fixed: 1 | Dynamic, as needed |
| Query handling | Raw question sent directly | Can decompose into sub-questions |
| Result evaluation | None | Model self-assesses retrieval quality |
| Best for | Single-hop, factual queries | Multi-hop, comparative, constraint queries |

Key capabilities: **iterative retrieval** (rewrite the query and re-retrieve when the first pass is insufficient) and **CRAG** (Corrective RAG — model scores retrieved chunk relevance after each pass; below threshold, it rewrites the query or switches strategy). Can be implemented with LangGraph, LlamaIndex Workflows, or a hand-written loop. Trade-off: higher latency and token cost than standard RAG.

### 4. Graph RAG

Proposed by Microsoft in 2024 — an architectural upgrade over standard RAG. Rather than chunking documents into text fragments, Graph RAG uses an LLM to extract entities and relationships from documents, builds a knowledge graph, and produces hierarchical summaries of densely connected "communities." Retrieval operates over these summaries, not raw text.

**Two retrieval modes:**

- **Local Search:** retrieves by entity and neighbors in the graph — suitable for specific "how do I use feature X" questions; comparable to standard RAG
- **Global Search:** retrieves over community-level summaries of the full graph — suited for "what is the relationship between these modules" questions that require a global view, which is exactly where standard RAG fails

**Leiden Algorithm: Community Detection**

After building the knowledge graph, Leiden detects which entities are most densely connected and groups them into communities. Running Leiden over this project's documents might automatically produce: Space / Permissions / User Roles, DataBuilder / View / Transformation, Catalog / Governance, etc.

**Global Search Query Flow (Map-Reduce):**

```
Offline index phase (one-time):
  Documents → chunks → LLM extracts entities and relations → knowledge graph
  → Leiden community detection → each community summarized by LLM

Online query phase (Global Search):
  Map:    send question to each community summary in parallel;
          each LLM call independently identifies relevant points
  Reduce: aggregate all non-empty partial answers;
          one final LLM call synthesizes the combined response
```

Global Search LLM calls = number of communities + 1. Higher cost, but covers the full graph — best for global questions.

**Dependencies:**

```bash
pip install graphrag

python -m graphrag index --init --root ./my_graphrag
# Place documents in my_graphrag/input/, edit settings.yaml with API key and model name
python -m graphrag index --root ./my_graphrag

python -m graphrag query --root ./my_graphrag --method global "What are the relationships between the modules?"
python -m graphrag query --root ./my_graphrag --method local  "How do I create a View in DataBuilder?"
```

For this project (~3000+ chunks), the index phase LLM call count is substantial. Recommended only when standard RAG falls short on global or cross-module questions.
