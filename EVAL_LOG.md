# Evaluation Log

This file records all RAGAS evaluation runs for AskDatasphere: parameters, results, analysis, and improvement plans.

For how each metric is computed (Context Recall, Context Precision, Faithfulness, Answer Relevancy), see the "Evaluating RAG Quality" section in [README.md](README.md). This file only records results and interpretation. All four metrics are better closer to 1.

For technical details on optimization directions, see [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md).

---

## Run #1 — Baseline (2026-10-26, 50 samples)

| Parameter | Value |
|---|---|
| Sample count | 50 (first 50 of the 400-row eval set) |
| Top K | 3 |
| Generation / judge model | claude-sonnet-4-5 |
| Embedding | BAAI/bge-base-en-v1.5 |
| Chunk size | 3200 chars / 400 overlap |

| Metric | Score | Count < 0.5 |
|---|---|---|
| Context Recall | 0.710 | 14 / 50 |
| Context Precision | 0.625 | 23 / 50 |
| Faithfulness | 0.916 | 1 / 50 |
| Answer Relevancy | 0.797 | 9 / 50 |

**Conclusion:** First successful eval run. Confirmed the bottleneck is on the retrieval side (Recall/Precision low; Faithfulness already strong). Top K=3 is too small; decided to raise to 5. Small sample — baseline reference only.

---

## Run #2 — Full Evaluation (2026-10-28, 400 samples)

| Parameter | Value |
|---|---|
| Sample count | 400 (full set, 200 chunks × 2 Q&A pairs each) |
| Top K | 5 |
| Generation / judge model | claude-sonnet-4-5 |
| Embedding | BAAI/bge-base-en-v1.5 |
| Chunk size | 3200 chars / 400 overlap |

| Metric | Mean | Count < 0.5 | Count = 0 |
|---|---|---|---|
| Context Recall | **0.793** | 77 / 400 (19%) | 77 / 400 |
| Context Precision | **0.631** | 165 / 400 (41%) | 73 / 400 |
| Faithfulness | **0.909** | 13 / 400 (3%) | 2 / 400 |
| Answer Relevancy | **0.796** | 74 / 400 (18.5%) | 74 / 400 |

### Interpretation

**Context Recall (0.793):** About 80% of questions had the necessary information retrieved. But 77 cases (19%) scored exactly 0 — the correct chunk was never retrieved and Claude had no basis to answer, outputting "insufficient information."

**Context Precision (0.631):** The weakest of the four. 165 cases (41%) scored below 0.5, meaning nearly half of all questions returned noisy chunks or had useful chunks ranked too low. This is the primary bottleneck.

**Faithfulness (0.909):** The strongest metric; only 13 cases below 0.5. The system prompt constraints ("answer only from the provided excerpts, say so if insufficient") are working well — Claude rarely fabricates. The 2 cases scoring 0 may involve Claude drawing on training-time SAP knowledge; worth investigating.

**Answer Relevancy (0.796):** 74 cases scored exactly 0 (18.5%), closely matching the 77 recall=0 cases. Root cause is retrieval failure: when retrieval finds nothing useful, Claude outputs "insufficient information" — a non-answer that generates candidate questions with no semantic relation to the original query, scoring 0. Not a generation problem.

### Low-Score Case Analysis (77 recall=0 cases)

Reviewing all recall=0 cases, three failure patterns account for most:

1. **Precise numerical / spec facts (~19 cases):** Questions asking for specific numbers (Pandas version `2.2.2`, range `1–366`, thread count `10`, preview `20` rows). The answer is a single sentence buried in a long passage or table. Dense embeddings don't encode exact numbers well, and the fact accounts for only a tiny fraction of the chunk vector, which gets dominated by surrounding context. Not ranked in Top 5.

2. **Rare compound terms:** Terms like `auxiliary measure`, `Calendar - Day of Year`, `Semantic Enrichment` — low-frequency in the corpus — are diluted by the broad topic of the surrounding chunk. For example, asking "what is an auxiliary measure" retrieves general measure content, never the definition sentence.

3. **Old product name / changelog meta-questions (4 cases — eval set noise):** Questions referencing `SAP Data Warehouse Cloud` (old name for SAP Datasphere), e.g. "earliest version mentioned in the docs" or "what was Administration renamed to in 2021.7". These require cross-document changelog retrieval that Top-5 vector search cannot handle.

**Hard bug discovered — chunks exceeding the embedding model's input limit:** `BAAI/bge-base-en-v1.5` has `max_seq_length` = 512 tokens. At 3200 characters (~800 tokens for English), SentenceTransformer silently truncates anything past the limit — roughly the last third of every chunk (~1200 characters) was never encoded. If the relevant fact happened to fall in the second half of a chunk, it was never in the index at all. This is not a ranking problem; it's missing data.

### Improvement Plan (priority order)

1. **Reduce chunk size (fix hard bug first):** Drop to ~1200 chars (~300 tokens, well within 512 limit), overlap ~150–200. Fixes both the truncation bug and semantic dilution. Requires rebuilding `chroma_db/`.
2. **Hybrid retrieval (BM25 + dense):** BM25 literal matching handles exact tokens (version numbers, `1–366`, rare terms) that dense search misses — directly addresses failure categories 1 and 2.
3. **CrossEncoder reranker:** Use Top K=10 as the recall pool, then rerank to Top 3–5. Directly improves Context Precision.
4. **Semantic chunking:** Cut at paragraph/heading boundaries rather than fixed character counts to further reduce fragmentation.

---

## Run #3 — Full Evaluation (2026-10-29, 400 samples, chunk 1500 + BM25 + Reranker)

| Parameter | Value |
|---|---|
| Sample count | 400 (full set, same eval_dataset as Run #2) |
| Top K | 5 |
| Generation / judge model | claude-sonnet-4-5 |
| Embedding | BAAI/bge-base-en-v1.5 |
| Chunk size | **1500 chars / 200 overlap** (~375 tokens — fixes the 512-token truncation bug from Run #2) |
| Retrieval strategy | **Dense (Chroma) + BM25 (rank_bm25) each retrieve 20 → RRF fusion Top 10 → CrossEncoder reranking Top 5** |
| Reranker | BAAI/bge-reranker-base (via ModelScope) |

| Metric | Mean | Change vs. Run #2 | Count < 0.5 | Count = 0 |
|---|---|---|---|---|
| Context Recall | **0.858** | +0.065 (↑8.2%) | 51 / 400 (13%) | 51 / 400 |
| Context Precision | **0.793** | +0.162 (↑25.7%) | 78 / 400 (20%) | — |
| Faithfulness | **0.919** | +0.010 (stable) | — | — |
| Answer Relevancy | **0.878** | +0.082 (↑10.3%) | — | — |

### Interpretation

**Context Recall (0.793 → 0.858):** recall=0 cases dropped from 77 to 51 (−34%). Smaller chunks fixed the silent truncation bug, and BM25 covered exact-token matches that dense search missed.

**Context Precision (0.631 → 0.793):** The largest gain — +0.162 (+25.7%). precision<0.5 cases dropped from 165 to 78 (−53%). The CrossEncoder reranker had the most impact here: rescoring the RRF Top 10 effectively filtered noisy chunks and pushed relevant ones higher.

**Faithfulness (0.909 → 0.919):** Essentially unchanged. Better retrieval means more questions have sufficient context, leaving Claude less need to improvise.

**Answer Relevancy (0.796 → 0.878):** Improved by 0.082, driven by the recall improvement. With 26 fewer recall=0 cases, fewer questions produce "insufficient information" non-answers, which previously scored 0 on answer relevancy by construction.

### Low-Score Case Analysis (51 remaining recall=0 cases)

The 34% reduction from Run #2's 77 cases reflects that the main failure modes then — rare term dilution and truncation — are now largely resolved. The remaining 51 cases show a different pattern:

1. **Constraint / negative-answer questions (~8 cases):** Questions of the form "Can I do X?" where the correct answer is a single prohibition ("No, you cannot…"). These constraints appear as a single sentence in the docs with no distinctive keywords to trigger BM25, and dense search doesn't surface short prohibitions reliably. Example: `Can I use views with input parameters as sources in a data flow?` → answer is "No" plus one explanatory sentence.

2. **Enumeration / exhaustive value lists (~10 cases):** Questions asking for all possible values (all deployment status states, all valid Calendar-Date formats, all four filter variable types). Ground truth is a complete enumeration, but these are scattered across multiple document sections, making single-pass Top-5 retrieval unlikely to cover all of them.

3. **Fine-grained UI behavior / edge-case features (~33 cases):** Questions about specific post-action behavior (what happens after clicking Auto-Map, what the Cancel Validation button does, the state of the export button after protection is enabled). These facts sit at very low density in large volumes of procedural documentation, with neither strong semantic signal nor enough keyword distinctiveness for BM25.

### Improvement Plan

1. **Semantic chunking (cut at paragraph / heading boundaries):** Fixed-character chunking still splits procedural descriptions mid-sentence or across chunk boundaries. Natural boundaries should reduce category-3 failures.
2. **Agentic loop / iterative retrieval:** For questions where one retrieval pass isn't enough, let the model self-assess whether retrieved context is sufficient and re-query with a rewritten query. Most useful for category-1 (constraint answers) and category-3 (procedural detail) failures.
3. **Eval set cleanup:** Some category-2 enumeration questions are inherently multi-chunk — the correct answer requires aggregating across multiple document sections. Removing these from the eval set would give a more accurate picture of single-pass retrieval quality.
