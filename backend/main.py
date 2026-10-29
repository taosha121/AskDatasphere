from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

load_dotenv(Path(__file__).parent.parent / ".env")

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CLAUDE_MODEL = "claude-sonnet-4-5"

# 混合检索：两路各召回 CANDIDATES 条候选，经 RRF 融合后取 top_k
_CANDIDATES = 20
_RRF_K = 60             # 标准 RRF 常数，越大越平滑两路权重
_RERANK_CANDIDATES = 10 # RRF 后送入 cross-encoder 精排的候选数

# BM25 索引（启动时构建，常驻内存）
_bm25_index: BM25Okapi | None = None
_bm25_docs: list[dict] = []       # [{"id": str, "text": str, "meta": dict}, ...]
_bm25_by_id: dict[str, dict] = {} # id → doc dict，供 RRF 后查元数据

RERANKER_NAME = "BAAI/bge-reranker-base"

SYSTEM_PROMPT = (
    "You are a helpful assistant for SAP Datasphere. "
    "Answer ONLY based on the documentation excerpts provided by the user. "
    "Do not include any information that is not explicitly stated in those excerpts. "
    "If the excerpts do not contain sufficient information to answer the question, "
    "respond with: \"The provided excerpts do not contain sufficient information to answer this question.\" "
    "Do not speculate, infer, or supplement with your own knowledge."
)


@lru_cache(maxsize=1)
def _get_embed_model():
    try:
        from modelscope import snapshot_download
        local = snapshot_download(MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(local)
    except Exception:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    try:
        from modelscope import snapshot_download
        local = snapshot_download(RERANKER_NAME)
    except ImportError:
        local = RERANKER_NAME
    return CrossEncoder(local)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("datasphere")


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _build_bm25_index() -> None:
    # Chroma 全量拉取，在内存里建 BM25 倒排索引
    global _bm25_index, _bm25_docs, _bm25_by_id
    result = _get_collection().get(include=["documents", "metadatas"])
    _bm25_docs = [
        {"id": id_, "text": doc, "meta": meta}
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    _bm25_index = BM25Okapi([_tokenize(d["text"]) for d in _bm25_docs])
    _bm25_by_id = {d["id"]: d for d in _bm25_docs}
    print(f"BM25 index built: {len(_bm25_docs)} docs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_embed_model()
    _get_reranker()
    _get_collection()
    _build_bm25_index()
    yield


app = FastAPI(title="AskDatasphere API", version="0.1.0", lifespan=lifespan)


@app.get("/api/chunks")
def list_chunks(
    offset: int = Query(default=0, ge=0),
    limit:  int = Query(default=20, ge=1, le=100),
):
    col = _get_collection()
    total = col.count()

    if total > 0 and offset >= total:
        raise HTTPException(status_code=400, detail=f"offset {offset} exceeds total {total}")

    result = col.get(
        limit=limit,
        offset=offset,
        include=["documents", "metadatas"],
    )

    items = [
        {
            "id":           chunk_id,
            "text":         doc,
            "deliverable":  meta.get("deliverable", ""),
            "topic_title":  meta.get("topic_title", ""),
            "source_file":  meta.get("source_file", ""),
            "chunk_index":  meta.get("chunk_index", 0),
        }
        for chunk_id, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]

    return {"total": total, "offset": offset, "limit": limit, "items": items}


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/api/ask")
def ask(req: AskRequest):
    model = _get_embed_model()
    col = _get_collection()

    # --- 向量检索（dense path）---
    vec = model.encode(req.question, normalize_embeddings=True).tolist()
    dense_res = col.query(
        query_embeddings=[vec],
        n_results=_CANDIDATES,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense_res["ids"][0]
    dense_by_id = {
        id_: {"text": doc, "meta": meta, "dist": dist}
        for id_, doc, meta, dist in zip(
            dense_ids,
            dense_res["documents"][0],
            dense_res["metadatas"][0],
            dense_res["distances"][0],
        )
    }

    # --- BM25 检索（sparse path）---
    tokens = _tokenize(req.question)
    if tokens and _bm25_index is not None:
        bm25_scores = _bm25_index.get_scores(tokens)
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:_CANDIDATES]
        bm25_ids = [_bm25_docs[i]["id"] for i in bm25_top_idx]
    else:
        bm25_ids = []

    # --- RRF 融合 ---
    rrf_scores: dict[str, float] = {}
    for rank, id_ in enumerate(dense_ids):
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, id_ in enumerate(bm25_ids):
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)

    top_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:_RERANK_CANDIDATES]

    # --- Reranker 精排（cross-encoder 对候选重新打分，取 top_k）---
    reranker = _get_reranker()
    pairs = [
        (req.question, _bm25_by_id[id_]["text"] if id_ not in dense_by_id else dense_by_id[id_]["text"])
        for id_ in top_ids
    ]
    rerank_scores = reranker.predict(pairs)
    top_ids = [id_ for _, id_ in sorted(zip(rerank_scores, top_ids), reverse=True)][:req.top_k]

    # --- 组装 chunk 列表（优先用 dense 的余弦相似度做 score 展示）---
    chunks = []
    for id_ in top_ids:
        if id_ in dense_by_id:
            d = dense_by_id[id_]
            text, meta = d["text"], d["meta"]
            score = round(1 - d["dist"], 4)
        else:
            d = _bm25_by_id[id_]
            text, meta = d["text"], d["meta"]
            # BM25-only 命中：用归一化后的 RRF 分数展示
            score = round(rrf_scores[id_] * (_RRF_K + 1), 4)
        chunks.append({
            "text": text,
            "deliverable": meta.get("deliverable", ""),
            "topic_title": meta.get("topic_title", ""),
            "source_file": meta.get("source_file", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "score": score,
        })

    context_parts = [
        f"[{i+1}] {c['deliverable']}\n{c['text']}" for i, c in enumerate(chunks)
    ]
    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        f"Documentation excerpts:\n\n{context}\n\n"
        f"Question: {req.question}\n\n"
        f"Instructions:\n"
        f"- Answer using ONLY the excerpts above.\n"
        f"- After each statement, cite the source with its number, e.g. [1] or [2].\n"
        f"- Do not add any information not explicitly present in the excerpts.\n\n"
        f"Answer:"
    )

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = message.content[0].text

    return {"answer": answer, "chunks": chunks, "prompt": prompt}


@app.get("/")
def root():
    return RedirectResponse(url="/ui/index.html")


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
