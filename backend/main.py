from __future__ import annotations

import os
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

load_dotenv(Path(__file__).parent.parent / ".env")

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CLAUDE_MODEL = "claude-sonnet-4-5"

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
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("datasphere")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_embed_model()
    _get_collection()
    yield


app = FastAPI(title="DS-doc RAG API", version="0.1.0", lifespan=lifespan)


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
    top_k: int = 3


@app.post("/api/ask")
def ask(req: AskRequest):
    model = _get_embed_model()
    col = _get_collection()

    vec = model.encode(req.question, normalize_embeddings=True).tolist()
    results = col.query(query_embeddings=[vec], n_results=req.top_k)

    chunks = [
        {
            "text": doc,
            "deliverable": meta.get("deliverable", ""),
            "topic_title": meta.get("topic_title", ""),
            "source_file": meta.get("source_file", ""),
            "chunk_index": meta.get("chunk_index", 0),
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

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
