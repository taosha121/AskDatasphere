"""
向量数据库构建脚本。

扫描 pdfs/ 下所有 PDF 文件，提取纯文本，切块后生成向量并写入 Chroma 本地数据库。
只需运行一次；重复运行会跳过已存在的 chunk（幂等）。

运行方式：
    python -m rag.ingest
"""
import re
from pathlib import Path

import fitz  # pymupdf
import chromadb
from sentence_transformers import SentenceTransformer

PDF_DIR    = Path(__file__).parent.parent / "pdfs"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

MODEL_NAME    = "BAAI/bge-base-en-v1.5"
CHUNK_CHARS   = 1500   # ~375 token，留足 bge-base 512 token 上限余量（3200≈800 token 会被尾部截断）
OVERLAP_CHARS = 200


def _load_model() -> SentenceTransformer:
    try:
        from modelscope import snapshot_download
        local = snapshot_download(MODEL_NAME)
        print(f"ModelScope model: {local}")
        return SentenceTransformer(local)
    except Exception as e:
        print(f"ModelScope unavailable ({e}), loading directly")
        return SentenceTransformer(MODEL_NAME)


def _extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    doc = fitz.open(pdf_path)

    # 优先 PDF 内嵌 metadata title，没有则从第一页正文提取
    title = (doc.metadata.get("title") or "").strip()
    if not title:
        first_page = doc[0].get_text() if len(doc) > 0 else ""
        lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
        title_lines, collecting = [], False
        for line in lines:
            if re.search(r"Document Version", line, re.IGNORECASE):
                collecting = True
                continue
            if collecting:
                if line.startswith("©"):
                    break
                title_lines.append(line)
        title = " ".join(title_lines).strip() or pdf_path.stem

    pages = [page.get_text() for page in doc]
    doc.close()

    text = "\n\n".join(pages)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return title, text.strip()


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + CHUNK_CHARS].strip()
        if len(piece) > 100:
            chunks.append(piece)
        start += CHUNK_CHARS - OVERLAP_CHARS
    return chunks


def build():
    print("Loading embedding model…")
    model = _load_model()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        name="datasphere",
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")

    added = skipped = 0
    for pdf_path in pdf_files:
        title, text = _extract_text_from_pdf(pdf_path)
        if not text:
            continue

        chunks = _chunk(text)
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"pdf__{pdf_path.stem}__{i}"

            if col.get(ids=[chunk_id])["ids"]:
                skipped += 1
                continue

            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()
            col.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[{
                    "deliverable": title,
                    "topic_title": title,
                    "source_file": str(pdf_path.relative_to(PDF_DIR.parent)),
                    "chunk_index": i,
                }],
            )
            added += 1

        print(f"  {title[:50]:<50} {len(chunks)} chunks")

    print(f"\nDone. Added: {added}  Skipped: {skipped}  Total: {col.count()}")


if __name__ == "__main__":
    build()
