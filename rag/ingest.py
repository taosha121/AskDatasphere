"""
向量数据库构建脚本。

扫描 html_docs/ 下所有 HTML topic 文件以及 pdfs/ 下所有 PDF 文件，提取纯文本，
切块后生成向量并写入 Chroma 本地数据库。只需运行一次；重复运行会跳过已存在的
chunk（幂等）。

运行方式：
    python -m rag.ingest
"""
import re
from pathlib import Path

import fitz  # pymupdf
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

# html_docs/、pdfs/ 和 chroma_db/ 都在项目根目录，相对于本文件向上两级
HTML_DIR   = Path(__file__).parent.parent / "html_docs"
PDF_DIR    = Path(__file__).parent.parent / "pdfs"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# ModelScope / HuggingFace 上的 embedding 模型名称
MODEL_NAME    = "BAAI/bge-base-en-v1.5"

# 每个 chunk 最大字符数（约 800 token）
CHUNK_CHARS   = 3200
# 相邻 chunk 之间的重叠字符数，避免语义在边界处被硬截断
OVERLAP_CHARS = 400


def _load_model() -> SentenceTransformer:
    """
    加载 embedding 模型。

    优先从 ModelScope 下载（国内网络友好），失败则直接用模型名让
    sentence-transformers 自己去 HuggingFace 拉取。
    """
    try:
        from modelscope import snapshot_download
        local = snapshot_download(MODEL_NAME)
        print(f"ModelScope model: {local}")
        return SentenceTransformer(local)
    except Exception as e:
        print(f"ModelScope unavailable ({e}), loading directly")
        return SentenceTransformer(MODEL_NAME)


def _extract_text(html_path: Path) -> tuple[str, str]:
    """
    从 HTML topic 文件中提取 (标题, 正文纯文本)。

    SAP Help Portal 的 topic 文件带有大量 <script>/<style>/<nav> 噪音，
    需要先删除再提取文本，否则向量化质量很差。
    连续空白压缩为段落分隔符，保留语义结构。
    """
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    # 优先取 <title>，没有则取第一个 <h1>
    title = ""
    if soup.title:
        title = soup.title.string or ""
    if not title and soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)

    # 删除不含正文内容的标签
    for tag in soup(["script", "style", "nav", "head"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    # 将 3 个及以上连续空白字符替换为段落换行，减少无意义空格
    text = re.sub(r"\s{3,}", "\n\n", text)
    return title.strip(), text.strip()


def _extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    """
    从 PDF 文件中提取 (标题, 正文纯文本)。

    逐页提取文本后拼接，优先使用 PDF 内嵌的 title 元数据作为标题，
    没有则用文件名。
    """
    doc = fitz.open(pdf_path)

    title = (doc.metadata.get("title") or "").strip()
    if not title:
        title = pdf_path.stem

    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    text = "\n\n".join(pages)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return title, text.strip()


def _chunk(text: str) -> list[str]:
    """
    将长文本按固定窗口切成多个 chunk，相邻 chunk 有 OVERLAP_CHARS 字符重叠。

    重叠的目的：防止一句话被切在两个 chunk 的边界，导致检索时语义不完整。
    过短的片段（<100 字符）直接丢弃，通常是页眉页脚等噪音。
    """
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + CHUNK_CHARS].strip()
        if len(piece) > 100:
            chunks.append(piece)
        start += CHUNK_CHARS - OVERLAP_CHARS
    return chunks


def build():
    """
    主流程：遍历所有 HTML 文件 → 提取文本 → 切块 → embed → 写入 Chroma。

    Chroma collection 使用余弦距离（cosine），与 bge 模型的归一化向量配合
    效果最好（归一化后余弦距离等价于点积，便于排序）。

    每个 chunk 存储的 metadata：
      - deliverable  : 所属文档目录名（如 Administering_SAP_Datasphere）
      - topic_title  : HTML 文件的标题
      - source_file  : 相对于项目根的文件路径，方便溯源
      - chunk_index  : 该 chunk 在文件内的序号
    """
    print("Loading embedding model…")
    model = _load_model()

    # PersistentClient 会把数据持久化到 chroma_db/ 目录，进程退出后不丢失
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        name="datasphere",
        metadata={"hnsw:space": "cosine"},
    )

    html_files = sorted(HTML_DIR.rglob("*.html"))
    print(f"Found {len(html_files)} HTML topic files")

    added = skipped = 0
    for html_path in html_files:
        # 父目录名即文档 deliverable 名称（由 download_html_docs.py 写入时命名）
        deliverable = html_path.parent.name
        title, text = _extract_text(html_path)
        if not text:
            continue

        chunks = _chunk(text)
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{html_path.stem}__{i}"

            if col.get(ids=[chunk_id])["ids"]:
                skipped += 1
                continue

            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()
            col.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[{
                    "deliverable": deliverable,
                    "topic_title": title,
                    "source_file": str(html_path.relative_to(HTML_DIR.parent)),
                    "chunk_index": i,
                }],
            )
            added += 1

        print(f"  {deliverable[:45]:<45} {len(chunks)} chunks")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files")

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
                    "deliverable": "pdf",
                    "topic_title": title,
                    "source_file": str(pdf_path.relative_to(PDF_DIR.parent)),
                    "chunk_index": i,
                }],
            )
            added += 1

        print(f"  {title[:45]:<45} {len(chunks)} chunks")

    print(f"\nDone. Added: {added}  Skipped: {skipped}  Total: {col.count()}")


if __name__ == "__main__":
    build()
