from pathlib import Path
import re
import unicodedata
from typing import List, Dict

from bs4 import BeautifulSoup
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME


# ---------- ID utilities ----------

def make_pinecone_id(raw: str, max_len: int = 64) -> str:
    """
    Turn an arbitrary string into a safe Pinecone ID:
    - Normalize unicode
    - Strip non-ASCII
    - Keep only letters, digits, hyphen, underscore
    - Collapse repeats and trim length
    """
    raw_norm = unicodedata.normalize("NFKD", raw)
    raw_ascii = raw_norm.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_ascii)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    safe = safe.lower()[:max_len] or "id"
    return safe


# ---------- HTML loading & extraction ----------

STOP_HEADINGS = {
    "was this article helpful?",
    "related articles",
}


def load_html_docs(root_dir: str = "../data/raw"):
    docs = []
    for path in Path(root_dir).glob("*.html"):
        with path.open("r", encoding="utf-8") as f:
            docs.append(
                {
                    "id": path.stem,
                    "source_path": str(path),
                    "html": f.read(),
                }
            )
    return docs

# def load_html_docs(raw_dir: str = "../data/raw") -> List[Dict]:
#     """
#     Load local HTML snapshots of help center articles.
#     """
#     docs: List[Dict] = []
#     for path in Path(raw_dir).glob("*.html"):
#         with path.open(encoding="utf-8") as f:
#             html = f.read()
#         docs.append({"source_path": str(path), "html": html})
#     return docs


def extract_article_from_html(html: str, source_path: str) -> Dict:
    """
    Parse a Typeform Help Center HTML snapshot and extract:
    - title (h1)
    - main article content (headings, paragraphs, list items)
    - metadata
    """
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main") or soup.body or soup
    title_tag = main.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else source_path

    content_parts: List[str] = []
    if title_tag:
        for el in title_tag.find_all_next():
            # Stop when we hit typical footer headings
            if el.name in ("h2", "h3", "h4"):
                heading_text = el.get_text(" ", strip=True).lower()
                if any(stop in heading_text for stop in STOP_HEADINGS):
                    break

            if el.name in ("p", "h2", "h3", "li"):
                text = el.get_text(" ", strip=True)
                if not text:
                    continue
                if el.name in ("h2", "h3"):
                    level = 2 if el.name == "h2" else 3
                    content_parts.append("#" * level + " " + text)
                elif el.name == "li":
                    content_parts.append(f"- {text}")
                else:
                    content_parts.append(text)

    content = "\n\n".join(content_parts)

    # Use title (or filename) as a base for article_id
    raw_id = title or Path(source_path).stem
    article_id = make_pinecone_id(raw_id)

    return {
        "id": article_id,
        "title": title,
        "content": content,
        "metadata": {
            "source": "typeform_help_center_snapshot",
            "source_path": source_path,
        },
    }


# ---------- Chunking ----------

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunker with overlap.
    - max_chars: max characters per chunk
    - overlap: chars to overlap between chunks
    """
    if not text:
        return []

    if overlap >= max_chars:
        overlap = max_chars // 4  # defensive

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks


def build_chunks(articles: List[Dict]) -> List[Dict]:
    """
    Build chunk dicts from extracted articles.
    Each chunk gets:
    - id (safe for Pinecone)
    - text
    - metadata (article_id, title, chunk_index, etc.)
    """
    chunks: List[Dict] = []

    for art in articles:
        article_id = art["id"]
        article_title = art["title"]
        article_meta = art["metadata"]
        article_content = art["content"]

        article_chunks = chunk_text(article_content, max_chars=1200, overlap=200)

        for i, chunk_text_ in enumerate(article_chunks):
            chunk_raw_id = f"{article_id}-chunk-{i}"
            chunk_id = make_pinecone_id(chunk_raw_id, max_len=96)

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text_,
                    "metadata": {
                        **article_meta,
                        "article_id": article_id,
                        "title": article_title,
                        "chunk_index": i,
                        # "url": ...  # if you later add URLs
                    },
                }
            )

    return chunks


# ---------- Embeddings & Pinecone upsert ----------

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def get_index():
    """
    Get (and lazily create) the Pinecone index used by this app.
    """
    existing = [idx.name for idx in pinecone_client.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pinecone_client.Index(PINECONE_INDEX_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using text-embedding-3-small.
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in resp.data]


def upsert_chunks(chunks: List[Dict]):
    """
    Upsert chunk vectors into Pinecone with metadata.
    """
    index = get_index()
    vectors = [
        {
            "id": c["id"],
            "values": c["embedding"],
            "metadata": {
                "text": c["text"],
                **c["metadata"],
            },
        }
        for c in chunks
    ]
    return index.upsert(vectors=vectors)


def run_ingestion_once() -> None:
    """
    End-to-end ingestion:
    - Load HTML snapshots
    - Extract articles
    - Chunk contents
    - Embed chunks
    - Upsert to Pinecone
    """
    html_docs = load_html_docs()
    articles = [
        extract_article_from_html(doc["html"], doc["source_path"])
        for doc in html_docs
    ]
    print(f"Loaded {len(articles)} articles from HTML snapshots.")

    chunks = build_chunks(articles)
    print(f"Built {len(chunks)} chunks from articles.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    for c, vec in zip(chunks, embeddings):
        c["embedding"] = vec

    upsert_resp = upsert_chunks(chunks)
    print(f"Ingestion completed. Upsert response: {upsert_resp}")
