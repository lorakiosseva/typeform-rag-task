from typing import List

from openai import OpenAI
from pinecone import Pinecone

from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)


# ---------- Embedding & retrieval ----------

def embed_query(query: str) -> List[float]:
    """
    Embed a user query using the same embedding model as the corpus.
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    return resp.data[0].embedding


def retrieve(query: str, top_k: int = 5):
    """
    Semantic search in Pinecone over help center chunks.
    Returns the raw Pinecone matches list.
    """
    query_vec = embed_query(query)
    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
    )
    # Pinecone client returns an object or dict-like response
    return res["matches"]


# ---------- Context building & LLM answer ----------

def build_context_from_matches(matches) -> str:
    """
    Concatenate retrieved chunks into a single context string
    for the LLM prompt.
    """
    parts = []
    for i, m in enumerate(matches):
        md = m["metadata"]
        chunk_text = md.get("text", "")
        title = md.get("title", "")
        parts.append(
            f"[{i}] Title: {title}\nChunk ID: {m['id']}\nContent:\n{chunk_text}"
        )
    return "\n\n---\n\n".join(parts)


def answer_with_rag(query: str, top_k: int = 5) -> str:
    """
    End-to-end RAG answer:
    - retrieve relevant chunks from Pinecone
    - build a context string
    - call GPT model to generate an answer grounded in context
    """
    matches = retrieve(query, top_k=top_k)
    if not matches:
        return (
            "I couldn't find any relevant information "
            "in the current set of help center articles."
        )

    context = build_context_from_matches(matches)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Typeform support assistant. "
                "Use ONLY the provided context to answer the user's question. "
                "If the answer is not in the context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content
