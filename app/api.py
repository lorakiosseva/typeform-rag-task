from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pinecone import Pinecone

from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from app.rag import retrieve, answer_with_rag


pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

app = FastAPI(
    title="Typeform Help Center RAG API",
    description="RAG chatbot over Typeform Help Center articles",
    version="0.1.0",
)


# ---------- Pydantic models ----------

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class ChunkMatch(BaseModel):
    id: str
    score: float
    text: str
    article_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    chunk_index: Optional[int] = None
        
# class ChunkMatch(BaseModel):
#     id: str
#     score: float
#     text: str
#     article_id: str | None = None
#     title: str | None = None
#     url: str | None = None
#     chunk_index: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChunkMatch]


# ---------- Endpoints ----------

@app.get("/health")
def health():
    """
    Simple health check: confirms we can talk to Pinecone.
    """
    try:
        _ = pinecone_client.list_indexes()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_question", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """
    RAG-powered chat endpoint:
    - Embeds the query
    - Retrieves top_k chunks from Pinecone
    - Generates an answer with GPT
    - Returns answer + sources
    """
    try:
        matches = retrieve(request.query, top_k=request.top_k)
        if not matches:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found in the knowledge base.",
            )

        answer = answer_with_rag(request.query, top_k=request.top_k)

        sources: List[ChunkMatch] = []
        for m in matches:
            md = m["metadata"]
            sources.append(
                ChunkMatch(
                    id=m["id"],
                    score=m["score"],
                    text=md.get("text", ""),
                    article_id=md.get("article_id"),
                    title=md.get("title"),
                    url=md.get("url"),
                    chunk_index=md.get("chunk_index"),
                )
            )

        return ChatResponse(answer=answer, sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
