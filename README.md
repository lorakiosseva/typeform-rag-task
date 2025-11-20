# Typeform Help Center RAG Prototype

This project is a small end-to-end Retrieval-Augmented Generation (RAG) prototype for a Typeform Help Center chatbot. It:

- Ingests a fixed set of Typeform Help Center articles (provided as local HTML snapshots)
- Cleans and chunks their content
- Embeds chunks with an embedding model
- Stores embeddings in a Pinecone vector index
- Retrieves relevant chunks given a user query
- Uses an LLM to generate answers grounded in the retrieved context
- Exposes a small FastAPI service with a single `/ask_question` endpoint

The solution is not containerised wiht Docker due to config issues and time consrtaints


## Project structure: 

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py         # env vars (OpenAI, Pinecone)
│   ├── ingest.py         # HTML; cleaned articles; chunks; embeddings; Pinecone
│   ├── rag.py            # retrieval + answer generation
│   └── api.py            # FastAPI app exposing /health to check connection and /ask_question to ask your Qs
│
├── data/
│   └── raw/              # local HTML snapshots of the 2 Help Center articles
│
├── notebooks/
│   └── exploration.ipynb # where the initial exploration and testing happened, 
                          # before extracting the code in cleaner files in app/
│
├── .env                  # real secrets (NOT committed)
├── .env.example          # template of required env vars
├── requirements.txt      # minimal requirements necessary to run the project
└── Dockerfile            # and attempt at one

```

How to run the project locally without Docker:

Crete your own .env file, using the .env.example and putting your own API keys.

```cp .env.example .env```

Do a one-time ingestion to populate pinecone. This:
- Loads the local HTML snapshots from data/raw/
- Extracts article content and cleans it
- Chunks the text
- Embeds each chunk
- Upserts all vectors into Pinecone

You can run: 

```python -c "from app.ingest import run_ingestion_once; run_ingestion_once()"```

You should see print statements stating that 2 articleas have been loaded, 14 chunks have been build and an upsert response

Then you can start the FastAPI app. From the project root run: 

```uvicorn app.api:app --reload --port 8000```

And then you can open http://127.0.0.1:8000/docs to play with POST/ask_question


Example request:

```
{
  "query": "How can I create a multi language form in Typeform?",
  "top_k": 5
}
```

