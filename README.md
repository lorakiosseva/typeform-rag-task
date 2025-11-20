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

## How to run the project locally without Docker:

Clone the repo, create and activate a virtual environent, install dependencies from the requirements file

```pip install -r requirements.txt```

Crete your own .env file, using the .env.example and add your own API keys.

```cp .env.example .env```

Do a one-time ingestion to populate pinecone. This:
- Loads the local HTML snapshots from data/raw/
- Extracts article content and cleans it
- Chunks the text
- Embeds each chunk
- Upserts all vectors into Pinecone

In the project root dir run: 

```python -c "from app.ingest import run_ingestion_once; run_ingestion_once()"```

You should see print statements stating that 2 articles have been loaded, 14 chunks have been build, and an upsert response.

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


## Summary of design decisions:

1. Ingestion strategy

Input format: Local HTML snapshots of the provided Help Center articles, placed under data/raw/. Simple and easy for the sake of the task. Tried with http request, got 403 forbidden. For actual implementation it would need a way to keep up with updates of the documention (e.g. scheduled pulling of data).

Extraction: BeautifulSoup used to parse HTML and identify the \<h1> as the article title. Extract the main article content (paragraphs, subheadings, list items). Stop at typical footer sections like “Was this article helpful?” and “Related articles”.
Convert subheadings into a markdown-like format (##, ###) to preserve structure.

IDs and metadata: Article and chunk IDs are normalized to be safe for Pinecone (ASCII-only, kebab-case, because I ran into an issue with an emdash in the title)

Each chunk stores article_id, title, chunk_index, and source_path in metadata.

2. Chunking & embeddings

Chunking: Character-based chunking with max_chars=1200 and overlap=200. This approximates ~300 tokens per chunk while keeping overlap for continuity.

The chunk size is a trade-off: Small enough to keep retrieval precise and avoid mixing unrelated sections. Large enough for each chunk to contain a meaningful piece of the documentation (e.g. one step or sub-section).

Embeddings: All chunks are embedded with text-embedding-3-small and stored in Pinecone. The same model is used to embed user queries at retrieval time, ensuring embedding space consistency.
This model was chosen because it has a strong sematic retrieval performance, low cost, integrates well with pinecone. 

3. Retrieval & RAG prompting

Semantic search: The user query is embedded and used to query Pinecone (top_k configurable, default 5, looks like 3–5 is standard for help-center RAG systems; 5 is the safer default during prototyping; relevant content captured, but stays within the context window). Matches include both vector similarity score and stored metadata.

Prompt construction: Retrieved chunks are concatenated into a “context” string with section boundaries and titles. The system prompt enforces: Use only the provided context. If the answer is not present, say “I don’t know”.

LLM: gpt-4o-mini is used for generation. temperature=0.1 to make answers deterministic and grounded.

4. API design

GET /health: Lightweight health check and quick Pinecone connectivity sanity check.

POST /ask_question:

Input: ```{"query": "...", "top_k": 5}```

Output: ```{"answer": "...", "sources": [...] }```

Sources expose:

``` 
id
score
text
article_id
title
chunk_index
```
Easy to debug and understand why a particular answer was produced.


5. Evaluation & Reliability

For the completion of the task quality was measured "manually" by observing retrieval relevance,
groundedness (answer must come directly from retrieved text) and response clarity. Tried a few questions - answers were relevant and concise or answered "I don't know" when it should. 

Must implement better evaluation. Suggestions:
- Create a small labeled test set of “question and expected passage” pairs to measure Recall@k.
- Track groundedness automatically by checking whether each sentence in the answer is supported by retrieved chunks.
- Add user-facing feedback signals (thumbs up/down) to capture real usage quality.
- Introduce confidence scoring based on retrieval similarity and LLM uncertainty.
- Monitor observability metrics like latency, token usage, retrieval failures, and hallucination rate in a dashboard.
- Run robustness tests (paraphrases, ambiguous queries, multilingual inputs) to stress-test reliability.