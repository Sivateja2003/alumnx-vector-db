# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv venv && uv sync

# Run locally (with auto-reload)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
uv run pytest -q

# Run a single test file
uv run pytest tests/test_api.py -q

# Run with Docker
docker-compose up --build
```

**Required environment variable:** `GOOGLE_API_KEY` in a `.env` file at the project root.

## Architecture

NexVec is a FastAPI-based RAG (Retrieval-Augmented Generation) service for PDF ingestion and semantic retrieval using Google Gemini embeddings.

### Core Pipelines

**Ingest (`POST /ingest`):** PDF upload → text extraction (pdfplumber) → chunking → Gemini embedding → normalise vectors → persist to JSONL store.

**Retrieve (`POST /retrieve`):** Query text → Gemini embedding → load JSONL store → KNN search per chunk/model group → return top-k results with similarity scores.

### Key Components

| Layer                | Location                             | Purpose                                                                      |
| -------------------- | ------------------------------------ | ---------------------------------------------------------------------------- |
| API routes           | `app/routers/`                       | Thin HTTP handlers delegating to services                                    |
| Ingestion pipeline   | `app/services/ingestion.py`          | Orchestrates extract → chunk → embed → store                                 |
| Retrieval service    | `app/services/retrieval_service.py`  | Orchestrates embed → load → KNN search                                       |
| Embedder             | `app/services/embedding/embedder.py` | Google Gemini via `langchain-google-genai`, batched in groups of 100         |
| Chunking strategies  | `app/services/chunking/`             | `fixed_length`, `paragraph` (NLTK), or `both` — registered via `registry.py` |
| Retrieval strategies | `app/services/retrieval/`            | KNN with cosine or dot-product distance — registered via `registry.py`       |
| Vector store         | `app/services/store/jsonl_store.py`  | One JSONL file per knowledge base in `./vector_store/`                       |
| Config               | `config.yaml` + `app/config.py`      | Chunk size, overlap, embedding model, KNN k, vector dimensions               |

### Design Patterns

- **Registry pattern** for chunking and retrieval strategies — extend by adding a class and registering it.
- **JSONL storage** (Phase 1 simplicity) — one file per knowledge base, each line is a JSON object with chunk text, embedding vector, and metadata.
- Vectors are **L2-normalised** before storage so cosine similarity reduces to a dot product.
- Retrieval groups chunks by `(chunking_strategy, embedding_model)` before running KNN.

### Configuration (`config.yaml`)

Key defaults: `chunk_size: 500`, `overlap_size: 50`, `default_chunking_strategy: both`, `embedding_model: models/gemini-embedding-001`, `output_dimensionality: 3072`, `vector_store_path: ./vector_store/`.

## Testing

Tests live in `tests/` with fixtures in `conftest.py`. Tests mock the Google Gemini API and use temp directories for the vector store. Run the full suite with `uv run pytest -q` before raising a PR.

## Deployment

CI/CD runs on push to `main` via `.github/workflows/deploy.yml` — SSH into AWS EC2, pull code, and restart with PM2. Never push directly to `main`.
