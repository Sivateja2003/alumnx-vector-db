from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from tests.helpers import MockPostgresStore, MockVectorFileStore


class FakeEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


@pytest.fixture()
def client(monkeypatch):
    cfg = SimpleNamespace(
        chunk_size=10,
        overlap_size=2,
        default_chunking_strategy="fixed_length",
        max_paragraph_size=20,
        knn_k=5,
        default_retrieval_strategy={"algorithm": "knn", "distance_metric": "cosine"},
        embedding_model="models/gemini-mock",
        output_dimensionality=3,
        vector_size=3,
        min_page_text_length=1,
        postgres_url="postgresql://mock",
    )

    pg = MockPostgresStore()
    vfs = MockVectorFileStore()

    monkeypatch.setattr("app.routers.ingest.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.ingestion.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.retrieval_service.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.embedding.embedder.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.ingestion.GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr("app.services.retrieval_service.GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr("app.services.ingestion.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.services.ingestion.VectorFileStore", lambda: vfs)
    monkeypatch.setattr("app.services.retrieval_service.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.services.retrieval_service.VectorFileStore", lambda: vfs)
    monkeypatch.setattr("app.routers.knowledgebases.PostgresStore", lambda: pg)
    monkeypatch.setattr(
        "app.services.ingestion.extract_pdf_pages",
        lambda _: [
            SimpleNamespace(page_number=1, text="alpha beta gamma delta epsilon"),
            SimpleNamespace(page_number=2, text="zeta eta theta iota kappa"),
        ],
    )

    yield TestClient(app)


def test_retrieval_strategies_lists_ann_as_unsupported(client):
    response = client.get("/retrieval-strategies")
    assert response.status_code == 200
    payload = response.json()
    assert any(item["name"] == "ann" for item in payload["algorithms"])


def test_ann_is_rejected_with_400_warning(client):
    response = client.post(
        "/retrieve",
        json={"query": "hello", "retrieval_strategy": {"algorithm": "ann", "distance_metric": "cosine"}},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "INVALID_RETRIEVAL_STRATEGY"
    assert "ann" in payload["message"].lower()


def test_ingest_creates_kb_and_retrieve_returns_results(client):
    ingest = client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Finance KB", "chunking_strategy": "fixed_length"},
    )
    assert ingest.status_code == 200
    payload = ingest.json()
    assert payload["kb_name"] == "finance_kb"

    retrieve = client.post("/retrieve", json={"query": "alpha", "kb_name": "finance_kb"})
    assert retrieve.status_code == 200
    result = retrieve.json()
    assert result["results"][0]["kb_name"] == "finance_kb"
    assert result["results"][0]["strategy_results"]


def test_ingest_uses_config_defaults_when_strategy_is_omitted(client):
    ingest = client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Defaults KB"},
    )
    assert ingest.status_code == 200
    payload = ingest.json()
    strategies = [item["strategy_name"] for item in payload["strategies_processed"]]
    assert strategies == ["fixed_length"]


def test_list_knowledgebases_returns_created_kb(client):
    client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Engineering KB", "chunking_strategy": "fixed_length"},
    )

    response = client.get("/knowledgebases")
    assert response.status_code == 200
    assert "engineering_kb" in response.json()["knowledgebases"]


def test_retrieve_can_exclude_embedding_vectors(client):
    client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Vectors KB", "chunking_strategy": "fixed_length"},
    )

    response = client.post(
        "/retrieve",
        json={"query": "alpha", "kb_name": "vectors_kb", "excludevectors": True},
    )
    assert response.status_code == 200
    payload = response.json()
    chunks = payload["results"][0]["strategy_results"][0]["chunks"]
    assert chunks
    assert "embedding_vector" not in chunks[0]
