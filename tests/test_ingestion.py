from __future__ import annotations
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np

# Core services to mock and use
from app.services.ingestion import ingest_file as ingest_pdf
from app.models import IngestResponse

class MockEmbedder:
    """Mock the embedding service; return arbitrary vectors based on text length."""
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Return a simple vector of length 3 (per our fake config)
        return [[float(len(t)), 0.5, 0.0] for t in texts]

@pytest.fixture
def mock_ingest_deps(tmp_path, monkeypatch):
    """Set up mocks for ingestion dependencies."""
    # Mock config
    cfg = SimpleNamespace(
        vector_store_path=tmp_path / "vector_store",
        chunk_size=10,
        overlap_size=2,
        embedding_model="models/gemini-mock",
        vector_size=3,
        default_chunking_strategy="fixed_length",
        postgres_url="postgresql://user:pass@localhost/db"
    )
    monkeypatch.setattr("app.services.ingestion.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.store.vector_file_store.get_config", lambda: cfg)
    
    # Mock Embedder
    monkeypatch.setattr("app.services.ingestion.GeminiEmbedder", MockEmbedder)
    
    # Mock PDF Extractor
    pages = [SimpleNamespace(page_number=1, text="The fox jump.")]
    monkeypatch.setattr("app.services.ingestion.extract_pdf_pages", lambda _: pages)
    
    # Mock Postgres - we'll capture its calls
    mock_pg = MagicMock()
    mock_pg.has_active_chunks.return_value = False
    monkeypatch.setattr("app.services.ingestion.PostgresStore", lambda: mock_pg)
    
    # Mock VectorFileStore - we'll capture its calls
    mock_vfs = MagicMock()
    monkeypatch.setattr("app.services.ingestion.VectorFileStore", lambda: mock_vfs)
    
    return SimpleNamespace(cfg=cfg, pg=mock_pg, vfs=mock_vfs)

def test_ingest_pdf_success_flow(mock_ingest_deps):
    """Test full ingestion process calls both stores."""
    response = ingest_pdf(
        file_name="test.pdf", file_path="m", kb_name="test_kb",
        chunking_strategy="fixed_length", chunk_size=None, overlap_size=None, 
        embedding_model=None, overwrite=False
    )
    
    # Check Postgres calls
    assert mock_ingest_deps.pg.insert_chunks.called
    # Check VectorFileStore calls
    assert mock_ingest_deps.vfs.append.called
    
    assert isinstance(response, IngestResponse)
    assert response.kb_name == "test_kb"

def test_ingest_pdf_duplicate_checks_pg(mock_ingest_deps):
    """Verify that duplicate checks go through PostgresStore."""
    mock_ingest_deps.pg.has_active_chunks.return_value = True
    
    with pytest.raises(FileExistsError):
        ingest_pdf(
            file_name="dup.pdf", file_path="m", kb_name="kb",
            chunking_strategy="fixed_length", chunk_size=None, overlap_size=None, 
            embedding_model=None, overwrite=False
        )

def test_ingest_pdf_overwrite_cleanup(mock_ingest_deps):
    """Test that overwrite cleans up both stores."""
    mock_ingest_deps.pg.has_active_chunks.return_value = True
    mock_ingest_deps.pg.deactivate_chunks.return_value = ["old-id-1"]
    
    ingest_pdf(
        file_name="v.pdf", file_path="m", kb_name="k",
        chunking_strategy="fixed_length", chunk_size=None, overlap_size=None, 
        embedding_model=None, overwrite=True
    )
    
    # Verify both were cleaned
    assert mock_ingest_deps.pg.deactivate_chunks.called
    mock_ingest_deps.vfs.remove_chunk_ids.assert_called_with("nex_vec", {"old-id-1"})
