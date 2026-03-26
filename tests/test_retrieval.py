from __future__ import annotations
import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.retrieval_service import retrieve_documents
from app.models import RetrieveRequest, RetrieveResponse

class MockEmbedder:
    """Mock embedder to keep things fast."""
    def __init__(self, model): pass
    def embed_query(self, text): return [1.0, 1.0, 0.0]

@pytest.fixture
def mock_retrieval_deps(tmp_path, monkeypatch):
    """Mocks for the new retrieval logic (PG + VFS)."""
    cfg = SimpleNamespace(
        vector_store_path=tmp_path / "vector_store",
        knn_k=5,
        default_retrieval_strategy={"algorithm": "knn", "distance_metric": "cosine"},
        postgres_url="postgresql://user:pass@localhost/db"
    )
    monkeypatch.setattr("app.services.retrieval_service.get_config", lambda: cfg)
    
    # Mock Embedder
    monkeypatch.setattr("app.services.retrieval_service.GeminiEmbedder", MockEmbedder)
    
    # Mock Stores
    mock_pg = MagicMock()
    mock_vfs = MagicMock()
    monkeypatch.setattr("app.services.retrieval_service.PostgresStore", lambda: mock_pg)
    monkeypatch.setattr("app.services.retrieval_service.VectorFileStore", lambda: mock_vfs)
    
    return SimpleNamespace(cfg=cfg, pg=mock_pg, vfs=mock_vfs)

def test_retrieve_documents_workflow(mock_retrieval_deps):
    """Test full retrieval: PG lookup -> Vector Search -> PG Enrichment."""
    # 1. PG says we have active chunks
    mock_retrieval_deps.pg.kb_exists.return_value = True
    mock_retrieval_deps.pg.get_active_group_index.return_value = [
        {"chunk_id": "c1", "chunking_strategy": "fixed_length", "embedding_model": "m1"}
    ]
    
    # 2. VFS provides vectors
    mock_retrieval_deps.vfs.read.return_value = (
        np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        ["c1"]
    )
    
    # 3. Mock KNN Retriever (returned by registry)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [("c1", 1.0)]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("app.services.retrieval_service.get_retriever_registry", 
                       lambda: {"knn": mock_retriever})
    
    # 4. PG provides final metadata
    mock_retrieval_deps.pg.get_chunks_by_ids.return_value = [
        {"chunk_id": "c1", "chunk_text": "hello test", "source_filename": "f.pdf", 
         "chunking_strategy": "fixed_length", "chunk_index": 0, "embedding_model": "m1", "created_at": "now"}
    ]
    
    request = RetrieveRequest(query="test", kb_name="kb1")
    response = retrieve_documents(request)
    
    assert isinstance(response, RetrieveResponse)
    assert response.results[0].strategy_results[0].chunks[0].chunk_text == "hello test"
    assert mock_retriever.retrieve.called

def test_retrieve_from_missing_kb_raises_error(mock_retrieval_deps):
    """Should raise FileNotFoundError when PG says KB doesn't exist."""
    mock_retrieval_deps.pg.kb_exists.return_value = False
    request = RetrieveRequest(query="ghost", kb_name="ghost_kb")
    with pytest.raises(FileNotFoundError):
        retrieve_documents(request)

def test_retrieve_slugifies_kb_name(mock_retrieval_deps):
    """kb_name with hyphens/spaces/caps must be slugified before PG lookup."""
    mock_retrieval_deps.pg.kb_exists.return_value = False
    request = RetrieveRequest(query="test", kb_name="Test-KB")
    with pytest.raises(FileNotFoundError):
        retrieve_documents(request)
    # Must have looked up the slugified name, not the raw one
    mock_retrieval_deps.pg.kb_exists.assert_called_once_with("test_kb")


def test_retrieve_all_kbs(mock_retrieval_deps):
    """Verify listing all KBs from Postgres when kb_name is None."""
    mock_retrieval_deps.pg.list_kb_names.return_value = ["kb1", "kb2"]
    mock_retrieval_deps.pg.get_active_group_index.return_value = [] # skip internal search
    
    request = RetrieveRequest(query="test", kb_name=None)
    response = retrieve_documents(request)
    
    assert mock_retrieval_deps.pg.list_kb_names.called
    assert len(response.results) == 2
