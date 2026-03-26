from __future__ import annotations

import numpy as np
import pytest

from app.services.retrieval.knn import KNNRetriever


@pytest.fixture
def retriever():
    return KNNRetriever()


def test_knn_cosine_similarity(retriever):
    query = np.array([1.0, 0.0], dtype=np.float32)
    vectors = np.array([
        [1.0, 0.0],   # score 1.0  — perfect match
        [0.0, 1.0],   # score 0.0  — orthogonal
        [-1.0, 0.0],  # score -1.0 — opposite
    ], dtype=np.float32)
    chunk_ids = ["perfect", "orthogonal", "opposite"]

    results = retriever.retrieve(query, vectors, chunk_ids, k=2, distance_metric="cosine")
    assert len(results) == 2
    assert results[0] == ("perfect", pytest.approx(1.0))
    assert results[1] == ("orthogonal", pytest.approx(0.0))


def test_knn_dot_product(retriever):
    query = np.array([1.0, 1.0], dtype=np.float32)
    vectors = np.array([
        [2.0, 2.0],  # score 4.0
        [0.5, 0.5],  # score 1.0
    ], dtype=np.float32)
    chunk_ids = ["large", "small"]

    results = retriever.retrieve(query, vectors, chunk_ids, k=10, distance_metric="dot_product")
    assert results[0][0] == "large"
    assert results[0][1] == pytest.approx(4.0)
    assert results[1][0] == "small"


def test_knn_empty_returns_empty(retriever):
    results = retriever.retrieve(
        np.array([1.0, 0.0], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        [],
        k=5,
    )
    assert results == []


def test_knn_k_limit_respected(retriever):
    query = np.array([1.0, 0.0], dtype=np.float32)
    vectors = np.array([[1.0, 0.0]] * 10, dtype=np.float32)
    chunk_ids = [str(i) for i in range(10)]

    results = retriever.retrieve(query, vectors, chunk_ids, k=3)
    assert len(results) == 3
