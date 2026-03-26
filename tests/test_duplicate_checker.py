from __future__ import annotations

# Duplicate checking is now handled by PostgresStore.has_active_chunks().
# These tests verify the in-memory mock behaviour used across the test suite.

import pytest
from tests.helpers import MockPostgresStore


def _chunk(chunk_id, is_active=True, strategy="fixed_length"):
    return {
        "chunk_id": chunk_id, "kb_name": "kb", "source_filename": "a.pdf",
        "chunking_strategy": strategy, "chunk_index": 0, "page_number": 1,
        "chunk_text": "hello", "embedding_model": "m1",
        "is_active": is_active, "created_at": "2024-01-01",
        "deactivated_at": None if is_active else "2024-01-02",
    }


def test_has_active_chunks_true():
    pg = MockPostgresStore()
    pg.insert_chunks([_chunk("c1")])
    assert pg.has_active_chunks("a.pdf", "fixed_length", "m1", "kb") is True


def test_has_active_chunks_inactive_not_matched():
    pg = MockPostgresStore()
    pg.insert_chunks([_chunk("c1", is_active=False)])
    assert pg.has_active_chunks("a.pdf", "fixed_length", "m1", "kb") is False


def test_has_active_chunks_different_strategy_no_match():
    pg = MockPostgresStore()
    pg.insert_chunks([_chunk("c1", strategy="fixed_length")])
    assert pg.has_active_chunks("a.pdf", "paragraph", "m1", "kb") is False


def test_deactivate_chunks_returns_ids():
    pg = MockPostgresStore()
    pg.insert_chunks([_chunk("c1"), _chunk("c2")])
    deactivated = pg.deactivate_chunks("a.pdf", ["fixed_length"], "m1", "kb", "2024-01-02")
    assert set(deactivated) == {"c1", "c2"}
    assert pg.has_active_chunks("a.pdf", "fixed_length", "m1", "kb") is False
