from __future__ import annotations

"""Shared in-memory mock implementations for PostgresStore and VectorFileStore.
Used across the test suite to avoid real DB/file-system dependencies.
"""

import numpy as np


class MockPostgresStore:
    """In-memory implementation of PostgresStore for tests."""

    def __init__(self) -> None:
        self._chunks: dict[str, dict] = {}  # chunk_id -> row

    def ensure_table(self) -> None:
        pass

    def insert_chunks(self, rows: list[dict]) -> None:
        for row in rows:
            self._chunks[row["chunk_id"]] = row.copy()

    def has_active_chunks(self, source_filename: str, chunking_strategy: str, embedding_model: str, kb_name: str) -> bool:
        return any(
            r["is_active"]
            and r["source_filename"] == source_filename
            and r["chunking_strategy"] == chunking_strategy
            and r["embedding_model"] == embedding_model
            and r["kb_name"] == kb_name
            for r in self._chunks.values()
        )

    def deactivate_chunks(self, source_filename: str, strategy_names: list[str], embedding_model: str, kb_name: str, deactivated_at: str) -> list[str]:
        ids: list[str] = []
        for cid, row in self._chunks.items():
            if (
                row["is_active"]
                and row["source_filename"] == source_filename
                and row["chunking_strategy"] in strategy_names
                and row["embedding_model"] == embedding_model
                and row["kb_name"] == kb_name
            ):
                row["is_active"] = False
                row["deactivated_at"] = deactivated_at
                ids.append(cid)
        return ids

    def get_active_group_index(self, kb_name: str, embedding_model: str | None = None) -> list[dict]:
        result = []
        for row in self._chunks.values():
            if row["kb_name"] == kb_name and row["is_active"]:
                if embedding_model is None or row["embedding_model"] == embedding_model:
                    result.append({
                        "chunk_id": row["chunk_id"],
                        "chunking_strategy": row["chunking_strategy"],
                        "embedding_model": row["embedding_model"],
                    })
        return result

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        return [self._chunks[cid].copy() for cid in chunk_ids if cid in self._chunks]

    def kb_exists(self, kb_name: str) -> bool:
        return any(r["kb_name"] == kb_name for r in self._chunks.values())

    def list_kb_names(self) -> list[str]:
        return sorted({r["kb_name"] for r in self._chunks.values() if r["is_active"]})


class MockVectorFileStore:
    """In-memory implementation of VectorFileStore for tests."""

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}   # kb_name -> float32 array (N, D)
        self._ids: dict[str, list[str]] = {}         # kb_name -> chunk_ids

    def read(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        if kb_name not in self._ids:
            return np.empty((0, 3), dtype=np.float32), []
        return self._vectors[kb_name].copy(), list(self._ids[kb_name])

    def append(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        if kb_name in self._ids:
            self._vectors[kb_name] = np.vstack([self._vectors[kb_name], vectors]).astype(np.float32)
            self._ids[kb_name].extend(chunk_ids)
        else:
            self._vectors[kb_name] = vectors.astype(np.float32)
            self._ids[kb_name] = list(chunk_ids)

    def rewrite(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        self._vectors[kb_name] = vectors.astype(np.float32)
        self._ids[kb_name] = list(chunk_ids)

    def remove_chunk_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        if kb_name not in self._ids:
            return
        old_ids = self._ids[kb_name]
        old_vecs = self._vectors[kb_name]
        mask = [cid not in ids_to_remove for cid in old_ids]
        self._ids[kb_name] = [cid for cid, keep in zip(old_ids, mask) if keep]
        self._vectors[kb_name] = old_vecs[np.array(mask)].astype(np.float32)

    def list_kb_names(self) -> list[str]:
        return sorted(self._ids.keys())

    def delete_kb(self, kb_name: str) -> None:
        self._vectors.pop(kb_name, None)
        self._ids.pop(kb_name, None)
