from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import get_config


class VectorFileStore:
    """Stores only chunk_id + normalised float32 vectors in numpy binary files.

    Per knowledge base:
      <kb_name>.npy     — float32 array of shape (N, vector_size)
      <kb_name>_ids.npy — unicode string array of shape (N,), one chunk_id per row

    Only active vectors are kept; deactivated vectors are removed on overwrite.
    """

    def __init__(self) -> None:
        self.config = get_config()

    def _ensure_path(self) -> Path:
        path = self.config.vector_store_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _vec_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}.npy"

    def _ids_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}_ids.npy"

    def read(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        """Return (vectors, chunk_ids). Empty arrays if KB does not exist."""
        vec_path = self._vec_path(kb_name)
        ids_path = self._ids_path(kb_name)
        if not vec_path.exists() or not ids_path.exists():
            return np.empty((0, self.config.vector_size), dtype=np.float32), []
        vectors = np.load(vec_path, mmap_mode='r')
        chunk_ids = np.load(ids_path, allow_pickle=False).tolist()
        return vectors, chunk_ids

    def append(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Append new chunk_ids and their normalised vectors to the KB file."""
        existing_vectors, existing_ids = self.read(kb_name)
        new_vectors = np.vstack([existing_vectors, vectors]).astype(np.float32) if existing_ids else vectors.astype(np.float32)
        new_ids = existing_ids + chunk_ids
        self._write(kb_name, new_ids, new_vectors)

    def rewrite(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Overwrite the KB file with the given chunk_ids and vectors."""
        self._write(kb_name, chunk_ids, vectors.astype(np.float32))

    def remove_chunk_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        """Remove specific chunk_ids (and their vectors) from the KB file."""
        existing_vectors, existing_ids = self.read(kb_name)
        if not existing_ids:
            return
        mask = np.array([cid not in ids_to_remove for cid in existing_ids])
        kept_ids = [cid for cid, keep in zip(existing_ids, mask) if keep]
        kept_vectors = existing_vectors[mask]
        self._write(kb_name, kept_ids, kept_vectors)

    def list_kb_names(self) -> list[str]:
        path = self._ensure_path()
        return sorted(
            p.stem for p in path.glob("*.npy") if not p.stem.endswith("_ids")
        )

    def delete_kb(self, kb_name: str) -> None:
        for p in (self._vec_path(kb_name), self._ids_path(kb_name)):
            if p.exists():
                p.unlink()

    def _write(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        np.save(self._vec_path(kb_name), vectors)
        np.save(self._ids_path(kb_name), np.array(chunk_ids, dtype=str))
