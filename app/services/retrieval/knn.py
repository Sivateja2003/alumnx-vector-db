from __future__ import annotations

import numpy as np

from app.services.retrieval.base import BaseRetriever


class KNNRetriever(BaseRetriever):
    @property
    def strategy_name(self) -> str:
        return "knn"

    def retrieve(
        self,
        query_vector: np.ndarray,
        vectors: np.ndarray,
        chunk_ids: list[str],
        k: int,
        distance_metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """
        Vectorised exact KNN using a single BLAS matrix-vector multiply.

        Complexity:
          - Scoring:  O(N × D) — one matrix multiply, fully parallelised by numpy/BLAS
          - Top-k:    O(N)     — argpartition, not sort
          - Sorting k winners: O(k log k) — negligible since k << N

        For 10k chunks × 3072 dims this runs in ~1-3ms on CPU.
        """
        n = len(chunk_ids)
        if n == 0 or k <= 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        mat = np.asarray(vectors, dtype=np.float32)  # no-op if already float32 ndarray

        if distance_metric == "cosine":
            # Stored vectors are already L2-normalised at ingest time.
            # Normalise the query once here → cosine = dot product.
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
        elif distance_metric != "dot_product":
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Single BLAS SGEMV call — entire scoring in one shot
        scores: np.ndarray = mat @ q  # shape (N,)

        # O(N) partial sort — only guarantee top-k are in the last k positions
        actual_k = min(k, n)
        top_idx = np.argpartition(scores, -actual_k)[-actual_k:]

        # Sort only the k winners — O(k log k), negligible
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(chunk_ids[i], float(scores[i])) for i in top_idx]
