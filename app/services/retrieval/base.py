from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query_vector: np.ndarray,
        vectors: np.ndarray,
        chunk_ids: list[str],
        k: int,
        distance_metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, similarity_score) pairs for the top-k results."""
        raise NotImplementedError

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        raise NotImplementedError
