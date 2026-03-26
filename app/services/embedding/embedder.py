from __future__ import annotations

from google import genai
from google.genai import types

from app.config import get_config


class GeminiEmbedder:
    def __init__(self, model: str | None = None) -> None:
        self.config = get_config()
        self.model = model or self.config.embedding_model

    def _client(self) -> genai.Client:
        return genai.Client()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks. Batched in groups of 100."""
        if not texts:
            return []
        client = self._client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            result = client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="retrieval_document",
                    output_dimensionality=self.config.output_dimensionality,
                ),
            )
            vectors.extend(e.values for e in result.embeddings)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single retrieval query."""
        client = self._client()
        result = client.models.embed_content(
            model=self.model,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="retrieval_query",
                output_dimensionality=self.config.output_dimensionality,
            ),
        )
        return result.embeddings[0].values
