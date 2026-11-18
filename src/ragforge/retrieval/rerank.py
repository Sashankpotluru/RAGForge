"""Optional real cross-encoder adapter; model downloads require explicit configuration."""

from collections.abc import Sequence
from typing import Any

from ragforge.domain.models import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model: str, revision: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model, revision=revision)

    def rank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in chunks])
        scored = sorted(
            zip(chunks, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].chunk.chunk_id),
        )
        return [
            c.model_copy(update={"rank": i, "scores": {**c.scores, "rerank": float(score)}})
            for i, (c, score) in enumerate(scored, 1)
        ]
