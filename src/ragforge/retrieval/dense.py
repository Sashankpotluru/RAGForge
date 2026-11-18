"""Embedding adapters. Hash embeddings are an offline demonstration, not semantic AI."""

import math
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Protocol


class Embedding(Protocol):
    dimensions: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedding:
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4]) % self.dimensions] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


class SentenceTransformerEmbedding:
    def __init__(self, model: str, revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model, revision=revision)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()  # type: ignore[no-any-return]
