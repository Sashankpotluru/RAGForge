"""Ports that isolate deterministic core logic from external services."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragforge.domain.models import (
    Chunk,
    Citation,
    MetadataFilter,
    ProviderAnswer,
    RetrievedChunk,
)


class DenseIndex(Protocol):
    """Dense-vector persistence and retrieval."""

    def upsert(self, version: str, chunks: Sequence[Chunk]) -> None: ...

    def search(
        self, query: str, metadata_filter: MetadataFilter, limit: int
    ) -> list[RetrievedChunk]: ...


class LexicalIndex(Protocol):
    """Persisted lexical retrieval over the same chunk records."""

    def refresh(self, chunks: Sequence[Chunk], version: str) -> None: ...

    def search(
        self, query: str, metadata_filter: MetadataFilter, limit: int
    ) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    """A relevance reranker that sees only fused candidates."""

    def rank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]: ...


class AnswerProvider(Protocol):
    """An explicitly configured generator; no provider is implicit."""

    def answer(
        self, question: str, evidence: Sequence[RetrievedChunk], citations: Sequence[Citation]
    ) -> ProviderAnswer: ...
