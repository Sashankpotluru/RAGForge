"""A local, atomically persisted BM25 index over RAGForge chunks."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ragforge.domain.filters import matches_filter
from ragforge.domain.models import Chunk, MetadataFilter, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


class Bm25Store:
    """A deterministic local lexical index for corpus sizes in the v1 envelope."""

    def __init__(
        self, path: Path, k1: float = 1.5, b: float = 0.75, *, load_snapshot: bool = True
    ) -> None:
        self._path = path
        self._k1 = k1
        self._b = b
        self._chunks = self._load() if load_snapshot else []

    def refresh(self, chunks: Sequence[Chunk], version: str) -> None:
        """Persist a complete replacement only after the new payload validates."""
        _ = version
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("duplicate chunk IDs cannot be indexed")
        payload = [chunk.model_dump(mode="json") for chunk in chunks]
        encoded = json.dumps(payload, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, self._path)
        self._chunks = list(chunks)

    def search(
        self,
        query: str,
        metadata_filter: MetadataFilter,
        limit: int,
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return lexical matches with stable ranking and score provenance."""
        candidates = [
            chunk
            for chunk in self._chunks
            if (allowed_ids is None or chunk.chunk_id in allowed_ids)
            and matches_filter(chunk.metadata, metadata_filter)
        ]
        query_terms = _tokens(query)
        if not query_terms or not candidates:
            return []
        tokenized = {chunk.chunk_id: _tokens(chunk.text) for chunk in candidates}
        average_length = sum(len(tokens) for tokens in tokenized.values()) / len(candidates)
        if average_length == 0:
            return []
        document_frequency = Counter(term for tokens in tokenized.values() for term in set(tokens))
        scored = [
            (
                chunk,
                _bm25_score(
                    query_terms,
                    tokenized[chunk.chunk_id],
                    document_frequency,
                    len(candidates),
                    average_length,
                    self._k1,
                    self._b,
                ),
            )
            for chunk in candidates
        ]
        ranked = sorted(
            ((chunk, score) for chunk, score in scored if score > 0),
            key=lambda item: (-item[1], item[0].chunk_id),
        )[:limit]
        return [
            RetrievedChunk(
                chunk=chunk,
                rank=rank,
                scores={"lexical": score},
                source_ranks={"lexical": rank},
            )
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ]

    def _load(self) -> list[Chunk]:
        if not self._path.is_file():
            return []
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("invalid BM25 index payload")
        return [Chunk.model_validate(item) for item in payload]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bm25_score(
    query_terms: list[str],
    document_terms: list[str],
    document_frequency: Counter[str],
    corpus_size: int,
    average_length: float,
    k1: float,
    b: float,
) -> float:
    frequencies = Counter(document_terms)
    denominator_length = 1 - b + b * len(document_terms) / average_length
    score = 0.0
    for term in query_terms:
        frequency = frequencies[term]
        if not frequency:
            continue
        inverse_frequency = math.log(
            1 + (corpus_size - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
        )
        score += inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * denominator_length)
    return score
