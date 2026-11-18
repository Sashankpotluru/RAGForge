"""Deterministic reciprocal-rank fusion with rank provenance."""

from collections.abc import Mapping, Sequence

from ragforge.domain.models import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    if k < 1:
        raise ValueError("RRF constant must be positive")
    merged: dict[str, RetrievedChunk] = {}
    totals: dict[str, float] = {}
    for source, ranking in rankings.items():
        seen: set[str] = set()
        for rank, result in enumerate(ranking, 1):
            key = result.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            previous = merged.get(key)
            scores = previous.scores.copy() if previous else {}
            source_score = result.scores.get(source.removesuffix("_rewritten"))
            if source_score is not None:
                scores[source] = source_score
            ranks = previous.source_ranks.copy() if previous else {}
            merged[key] = result.model_copy(
                update={
                    "scores": scores,
                    "source_ranks": {**ranks, source: rank},
                }
            )
            totals[key] = totals.get(key, 0.0) + 1 / (k + rank)
    return [
        merged[key].model_copy(
            update={"rank": rank, "scores": {**merged[key].scores, "rrf": totals[key]}}
        )
        for rank, key in enumerate(sorted(merged, key=lambda key: (-totals[key], key)), 1)
    ]
