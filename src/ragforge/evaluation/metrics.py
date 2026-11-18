"""Recall counts distinct annotated spans; hit rate is separately named."""

import math
from collections.abc import Sequence


def binary_ndcg(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Binary chunk-level nDCG@10 with an ideal over all eligible relevant chunks."""
    seen: set[str] = set()
    gain = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:10], 1):
        if chunk_id in relevant_ids and chunk_id not in seen:
            gain += 1 / math.log2(rank + 1)
        seen.add(chunk_id)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant_ids)) + 1))
    return gain / ideal if ideal else 0.0


def retrieval_metrics(matches: Sequence[set[int]], total_spans: int) -> dict[str, float]:
    if total_spans < 1:
        raise ValueError("at least one gold span is required")
    metrics: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        found = set().union(*matches[:k])
        metrics[f"recall@{k}"] = len(found) / total_spans
        metrics[f"hit_rate@{k}"] = float(bool(found))
    metrics["mrr@10"] = next((1 / rank for rank, m in enumerate(matches[:10], 1) if m), 0.0)
    # Binary novelty gain: repeated coverage of an already seen gold span earns zero.
    seen: set[int] = set()
    dcg = 0.0
    for rank, matched in enumerate(matches[:10], 1):
        if matched - seen:
            dcg += 1 / math.log2(rank + 1)
        seen.update(matched)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, total_spans) + 1))
    metrics["novelty_ndcg@10"] = dcg / ideal
    return metrics
