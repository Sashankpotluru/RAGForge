"""One canonical implementation of v1 metadata-filter semantics."""

from __future__ import annotations

from collections.abc import Mapping

from ragforge.domain.models import MetadataCondition, MetadataFilter, MetadataOperator


def matches_filter(metadata: Mapping[str, object], metadata_filter: MetadataFilter) -> bool:
    """Apply AND-only conditions; absent fields intentionally never match."""
    return all(_matches_condition(metadata, condition) for condition in metadata_filter.conditions)


def _matches_condition(metadata: Mapping[str, object], condition: MetadataCondition) -> bool:
    actual = metadata.get(condition.field)
    if actual is None:
        return False
    if condition.operator is MetadataOperator.EQ:
        if isinstance(actual, bool) != isinstance(condition.value, bool):
            return False
        return condition.value in actual if isinstance(actual, list) else actual == condition.value
    if condition.operator is MetadataOperator.IN:
        if not isinstance(condition.value, list):
            return False
        values = actual if isinstance(actual, list) else [actual]
        return any(value in condition.value for value in values)
    if isinstance(actual, bool) or isinstance(condition.value, bool):
        return False
    if not isinstance(actual, int | float) or not isinstance(condition.value, int | float):
        return False
    if condition.operator is MetadataOperator.GTE:
        return actual >= condition.value
    return actual <= condition.value
