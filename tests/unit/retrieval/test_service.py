from pathlib import Path

import pytest

from ragforge.config import Settings
from ragforge.domain.models import DocumentInput, MetadataCondition, MetadataFilter, SearchRequest
from ragforge.runtime import Runtime


def test_search_provenance_filters_and_unconfigured_features(tmp_path: Path) -> None:
    from ragforge.retrieval.service import SearchService

    runtime = Runtime(Settings(data_dir=tmp_path))
    runtime.ingest(
        DocumentInput(source="memory:a", content="violet garden", metadata={"team": "a"})
    )
    runtime.ingest(
        DocumentInput(source="memory:b", content="violet garden", metadata={"team": "b"})
    )
    search = SearchService(runtime)
    request = SearchRequest(
        query="violet",
        metadata_filter=MetadataFilter(
            conditions=(MetadataCondition(field="team", operator="eq", value="a"),)
        ),
    )
    found = search.search(request)
    assert len(found) == 1
    assert found[0].chunk.metadata == {"team": "a"}
    assert set(found[0].source_ranks) == {"dense", "lexical"}
    with pytest.raises(ValueError, match="rewrit"):
        search.search(request.model_copy(update={"rewrite_query": True}))
    with pytest.raises(ValueError, match="rerank"):
        search.search(request.model_copy(update={"mode": "hybrid_rerank"}))
    runtime.close()


def test_rrf_deduplicates_and_aggregates_exact_ranks() -> None:
    from ragforge.domain.models import Chunk, RetrievedChunk
    from ragforge.retrieval.fusion import reciprocal_rank_fusion

    a = RetrievedChunk(
        chunk=Chunk(
            chunk_id="a",
            document_id="d",
            version="v",
            ordinal=0,
            text="a",
            source_location="memory:a",
        ),
        rank=1,
        scores={"dense": 0.9},
    )
    b = a.model_copy(update={"chunk": a.chunk.model_copy(update={"chunk_id": "b"})})
    ranked = reciprocal_rank_fusion({"dense": [a, b], "lexical": [b, a, a]}, 60)
    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_id == "a"
    assert ranked[0].scores["rrf"] == pytest.approx(1 / 61 + 1 / 62)
    assert ranked[0].source_ranks == {"dense": 1, "lexical": 2}


def test_collection_scope_filters_both_branches_before_limit(tmp_path):
    from ragforge.retrieval.service import SearchService

    runtime = Runtime(Settings(data_dir=tmp_path))
    runtime.ingest(DocumentInput(source="memory:a", content="violet", collection="other"))
    runtime.ingest(DocumentInput(source="memory:b", content="violet garden"))
    found = SearchService(runtime).search(SearchRequest(query="violet", candidate_limit=1))
    assert len(found) == 1 and found[0].chunk.source == "memory:b"
    runtime.close()


@pytest.mark.parametrize(
    "metadata,value,expected",
    [({"x": True}, 1, False), ({"x": 1}, True, False), ({"x": ["a"]}, "a", True), ({}, "a", False)],
)
def test_filter_eq_type_parity(metadata, value, expected) -> None:
    from ragforge.domain.filters import matches_filter

    assert (
        matches_filter(
            metadata,
            MetadataFilter(conditions=(MetadataCondition(field="x", operator="eq", value=value),)),
        )
        is expected
    )
