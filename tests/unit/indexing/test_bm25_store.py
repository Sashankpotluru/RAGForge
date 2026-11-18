from pathlib import Path

import pytest


def _chunk(chunk_id: str, text: str, metadata: dict[str, object] | None = None):
    from ragforge.domain.models import Chunk

    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        version="v1",
        ordinal=int(chunk_id[-1]),
        text=text,
        source_location="file:///knowledge.md",
        metadata=metadata or {},
    )


def test_bm25_ranks_matching_terms_and_persists(tmp_path: Path) -> None:
    from ragforge.domain.models import MetadataFilter
    from ragforge.indexing.bm25_store import Bm25Store

    index_path = tmp_path / "bm25.json"
    store = Bm25Store(index_path)
    store.refresh(
        [_chunk("chunk-1", "incident response guide"), _chunk("chunk-2", "vacation guide")], "v1"
    )

    result = Bm25Store(index_path).search("incident", MetadataFilter(), limit=2)

    assert [item.chunk.chunk_id for item in result] == ["chunk-1"]
    assert result[0].source_ranks == {"lexical": 1}


def test_bm25_excludes_missing_metadata_and_preserves_previous_index_on_bad_refresh(
    tmp_path: Path,
) -> None:
    from ragforge.domain.models import MetadataCondition, MetadataFilter, MetadataOperator
    from ragforge.indexing.bm25_store import Bm25Store

    store = Bm25Store(tmp_path / "bm25.json")
    original = _chunk("chunk-1", "incident response guide", {"department": "security"})
    store.refresh([original], "v1")
    security_only = MetadataFilter(
        conditions=(
            MetadataCondition(field="department", operator=MetadataOperator.EQ, value="security"),
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        store.refresh([original, original], "v2")

    assert [item.chunk.chunk_id for item in store.search("incident", security_only, limit=5)] == [
        "chunk-1"
    ]
    assert (
        store.search(
            "incident",
            MetadataFilter(
                conditions=(
                    MetadataCondition(field="missing", operator=MetadataOperator.EQ, value="x"),
                )
            ),
            limit=5,
        )
        == []
    )
