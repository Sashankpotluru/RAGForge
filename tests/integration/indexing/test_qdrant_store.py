from pathlib import Path

from ragforge.domain.models import Chunk, MetadataCondition, MetadataFilter


def test_embedded_qdrant_active_filter_and_metadata_parity(tmp_path: Path) -> None:
    from ragforge.indexing.qdrant_store import QdrantStore
    from ragforge.retrieval.dense import HashEmbedding

    store = QdrantStore(HashEmbedding(32), path=tmp_path / "qdrant")
    chunks = [
        Chunk(
            chunk_id=str(i),
            document_id="doc",
            version=str(i),
            ordinal=i,
            text="violet",
            source_location="memory:a",
            metadata=metadata,
        )
        for i, metadata in enumerate([{"team": "x"}, {"team": "y"}, {}])
    ]
    store.upsert("v", chunks)
    found = store.search(
        "violet",
        MetadataFilter(conditions=(MetadataCondition(field="team", operator="eq", value="x"),)),
        1,
        allowed_ids=["0", "1", "2"],
    )
    assert [r.chunk.chunk_id for r in found] == ["0"]
    assert store.search("violet", MetadataFilter(), 5, allowed_ids=[]) == []
    store.close()
