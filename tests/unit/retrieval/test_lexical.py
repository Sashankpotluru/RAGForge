def test_lexical_store_breaks_equal_scores_by_chunk_id(tmp_path) -> None:
    from ragforge.domain.models import Chunk, MetadataFilter
    from ragforge.indexing.bm25_store import Bm25Store

    chunks = [
        Chunk(
            chunk_id="b",
            document_id="doc",
            version="v",
            ordinal=0,
            text="policy",
            source_location="b",
        ),
        Chunk(
            chunk_id="a",
            document_id="doc",
            version="v",
            ordinal=1,
            text="policy",
            source_location="a",
        ),
    ]
    store = Bm25Store(tmp_path / "bm25.json")
    store.refresh(chunks, "v")

    assert [item.chunk.chunk_id for item in store.search("policy", MetadataFilter(), limit=2)] == [
        "a",
        "b",
    ]
