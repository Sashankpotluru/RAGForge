"""Explicit server exercise, isolated in a unique temporary collection."""

import os
from uuid import uuid4

import pytest

from ragforge.domain.models import Chunk, MetadataFilter
from ragforge.indexing.qdrant_store import QdrantStore
from ragforge.retrieval.dense import HashEmbedding


@pytest.mark.skipif(
    os.environ.get("RUN_QDRANT_INTEGRATION") != "1",
    reason="requires an explicitly enabled Qdrant server",
)
def test_server_roundtrip():
    store = QdrantStore(
        HashEmbedding(32),
        collection="ragforge_test_" + uuid4().hex,
        url=os.environ.get("RAGFORGE_QDRANT_URL", "http://127.0.0.1:6333"),
    )
    try:
        chunk = Chunk(
            chunk_id="one",
            document_id="doc",
            version="v",
            ordinal=0,
            text="violet",
            source_location="memory:a",
        )
        store.upsert("v", [chunk])
        assert store.search("violet", MetadataFilter(), 1, allowed_ids=["one"])[0].chunk == chunk
        store.delete_ids(["one"])
        assert not store.search("violet", MetadataFilter(), 1, allowed_ids=["one"])
    finally:
        store.client.delete_collection(store.collection)
        store.close()
