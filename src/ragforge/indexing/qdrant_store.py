"""Qdrant persistence with exact active-set filtering before candidate limits."""

from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from ragforge.domain.filters import matches_filter
from ragforge.domain.models import Chunk, MetadataFilter, RetrievedChunk
from ragforge.retrieval.dense import Embedding


def point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(
        self,
        embedding: Embedding,
        *,
        path: Path | None = None,
        url: str | None = None,
        collection: str = "ragforge",
    ) -> None:
        self.embedding = embedding
        self.collection = collection
        self.client = QdrantClient(url=url) if url else QdrantClient(path=str(path))
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=embedding.dimensions, distance=models.Distance.COSINE
                ),
            )

    def upsert(self, version: str, chunks: Sequence[Chunk]) -> None:
        vectors = self.embedding.encode([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding count mismatch")
        if chunks:
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(c.chunk_id), vector=v, payload=c.model_dump(mode="json")
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
                wait=True,
            )

    def search(
        self, query: str, metadata_filter: MetadataFilter, limit: int, *, allowed_ids: Sequence[str]
    ) -> list[RetrievedChunk]:
        if not allowed_ids:
            return []
        # The catalog prefilters using the canonical predicate; this adapter also
        # supports direct calls by retrieving metadata and filtering IDs first.
        records = self.client.retrieve(
            self.collection, ids=[point_id(i) for i in allowed_ids], with_payload=True
        )
        eligible = [
            r.id
            for r in records
            if r.payload is not None
            and matches_filter(Chunk.model_validate(r.payload).metadata, metadata_filter)
        ]
        if not eligible:
            return []
        points = self.client.query_points(
            self.collection,
            query=self.embedding.encode([query])[0],
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=eligible)]),
            limit=limit,
            with_payload=True,
        ).points
        return [
            RetrievedChunk(
                chunk=Chunk.model_validate(p.payload),
                rank=i,
                scores={"dense": p.score},
                source_ranks={"dense": i},
            )
            for i, p in enumerate(sorted(points, key=lambda p: (-p.score, str(p.id))), 1)
        ]

    def delete_ids(self, ids: Sequence[str]) -> None:
        if ids:
            self.client.delete(
                self.collection, models.PointIdsList(points=[point_id(i) for i in ids]), wait=True
            )

    def ready(self) -> bool:
        return self.client.collection_exists(self.collection)

    def close(self) -> None:
        self.client.close()
