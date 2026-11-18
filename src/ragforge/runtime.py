"""Single-worker service composition and authoritative index lifecycle."""

import contextlib
import json
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

from ragforge.catalog.sqlite import SqliteCatalog
from ragforge.config import Settings
from ragforge.domain.models import DocumentInput, IngestDecision
from ragforge.indexing.bm25_store import Bm25Store
from ragforge.indexing.qdrant_store import QdrantStore
from ragforge.ingestion.chunker import SemanticChunker
from ragforge.ingestion.loaders import LoadedDocument, LoadedSegment
from ragforge.retrieval.dense import Embedding, HashEmbedding, SentenceTransformerEmbedding


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    version: str
    decision: str
    chunks: int


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = RLock()
        self.catalog = SqliteCatalog(settings.data_dir / "catalog.sqlite3")
        config = {
            k: v
            for k, v in settings.model_dump(mode="json").items()
            if k
            in {
                "embedding_model",
                "embedding_revision",
                "embedding_dimensions",
                "target_chars",
                "overlap_chars",
                "qdrant_collection",
                "qdrant_url",
            }
        }
        encoded = json.dumps(config, sort_keys=True)
        self.config_id = sha256(encoded.encode()).hexdigest()
        self.catalog.check_configuration(encoded, record=False)
        embedding: Embedding = (
            HashEmbedding(settings.embedding_dimensions)
            if settings.embedding_model == "hash-demo-v1"
            else SentenceTransformerEmbedding(settings.embedding_model, settings.embedding_revision)
        )
        self.dense = QdrantStore(
            embedding,
            path=settings.data_dir / "qdrant",
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
        )
        try:
            self.lexical = Bm25Store(settings.data_dir / "bm25.json", load_snapshot=False)
            self.chunker = SemanticChunker(settings.target_chars, settings.overlap_chars)
            self.cleanup()
            self.lexical.refresh(self.catalog.active_chunks(), "startup")
            self.catalog.check_configuration(encoded)
        except BaseException:
            self.dense.close()
            raise

    def cleanup(self) -> None:
        active_ids = {c.chunk_id for c in self.catalog.active_chunks()}
        self.dense.delete_ids(
            [c.chunk_id for c in self.catalog.all_chunks() if c.chunk_id not in active_ids]
        )
        self.catalog.remove_inactive()

    def ingest(self, document: DocumentInput, loaded: LoadedDocument | None = None) -> IngestResult:
        with self.lock:
            loaded = loaded or LoadedDocument(
                document.source, (LoadedSegment(document.content, document.source),)
            )
            digest = sha256(
                json.dumps(
                    {
                        "document": document.model_dump(mode="json"),
                        "segments": [(s.text, s.source_location) for s in loaded.segments],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            chunks = self.chunker.chunk(loaded, "pending", digest)
            if not chunks:
                raise ValueError("document has no extractable text")
            decision = self.catalog.register(document, digest)
            from ragforge.ingestion.identity import canonical_source

            identity = (
                f"external:{document.external_id}"
                if document.external_id
                else f"source:{canonical_source(document.source)}"
            )
            record = self.catalog.get_by_identity(document.collection, identity)
            assert record is not None
            if decision is IngestDecision.NOOP:
                return IngestResult(record.document_id, digest, decision.value, 0)
            chunks = [
                c.model_copy(
                    update={
                        "metadata": document.metadata,
                        "title": document.title or document.source,
                    }
                )
                for c in self.chunker.chunk(loaded, record.document_id, digest)
            ]
            old = self.catalog.active_chunks()
            try:
                self.catalog.stage_chunks(chunks)
                self.dense.upsert(digest, chunks)
                replacement = [c for c in old if c.document_id != record.document_id] + chunks
                self.lexical.refresh(replacement, digest)
                self.catalog.promote(record.document_id, digest)
            except Exception as error:
                self.catalog.mark_failed(
                    record.document_id, "index backend failure; retry ingestion"
                )
                self.lexical.refresh(old, "rollback")
                raise RuntimeError(
                    f"indexing failed for {record.document_id}; retry ingestion"
                ) from error
            # Cleanup is recoverable at startup; never undo an already committed promotion.
            with contextlib.suppress(Exception):
                self.cleanup()
            return IngestResult(record.document_id, digest, decision.value, len(chunks))

    def delete(self, document_id: str) -> None:
        with self.lock:
            self.catalog.delete(document_id)
            self.lexical.refresh(self.catalog.active_chunks(), "delete")
            self.cleanup()

    def close(self) -> None:
        self.dense.close()
