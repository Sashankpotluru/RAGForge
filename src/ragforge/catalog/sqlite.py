"""SQLite ownership of logical document identity and active index versions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from ragforge.domain.models import Chunk, Document, DocumentInput, IndexStatus, IngestDecision
from ragforge.ingestion.identity import canonical_source, document_id


class SqliteCatalog:
    """A small persistent catalog with explicit staging and failure states."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register(self, document: DocumentInput, digest: str) -> IngestDecision:
        """Record a new/staged document version or recognize a no-op update."""
        source = canonical_source(document.source)
        logical_identity = (
            f"external:{document.external_id}"
            if document.external_id is not None
            else f"source:{source}"
        )
        stable_id = document_id(document.collection, logical_identity)
        version = digest
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT content_hash, status FROM documents
                WHERE collection = ? AND logical_identity = ?
                """,
                (document.collection, logical_identity),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO documents (
                        collection, logical_identity, document_id, source, content_hash,
                        active_version, staged_version, status, metadata_json, failure_message
                    ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL)
                    """,
                    (
                        document.collection,
                        logical_identity,
                        stable_id,
                        source,
                        digest,
                        version,
                        IndexStatus.STAGING.value,
                        json.dumps(document.metadata, sort_keys=True),
                    ),
                )
                return IngestDecision.NEW
            if existing["content_hash"] == digest and existing["status"] == "active":
                connection.execute(
                    """
                    UPDATE documents SET source = ?, metadata_json = ?
                    WHERE collection = ? AND logical_identity = ?
                    """,
                    (
                        source,
                        json.dumps(document.metadata, sort_keys=True),
                        document.collection,
                        logical_identity,
                    ),
                )
                return IngestDecision.NOOP
            connection.execute(
                """
                UPDATE documents
                SET source = ?, content_hash = ?, staged_version = ?, status = ?,
                    metadata_json = ?, failure_message = NULL
                WHERE collection = ? AND logical_identity = ?
                """,
                (
                    source,
                    digest,
                    version,
                    IndexStatus.STAGING.value,
                    json.dumps(document.metadata, sort_keys=True),
                    document.collection,
                    logical_identity,
                ),
            )
            return IngestDecision.REPLACE

    def get_by_identity(self, collection: str, logical_identity: str) -> Document | None:
        """Return the persistent document record for a logical identity."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, collection, logical_identity, source, content_hash,
                       active_version, staged_version, failure_message, status, metadata_json
                FROM documents WHERE collection = ? AND logical_identity = ?
                """,
                (collection, logical_identity),
            ).fetchone()
        return _document_from_row(row) if row is not None else None

    def get_by_document_id(self, stable_id: str) -> Document | None:
        """Return a document by the API-visible stable ID."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, collection, logical_identity, source, content_hash,
                       active_version, staged_version, failure_message, status, metadata_json
                FROM documents WHERE document_id = ?
                """,
                (stable_id,),
            ).fetchone()
        return _document_from_row(row) if row is not None else None

    def promote(self, stable_id: str, version: str) -> None:
        """Atomically make a completely indexed staged version queryable."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET active_version = ?, staged_version = NULL, status = ?, failure_message = NULL
                WHERE document_id = ? AND staged_version = ?
                """,
                (version, IndexStatus.ACTIVE.value, stable_id, version),
            )

    def mark_failed(self, stable_id: str, message: str) -> None:
        """Record a safe error while leaving any existing active version intact."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, failure_message = ?
                WHERE document_id = ?
                """,
                (IndexStatus.FAILED.value, message, stable_id),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chunks "
                "(id TEXT PRIMARY KEY, document_id TEXT, version TEXT, payload TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS configuration (id INTEGER PRIMARY KEY, value TEXT)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    collection TEXT NOT NULL,
                    logical_identity TEXT NOT NULL,
                    document_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    active_version TEXT,
                    staged_version TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    failure_message TEXT,
                    PRIMARY KEY (collection, logical_identity)
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def check_configuration(self, value: str, *, record: bool = True) -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM configuration WHERE id=1").fetchone()
            if row is not None and row[0] != value:
                raise ValueError(
                    "index configuration changed; use a new data directory and collection"
                )
            if record:
                connection.execute("INSERT OR IGNORE INTO configuration VALUES (1, ?)", (value,))

    def stage_chunks(self, chunks: Sequence[Chunk]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)",
                [(c.chunk_id, c.document_id, c.version, c.model_dump_json()) for c in chunks],
            )

    def active_chunks(self, collection: str | None = None) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT c.payload FROM chunks c JOIN documents d "
                "ON c.document_id=d.document_id AND c.version=d.active_version "
                "WHERE (? IS NULL OR d.collection=?) ORDER BY c.id",
                (collection, collection),
            ).fetchall()
        return [Chunk.model_validate_json(row[0]) for row in rows]

    def all_chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM chunks").fetchall()
        return [Chunk.model_validate_json(row[0]) for row in rows]

    def remove_inactive(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE NOT EXISTS (SELECT 1 FROM documents d "
                "WHERE d.document_id=chunks.document_id AND d.active_version=chunks.version)"
            )
            connection.execute(
                "UPDATE documents SET status='failed', failure_message='interrupted indexing' "
                "WHERE status='staging'"
            )

    def delete(self, stable_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE document_id=?", (stable_id,))
            # Keep chunk IDs until vector deletion succeeds (or startup retries it).


def _document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        document_id=row["document_id"],
        collection=row["collection"],
        logical_identity=row["logical_identity"],
        source=row["source"],
        content_hash=row["content_hash"],
        active_version=row["active_version"],
        staged_version=row["staged_version"],
        failure_message=row["failure_message"],
        status=IndexStatus(row["status"]),
        metadata=json.loads(row["metadata_json"]),
    )
