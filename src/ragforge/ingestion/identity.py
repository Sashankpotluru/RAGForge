"""Deterministic source, document, and content identifiers."""

from __future__ import annotations

from hashlib import sha256
from posixpath import normpath
from urllib.parse import urlsplit, urlunsplit


def canonical_source(value: str) -> str:
    """Normalize URL identity without changing its query semantics."""
    cleaned = value.strip()
    parsed = urlsplit(cleaned)
    if not parsed.scheme:
        return cleaned

    path = normpath(parsed.path) if parsed.path else ""
    if parsed.path.startswith("/") and not path.startswith("/"):
        path = f"/{path}"
    if path == ".":
        path = ""
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def document_id(collection: str, logical_key: str) -> str:
    """Create a stable ID from caller-visible collection and logical identity."""
    return sha256(f"{collection}\x00{logical_key}".encode()).hexdigest()


def content_hash(text: str) -> str:
    """Fingerprint exact normalized document content."""
    return sha256(text.encode("utf-8")).hexdigest()
