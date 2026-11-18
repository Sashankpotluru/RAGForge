"""Local document loaders that retain source provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
from bs4 import BeautifulSoup


class IngestionError(ValueError):
    """Raised when a source cannot be safely normalized for indexing."""


@dataclass(frozen=True)
class LoadedSegment:
    """Text from one source location, such as a PDF page."""

    text: str
    source_location: str


@dataclass(frozen=True)
class LoadedDocument:
    """A locally loaded source split only where provenance changes."""

    source: str
    segments: tuple[LoadedSegment, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(segment.text for segment in self.segments)


def load_local_document(path: Path) -> LoadedDocument:
    """Load supported local documents without losing their source location."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise IngestionError(f"source does not exist: {resolved}")
    source = resolved.as_uri()
    suffix = resolved.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return _single_segment(source, _read_text(resolved))
    if suffix in {".html", ".htm"}:
        return _single_segment(source, _read_html(resolved))
    if suffix == ".pdf":
        return _read_pdf(resolved, source)
    raise IngestionError(f"unsupported document type: {suffix or 'no extension'}")


def _single_segment(source: str, text: str) -> LoadedDocument:
    segment = LoadedSegment(text=text, source_location=source)
    return LoadedDocument(source=source, segments=(segment,))


def load_bytes(data: bytes, mime: str, source: str) -> LoadedDocument:
    """Parse bounded uploaded bytes without granting server-filesystem access."""
    if len(data) > 20 * 1024 * 1024:
        raise IngestionError("document exceeds 20 MiB")
    if mime == "application/pdf":
        try:
            with fitz.open(stream=data, filetype="pdf") as pdf:
                segments = tuple(
                    LoadedSegment(page.get_text("text"), f"{source}#page={page.number + 1}")
                    for page in pdf
                )
            if not any(s.text.strip() for s in segments):
                raise IngestionError("PDF has no extractable text")
            return LoadedDocument(source, segments)
        except (RuntimeError, fitz.FileDataError) as error:
            raise IngestionError("invalid PDF") from error
    if mime not in {"text/plain", "text/markdown", "text/html"}:
        raise IngestionError("unsupported document type")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IngestionError("document must be UTF-8") from error
    if mime == "text/html":
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        text = soup.get_text("\n", strip=True)
    return _single_segment(source, text)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise IngestionError(f"unable to decode text source: {path}") from error


def _read_html(path: Path) -> str:
    soup = BeautifulSoup(_read_text(path), "html.parser")
    return soup.get_text("\n", strip=True)


def _read_pdf(path: Path, source: str) -> LoadedDocument:
    try:
        pdf = fitz.open(path)
    except (fitz.FileDataError, RuntimeError) as error:
        raise IngestionError(f"unable to read PDF source: {path}") from error
    try:
        segments = tuple(
            LoadedSegment(text=text, source_location=f"{source}#page={page.number + 1}")
            for page in pdf
            if (text := page.get_text("text").strip())
        )
    finally:
        pdf.close()
    if not segments:
        raise IngestionError(f"PDF has no extractable text: {path}")
    return LoadedDocument(source=source, segments=segments)
