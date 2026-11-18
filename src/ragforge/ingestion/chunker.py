"""Structure-aware bounded windows with exact normalized, location-local offsets."""

import re
from hashlib import sha256

from ragforge.domain.models import Chunk
from ragforge.ingestion.loaders import LoadedDocument


class SemanticChunker:
    """Prefer sentence boundaries, then words, then a hard character limit."""

    def __init__(self, target_chars: int = 800, overlap_chars: int = 120) -> None:
        if target_chars < 1 or not 0 <= overlap_chars < target_chars:
            raise ValueError("require positive target and 0 <= overlap < target")
        self._target_chars = target_chars
        self._overlap_chars = overlap_chars

    def chunk(self, document: LoadedDocument, document_id: str, version: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for segment in document.segments:
            normalized = " ".join(segment.text.split())
            start = 0
            previous_end = 0
            while start < len(normalized):
                while start < len(normalized) and normalized[start].isspace():
                    start += 1
                end = min(start + self._target_chars, len(normalized))
                if end < len(normalized):
                    boundaries = list(re.finditer(r"[.!?] +", normalized[start:end]))
                    if boundaries and start + boundaries[-1].end() - 1 > previous_end:
                        end = start + boundaries[-1].end() - 1
                    else:
                        word_end = normalized.rfind(" ", start, end)
                        if word_end > previous_end:
                            end = word_end
                while end > start and normalized[end - 1].isspace():
                    end -= 1
                if end <= start:
                    break
                text = normalized[start:end]
                ordinal = len(chunks)
                key = f"{document_id}\x00{version}\x00{ordinal}\x00{text}"
                chunks.append(
                    Chunk(
                        chunk_id=sha256(key.encode()).hexdigest(),
                        document_id=document_id,
                        version=version,
                        ordinal=ordinal,
                        text=text,
                        source=document.source,
                        source_location=segment.source_location,
                        start_offset=start,
                        end_offset=end,
                    )
                )
                if end == len(normalized):
                    break
                previous_end = end
                start = max(start + 1, end - self._overlap_chars)
        return chunks
