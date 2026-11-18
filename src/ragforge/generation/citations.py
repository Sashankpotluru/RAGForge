"""Validate reference identity, not whether prose is entailed by source evidence."""

import re
from collections.abc import Sequence

from ragforge.domain.models import Citation, ProviderAnswer, RetrievedChunk


def build_citations(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            citation_id=f"C{i}",
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            source=c.source or c.source_location,
            source_location=c.source_location,
            quote=c.text,
        )
        for i, c in enumerate((r.chunk for r in chunks), 1)
    ]


def validate_citation_references(answer: ProviderAnswer, citations: Sequence[Citation]) -> None:
    available = {c.citation_id for c in citations}
    declared = set(answer.citation_ids)
    inline = set(re.findall(r"\[(C\d+)\]", answer.text))
    if not declared or not declared <= available or inline != declared:
        raise ValueError("answer must use matching, known inline and declared citation IDs")
