import pytest

from ragforge.domain.models import Chunk, ProviderAnswer, RetrievedChunk


def test_citations_bind_source_and_reject_unknown_or_missing_references() -> None:
    from ragforge.generation.citations import build_citations, validate_citation_references

    evidence = [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="a",
                document_id="d",
                version="v",
                ordinal=0,
                text="source text",
                source="memory:a",
                source_location="memory:a#page=1",
            ),
            rank=1,
        )
    ]
    citations = build_citations(evidence)
    assert citations[0].quote == "source text"
    assert citations[0].source == "memory:a"
    validate_citation_references(ProviderAnswer(text="Claim [C1]", citation_ids=("C1",)), citations)
    for answer in [
        ProviderAnswer(text="Claim [C2]", citation_ids=("C2",)),
        ProviderAnswer(text="Claim", citation_ids=()),
        ProviderAnswer(text="Claim [C99]", citation_ids=("C1",)),
    ]:
        with pytest.raises(ValueError):
            validate_citation_references(answer, citations)
