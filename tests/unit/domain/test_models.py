import pytest
from pydantic import ValidationError


def test_external_id_keeps_logical_identity_when_source_moves() -> None:
    from ragforge.domain.models import DocumentInput

    original = DocumentInput(
        source="file:///knowledge/employee-handbook.md",
        content="Original handbook text.",
        external_id="employee-handbook",
    )
    moved = DocumentInput(
        source="file:///archive/2026/employee-handbook.md",
        content="Updated handbook text.",
        external_id="employee-handbook",
    )

    assert original.logical_identity == "external:employee-handbook"
    assert moved.logical_identity == original.logical_identity


def test_rejects_nested_metadata_values() -> None:
    from ragforge.domain.models import DocumentInput

    with pytest.raises(ValidationError, match="flat"):
        DocumentInput(
            source="file:///knowledge/security.md",
            content="Security guidance.",
            metadata={"owner": {"name": "security"}},
        )


def test_rejects_unknown_metadata_operator() -> None:
    from ragforge.domain.models import MetadataCondition

    with pytest.raises(ValidationError):
        MetadataCondition(field="department", operator="contains", value="engineering")


def test_chunk_requires_a_nonempty_source_location() -> None:
    from ragforge.domain.models import Chunk

    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            version="version-1",
            ordinal=0,
            text="A source-backed chunk.",
            source_location="",
        )
