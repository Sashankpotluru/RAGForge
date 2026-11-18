"""Domain records shared across ingestion, retrieval, and delivery layers."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type MetadataScalar = str | int | float | bool
type MetadataValue = MetadataScalar | list[str]
type Metadata = dict[str, MetadataValue]


class MetadataOperator(StrEnum):
    """The small, shared metadata-filter language supported in v1."""

    EQ = "eq"
    IN = "in"
    GTE = "gte"
    LTE = "lte"


class IndexStatus(StrEnum):
    """A document version's visibility in the hybrid index."""

    STAGING = "staging"
    ACTIVE = "active"
    FAILED = "failed"


class IngestDecision(StrEnum):
    """The catalog's deterministic decision for an ingestion request."""

    NEW = "new"
    NOOP = "noop"
    REPLACE = "replace"


class MetadataCondition(BaseModel):
    """One conjunctive metadata predicate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    operator: MetadataOperator
    value: MetadataValue

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object) -> MetadataValue:
        return _validated_metadata_value(value)

    @model_validator(mode="after")
    def validate_operand(self) -> MetadataCondition:
        if self.operator is MetadataOperator.IN and not isinstance(self.value, list):
            raise ValueError("in requires a list of strings")
        if self.operator is MetadataOperator.EQ and isinstance(self.value, list):
            raise ValueError("eq requires a scalar")
        if self.operator in (MetadataOperator.GTE, MetadataOperator.LTE) and (
            isinstance(self.value, bool) or not isinstance(self.value, (int, float))
        ):
            raise ValueError("range operators require a numeric operand")
        return self


class MetadataFilter(BaseModel):
    """An AND-only collection of metadata conditions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conditions: tuple[MetadataCondition, ...] = ()


class DocumentInput(BaseModel):
    """A normalized document ready for catalog registration."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(min_length=1)
    content: str
    external_id: str | None = Field(default=None, min_length=1)
    title: str | None = None
    collection: str = Field(default="default", min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, metadata: object) -> Metadata:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a flat object")
        for key, value in metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metadata keys must be non-empty strings")
            _validate_metadata_value(value)
        return cast(Metadata, metadata)

    @property
    def logical_identity(self) -> str:
        """Prefer caller-controlled identity so a document may safely move."""
        if self.external_id is not None:
            return f"external:{self.external_id}"
        return f"source:{self.source}"


class Document(BaseModel):
    """The catalog's stable view of a logical document."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    logical_identity: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    active_version: str | None = None
    staged_version: str | None = None
    failure_message: str | None = None
    status: IndexStatus
    metadata: Metadata = Field(default_factory=dict)


class Chunk(BaseModel):
    """A citation-ready slice of a document version."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source: str = ""
    title: str = ""
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    metadata: Metadata = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """A caller's retrieval request before optional answer generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    collection: str = Field(default="default", min_length=1)
    metadata_filter: MetadataFilter = Field(default_factory=MetadataFilter)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    result_limit: int = Field(default=5, ge=1, le=50)
    rewrite_query: bool = False
    mode: str = Field(default="hybrid", pattern="^(dense|hybrid|hybrid_rerank)$")

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class RetrievedChunk(BaseModel):
    """A returned chunk with rank-stage provenance."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    rank: int = Field(ge=1)
    scores: dict[str, float] = Field(default_factory=dict)
    source_ranks: dict[str, int] = Field(default_factory=dict)


class Citation(BaseModel):
    """A source-bound reference to retrieved evidence."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ProviderAnswer(BaseModel):
    """A generated answer paired with the citations it claims to use."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    citation_ids: tuple[str, ...] = ()


def _validated_metadata_value(value: object) -> MetadataValue:
    _validate_metadata_value(value)
    return cast(MetadataValue, value)


def _validate_metadata_value(value: object) -> None:
    """Reject structures that cannot be faithfully filtered in both stores."""
    if isinstance(value, bool | str | int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return
    raise ValueError("metadata must be flat scalar values or lists of strings")
