"""Source-span labels remain stable when chunk boundaries change."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ragforge.domain.models import Chunk, MetadataFilter


class EvidenceSpan(BaseModel):
    source: str
    source_location: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> EvidenceSpan:
        if self.end <= self.start:
            raise ValueError("span end must exceed start")
        return self


class EvaluationCase(BaseModel):
    id: str
    question: str
    spans: list[EvidenceSpan] = Field(min_length=1)
    metadata_filter: MetadataFilter = Field(default_factory=MetadataFilter)

    @model_validator(mode="after")
    def distinct(self) -> EvaluationCase:
        keys = [(s.source, s.source_location, s.start, s.end) for s in self.spans]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate gold spans")
        return self


def span_overlaps(c: Chunk, span: EvidenceSpan) -> bool:
    return (
        c.source == span.source
        and c.source_location == span.source_location
        and max(c.start_offset, span.start) < min(c.end_offset, span.end)
    )
