"""Environment-backed service configuration."""

from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration that is safe to construct in tests and local development."""

    model_config = SettingsConfigDict(env_prefix="RAGFORGE_", extra="ignore")

    data_dir: Path = Path("data/runtime")
    qdrant_url: str | None = None
    provider_base_url: str | None = None
    provider_model: str | None = None
    provider_token: SecretStr | None = Field(default=None, exclude=True)
    qdrant_collection: str = "ragforge"
    url_ingestion_enabled: bool = False
    url_allowed_hosts: tuple[str, ...] = ()
    request_timeout_seconds: float = Field(default=20.0, gt=0)
    embedding_model: str = "hash-demo-v1"
    embedding_revision: str | None = None
    embedding_dimensions: int = Field(default=256, ge=16, le=4096)
    reranker_model: str | None = None
    reranker_revision: str | None = None
    target_chars: int = Field(default=800, ge=20)
    overlap_chars: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def coherent_settings(self) -> Self:
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be smaller than target_chars")
        if bool(self.provider_base_url) != bool(self.provider_model):
            raise ValueError("provider_base_url and provider_model must be set together")
        return self
