from pathlib import Path


def test_settings_reads_ragforge_prefixed_environment_values(monkeypatch, tmp_path: Path) -> None:
    from ragforge.config import Settings

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("RAGFORGE_DATA_DIR", str(runtime_dir))
    monkeypatch.setenv("UNRELATED_DATA_DIR", "/ignored")

    settings = Settings()

    assert settings.data_dir == runtime_dir
    assert settings.qdrant_url is None  # Embedded storage needs no running service.
    assert settings.url_ingestion_enabled is False
