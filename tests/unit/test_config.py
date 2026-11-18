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


def test_model_device_reaches_encoder_and_preserves_legacy_catalog_default(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    from ragforge.config import Settings
    from ragforge.runtime import Runtime

    calls = []

    class Encoder:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))

        def get_sentence_embedding_dimension(self):
            return 128

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder)
    )
    settings = Settings(data_dir=tmp_path, embedding_model="test", embedding_revision="r1")
    legacy_config = {
        key: getattr(settings, key)
        for key in (
            "embedding_model",
            "embedding_revision",
            "embedding_dimensions",
            "target_chars",
            "overlap_chars",
            "qdrant_collection",
            "qdrant_url",
        )
    }
    legacy_id = hashlib.sha256(json.dumps(legacy_config, sort_keys=True).encode()).hexdigest()
    runtime = Runtime(settings)
    assert runtime.config_id == legacy_id
    runtime.close()

    monkeypatch.setenv("RAGFORGE_EMBEDDING_DEVICE", "cpu")
    runtime = Runtime(
        Settings(data_dir=tmp_path / "explicit", embedding_model="test", embedding_revision="r1")
    )
    assert runtime.config_id != legacy_id
    assert calls == [
        ("test", {"revision": "r1", "device": None}),
        ("test", {"revision": "r1", "device": "cpu"}),
    ]
    runtime.close()
