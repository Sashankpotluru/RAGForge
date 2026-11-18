from pathlib import Path

import pytest

from ragforge.domain.models import DocumentInput, MetadataFilter


def test_failed_replacement_retry_and_restart(tmp_path: Path, monkeypatch) -> None:
    from ragforge.config import Settings
    from ragforge.runtime import Runtime

    runtime = Runtime(Settings(data_dir=tmp_path))
    original = DocumentInput(source="memory:a", external_id="a", content="violet flowers")
    first = runtime.ingest(original)
    assert runtime.ingest(original).decision == "noop"
    replacement = original.model_copy(update={"content": "orange flowers"})
    real_upsert = runtime.dense.upsert
    monkeypatch.setattr(runtime.dense, "upsert", lambda *a: (_ for _ in ()).throw(OSError()))
    with pytest.raises(RuntimeError, match="indexing failed"):
        runtime.ingest(replacement)
    assert runtime.catalog.get_by_document_id(first.document_id).status == "failed"
    assert [c.text for c in runtime.catalog.active_chunks()] == ["violet flowers"]
    monkeypatch.setattr(runtime.dense, "upsert", real_upsert)
    assert runtime.ingest(replacement).decision == "replace"
    assert [c.text for c in runtime.catalog.active_chunks()] == ["orange flowers"]
    runtime.close()
    restarted = Runtime(Settings(data_dir=tmp_path))
    assert restarted.lexical.search("orange", MetadataFilter(), 5)
    restarted.delete(first.document_id)
    assert restarted.catalog.active_chunks() == []
    restarted.close()


def test_changed_metadata_source_and_model_config(tmp_path: Path) -> None:
    from ragforge.config import Settings
    from ragforge.runtime import Runtime

    runtime = Runtime(Settings(data_dir=tmp_path))
    doc = DocumentInput(source="memory:a", external_id="a", content="violet")
    first = runtime.ingest(doc)
    result = runtime.ingest(
        doc.model_copy(update={"source": "memory:b", "metadata": {"team": "x"}})
    )
    assert result.document_id == first.document_id
    assert result.decision == "replace"
    assert runtime.catalog.active_chunks()[0].metadata == {"team": "x"}
    assert runtime.catalog.active_chunks()[0].source == "memory:b"
    runtime.close()
    with pytest.raises(ValueError, match="configuration"):
        Runtime(Settings(data_dir=tmp_path, embedding_dimensions=64))


def test_retry_original_after_failed_replacement_keeps_dense_points(tmp_path, monkeypatch):
    from ragforge.config import Settings
    from ragforge.runtime import Runtime

    runtime = Runtime(Settings(data_dir=tmp_path))
    original = DocumentInput(source="memory:a", content="violet garden")
    runtime.ingest(original)
    real = runtime.dense.upsert
    monkeypatch.setattr(runtime.dense, "upsert", lambda *a: (_ for _ in ()).throw(OSError()))
    with pytest.raises(RuntimeError):
        runtime.ingest(original.model_copy(update={"content": "orange"}))
    monkeypatch.setattr(runtime.dense, "upsert", real)
    runtime.ingest(original)
    found = runtime.dense.search(
        "violet",
        MetadataFilter(),
        5,
        allowed_ids=[c.chunk_id for c in runtime.catalog.active_chunks()],
    )
    assert len(found) == 1
    runtime.close()
