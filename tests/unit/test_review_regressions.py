import pytest

from ragforge.config import Settings
from ragforge.domain.models import Chunk, DocumentInput, MetadataFilter


def test_non_latin_corpus_does_not_crash_bm25(tmp_path):
    from ragforge.indexing.bm25_store import Bm25Store

    store = Bm25Store(tmp_path / "bm25.json")
    store.refresh(
        [
            Chunk(
                chunk_id="a",
                document_id="d",
                version="v",
                ordinal=0,
                text="你好世界",
                source_location="memory:a",
            )
        ],
        "v",
    )
    assert store.search("hello", MetadataFilter(), 5) == []


@pytest.mark.parametrize("ip", ["224.0.0.1", "ff02::1", "0.0.0.0", "169.254.169.254"])
def test_forbidden_ip_categories_rejected(monkeypatch, ip):
    from ragforge.ingestion.url_loader import SafeUrlLoader

    loader = SafeUrlLoader(
        Settings(url_ingestion_enabled=True, url_allowed_hosts=("docs.example",))
    )
    monkeypatch.setattr(loader, "_resolve", lambda host: [ip])
    monkeypatch.setattr(loader, "_fetch", lambda *a: (200, {"content-type": "text/plain"}, b"bad"))
    with pytest.raises(ValueError, match="public"):
        loader.load("https://docs.example/a")


def test_unrelated_ingest_cleans_failed_staged_vectors(tmp_path, monkeypatch):
    from ragforge.runtime import Runtime

    runtime = Runtime(Settings(data_dir=tmp_path))
    runtime.ingest(DocumentInput(source="memory:a", content="violet"))
    real = runtime.lexical.refresh

    def fail_once(chunks, version):
        if version != "rollback":
            raise OSError("disk failure")
        real(chunks, version)

    monkeypatch.setattr(runtime.lexical, "refresh", fail_once)
    with pytest.raises(RuntimeError):
        runtime.ingest(DocumentInput(source="memory:a", content="orange"))
    monkeypatch.setattr(runtime.lexical, "refresh", real)
    runtime.ingest(DocumentInput(source="memory:b", content="green"))
    assert runtime.dense.client.count(runtime.dense.collection).count == 2
    runtime.close()


def test_failed_catalog_delete_preserves_search(tmp_path, monkeypatch):
    from ragforge.runtime import Runtime

    runtime = Runtime(Settings(data_dir=tmp_path))
    doc = runtime.ingest(DocumentInput(source="memory:a", content="violet"))

    def fail(*args):
        raise OSError("catalog unavailable")

    monkeypatch.setattr(runtime.catalog, "delete", fail)
    with pytest.raises(OSError):
        runtime.delete(doc.document_id)
    assert runtime.lexical.search("violet", MetadataFilter(), 1)
    runtime.close()


def test_failed_model_startup_allows_new_config(tmp_path, monkeypatch):
    from ragforge.runtime import Runtime

    def fail(*args):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("ragforge.runtime.SentenceTransformerEmbedding", fail)
    with pytest.raises(RuntimeError):
        Runtime(Settings(data_dir=tmp_path, embedding_model="missing"))
    Runtime(Settings(data_dir=tmp_path)).close()


def test_overlap_validation_before_startup():
    with pytest.raises(ValueError, match="overlap"):
        Settings(target_chars=30, overlap_chars=120)
