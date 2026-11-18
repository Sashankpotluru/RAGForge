from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragforge.demo.app import create_demo_app


def test_demo_boundary(tmp_path: Path) -> None:
    with TestClient(create_demo_app(tmp_path)) as client:
        for path in ("/v1/documents", "/v1/documents/url", "/v1/evaluate"):
            assert client.post(path).status_code == 404
        assert client.get("/").status_code == 200
        assert (
            client.get("/assets/architecture-evaluation.png").headers["content-type"] == "image/png"
        )
        assert client.get("/assets/evaluate.mjs").status_code != 200
        assert len(client.get("/api/corpus").json()["documents"]) == 6
        assert (
            client.post(
                "/api/search", json={"query": "retention", "collection": "private"}
            ).status_code
            == 422
        )
        assert client.post("/api/search", content=b"x" * 8193).status_code == 413
        response = client.post(
            "/api/search", json={"query": "When do temporary credentials expire?"}
        )
        assert response.status_code == 200
        assert response.json()["hits"][0]["source"] == "fixture:security"


def test_refuse_existing_storage(tmp_path: Path) -> None:
    (tmp_path / "catalog.sqlite3").touch()
    with pytest.raises(ValueError, match="empty"), TestClient(create_demo_app(tmp_path)):
        pass


def test_passage_matching_requires_unique_verbatim_text() -> None:
    from ragforge.demo.local import cite_verbatim
    from ragforge.domain.models import Citation

    citation = Citation(
        citation_id="C1",
        chunk_id="x",
        document_id="y",
        source="s",
        source_location="s",
        quote="Backups run every six hours. Retain them for a month.",
    )
    assert (
        cite_verbatim("Backups run every six hours.", [citation])
        == "Backups run every six hours. [C1]"
    )
    assert cite_verbatim("Six hours", [citation]) == "Six hours [C1]"
    for text in ["every si", "Every day.", "Backups run every six hours. [C9]"]:
        with pytest.raises(ValueError):
            cite_verbatim(text, [citation])
    with pytest.raises(ValueError):
        cite_verbatim(
            "Backups run every six hours.",
            [citation, citation.model_copy(update={"citation_id": "C2"})],
        )


def test_limits_ignore_forwarded_addresses(tmp_path: Path) -> None:
    with TestClient(create_demo_app(tmp_path)) as client:
        for n in range(30):
            result = client.post(
                "/api/search", json={"query": " "}, headers={"X-Forwarded-For": str(n)}
            )
            assert result.status_code == 422
        assert client.post("/api/search", json={"query": "backup"}).status_code == 429


def test_no_private_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGFORGE_URL_ALLOWED_HOSTS", "example.com")
    monkeypatch.setenv("RAGFORGE_QDRANT_URL", "http://invalid.invalid")
    monkeypatch.setenv("RAGFORGE_PROVIDER_BASE_URL", "http://invalid.invalid")
    monkeypatch.setenv("RAGFORGE_DATA_DIR", str(tmp_path / "private"))
    with TestClient(create_demo_app(tmp_path / "public")) as client:
        assert client.get("/api/capabilities").json()["generation"] is None
        assert client.post("/api/answer", json={"query": "backup"}).status_code == 503
        assert client.get("/assets/%2e%2e%2fapp.py").status_code == 404
    assert not (tmp_path / "private").exists()


def test_generation_capacity_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from ragforge.demo.local import LocalGenerator

    entered, release = Event(), Event()

    def busy(*args: object) -> str:
        entered.set()
        assert release.wait(4)
        raise ValueError("unvalidated output")

    monkeypatch.setattr(LocalGenerator, "answer", busy)
    with (
        TestClient(create_demo_app(tmp_path, generate=True)) as client,
        ThreadPoolExecutor() as pool,
    ):
        pending = pool.submit(client.post, "/api/answer", json={"query": "backup"})
        try:
            assert entered.wait(4)
            assert client.post("/api/answer", json={"query": "backup"}).status_code == 429
        finally:
            release.set()
        response = pending.result().json()
        assert response["hits"] and response["generation_error"] and "answer" not in response
