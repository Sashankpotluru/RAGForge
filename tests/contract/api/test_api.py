from pathlib import Path

from fastapi.testclient import TestClient

from ragforge.config import Settings


def test_real_api_ingest_search_status_delete_and_errors(tmp_path: Path) -> None:
    from ragforge.api.app import create_app

    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        created = client.post(
            "/v1/documents", json={"source": "memory:a", "content": "violet garden"}
        )
        assert created.status_code == 200, created.text
        doc_id = created.json()["document_id"]
        found = client.post("/v1/search", json={"query": "violet"})
        assert found.status_code == 200, found.text
        body = found.json()
        assert body["citations"][0]["chunk_id"] == body["evidence"][0]["chunk"]["chunk_id"]
        assert body["request_id"] == found.headers["x-request-id"]
        assert body["config_id"] and body["index_id"]
        assert client.get(f"/v1/documents/{doc_id}").json()["status"] == "active"
        assert client.post("/v1/ask", json={"query": "violet"}).status_code == 503
        assert (
            client.post("/v1/search", json={"query": "violet", "rewrite_query": True}).status_code
            == 400
        )
        invalid = client.post("/v1/search", json={"query": " "})
        assert invalid.status_code == 422 and invalid.json()["request_id"]
        for invalid_body in [
            {"colletion": "other"},
            {"metadata_filter": {"condition": []}},
            {
                "metadata_filter": {
                    "conditions": [{"field": "team", "operator": "eq", "value": "x", "typo": 1}]
                }
            },
        ]:
            assert (
                client.post("/v1/search", json={"query": "violet", **invalid_body}).status_code
                == 422
            )
        assert (
            client.post("/v1/documents/url", json={"url": "https://example.com"}).status_code == 400
        )
        assert client.delete(f"/v1/documents/{doc_id}").status_code == 200
        assert client.post("/v1/search", json={"query": "violet"}).json()["evidence"] == []
        assert client.get(f"/v1/documents/{doc_id}").status_code == 404


def test_uploaded_markdown_and_generator_validation(tmp_path: Path) -> None:
    import base64

    from ragforge.api.app import create_app
    from ragforge.domain.models import ProviderAnswer

    class BadProvider:
        def answer(self, *args):
            return ProviderAnswer(text="false [C9]", citation_ids=("C9",))

    with TestClient(
        create_app(Settings(data_dir=tmp_path), answer_provider=BadProvider())
    ) as client:
        response = client.post(
            "/v1/documents",
            json={
                "source": "upload:notes.md",
                "file_base64": base64.b64encode(b"violet notes").decode(),
                "mime_type": "text/markdown",
            },
        )
        assert response.status_code == 200, response.text
        assert client.post("/v1/ask", json={"query": "violet"}).status_code == 502
