"""Run against a separately started Compose stack."""

import os
import time
from uuid import uuid4

import httpx
import pytest


@pytest.mark.skipif(os.environ.get("RUN_COMPOSE_SMOKE") != "1", reason="requires a running stack")
def test_live_ingestion_and_citation():
    with httpx.Client(
        base_url=os.environ.get("RAGFORGE_API_URL", "http://127.0.0.1:8000")
    ) as client:
        for attempt in range(30):
            try:
                if client.get("/readyz").is_success:
                    break
            except httpx.TransportError:
                pass
            if attempt == 29:
                pytest.fail("service readiness deadline exceeded")
            time.sleep(1)
        collection = "smoke-" + uuid4().hex
        source = "memory:compose"
        created = client.post(
            "/v1/documents", json={"source": source, "content": "violet", "collection": collection}
        )
        created.raise_for_status()
        try:
            found = client.post("/v1/search", json={"query": "violet", "collection": collection})
            found.raise_for_status()
            evidence = found.json()
            assert evidence["citations"][0]["source"] == source
            assert (
                evidence["citations"][0]["chunk_id"] == evidence["evidence"][0]["chunk"]["chunk_id"]
            )
        finally:
            client.delete("/v1/documents/" + created.json()["document_id"]).raise_for_status()
