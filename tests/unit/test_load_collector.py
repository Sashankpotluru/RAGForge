"""Exercise timeout, HTTP, transport and invalid-response accounting without model downloads."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_closed_loop_drains_and_counts_every_outcome(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from benchmarks.load.collector import collect, diagnostics, metrics, summarize

    # Keep the issuance window deterministic on slow CI hosts. The transport's last
    # response crosses the deadline and asyncio's real timeout cancels its wait.
    elapsed_ns = 0
    monkeypatch.setattr("benchmarks.load.collector.perf_counter_ns", lambda: elapsed_ns)
    calls = []
    search = dict(mode="hybrid", collection="scope", result_limit=10, candidate_limit=100)
    chunk = dict(
        chunk_id="c", document_id="d", text="hello", source="scifact:1", source_location="p"
    )
    citation = dict(
        citation_id="C1",
        chunk_id="c",
        document_id="d",
        quote="hello",
        source="scifact:1",
        source_location="p",
    )

    async def handler(req):
        nonlocal elapsed_ns
        elapsed_ns += 1_000_000
        i = len(calls)
        query = json.loads(req.content)["query"]
        calls.append(query)
        if i == 1:
            return httpx.Response(503)
        if i == 2:
            raise httpx.ConnectError("disconnected")
        if i >= 4:
            elapsed_ns += 90_000_000
            await asyncio.sleep(1)
        response = dict(
            request_id="id",
            config_id="config",
            index_id="index",
            mode="hybrid",
            query=query,
            retrieval_config=search,
            evidence=[dict(rank=1, chunk=chunk)],
            citations=[{**citation, "source": "other" if i == 3 else "scifact:1"}],
        )
        return httpx.Response(200, json=response, headers={"x-request-id": "id"})

    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(handler)
    ) as client:
        trial = await collect(
            client,
            [("0", "hello")],
            search,
            {"d": "scifact:1"},
            "config",
            "index",
            clients=1,
            seconds=0.04,
            timeout=0.08,
        )
    rows = trial["requests"]
    assert len(calls) == len(rows) == 5
    assert [r["failure"] for r in rows] == [
        None,
        "http",
        "transport:ConnectError",
        "validation:ValueError",
        "timeout",
    ]
    assert diagnostics(rows)["status_counts"] == {"200": 2, "503": 1, "no_response": 2}
    assert trial["elapsed_s"] >= 0.08
    summary = summarize([trial, trial])[0]
    assert summary["requests"] == 10 and summary["successes"] == 2 and summary["failures"] == 8
    assert summary["latency_ms"] == dict.fromkeys(["p50", "p95", "p99"], rows[0]["latency_ms"])
    assert summary["successful_requests_per_s"] == 1 / trial["elapsed_s"]
    assert metrics(rows[1:], 1)["latency_ms"] == dict.fromkeys(["p50", "p95", "p99"])
