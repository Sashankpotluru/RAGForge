"""Closed-loop HTTP observations. Times are monotonic nanoseconds relative to trial start."""

import asyncio
import math
from collections import Counter
from datetime import UTC, datetime
from time import perf_counter_ns

import httpx


def validate(response, payload, sources, config_id, index_id):
    body = response.json()
    request_id = response.headers.get("x-request-id")
    if not request_id or body["request_id"] != request_id:
        raise ValueError("request_id")
    if body["config_id"] != config_id or body["index_id"] != index_id:
        raise ValueError("index/config identity")
    if body["query"] != payload["query"] or body["mode"] != payload["mode"]:
        raise ValueError("search identity")
    if any(body["retrieval_config"].get(k) != v for k, v in payload.items() if k != "query"):
        raise ValueError("retrieval configuration")
    evidence = body["evidence"]
    citations = body["citations"]
    if not isinstance(evidence, list) or not 0 < len(evidence) <= payload["result_limit"]:
        raise ValueError("evidence length")
    seen = set()
    for rank, (hit, citation) in enumerate(zip(evidence, citations, strict=True), 1):
        chunk = hit["chunk"]
        if hit["rank"] != rank or sources[chunk["document_id"]] != chunk["source"]:
            raise ValueError("scoped source/rank")
        if not chunk["chunk_id"] or chunk["chunk_id"] in seen or not chunk["text"]:
            raise ValueError("duplicate or empty chunk")
        seen.add(chunk["chunk_id"])
        if citation["citation_id"] != f"C{rank}" or citation["quote"] != chunk["text"]:
            raise ValueError("citation identity/text")
        if any(
            citation[k] != chunk[k]
            for k in ("source", "source_location", "document_id", "chunk_id")
        ):
            raise ValueError("citation source")
    return request_id


async def collect(
    client, queries, search, sources, config_id, index_id, clients, seconds, timeout=30
):
    utc = datetime.now(UTC).isoformat()
    origin = perf_counter_ns()
    deadline = origin + int(seconds * 1e9)
    rows = []

    async def worker(worker_id):
        sequence = 0
        while (scheduled := perf_counter_ns()) < deadline:
            qid, query = queries[(worker_id + sequence * clients) % len(queries)]
            payload = {**search, "query": query}
            status = request_id = failure = None
            started = perf_counter_ns()
            try:
                async with asyncio.timeout(timeout):
                    response = await client.post("/v1/search", json=payload)
                    status = response.status_code
                    request_id = response.headers.get("x-request-id")
                    if status == 200:
                        validate(response, payload, sources, config_id, index_id)
                    else:
                        failure = "http"
            except (TimeoutError, httpx.TimeoutException):
                failure = "timeout"
            except httpx.RequestError as exc:
                failure = "transport:" + type(exc).__name__
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                failure = "validation:" + type(exc).__name__
            finished = perf_counter_ns()
            if failure is None and finished - started > timeout * 1e9:
                failure = "timeout"
            rows.append(
                dict(
                    worker=worker_id,
                    sequence=sequence,
                    query_id=qid,
                    scheduled_ns=scheduled - origin,
                    start_ns=started - origin,
                    finish_ns=finished - origin,
                    latency_ms=(finished - started) / 1e6,
                    status=status,
                    request_id=request_id,
                    failure=failure,
                )
            )
            sequence += 1

    await asyncio.gather(*(worker(i) for i in range(clients)))
    elapsed = (perf_counter_ns() - origin) / 1e9
    return dict(
        start_utc=utc,
        clients=clients,
        elapsed_s=elapsed,
        requests=sorted(rows, key=lambda row: row["start_ns"]),
    )


def metrics(rows, elapsed):
    successes = sorted(row["latency_ms"] for row in rows if row["failure"] is None)
    failures = len(rows) - len(successes)
    return dict(
        requests=len(rows),
        successes=len(successes),
        failures=failures,
        elapsed_s=elapsed,
        successful_requests_per_s=len(successes) / elapsed,
        latency_ms={
            f"p{p}": successes[math.ceil(p * len(successes) / 100) - 1] if successes else None
            for p in (50, 95, 99)
        },
    )


def diagnostics(rows):
    return dict(
        failure_counts=dict(Counter(row["failure"] for row in rows if row["failure"])),
        status_counts=dict(
            Counter(
                str(row["status"]) if row["status"] is not None else "no_response" for row in rows
            )
        ),
        error_rate=sum(row["failure"] is not None for row in rows) / len(rows) if rows else None,
    )


def summarize(trials):
    scenarios = []
    for clients in sorted({t["clients"] for t in trials}):
        group = [t for t in trials if t["clients"] == clients]
        rows = [row for trial in group for row in trial["requests"]]
        scenarios.append(
            dict(
                clients=clients,
                trials=len(group),
                **metrics(rows, sum(t["elapsed_s"] for t in group)),
            )
        )
    return scenarios
