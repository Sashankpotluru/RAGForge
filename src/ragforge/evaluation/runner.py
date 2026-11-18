"""Offline fixture runner; explicitly separates synthetic from semantic measurements."""

import importlib.metadata
import json
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Any

from ragforge.config import Settings
from ragforge.domain.filters import matches_filter
from ragforge.domain.models import DocumentInput, SearchRequest
from ragforge.evaluation.datasets import EvaluationCase, span_overlaps
from ragforge.evaluation.metrics import binary_ndcg, retrieval_metrics
from ragforge.generation.citations import build_citations
from ragforge.retrieval.rerank import CrossEncoderReranker
from ragforge.retrieval.service import SearchService
from ragforge.runtime import Runtime


def index_id(runtime: Runtime) -> str:
    return sha256(
        "\n".join(c.chunk_id for c in runtime.catalog.active_chunks()).encode()
    ).hexdigest()


def source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    provenance: dict[str, Any] = {
        "version": importlib.metadata.version("ragforge"),
        "kind": "installed-package",
        "commit": None,
        "dirty": None,
    }
    if (root / ".git").exists() and (root / "src/ragforge/evaluation/runner.py").is_file():
        try:

            def git(*args: str) -> str:
                return subprocess.check_output(
                    ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
                ).strip()

            provenance.update(
                kind="git-checkout",
                commit=git("rev-parse", "HEAD"),
                dirty=bool(git("status", "--porcelain")),
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return provenance


def run_fixture(settings: Settings, modes: list[str] | None = None) -> dict[str, Any]:
    raw = files("ragforge.evaluation").joinpath("fixture.json").read_text()
    dataset = json.loads(raw)
    cases = [EvaluationCase.model_validate(c) for c in dataset["cases"]]
    modes = modes or (
        ["dense", "hybrid", "hybrid_rerank"] if settings.reranker_model else ["dense", "hybrid"]
    )
    provenance = source_provenance()
    artifact: dict[str, Any] = {
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": sha256(raw.encode()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "source": provenance,
        "git_commit": provenance["commit"] or "unavailable",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "concurrency": 1,
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["ragforge", "qdrant-client", "fastapi", "pydantic"]
        },
        "limitations": [
            "12 synthetic cases, 6 documents; not a quality benchmark",
            "Hash embeddings are token hashing, not semantic embeddings",
        ],
        "timing_boundary": (
            "search through citation construction, excluding validation, "
            "model load, ingestion and generation"
        ),
        "runs": [],
    }
    if settings.embedding_model != "hash-demo-v1":
        artifact["limitations"].pop()
        artifact["dependencies"]["sentence-transformers"] = importlib.metadata.version(
            "sentence-transformers"
        )
    with tempfile.TemporaryDirectory(prefix="ragforge-eval-") as directory:
        runtime = Runtime(
            settings.model_copy(update={"data_dir": Path(directory), "qdrant_url": None})
        )
        try:
            for doc in dataset["documents"]:
                runtime.ingest(DocumentInput.model_validate(doc))
            reranker = (
                CrossEncoderReranker(settings.reranker_model, settings.reranker_revision)
                if settings.reranker_model
                else None
            )
            search = SearchService(runtime, reranker)
            for mode in modes:
                results: list[dict[str, Any]] = []
                for case in cases:
                    request = SearchRequest(
                        query=case.question,
                        metadata_filter=case.metadata_filter,
                        mode=mode,
                        result_limit=10,
                        candidate_limit=20,
                    )
                    started = perf_counter()
                    evidence = search.search(request)
                    build_citations(evidence)
                    elapsed = (perf_counter() - started) * 1000
                    matches = [
                        {i for i, span in enumerate(case.spans) if span_overlaps(r.chunk, span)}
                        for r in evidence
                    ]
                    relevant_ids = {
                        c.chunk_id
                        for c in runtime.catalog.active_chunks(request.collection)
                        if matches_filter(c.metadata, case.metadata_filter)
                        and any(span_overlaps(c, span) for span in case.spans)
                    }
                    metrics = retrieval_metrics(matches, len(case.spans))
                    metrics["ndcg@10"] = binary_ndcg(
                        [r.chunk.chunk_id for r in evidence], relevant_ids
                    )
                    results.append(
                        {
                            "case_id": case.id,
                            "latency_ms": elapsed,
                            "ranks": [
                                {
                                    "chunk_id": r.chunk.chunk_id,
                                    "source": r.chunk.source,
                                    "spans": sorted(m),
                                }
                                for r, m in zip(evidence, matches, strict=True)
                            ],
                            "eligible_relevant_chunk_ids": sorted(relevant_ids),
                            "metrics": metrics,
                        }
                    )
                latencies = sorted(r["latency_ms"] for r in results)
                artifact["runs"].append(
                    {
                        "mode": mode,
                        "config_id": runtime.config_id,
                        "index_id": index_id(runtime),
                        "settings": settings.model_dump(
                            mode="json",
                            exclude={
                                "provider_base_url",
                                "provider_model",
                                "provider_token",
                                "qdrant_url",
                                "data_dir",
                            },
                        ),
                        "candidate_limit": 20,
                        "result_limit": 10,
                        "rrf_k": 60,
                        "cases": results,
                        "failures": [],
                        "metrics": {
                            metric: sum(r["metrics"][metric] for r in results) / len(results)
                            for metric in results[0]["metrics"]
                        },
                        "latency_ms": {
                            "p50": latencies[(len(latencies) - 1) // 2],
                            "p95": latencies[__import__("math").ceil(0.95 * len(latencies)) - 1],
                        },
                    }
                )
        finally:
            runtime.close()
    return artifact


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
