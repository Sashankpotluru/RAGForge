# Release verification

Verified on 2026-11-18 on macOS ARM64 with Python 3.13.2. This record separates
local checks, model measurements, and hosted build verification.

## Local quality gates

Run these commands from the repository root after the locked development setup:

| Command | Observed result |
| --- | --- |
| `python -m pytest -q` | **73 passed, 2 skipped** |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed |
| `mypy src` | No issues in 36 source files; one non-failing unused PyMuPDF override note |
| `uv pip check` | 83 installed packages compatible in the development environment |
| `git diff --check` | Passed |
| `uv build --wheel` | Built `ragforge-0.1.0-py3-none-any.whl` |

The default test run skips the explicitly enabled remote-Qdrant and running-Compose
tests. Embedded Qdrant tests run in the default suite. The separate native-server
run below covers remote Qdrant. Docker was then verified on GitHub's Linux runner.

## Hosted container verification

[GitHub Actions run 44380235610](https://github.com/Sashankpotluru/RAGForge/actions/runs/44380235610)
passed on the clean public release. The run installed the locked dependencies,
passed formatting, lint and typing, ran **73 tests with two opt-in skips**, and
executed the fixture evaluator. It then built the non-root API image, started
Docker Compose with Qdrant v1.20.0, and passed **all three integration tests**,
including live ingestion, cited retrieval, cleanup, and a remote vector round trip.
Compose was shut down and its test volumes removed afterward.

## End-to-end checks

The actual Uvicorn service was launched with temporary storage and
used HTTP requests to verify ingestion, identical-submission no-op, collection
isolation, citations, request identifiers, invalid-query rejection, deletion,
status, and persistence across restart. An unconfigured answer endpoint returned
503. After configuring a loopback test gateway, answer generation and query
rewriting both exercised `/v1/chat/completions` successfully. No hosted model
API or real API token was used in this check.

A second, clean virtual environment installed the **non-editable wheel** and ran
the CLI from a directory outside this checkout. Checks covered help, ingestion,
search, clean JSON output, missing-input errors, and the packaged twelve-case
evaluation. Evaluation provenance correctly identified an installed package
instead of attributing an unrelated working directory's Git commit.

The freely resolved wheel environment also exercised newer compatible dependencies.
Starlette emitted a TestClient deprecation warning on stderr; CLI JSON stdout
remained parseable. Use `uv.lock` for the documented, reproducible environment.

## Real vector server and models

An official native **Qdrant v1.20.0** server ran on
loopback in temporary storage, with telemetry disabled. Its macOS ARM64 archive
was checked against SHA-256
`531904e822050790bb8c1da6705f801ee48b91eb4c229d0d605ff06f6331b20a`.
With that server listening on port 16333:

```bash
RUN_QDRANT_INTEGRATION=1 \
RAGFORGE_QDRANT_URL=http://127.0.0.1:16333 \
uv run --no-sync pytest -q tests/integration/indexing
```

Both the embedded and real server round-trip tests passed. The temporary server
was stopped afterward. See the [integration tests](../tests/integration/indexing)
and [operations runbook](runbook.md) for the reproducible contract.

Actual SentenceTransformer and CrossEncoder weights were provisioned, then the
fixture was run with `HF_HUB_OFFLINE=1` using pinned revisions. Five measured
configurations are checked in: two token-hash runs and three neural-model runs.
The [evaluation report](evaluation.md) links all per-case results, settings,
model revisions, source provenance, and latency measurements. These are small
fixture measurements, not general quality or capacity claims.

## Python-backed demo (initial release)

The separate sample workbench was exercised in a browser at desktop (1280px)
and phone (390px) widths. Search, generated answers, citation navigation, corpus,
recorded results, and the packaged architecture image were checked. The page did
not overflow horizontally at either width.

The real local Qwen model answered three sequential sample questions with text
that matched retrieved evidence. The service attached the uniquely matching
source citation. [Outputs and observed timings](../evaluations/results/local-generation-v1.json)
are recorded separately from retrieval evaluation. These examples are not a
general answer-quality benchmark. The demo performs constrained extractive
generation; it does not guarantee that an extracted phrase answers every query.

The public application has no administration routes. Tests cover the isolated
sample index, rejection of existing storage, independence from private backend
configuration, body limits, globally shared rate limits, and concurrent-generation
rejection. [Demo operation and limits](demo.md)

The original temporary HTTPS session was checked on 2026-11-18: page, health,
corpus, results and diagram returned 200;
real local generation returned cited evidence; administration routes returned 404,
unknown query fields 422, oversized bodies 413, and path traversal 404. The browser
also completed generation and followed its citation to the matching passage.
That deployment used a temporary tunnel to the read-only Python demo.

## Review and boundaries

Independent review identified failure cases in initialization, deletion, cache
recovery, staged-vector cleanup, chunking, metadata scope validation, and query
score provenance. The release includes regression tests for the accepted findings.
CLI input-error handling is also covered by regression tests.

The original architecture reference was rendered and visually inspected. The
current presentation image is the supplied PNG, copied unchanged into the docs
and demo assets (SHA-256 `0150e1c11c4b22821c6a2fb6874a61ff260bad16b23073d4cefb7361f5c29afc`).
The portfolio root contains exactly the ten requested project directories; only
RAGForge is implemented.

Not verified or delivered in this release:

- Docker/Compose execution on this Mac (covered instead by the hosted Linux run).
- Hosted LLM compatibility or answer entailment using a real provider.
- Production load, multi-worker consistency, tenant authorization, or general
  retrieval improvement on an independent real-world benchmark.
- Managed Python backend hosting or the other nine portfolio projects.

The [security boundaries](security.md) and [implementation decisions](implementation-notes.md)
are part of the release contract, not optional production-hardening claims.
