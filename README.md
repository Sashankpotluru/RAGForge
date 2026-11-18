# RAGForge

[![CI](https://github.com/Sashankpotluru/RAGForge/actions/workflows/ci.yml/badge.svg)](https://github.com/Sashankpotluru/RAGForge/actions/workflows/ci.yml)

RAGForge searches documents, retrieves useful passages, and ties answers back to
their sources. It combines vector search with BM25, merges their rankings, and
can rerank the results before passing evidence to a language model.

This repository contains the full Python backend and a browser demo. Document
ingestion, versioning, persistence, and evaluation live in the backend. The demo
gives you a quick way to try retrieval without installing anything.

## Try it

**[Open the demo](https://sashankpotluru.github.io/RAGForge/)**

Ask **“How often do database backups run?”** The sample document says backups run
every six hours. Inspect the retrieved passage and follow the answer's citation
back to its source.

The hosted demo runs on GitHub Pages. Search happens in your browser, so it works
when the maintainer's computer is off. Keyword search is ready immediately.
Neural search and model answers are optional: the page explains the download
before loading weights onto your device. No account or API key is needed.

The six sample documents are fictional policies written for this project. The
page doesn't upload documents or connect to a private index. Model answers require
a supported browser and enough memory; keyword search works without loading a
model. [Browser setup and limitations](docs/browser-demo.md)

## What's in the backend

- **Document ingestion:** PDF, Markdown, HTML and plain text, with bounded chunks
  and source offsets. HTTPS ingestion is optional and restricted to allowed hosts.
- **Hybrid retrieval:** dense vectors in Qdrant, a persisted BM25 index, reciprocal
  rank fusion, and an optional cross-encoder reranker.
- **Document lifecycle:** SQLite tracks active versions. Repeated ingestion is a
  no-op, replacements are staged, and failed updates can be recovered.
- **Cited answers:** quotes and identifiers travel with retrieval results. A
  configured chat-completions provider can answer questions or rewrite queries.
- **Evaluation:** versioned fixtures, separate baselines, per-question results,
  model revisions, and clearly defined timing boundaries.

The code uses explicit components so each ranking stage and failure path can be
inspected. [Implementation decisions](docs/implementation-notes.md)

## Run the sample UI locally

Run the browser demo locally with an empty dedicated storage directory. Its six synthetic documents are bundled with the application.

```bash
uv sync --locked
uv run --no-sync ragforge demo --data-dir data/public-demo --port 8787
```

Open [the workbench](http://127.0.0.1:8787). The [local demo guide](docs/demo.md)
covers neural retrieval and running Qwen on your own machine.

## Run the backend

Use Python 3.12 or 3.13 and uv 0.13.9. Run from the repository root.

```bash
python3 -m pip install uv==0.13.9
uv sync --locked --extra dev
uv run --no-sync ragforge ingest data/fixtures/sample.md
uv run --no-sync ragforge search "What does this document describe?"
uv run --no-sync ragforge evaluate --output results/fixture.json
uv run --no-sync ragforge serve
```

Visit [OpenAPI documentation](http://127.0.0.1:8000/docs) while the server runs. The default uses embedded Qdrant and a deterministic token-hash embedding. After installation the default retrieval and evaluation paths run offline. Use the service HTTP endpoints for concurrent clients. Stop the server before opening the same embedded data directory from another CLI process.

```bash
curl -sS http://127.0.0.1:8000/v1/documents -H 'Content-Type: application/json' \
  -d '{"source":"memory:guide","content":"Violet flowers grow in the garden.","metadata":{"team":"garden"}}'
curl -sS http://127.0.0.1:8000/v1/search -H 'Content-Type: application/json' \
  -d '{"query":"flowers","collection":"default","mode":"hybrid","metadata_filter":{"conditions":[{"field":"team","operator":"eq","value":"garden"}]}}'
```

## API reference

| Interface | Behavior |
|---|---|
| POST /v1/documents | Supply source and content, or source, MIME type and file_base64. Optional collection and external_id identify a document across updates. |
| POST /v1/documents/url | Fetch a permitted HTTPS URL with redirects and bounded content. Disabled by default. |
| GET or DELETE /v1/documents/{document_id} | Inspect the active/staged version and failure state or remove the document. |
| POST /v1/search | Search one collection using dense, hybrid or hybrid_rerank mode. Optional query rewriting adds another ranking. |
| POST /v1/ask | Generate a cited answer from evidence using the configured provider. Returns 503 when unconfigured. |
| POST /v1/evaluations/run | Submit an empty JSON object to evaluate the packaged fixture and persist its results. |
| GET /healthz or /readyz | Inspect process health or Qdrant collection availability. |

Use `ragforge --help` for CLI commands. Configure retrieval models and optional providers through the environment. [Operations runbook](docs/runbook.md)

## What the evaluation shows

The Python backend was tested on twelve labeled questions across six sample
documents. These runs used pinned tiny neural models:

| Neural retrieval | Hit rate@1 | MRR@10 | Recall@1 |
| --- | ---: | ---: | ---: |
| Dense | 83.3% | 0.889 | 79.2% |
| Hybrid | 91.7% | 0.958 | 87.5% |
| Hybrid + cross-encoder | 100.0% | 1.000 | 95.8% |

Reranking found a relevant first passage for every question in this fixture. That
doesn't establish the same result on a larger or unfamiliar collection. The
[evaluation notes](docs/evaluation.md) explain the labels, metrics and limits;
the [raw results](evaluations/results) include per-question evidence and timings.
These measurements describe the Python backend, not the browser's ONNX models.

## Architecture

![System architecture](docs/architecture-1.png)

This diagram shows the backend and the original tunnel-based demo. Its “Mac must
stay running” label applies to that older deployment. The current GitHub Pages
demo runs in the visitor's browser. [Current hosting topology](docs/architecture.md)

## Tests and containers

```bash
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src
docker compose up --build -d
```

CI runs the Python tests, type checks, fixture evaluation, and integration checks
against Docker Compose and Qdrant. The browser demo has separate ranking tests and
static build checks. [Verification record](docs/verification.md)

## Project layout

| Path | Contents |
| --- | --- |
| `src/ragforge/` | Python ingestion, storage, retrieval, generation and API |
| `src/ragforge/demo/` | Shared workbench UI and isolated Python demo |
| `web/` | Browser retrieval, model runtime, tests and Pages build |
| `tests/` | Python unit, contract and integration tests |
| `evaluations/results/` | Recorded measurements and provenance |
| `docs/` | Design decisions, setup, operations and limitations |

[Architecture diagram](docs/architecture.md) · [Verification](docs/verification.md) · [Security and trust boundaries](docs/security.md) · [Implementation decisions](docs/implementation-notes.md)

The backend is designed for one trusted operator. The separate browser workbench exposes a fixed sample corpus. Original code is MIT licensed. PDF extraction and optional model weights have separate terms. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
