# Operations runbook

## Storage and deployment

The default runtime writes SQLite records, Qdrant files and the BM25 snapshot under `data/runtime`. Set `RAGFORGE_DATA_DIR` to select a durable directory. Use one writer and one service worker per data directory and dedicated Qdrant collection. Shut down the process before copying or restoring the directory. In remote mode back up and restore the corresponding Qdrant collection together with the local files. Use a new directory and collection when changing embedding or chunking settings.

For a separate server, set `RAGFORGE_QDRANT_URL=http://127.0.0.1:6333`. The physical collection defaults to `ragforge` and is configurable with `RAGFORGE_QDRANT_COLLECTION`. A request's logical collection is separately scoped in the catalog. All local operators with filesystem or service access can select any logical collection.

```bash
docker compose up --build -d
curl -f http://127.0.0.1:8000/readyz
RUN_COMPOSE_SMOKE=1 RUN_QDRANT_INTEGRATION=1 uv run --no-sync pytest tests/integration
docker compose down
```

Compose uses named volumes for API state and Qdrant storage. The API runs as UID 10001. Container execution requires Docker and Compose; local verification used native Qdrant. CI defines the Compose smoke job.

## Models and generation

```bash
uv sync --locked --extra dev --extra ml
export RAGFORGE_EMBEDDING_MODEL=sentence-transformers-testing/stsb-bert-tiny-safetensors
export RAGFORGE_EMBEDDING_REVISION=f3cb857cba53019a20df283396bcca179cf051a4
export RAGFORGE_RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L2-v2
export RAGFORGE_RERANKER_REVISION=81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc
export RAGFORGE_DATA_DIR=data/runtime-ml
uv run --no-sync ragforge evaluate --output results/tiny.json
```

Initial model provisioning downloads weights from Hugging Face. Once cached, use `HF_HUB_OFFLINE=1`. The default hybrid mode fuses dense and BM25 with reciprocal-rank fusion (k=60). Specify `hybrid_rerank` to score fused candidates with the cross-encoder. Configuration errors surface when an unavailable stage is requested.

`RAGFORGE_EMBEDDING_DEVICE=cpu` selects CPU execution for the SentenceTransformer
adapter, as used in the [API load test](load-testing.md). When unset, the model
library selects its device as before. This setting does not select a model or
enable generation.

For chat completions, set `RAGFORGE_PROVIDER_BASE_URL` (including any /v1 prefix) and `RAGFORGE_PROVIDER_MODEL`. Supply `RAGFORGE_PROVIDER_TOKEN` if required. The configured endpoint receives questions and cited text for answers, or query text for rewriting. It must return JSON content with text and citation_ids for answers, or query for rewriting. Enable rewriting per request with `rewrite_query=true`. Retrieval alone needs no provider.

## Recovery and checks

Stable identity prefers external_id, then canonical source, within a collection. Identical submissions are no-ops. A replacement stages vectors, refreshes the lexical snapshot, and promotes the active version in SQLite. A failed replacement keeps the previous version searchable. Retry the original request to resume. Startup removes inactive vectors using retained catalog IDs, then rebuilds BM25 from active chunks.

Run `uv run --no-sync ruff format --check .`, `uv run --no-sync ruff check .`, `uv run --no-sync mypy src`, and `uv run --no-sync pytest`. To exercise a separate Qdrant server run `RUN_QDRANT_INTEGRATION=1 RAGFORGE_QDRANT_URL=http://127.0.0.1:6333 uv run --no-sync pytest tests/integration/indexing`. This creates and deletes a uniquely named test collection.

