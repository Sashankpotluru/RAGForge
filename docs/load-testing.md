# Python API load testing

This experiment measures the Python search service under concurrent HTTP
requests. It exercises FastAPI, the runtime lock, SQLite, BM25, trained sentence
embeddings and a separate native Qdrant server. It is a reproducible single-host
capacity profile, not a hosted availability guarantee or a claim about maximum
production capacity.

The GitHub Pages demo runs its own retrieval in the browser. Its traffic does
not pass through this Python service. The [retrieval benchmarks](scifact-evaluation.md)
and their Node CPU timings answer a different question from this experiment.

## Workload

The dataset is a deterministic subset of 1,000 SciFact documents. Original IDs
and content are retained. Selection is independent of retrieval outcomes. The
query pool contains 32 distinct queries from SciFact's training split; test
relevance judgments are not used to choose load-test queries.

Documents are ingested through the real API into fresh temporary storage and
an isolated Qdrant collection. Preparation, model loading and warmup are timed
separately from the measured searches. Searches use `POST /v1/search`, hybrid
mode, `candidate_limit=100`, `result_limit=10` and the isolated logical collection.
There is no generation, query rewriting, reranking or concurrent ingestion in
this profile.

The harness runs 1, 4, 8 and 16 concurrent clients, with three 30-second trials
at each level. The fixed client-count order is `1, 4, 8, 16, 8, 16, 1, 4, 16, 8,
4, 1`, with all 32 queries warmed sequentially before each trial. Clients wait
for a response before sending their next request,
with no additional think time. This is a **closed-loop** workload: when responses
slow down, the arrival rate also falls. It does not simulate a fixed external
request arrival rate, and it can understate overload compared with an open-loop
test. [Open and closed workload models](https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/)

## What is counted

Latency spans the complete HTTP request, response-body consumption and response
validation. A successful response must have the expected status and structure,
a request identifier, correctly scoped source documents and consistent evidence
and citations. Timeouts, HTTP errors and validation failures are counted; retries
do not hide unsuccessful attempts.

The issuance window lasts 30 seconds, then outstanding requests are allowed to
finish within their timeout. Throughput divides successful requests by the
actual wall interval including this drain. Percentiles describe the observed
**successful requests**, using nearest-rank p50/p95/p99. Failures remain in the
attempt count and error rate, with their own timings in the raw report. These
numbers are not a promised service-level objective. Per-trial observations remain
available so a pooled mean cannot hide an unsuccessful trial.

## Measured results

All **2,967 measured requests succeeded**, with no HTTP errors, timeouts or
response-validation failures in any of the 12 trials. The 1,000 documents
produced **2,497 active chunks and matching Qdrant points**.

| Concurrent clients | Trials | Successful requests | Successful req/s | p50 | p95 | p99 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 707 | 7.839 | 122 ms | 205 ms | 230 ms | 0 / 707 |
| 4 | 3 | 743 | 8.142 | 470 ms | 601 ms | 627 ms | 0 / 743 |
| 8 | 3 | 753 | 8.117 | 979 ms | 1,087 ms | 1,163 ms | 0 / 753 |
| 16 | 3 | 764 | 7.986 | 2,011 ms | 2,171 ms | 2,382 ms | 0 / 764 |

The totals above pool the three trials at each level. Trial-to-trial variation
remains visible:

| Clients | Trial req/s range | Trial p95 range |
| --- | ---: | ---: |
| 1 | 7.613–8.231 | 202–221 ms |
| 4 | 7.551–8.498 | 550–624 ms |
| 8 | 7.816–8.558 | 994–1,106 ms |
| 16 | 7.656–8.470 | 1,936–2,353 ms |

Throughput stayed near eight successful requests per second as client count
increased, while latency rose. That is consistent with queueing around the
backend's serialized retrieval path. This experiment does not isolate how much
time is spent waiting for the runtime lock versus computing embeddings, reading
the catalog or searching Qdrant. Adding clients did not provide a useful
throughput increase in this configuration.

The measured windows plus drain totaled **369.889 seconds**, compared with
360 planned issuance seconds. Preparation was separate: source selection
0.026 seconds, service startup 5.119 seconds, ingestion 31.121 seconds and index
verification 0.010 seconds. The 384 warmup searches took 48.919 seconds in total
and are excluded from the measured request count.

Execution used an Apple M4 Pro with 24 GiB RAM, Python 3.13.2, Qdrant 1.20.0,
one Uvicorn worker, 40 AnyIO thread tokens, and CPU PyTorch with six intra-op and
one inter-op thread. The embedding checkpoint was
`sentence-transformers-testing/stsb-bert-tiny-safetensors` at revision
`f3cb857cba53019a20df283396bcca179cf051a4`. Process CPU and RSS were not measured.

Measured source commit: `6279dc80d0e006fc2a521b40448b0fc23039ffdf`.
Full report SHA-256:
`df100ca61213acbd442be43702a9304a5fcaf2976a6fec7b2f319c49791130be`.
The report records exact dependency versions, selected source IDs and query
texts, data/model hashes, configuration identity, scheduling and every request.
Its measured source paths were clean; documentation edits elsewhere in the
checkout were in progress and are recorded as such.

## Reproduce the run

Use Python 3.13 and the locked ML environment:

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    revision="f3cb857cba53019a20df283396bcca179cf051a4",
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
)
PY
```

Prepare the official source files using the [SciFact setup](scifact-evaluation.md#reproduce-the-run).
The harness verifies `corpus.jsonl`, `queries.jsonl` and `qrels/train.tsv` under
`data/scifact`. It sorts document and training-query IDs by SHA-256, takes the
first 1,000 documents and the first 32 distinct trimmed, case-insensitive query
texts. Selection does not inspect relevance scores.

For the measured macOS ARM64 configuration, download the official native release:

```sh
ragforge_tools=$(mktemp -d)
curl -fL https://github.com/qdrant/qdrant/releases/download/v1.20.0/qdrant-aarch64-apple-darwin.tar.gz \
  -o "$ragforge_tools/qdrant.tar.gz"
tar -xzf "$ragforge_tools/qdrant.tar.gz" -C "$ragforge_tools"
QDRANT_BINARY="$ragforge_tools/qdrant" \
QDRANT_ARCHIVE="$ragforge_tools/qdrant.tar.gz" \
uv run --no-sync python -m benchmarks.load.run
```

The harness requires committed measurement code. It checks the native archive,
binary and cached model hashes before launching the services, selects available
loopback ports and creates new temporary storage. Downloads require network
access; the measured model execution uses the local cache. Other platforms need
the matching official Qdrant binary and will produce different timings.

The run writes a detailed report at
[`evaluations/results/backend-load-v1.json`](../evaluations/results/backend-load-v1.json)
and an aggregate at
[`benchmarks/data/backend-load-v1/summary.json`](../benchmarks/data/backend-load-v1/summary.json).
Reproduction overwrites those output files, so retain a copy when comparing runs.
The static build binds the aggregate to the full report's SHA-256; only the
aggregate goes to Pages. Runtime logs remain in ignored local storage.

## Boundaries

- The API uses one worker and serializes retrieval snapshots with its runtime
  lock. More clients can increase queueing without improving throughput.
- The client, API and Qdrant run on the same machine. They share compute and
  memory; traffic uses loopback rather than a public network.
- The model is a trained tiny SentenceTransformer checkpoint, separate from
  the browser's quantized MiniLM. Its timings cannot be used as measurements
  of a larger embedding model or a hosted LLM.
- This is read-only, warm search on a fixed collection. It does not establish
  write throughput, large-index scaling, sustained soak reliability, failover,
  multi-worker consistency or performance across machines.
- Concurrent clients are not authenticated users. Multi-user authorization and
  tenant isolation remain separate features; the private API stays on loopback.
- Generated-answer accuracy and citation entailment are not evaluated here.

The temporary API and Qdrant instances use dedicated directories and are stopped
after the experiment. Existing user indexes are not part of the test.
