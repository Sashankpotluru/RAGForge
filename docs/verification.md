# Release verification

Verified on 2026-11-18 on macOS ARM64 with Python 3.13.2. This record separates
local checks, model measurements, and hosted build verification.

## SciFact benchmark release

The public workbench now uses 5,183 abstracts rather than the six-policy fixture.
All 300 official test queries completed in each of four retrieval modes. The
[measurement report](scifact-evaluation.md) records the fixed protocol, model/data
hashes, timings, bootstrap intervals and both gains and regressions.

- **13 Node tests passed**, covering metrics, incomplete evaluations, qrels,
  vector integrity, candidate bounds, excerpt citation checks, cancellation and
  the exact public artifact allowlist.
- **73 Python tests passed, two opt-in integration tests skipped locally**.
  Ruff formatting/lint and mypy passed. The existing unused PyMuPDF mypy
  configuration note is non-failing.
- **6,000 metric comparisons matched exactly** against `pytrec-eval-terrier`
  0.5.11 using the original test qrels. This checks every query/method/metric,
  without rerunning or substituting the model rankings. The audit does not
  validate bootstrap coverage or generated answers.
- Two preparations produced identical 7,961,088-byte document vectors. The
  static build checks corpus/vector alignment and binds its aggregate results
  to the full report's SHA-256.
- Real Chromium checks exercised all four search modes with pinned WASM models,
  corpus filtering/paging and cancel/retry at mobile width. Keyword search made
  no neural model/runtime/vector request. A real Flan-T5 run without a unique
  phrase match correctly withheld its generated answer and kept the evidence.

[CI run 44394079909](https://github.com/Sashankpotluru/RAGForge/actions/runs/44394079909)
passed for release commit `982657c`: 13 browser tests, 73 Python tests with two
opt-in skips, and three additional live Docker Compose/Qdrant integration tests.
The Python run retained the upstream Starlette deprecation warning described
below. [Pages deployment 44394080071](https://github.com/Sashankpotluru/RAGForge/actions/runs/44394080071)
also passed. The hosted corpus, manifest, vectors, aggregate results and diagram
returned HTTP 200 and matched the committed bytes exactly. The public website
completed actual hybrid reranking and BM25 searches over SciFact; its Results
view displayed the recorded benchmark and its Architecture view the new diagram.

The sections below retain the earlier backend and six-policy release checks;
their fixture results do not describe SciFact retrieval quality.

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

## GitHub Pages browser edition

The browser release adds nine Node tests for retrieval rankings, fusion,
citation matching, worker cancellation and failures, and the static artifact's
contents. All nine passed. The full Python suite still passed 73 tests with two
opt-in service skips; formatting, lint, typing and fixture evaluation passed.

The workbench was tested from a static server at the `/RAGForge/` project path.
Real MiniLM embeddings and the MS MARCO cross-encoder both ranked the backup
policy first for the backup-frequency question. Keyword search required no
model download. The shared Python UI also completed dense and hybrid searches.

Actual Flan-T5 Base inference in the browser produced these observations:

| Question | Displayed answer |
| --- | --- |
| How often do database backups run? | every six hours [C1] |
| How long are application logs retained? | seven days [C1] |
| How long are database backups retained? | thirty days [C1] |
| What color is the database server? | No cited answer; the model abstained |

These are individual functional checks, not an accuracy benchmark. Each displayed
answer matched exactly one supplied source passage. That match checks provenance;
it does not prove that every generated answer is relevant to the question.

Canceling generation restored the controls, and the next search succeeded.
Desktop and 390px layouts showed no horizontal page overflow. Corpus, recorded
results, citation navigation and the supplied architecture image were checked.

The release was then published to
[GitHub Pages](https://sashankpotluru.github.io/RAGForge/).
[Deployment run 44386524662](https://github.com/Sashankpotluru/RAGForge/actions/runs/44386524662)
passed for commit `c69ae47`.
[CI run 44386524707](https://github.com/Sashankpotluru/RAGForge/actions/runs/44386524707)
passed all nine browser tests, 73 Python tests and three additional live
Docker Compose/Qdrant integration tests. The Python run emitted the upstream
Starlette TestClient deprecation warning described above.

The public page completed real hybrid retrieval, cross-encoder reranking and
Flan-T5 generation, returning “every six hours [C1]” for the backup-frequency
question. The old tunnel, Python demo and local static preview were shut down.
The GitHub Pages site remained available. The repository homepage now points to
this stable URL; no request depends on a service running on the maintainer's Mac.

## Review and boundaries

Independent review identified failure cases in initialization, deletion, cache
recovery, staged-vector cleanup, chunking, metadata scope validation, and query
score provenance. The release includes regression tests for the accepted findings.
CLI input-error handling is also covered by regression tests.

The architecture image now shows the GitHub Actions deployment, static GitHub
Pages hosting, browser retrieval and optional generation, and the separately run
Python backend. The README and website use identical copies. The previous
tunnel diagram was replaced, and the surrounding descriptions were updated.
The portfolio root contains exactly the ten requested project directories; only
RAGForge is implemented.

Not verified or delivered in this release:

- Docker/Compose execution on this Mac (covered instead by the hosted Linux run).
- Hosted LLM compatibility or answer entailment using a real provider.
- Production load, multi-worker consistency, tenant authorization, or retrieval
  improvement across multiple independent real-world collections.
- Managed Python backend hosting or the other nine portfolio projects.

The [security boundaries](security.md) and [implementation decisions](implementation-notes.md)
are part of the release contract, not optional production-hardening claims.
