# RAGForge architecture

RAGForge has two execution paths: a public browser workbench hosted on GitHub
Pages and a Python backend that can be run separately. Both implementations live
in the same GitHub repository. The website does not depend on the Python service.

## Deployment diagram

![SciFact browser search, two retrieval benchmarks and isolated HTTP capacity measurement](architecture-evaluation.png)

GitHub Actions tests the code, builds an allowlisted static artifact, and deploys
that artifact to GitHub Pages. The browser loads the workbench and all 5,183
SciFact abstracts. Keyword search uses the loaded corpus; neural retrieval and
answer generation download their models only when requested.

Preparation computes a pinned MiniLM document index once. A separate offline
runner scores the shared retrieval code on all 300 SciFact test queries and all
1,406 ArguAna counterargument queries. It publishes rankings, metrics, confidence
intervals and provenance. Test judgments are not deployed to the browser.
ArguAna is a recorded offline benchmark; SciFact remains the interactive search
corpus. The original six-policy fixture remains with the Python sample app.

The backend has its own load harness. It starts isolated loopback services,
ingests a fixed 1,000-document SciFact subset and drives actual HTTP search
requests with 1, 4, 8 and 16 concurrent clients. That path measures the Python
service, SQLite, native Qdrant and trained embeddings together. It is separate
from browser searches and the Node retrieval-quality evaluation. Only aggregate
load results are published to the website; temporary service state and logs are
not part of the static build.

## Current public hosting

```mermaid
flowchart LR
    data["BEIR SciFact<br/>5,183 abstracts"] --> prep["Pinned MiniLM index preparation"]
    prep --> repo["GitHub repository<br/>source + corpus + vectors"]
    data --> eval["Shared retrieval benchmark<br/>BM25 / dense / hybrid / reranked"]
    prep --> eval
    labels["SciFact: 300 test queries + qrels"] --> eval
    arguana["ArguAna: 8,674 arguments<br/>1,406 test queries + qrels"] --> eval
    eval --> results["Versioned metrics + intervals"]
    load["Isolated Python API load test<br/>1,000 documents · 32 train queries"] --> capacity["HTTP throughput, latency + errors"]
    capacity --> ci
    repo --> ci["GitHub Actions<br/>test + static build"]
    results --> ci
    ci --> pages["GitHub Pages<br/>interface, corpus, vectors + summary"]
    pages --> browser["Retrieval worker<br/>BM25 / MiniLM / RRF / rerank"]
    models["Pinned public ONNX weights"] -->|optional download| browser
    browser --> evidence["Ranked passages + citations"]
    evidence --> model["Optional Flan-T5 generation<br/>WebAssembly worker"]
    models -->|optional download| model
```

GitHub Pages serves static files; it does not run FastAPI, SQLite, Qdrant, or a
Python model server. The browser edition searches the packaged SciFact corpus in
memory. Its model downloads come from public model repositories, while questions
and retrieved context stay in the browser. Corpus and vector hashes are checked
before neural search. Precomputed vectors avoid embedding thousands of abstracts
on each visit; the browser embeds only the query.

The browser and Python implementations have separate model configurations. The
retrieval benchmarks use the browser ranking implementation with Node CPU inference.
Live browser inference uses WebAssembly; timings depend on the device. Historical
Python fixture measurements remain separately labeled.
[SciFact](scifact-evaluation.md) · [ArguAna](arguana-evaluation.md) ·
[API load test](load-testing.md)

<details>
<summary>Python backend components</summary>

```mermaid
flowchart TB
    accTitle: RAGForge Python backend
    accDescr: A trusted API and CLI ingest documents into a single-worker runtime. SQLite tracks active versions, Qdrant stores dense vectors, and BM25 provides lexical rankings. Retrieval returns cited evidence before optional generation. The evaluation runner measures this same retrieval path. This backend is run separately from the GitHub Pages website.
    caller["Trusted local API / CLI"]
    entry["Private runtime<br/>single-worker lock"]
    caller --> entry
    ingest["Ingestion<br/>load · chunk · stage · publish"]
    entry -->|documents| ingest
    catalog[("SQLite<br/>active versions")]
    vectors[("Qdrant<br/>dense vectors")]
    lexical[("BM25<br/>lexical snapshot")]
    ingest --> catalog
    ingest -->|embeddings| vectors
    ingest --> lexical
    search["Retrieval<br/>scope · candidates · RRF · rerank"]
    entry -->|private queries| search
    catalog -->|active IDs + filters| search
    vectors -->|dense ranks| search
    lexical -->|lexical ranks| search
    evidence["Evidence response<br/>sources · spans · scores · timings"]
    search --> evidence
    provider["Optional generation<br/>local Qwen / configured API<br/>citation-reference validation"]
    evidence -.-> provider
    evaluation["Evaluation runner<br/>dense / hybrid / reranked"]
    evaluation --> search
    evaluation --> results["Versioned results<br/>Recall · HitRate · MRR · nDCG"]
```

The API and CLI compose the same Python components. SQLite owns which document
versions are visible; Qdrant and BM25 supply rankings for eligible chunks. Private
queries and the evaluation runner both call the retrieval service.

The optional local Python sample app constructs its own fresh runtime and
sample-only stores. It never mounts the private API or reads an existing user
index. Only the trusted API/CLI can ingest arbitrary documents. The Python sample
app has no mutation routes. It is a separate way to run the workbench locally.

Solid lines show the core path. Dashed lines require an enabled answer provider.
Dense-only evaluation bypasses BM25/fusion; hybrid evaluation adds BM25 and RRF;
the reranked configuration also reranks the fused candidates.

</details>

The [PNG](architecture-evaluation.png) is the presentation diagram used in the README and
website. The Mermaid references provide editable descriptions of the deployment
and backend paths.

## Consistency boundary

One runtime lock serializes index mutations and retrieval snapshots within one
process. Run one API worker, and do not run a second process against the same data
directory. This deliberately favors a comprehensible consistency model over
throughput. It is not a distributed transaction or multi-worker service.

For replacement, the prior active chunks remain visible while the new version is
staged. Vector upserts and the lexical snapshot must succeed before the SQLite
active-version pointer is promoted. Search restricts vector candidates to active,
eligible chunk identifiers **before** applying the candidate limit. Failed or
obsolete versions therefore cannot consume result slots. The shared metadata
predicate is applied before retrieval in both branches.

SQLite retains normalized chunk text, source locations, and spans. Startup uses
the catalog to recover the lexical snapshot and discard staged/inactive data;
recovery does not silently download source documents. This is recoverable
coordination across independent stores, not an atomic transaction spanning all
three stores.

## ADR-001: Explicit retrieval components

**Status:** Accepted for the initial backend milestone  
**Date:** 2026-11-18

### Context

The portfolio needs inspectable ranking, lifecycle behavior, and reproducible
measurements. The backend is local and single-node. A separate read-only web demo
exposes a synthetic sample corpus. A browser edition is hosted on GitHub Pages;
tenant authorization, managed Python hosting, and distributed workers remain
outside the release scope.

### Decision

Use a small Python composition layer with independently testable loaders,
embedding adapters, Qdrant storage, BM25 scoring, fusion, rerankers, citation
assembly, and evaluation. Keep SQLite authoritative for version visibility.

### Alternatives and trade-offs

| Option | Benefit | Cost in this milestone |
| --- | --- | --- |
| Explicit components, selected | Ranking and failure behavior are visible and testable | More lifecycle code to own |
| Framework-owned RAG chain | Faster initial orchestration | More behavior hidden behind framework defaults |
| Distributed search and worker services | Greater scaling flexibility | More failure modes and deployment dependencies |

### Consequences

The offline path can run without credentials or model downloads. Its hash
embedding is a test/demo mechanism, not a neural model or evidence of semantic
retrieval quality. The default path does not rerank. Real SentenceTransformer
embedding and CrossEncoder reranking adapters are separate, optional dependencies.

The local BM25 snapshot and active-ID filtering prioritize correctness and
auditability at a small scale. Larger corpora need measured capacity testing,
more efficient metadata indexing, and a coordinated distributed visibility
protocol before increasing worker count.

## Trust boundaries

- Source text is untrusted data, including instructions embedded in documents.
  Retrieval is not permission to execute those instructions.
- Source-bound citations establish provenance. Valid citation identifiers do not
  establish that generated claims are entailed by the cited text.
- External generation and rewriting are opt-in. A provider receives retrieved
  text; check data-sharing requirements before enabling it.
- Metadata filtering is a search feature, not tenant authorization. Do not expose
  the private administration API directly to the public internet. The dedicated
  demo app exposes only sample data, with body/query limits, a global request
  budget, and bounded concurrency.
- Model, chunking, and index configuration changes require an explicit reindex
  into a new data directory/collection rather than mixing incompatible vectors.

See [security](security.md), [operations](runbook.md),
[evaluation methodology](evaluation.md), and
[implementation decisions](implementation-notes.md) for the release-specific
contracts and limitations.

## Upstream references

- [Qdrant filtering](https://qdrant.tech/documentation/search/filtering/)
- [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
