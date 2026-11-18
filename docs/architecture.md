# RAGForge architecture

RAGForge is an explicit, single-worker retrieval service. The API and CLI compose
the same Python components; the evaluation runner exercises the same retrieval
path. SQLite owns which document versions are visible. Qdrant and the lexical
index are retrieval backends, not competing sources of truth.

## System diagram

```mermaid
flowchart TB
    accTitle: RAGForge architecture and public demo boundary
    accDescr: A trusted API and CLI ingest documents into a private runtime. The public browser reaches a separate sample-only read-only demo through a temporary HTTPS tunnel. Each runtime owns separate SQLite, Qdrant and BM25 stores. Retrieval fuses dense and lexical ranks, optionally reranks, and returns cited evidence before optional generation.
    caller["Trusted local API / CLI"]
    entry["Private runtime<br/>single-worker lock"]
    caller --> entry
    browser["Public browser<br/>evidence workbench"]
    tunnel["Temporary HTTPS tunnel<br/>Mac must stay running"]
    demo["Isolated read-only demo<br/>sample corpus · request limits"]
    browser --> tunnel --> demo
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
    demo -->|sample queries| search
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

The diagram shows shared component types, **not shared data**: the public demo
constructs its own fresh runtime and sample-only stores. It never mounts the
private API or reads an existing user index. Only the trusted API/CLI can ingest
arbitrary documents. The demo publishes evidence, answers, sample corpus, and
checked-in evaluation results; it has no mutation routes.

Solid lines show the core path. Dashed lines require an enabled answer provider.
Dense-only evaluation bypasses BM25/fusion; hybrid evaluation adds BM25 and RRF;
the reranked configuration also reranks the fused candidates. The diagram is
maintained in this file so GitHub can render it without a diagramming service.

An exported [PNG](architecture-1.png) and [SVG](architecture-1.svg) are included
for portfolio presentations. To regenerate with Mermaid CLI 12.2.0:

```bash
npx --package @mermaid-js/mermaid-cli@12.2.0 mmdc \
  -i docs/architecture.md -o docs/architecture.svg \
  -c docs/architecture-render.json --no-font-embed
npx --package @mermaid-js/mermaid-cli@12.2.0 mmdc \
  -i docs/architecture.md -o docs/architecture.png \
  -c docs/architecture-render.json -s 1.5
```

The renderer requires a Chromium browser. The CLI numbers diagrams extracted
from Markdown, hence the `-1` suffix on the exports. This is documentation tooling,
not a runtime dependency of RAGForge.

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
now exposes a synthetic sample corpus; tenant authorization, permanent managed
hosting, and distributed workers remain outside the release scope.

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
