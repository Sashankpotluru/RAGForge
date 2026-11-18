# Retrieval fixture evaluation

The packaged fixture contains six original synthetic documents and twelve questions, including a multi-span question. Each gold span identifies a source, a location and a half-open range in its whitespace-normalized text. A retrieved chunk covers a span when the ranges intersect at that source and location. Cases carry the metadata predicates used at search time. The small corpus permits near-saturated rankings and does not establish quality on real collections.

| Configuration | Recall@1 | Hit rate@1 | MRR@10 | Binary nDCG@10 | Novelty nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Token hash, dense | .833 | .833 | .917 | .938 | .938 |
| Token hash, hybrid | .875 | .917 | .958 | .963 | .963 |
| Tiny transformer, dense | .792 | .833 | .889 | .916 | .916 |
| Tiny transformer, hybrid | .875 | .917 | .958 | .969 | .969 |
| Tiny transformer, hybrid + cross-encoder | .958 | 1.000 | 1.000 | 1.000 | 1.000 |

Binary nDCG@10 credits each distinct relevant chunk once, with logarithmic rank discount. Its ideal ordering includes all relevant chunks in the eligible corpus, including relevant chunks missed by retrieval. The evaluator records their identifiers for each case. Binary and novelty nDCG coincide on these runs but can diverge when chunks cover overlapping or multiple evidence spans.

The tiny semantic dense model scored lower on Recall@1 than the token-hash dense baseline in this fixture. The reranked hybrid run produced a relevant first result for every question and still missed some first-rank gold evidence. The examples are too few to infer a general improvement.

Recall@k divides the number of distinct gold spans found through rank k by the number labeled for that question. Hit rate@k gives one point for any hit. MRR@10 uses the reciprocal rank of the first hit. Novelty nDCG@10 credits a rank once if it adds any previously unseen span, discounts by 1/log2(rank+1), and normalizes by the ideal discounted sum over min(10, gold span count) ranks. A single chunk covering multiple spans still earns one gain at that rank. This custom metric should only be compared under that definition.

Aggregates are case means. Candidate budget is 20, result budget 10 and RRF k is 60. Latency includes search through citation construction; model startup, ingestion, HTTP and generation are excluded. Per-mode p50/p95 use nearest ranks over twelve sequential queries. Hardware, versions, timestamps, model revisions, per-case rankings and actual timings are in the artifacts. The first run's warm-up and cache effects make these timings unsuitable as service SLOs.

[Offline artifact](../evaluations/results/offline-fixture-v1.json) · [Pinned transformer artifact](../evaluations/results/tiny-model-fixture-v1.json) · [Original fixture](../src/ragforge/evaluation/fixture.json) · [Reproduction commands](runbook.md)
