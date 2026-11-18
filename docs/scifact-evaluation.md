# SciFact retrieval evaluation

This page describes **scifact-v2**, measured after correcting cross-encoder pair
truncation. The original [scifact-v1 report](../evaluations/results/scifact-v1.json)
and its [metric audit](../evaluations/results/scifact-v1-metric-audit.json) remain
unchanged for comparison. The new [ArguAna benchmark](arguana-evaluation.md) uses
the same corrected ranking implementation and model settings.

RAGForge searches all **5,183 scientific abstracts** in BEIR SciFact. This report
uses all **300 official BEIR test queries** and their **339 positive relevance
judgments**. The collection and query split come from the published benchmark;
they were not written for this project. A relevant paper may support or refute a
claim. Retrieval relevance does not establish that the claim is true.
[SciFact paper](https://aclanthology.org/2020.emnlp-main.609/) ·
[BEIR collection](https://github.com/beir-cellar/beir)

## Results

Measured on 2026-11-18 using Node.js 22.23.1, ONNX Runtime 1.21.0 and an Apple M4
Pro with 24 GiB RAM. Each mode performed its own full search. Times exclude model
loading, document preparation and the training-query warmup.

| Method | HitRate@1 | MRR@10 | nDCG@10 | Recall@10 | Recall@100 | Median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.5433 | 0.6316 | 0.6629 | 0.7833 | 0.8826 | 0.95 / 2.16 |
| Dense MiniLM | 0.5033 | 0.6056 | 0.6465 | 0.7867 | 0.9250 | 6.46 / 8.55 |
| Hybrid (RRF) | 0.5400 | 0.6448 | 0.6802 | 0.8096 | 0.9550 | 6.68 / 9.47 |
| Hybrid + reranking | 0.5600 | 0.6484 | 0.6809 | 0.8096 | 0.9550 | 466.54 / 513.94 |

| Method | nDCG@10 95% interval | Paired change from BM25 | 95% interval for change |
| --- | --- | ---: | --- |
| BM25 | 0.6179–0.7106 | — | — |
| Dense MiniLM | 0.6018–0.6934 | −0.0164 | −0.0543 to +0.0212 |
| Hybrid (RRF) | 0.6389–0.7242 | +0.0173 | −0.0055 to +0.0398 |
| Hybrid + reranking | 0.6377–0.7258 | +0.0180 | −0.0161 to +0.0518 |

Hybrid improved observed Recall@100 by **7.24 percentage points** over BM25.
The top-10 ranking differences are less conclusive: every paired nDCG interval
above includes zero. Reranking increased nDCG@10 by only 0.0007 over hybrid while
increasing median latency about 70-fold. Recall@10 was unchanged.
BM25 remains a strong, inexpensive baseline for this collection.

The reranked method retains hybrid's candidate set, so their identical Recall@100
is expected. Reranking changes order; it cannot recover a paper outside the first
100 fused candidates. Results do not establish general superiority or a new
state of the art.

## Fixed protocol

The benchmark imports the same retrieval and neural-input code as the website.
The settings below were committed before the final test run. There was no
SciFact fine-tuning, test-set parameter search or query exclusion. Train queries
provide the warmup and the six example claims shown in the UI; these examples
are selected by numeric ID, not measured retrieval quality. The 809 train-query
IDs and 300 test-query IDs do not overlap.

| Component | Setting |
| --- | --- |
| Retrieval unit | Complete title plus newline plus abstract; official document ID |
| BM25 | k1=1.2, b=0.75; lowercase Unicode letter/number tokens, fixed stop list, unique query terms |
| Dense model | `Xenova/all-MiniLM-L6-v2`, q8; revision `751bff37182d3f1213fa05d7196b954e230abad9` |
| Embeddings | 384 dimensions; attention-masked mean pooling, L2 normalization, dot product |
| Dense input | First 256 tokenizer positions including special tokens; batch 8 |
| Fusion | First 100 BM25 and dense candidates; reciprocal rank fusion with k=60; keep first 100 |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2`, q8; revision `a09144355adeed5f58c8ed011d209bf8ee5a1fec` |
| Reranker input | 256 positions; reserve CLS and two SEP tokens, longest-first right truncation of query/document content, tie trims document; batch 8 |
| Reranking | Score first 50 fused candidates; preserve the remaining 50 in fused order |
| Ties | Ascending official corpus row |
| Evaluation order | Numeric test-query ID; BM25, dense, hybrid, reranked sequentially |
| Runtime | Transformers.js 3.8.1; Node CPU, 6 intra-op threads, 1 inter-op thread |
| Confidence intervals | 2,000 shared query-bootstrap samples; seed 20261118; percentile 95% intervals |

Test qrels go only to the scorer. The Pages artifact contains neither test
judgments nor saved test rankings. It searches the corpus when a visitor submits
a query. The separate full report publishes judgments and rankings for audit.

The v2 pair encoder fixes an input defect discovered while adding ArguAna:
the prior tokenizer could truncate all document tokens for long queries. The
corrected encoder preserves both nonempty segments, their separator tokens,
token-type IDs and attention masks. Short pairs that fit are checked against
the native tokenizer. This is a correctness change, selected from token-level
inspection rather than retrieval-score tuning. The diagnostic and its scope
are described in the [ArguAna report](arguana-evaluation.md).

## Metrics and independent check

Each metric is calculated per query and averaged across all 300 queries.
HitRate@1 checks whether the first result has a positive judgment. MRR@10 is the
reciprocal rank of the first relevant result within ten, or zero. Recall@10 and
@100 divide retrieved relevant documents by the query's number of judged
relevant documents. nDCG@10 discounts relevance by rank and normalizes against
the ideal ranking. Unjudged documents receive zero gain. All SciFact test gains
in this distribution are binary.

The 95% bootstrap intervals resample query IDs with replacement. Paired changes
resample the same IDs for each method and BM25. Related claims may violate the
independence assumption; these intervals do not describe a new collection or
the probability that an answer is correct.

The saved rankings were independently scored with
[`pytrec-eval-terrier` 0.5.11](https://pypi.org/project/pytrec-eval-terrier/0.5.11/),
a binding to `trec_eval`, using the original checksummed test qrels. All **6,000
comparisons matched exactly** (maximum absolute difference 0.0, tolerance 1e-12).
The audit uses tuple order as the ranking, because cross-encoder logits and the
unreranked RRF tail have different score scales. MRR is evaluated on only the
first ten results. This check validates metric values, not model inference or
bootstrap coverage. [Audit script](../benchmarks/verify_metrics.py) ·
[Audit record](../evaluations/results/scifact-v2-metric-audit.json)

## Failure cases

These examples are the largest nDCG@10 change against BM25 for each method;
ties use numeric query ID. Both gains and losses are retained in the report.
The query text is a benchmark claim, not an assertion by RAGForge.

| Method | Largest gain | Largest loss |
| --- | --- | --- |
| Dense | Query 514, dietary calcium and hyperparathyroidism: +1.0000 | Query 70, PPM1D and p53: −1.0000 |
| Hybrid | Query 72, Admpchordin: +0.6990 | Query 660, ivermectin and onchocerciasis: −0.6667 |
| Hybrid + reranking | Query 575, yeast chromosome aneuploidy: +1.0000 | Query 70, PPM1D and p53: −1.0000 |

These cases expose ranking failures, but do not identify their cause. Lexical
matching, truncation and model behavior would need controlled follow-up tests
before attributing an individual failure to one component.

## Reproduce the run

Use Node.js 22. No API key or paid model service is required. From the repository
root:

```sh
npm ci --prefix benchmarks
npm run prepare:data --prefix benchmarks -- scifact
npm run evaluate --prefix benchmarks -- scifact
node --test web/test/*.test.mjs
node web/build.mjs
```

Preparation checks the archive and extracted dataset files against their pinned
checksums. It loads the embedding model by immutable revision, then writes the
corpus, index manifest and binary vectors. Evaluation records SHA-256 hashes of
the embedding and reranking ONNX weight files in the full report and writes the
browser summary. It fails on missing queries,
invalid metrics, changed dataset bytes or misaligned vectors. Rerunning replaces
these generated artifacts; use a clean checkout to keep provenance interpretable.

To independently audit the resulting metrics without rerunning models:

```sh
python3 -m venv /tmp/ragforge-metric-audit
/tmp/ragforge-metric-audit/bin/pip install pytrec-eval-terrier==0.5.11
/tmp/ragforge-metric-audit/bin/python benchmarks/verify_metrics.py --report evaluations/results/scifact-v2.json --qrels data/scifact/qrels/test.tsv
```

In the recorded run, document embedding took about 54 seconds, loading and
warmup 0.65 seconds, and the 1,200 measured searches plus scoring took about 126
seconds. Downloads are additional. The two q8 retrieval models total about 46 MB;
the prepared float32 vector file is **7,961,088 bytes**. Two independent preparation
runs produced identical vector bytes. Browser inference uses WebAssembly, so its
elapsed times need not match the Node CPU table.

## Artifacts and provenance

- [Full report](../evaluations/results/scifact-v2.json): every query, judgment,
  ranked document/score tuple, metric, latency and environment detail.
- [Browser summary](../benchmarks/data/scifact-v2/summary.json): aggregate
  metrics, intervals, hardware and the full report's SHA-256.
- [Index manifest](../benchmarks/data/scifact-v1/index.json): corpus, row-order,
  vector and source-archive hashes.
- [Protocol source](../web/protocol.mjs) and [evaluation runner](../benchmarks/evaluate.mjs).

The measured code was committed with no changes to the measured files at
`01963d71a6c5454b1ecce26ac16766553f0de257`. Per-file source and model-weight hashes
are recorded in the report. Later documentation and interface changes do not
change the measured ranking implementation.

Archive SHA-256:
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
Vector SHA-256:
`1dfaf5677ca4b843a52d5dcf8b54c4da7d0ffc1bad9b50ce40096e1752a94e17`.
Full report SHA-256:
`a815317ef58a64a0c8104f10cc0c95f7e799b872232e0812b1111710e55d9332`.

## Scope and limits

- This is a single scientific-abstract collection with sparse judgments, not a
  general-search or production-load benchmark.
- It evaluates document retrieval, not answer accuracy, scientific claim
  verification, or the Python API's ingestion and Qdrant performance.
- Neural input limits can omit the end of long abstracts. BM25 and the corpus
  reader retain the full text.
- Every test query has a positive judgment; no-evidence detection is unmeasured.
- Public pretrained models may have encountered related benchmark material.
  No fine-tuning here does not prove absence of pretraining contamination.
- All 300 × 4 searches completed. Timings are sequential, warm, single-machine
  observations without a throughput or availability guarantee.

The original [six-policy Python fixture](evaluation.md) remains useful for
regression checks. It is separate from these measurements.

## Data credit

SciFact is by David Wadden and colleagues (2020),
[Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).
The retrieval distribution is from Nandan Thakur and colleagues' (2021)
[BEIR project](https://github.com/beir-cellar/beir). Claims and annotations are
CC BY 4.0; the abstract database is ODC-By 1.0, as stated in the
[original license](https://github.com/allenai/scifact/blob/master/LICENSE.md).
See [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution and adaptations.
