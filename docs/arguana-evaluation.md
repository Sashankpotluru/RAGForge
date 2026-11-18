# ArguAna counterargument retrieval

ArguAna adds a different retrieval task to RAGForge: given a debate argument,
find its labeled counterargument. The BEIR distribution contains **8,674
arguments and 1,406 official test queries**. The same four retrieval methods and
fixed model settings used for SciFact are evaluated here. A relevant result may
oppose the input argument; relevance is not agreement, truth or endorsement.
[Original paper](https://aclanthology.org/P18-1023/) ·
[BEIR distribution](https://huggingface.co/datasets/BeIR/arguana)

The public workbench continues to search SciFact. ArguAna is an offline benchmark
with its own recorded Results panel, full report and reproduction commands.
Keeping the tasks separate makes differences in model behavior easier to see.

## Dataset checks

The archive and each input file are checked against pinned hashes before use.
All official queries are retained. Two details matter for this collection:

- **1,298 query IDs also occur in the corpus.** The query's own document ID is
  excluded before candidate cutoffs in every retrieval mode. This prevents
  returning the original argument as its counterargument. Exclusion uses the
  query ID, never its relevance judgments.
- **Five labeled relevant documents are absent from the supplied corpus.**
  These queries remain in the denominator and score zero. The evaluator records
  the five known pairs explicitly and rejects unexpected missing references.

| Query ID | Missing relevant document ID |
| --- | --- |
| `test-free-speech-debate-yfsdfkhbwu-con03a` | `test-free-speech-debate-yfsdfkhbwu-con03b` |
| `test-education-ufsdfkhbwu-con03a` | `test-education-ufsdfkhbwu-con03b` |
| `test-politics-dhwem-pro06a` | `test-politics-dhwem-pro06b` |
| `test-science-sghwbdgmo-con03a` | `test-science-sghwbdgmo-con03b` |
| `test-society-asfhwapg-con04a` | `test-society-asfhwapg-con04b` |

There is one positive judgment per query. The missing documents limit maximum
possible mean recall to 1,401/1,406, about 99.64%, under this distribution. They
are a small part of the dataset and do not explain other ranking failures.

## Protocol

BM25, dense MiniLM, reciprocal rank fusion and cross-encoder reranking use the
same frozen settings as the [SciFact experiment](scifact-evaluation.md). No
model is fine-tuned and no parameter is selected using ArguAna test outcomes.
ArguAna has no supplied training split; model warmup uses an authored query.

Dense retrieval uses pinned q8 MiniLM embeddings with masked mean pooling and
L2 normalization. BM25 uses k1=1.2 and b=0.75. Hybrid fuses the first 100 results
from each branch with RRF k=60. The cross-encoder reranks the first 50 fused
candidates and preserves the remaining candidate order. The complete report
retains the first 100 results per method and query.

Official query text is retained, including arguments up to 5,500 characters.
Neural inputs remain bounded at 256 tokenizer positions. For the BERT
cross-encoder, three positions are reserved for special tokens. The longer of
the query and document segments is shortened from the right until both fit;
ties shorten the document. This preserves tokens from both nonempty segments.
BM25 uses the full tokenized text.
This truncation is a material limitation for long arguments and may affect the
methods differently. The experiment does not identify the cause of any
individual ranking failure.

A token-level audit before final measurement found that the inherited
Transformers.js pair truncation removed every candidate-document position for
519 of the 1,406 queries. Different candidate texts then produced identical
inputs for those queries. The final protocol corrects that input defect with
standard longest-first truncation, shared by the browser and Node evaluator.
The same correction is included in the new SciFact run. It was selected from
input inspection and synthetic tests, not a search over benchmark scores. The
historical SciFact-v1 report remains available with its original protocol.

Metrics are macro-averaged over all 1,406 queries. Unjudged documents receive
zero gain. The report includes HitRate@1, MRR@10, nDCG@10, Recall@10 and
Recall@100, with 2,000 query-bootstrap samples and paired nDCG differences from
BM25. The bootstrap seed is 20261118. Related arguments can violate the
independence assumption, so these intervals are not guarantees about new topics.

Measurements use the shared JavaScript retrieval engine under Node CPU
inference. They do not measure browser WebAssembly latency, generated-answer
quality, Python API capacity or scientific claim verification. Public pretrained
model exposure to related benchmark material cannot be ruled out.

## Measured results

All 1,406 queries completed in all four modes, with no search errors. The five
queries with absent relevant documents remain in every denominator.

| Method | nDCG@10 (95% interval) | MRR@10 | Recall@10 | Recall@100 | Median / p95 search |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4209 [0.4043, 0.4391] | 0.3358 | 0.6942 | 0.9189 | 9.05 / 13.20 ms |
| Dense MiniLM | 0.5002 [0.4826, 0.5188] | 0.4114 | 0.7859 | 0.9787 | 11.11 / 12.38 ms |
| Hybrid (RRF) | 0.5024 [0.4850, 0.5207] | 0.4137 | 0.7866 | 0.9822 | 19.30 / 24.92 ms |
| Hybrid + reranking | 0.4809 [0.4629, 0.4996] | 0.3910 | 0.7724 | 0.9822 | 281.12 / 296.75 ms |

Hybrid's nDCG@10 difference from BM25 is **+0.0816**, with a paired 95% interval
of **[+0.0707, +0.0925]**. Dense retrieval also improves on BM25 in this run:
+0.0793 [+0.0594, +0.0983]. The reranked variant improves on BM25 by
+0.0601 [+0.0402, +0.0801], but its mean score is lower than hybrid's and its
median latency is about 14.6 times higher. The reported paired intervals compare
each method with BM25; they do not test hybrid against dense or reranking.

The counterargument task rewards finding a relevant opposing argument, not merely
a similar passage. This run shows that the extra reranking stage is not
automatically useful. It does not isolate whether the regression comes from
training-domain mismatch, truncation or another cause. SciFact gives a different
pattern, so the two datasets remain separate rather than being pooled into one
headline score.

Average gains also hide individual failures. For example, hybrid loses a full
nDCG point relative to BM25 on query `test-philosophy-pppthbtcb-con01a`, while it
gains a full point on `test-economy-epsihbdns-pro01a`. These are the automatically
selected largest loss and gain in the report, not representative case studies.
The [full report](../evaluations/results/arguana-v1.json) retains every query and
ranking so these cases can be inspected.

An independent `pytrec-eval-terrier` 0.5.11 audit compared every query, mode and
reported metric against the original judgments: **28,120 comparisons, maximum
absolute difference 0**. This validates metric calculation, not model quality,
bootstrap assumptions or generated answers.
[Audit record](../evaluations/results/arguana-v1-metric-audit.json)

Document preparation took 83.6 seconds. Loading the prepared corpus/index took
0.42 seconds, cached model loading and warmup 0.46 seconds, and the sequential
evaluation 452.3 seconds. These measurements used an Apple M4 Pro, 24 GiB RAM,
Node 22.23.1, Transformers.js 3.8.1 and ONNX Runtime 1.21.0, with six intra-op
threads and one inter-op thread. Model downloads are excluded.

## Reproduce the experiment

Use Node.js 22 from the repository root. Preparation verifies the official
archive, then creates the local document vectors. Installing dependencies does
not automatically run model preparation.

```sh
npm ci --prefix benchmarks
npm run prepare:data --prefix benchmarks -- arguana
npm run evaluate --prefix benchmarks -- arguana
node benchmarks/verify_arguana_tokens.mjs
```

Before testing or building the site, save the audit of the new report using
`pytrec-eval-terrier==0.5.11` in a separate Python environment:

```sh
python benchmarks/verify_metrics.py --report evaluations/results/arguana-v1.json --qrels data/arguana/qrels/test.tsv --output evaluations/results/arguana-v1-metric-audit.json
```

The corpus and binary vectors remain in the ignored local cache. Evaluation
writes the full versioned report and its compact public summary. Reproduction
requires downloads on first use; no model-provider account or API key is needed.

## Provenance and data credit

Archive SHA-256:
`cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b`.
Test-qrels SHA-256:
`0c47b481fa8b47fb9ca5e74bc8a017bdea14518c0e4a5d21345291180b3aca76`.
The full report and index manifest record the remaining source, model and code
hashes, exact settings, hardware and per-query observations.

Measured code: `01963d71a6c5454b1ecce26ac16766553f0de257`.
Full report SHA-256:
`ded3b36e7bed4d083f01db49bfafe50addff9bad8cb57e5e39affbea0b926910`.
[Public summary](../benchmarks/data/arguana-v1/summary.json) ·
[Index manifest](../benchmarks/data/arguana-v1/index.json) ·
[Pair-token audit](../evaluations/results/arguana-v1-token-audit.json)

ArguAna Counterargs is by Henning Wachsmuth, Shahbaz Syed and Benno Stein (2018),
[Retrieval of the Best Counterargument without Prior Topic Knowledge](https://aclanthology.org/P18-1023/).
The [original Zenodo release](https://zenodo.org/records/3973258) identifies CC BY
4.0; the [BEIR mirror](https://huggingface.co/datasets/BeIR/arguana) labels its
distribution CC BY-SA 4.0. The redistributed BEIR-derived query/report material
retains that share-alike notice. RAGForge's original software remains MIT
licensed. [Third-party notices](../THIRD_PARTY_NOTICES.md)
