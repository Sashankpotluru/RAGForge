# Implementation decisions

The implementation uses bounded structure-aware chunking. It normalizes whitespace per source location, prefers sentence boundaries, then word boundaries, then hard character limits. Original PDF pages remain distinct. The historical class name SemanticChunker describes the design lineage; the splitter performs no embedding-based semantic segmentation. Offsets refer to normalized location-local text.

The authoritative catalog tracks active versions. Both retrieval branches use the same eligible active chunk IDs for the requested logical collection. Dense retrieval applies the shared metadata predicate to retrieved payloads and supplies Qdrant with a matching point-ID filter before limiting vector results. BM25 uses the same predicate before scoring. This design prioritizes parity for a small local corpus. Large collections require a more scalable metadata and active-set strategy.

Metadata remains nested under its own key. It cannot replace the top-level source, document identifier, offsets, or citation location. Equality accepts a scalar (including membership in a stored string list), membership accepts a list of strings, and ordered comparisons accept finite numbers. A missing value fails the predicate. Unknown filter and search keys are rejected.

The evaluation contract supersedes the design's earlier naming. Recall and hit rate are distinct. Standard binary ndcg@10 uses unique relevant chunk IDs. The additional novelty_ndcg@10 rewards newly covered evidence and deduplicates repeated matches. See [evaluation methods](evaluation.md).

Source and index identifiers, ranked evidence and citation references are inspectable.
The GitHub Pages workbench uses the full BEIR SciFact corpus and shares its ranking
implementation and pinned ONNX model artifacts with the SciFact and ArguAna
retrieval benchmarks. Their separate results show how the same methods behave
on scientific evidence retrieval and counterargument retrieval. Long query and
passage pairs use explicit longest-first truncation so one input cannot consume
the reranker's entire token budget.

The Python service retains its separate ingestion, persistence and fixture checks.
Its HTTP load profile uses a trained tiny embedding model, one API worker and
an isolated native Qdrant server. Concurrent clients can queue behind the runtime
lock; the measurements describe this configuration rather than a multi-worker
deployment. Generated-answer faithfulness and multi-user authorization remain
future work. Local installation and passing tests do not establish independent
dependency-risk approval. See the [SciFact evaluation](scifact-evaluation.md),
[ArguAna evaluation](arguana-evaluation.md) and [API load test](load-testing.md).
