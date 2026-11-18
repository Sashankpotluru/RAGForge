# Browser workbench

[Open the workbench on GitHub Pages](https://sashankpotluru.github.io/RAGForge/). It searches all **5,183 scientific abstracts in BEIR SciFact**. Enter a claim or question, or choose an example, then select **Find evidence**. Each citation links to the retrieved abstract. The **Corpus** view provides text filtering and pagination across the full collection. Example claims come from the training split; they are not assertions that the claims are true.

## Search modes

| Mode | How it ranks the abstracts | First-use model download |
| --- | --- | --- |
| Keyword (BM25) | Scores matching words using term frequency, document frequency and passage length. | None |
| Dense | Compares the query with precomputed normalized MiniLM document embeddings. | About 24 MB |
| Hybrid | Combines the dense and BM25 rankings with reciprocal rank fusion. | Shares the dense model |
| Hybrid + reranking | Scores each question and candidate passage together with a cross-encoder after fusion. | About 48 MB for both models |

Changing the menu selects a mode. Starting a search in that mode loads its models. Neural modes also load a shared JavaScript and WebAssembly runtime of about 23 MB before compression and a 7,961,088-byte document-vector file. Only the query needs embedding at search time. The document vectors are prepared from the same pinned q8 model and 256-token input limit. The worker validates corpus, vector and row-order hashes before neural search.

The interface loads and decodes the corpus for browsing. On the first search, the worker also loads and decodes it for ranking. Keeping both copies in memory increases startup work and memory use.

The worker keeps ranking work off the interface thread. BM25 needs the corpus but no neural runtime or vectors. Hybrid fuses the first 100 candidates from each retriever; reranking scores the first 50 fused candidates. The UI shows the first ten. **Cancel** terminates the worker. The next request starts a fresh worker and can reuse cached downloads.

## Cited answers

Expand **Cited answer** and select **Generate answer** to run Flan-T5 Base on your device. The first use downloads about 278 MB for its quantized encoder, decoder, configuration and tokenizer files, plus the shared runtime if needed. Generation runs in a WebAssembly worker and works without WebGPU. Available memory and browser support determine whether the model can run.

The model sees the question and bounded excerpts from up to four retrieved abstracts. Prompt packing keeps input within 512 tokens; output is capped at 40 new tokens with greedy decoding. To receive a citation, the entire answer must contain at least two words, occur in an excerpt actually supplied to the model and match exactly one supplied excerpt. Matching ignores case and trailing sentence punctuation. A matched quotation establishes its source. The reader must judge whether it answers the question. If the model abstains, fails or produces ambiguous text, the workbench shows the evidence and the reason it withheld the answer. This optional generation is not evaluated by the retrieval benchmark.

## Runtime and weights

GitHub Pages serves the application and public data. Optional inference uses Transformers.js 3.8.1 from jsDelivr and fixed ONNX files from Hugging Face. The external requests fetch runtime files, model configuration, tokenizers and weights. The visitor’s question and passages stay in the browser. Cache persistence depends on browser storage settings. [Transformers.js 3.8.1](https://github.com/huggingface/transformers.js/tree/3.8.1)

| Task | ONNX file size | Pinned model |
| --- | ---: | --- |
| Embeddings | 22,972,370 bytes, q8 | [MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/751bff37182d3f1213fa05d7196b954e230abad9/onnx) |
| Cross-encoder reranking | 23,143,499 bytes, q8 | [MS MARCO MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/tree/a09144355adeed5f58c8ed011d209bf8ee5a1fec/onnx) |
| Generation | 110,502,358-byte encoder and 164,739,632-byte merged decoder, q8 | [Flan-T5 Base](https://huggingface.co/Xenova/flan-t5-base/tree/c6a1b98a4a35e1063d88edaceef4dc52cf3b2784/onnx) |

The **Results** view separates the 300-query SciFact and 1,406-query ArguAna evaluations, including confidence intervals, paired differences against BM25 and runtime details. Both measure the shared ranking implementation under Node.js CPU inference. ArguAna tests counterargument retrieval; it is not the interactive workbench corpus. Browser inference uses WebAssembly. The elapsed time under the evidence results measures the current search and includes any first-use loading; it should not be compared directly to warm benchmark latency. Backend capacity results are shown separately because they measure actual HTTP requests to the Python service. [Local Python demo](demo.md) · [SciFact evaluation](scifact-evaluation.md) · [ArguAna evaluation](arguana-evaluation.md) · [API load test](load-testing.md)

The collection is credited to SciFact and BEIR. Abstract database rights are ODC-By 1.0; claims and annotations are CC BY 4.0. The interface links the original paper, data source and terms. [Third-party notices](../THIRD_PARTY_NOTICES.md)

## Preview a project-site build

From the repository root, use Node.js 22 and a local static HTTP server.

```sh
node --test web/test/*.test.mjs
node web/build.mjs
python3 -m http.server 4173 --bind 127.0.0.1 --directory dist
```

Open [the local preview](http://127.0.0.1:4173/RAGForge/). The build writes only to `dist/RAGForge/`. The Pages workflow publishes that directory from `main`. It contains the interface, browser code, full SciFact corpus, training examples, document vectors, aggregate measurements and architecture image. Test judgments and saved test rankings remain in the repository, outside the deployed artifact. Models are downloaded on demand, not bundled with the site.
