# Browser workbench

[Open the workbench on GitHub Pages](https://sashankpotluru.github.io/RAGForge/). It opens with a live keyword search across six sample policies. Enter a question or choose one from **Sample questions**, then select **Find evidence**. Each citation links to its source passage. The **Corpus** view contains the complete synthetic policies.

## Search modes

| Mode | How it ranks the policies | First-use model download |
| --- | --- | --- |
| Keyword (BM25) | Scores matching words using term frequency, document frequency and passage length. | None |
| Dense | Compares the question with each policy using normalized MiniLM sentence embeddings. | About 24 MB |
| Hybrid | Combines the dense and BM25 rankings with reciprocal rank fusion. | Shares the dense model |
| Hybrid + reranking | Scores each question and candidate passage together with a cross-encoder after fusion. | About 48 MB for both models |

Changing the menu selects a mode. Starting a search in that mode loads its models. Neural modes also load a shared JavaScript and WebAssembly runtime of about 23 MB before compression. The worker runs retrieval on the CPU and reuses the six document embeddings until the page closes or the worker stops. **Cancel** terminates the worker. The next request starts a fresh one and can reuse cached downloads.

## Cited answers

Expand **Cited answer** and select **Generate answer** to run Flan-T5 Base on your device. The first use downloads about 278 MB for its quantized encoder, decoder, configuration and tokenizer files, plus the shared runtime if needed. Generation runs in a WebAssembly worker and works without WebGPU. Available memory and browser support determine whether the model can run.

The model sees the question and up to four retrieved passages. It generates a short answer with greedy decoding. Input is capped at 512 tokens and output at 40 new tokens. To receive a citation, the entire answer must contain at least two words, occur in a passage supplied to the model and match exactly one retrieved passage. Matching ignores case and trailing sentence punctuation. A matched quotation establishes its source. The reader can judge whether it answers the question. If the model abstains, fails or produces ambiguous text, the workbench shows the passages and the reason it withheld the answer.

## Runtime and weights

GitHub Pages serves the application and public data. Optional inference uses Transformers.js 3.8.1 from jsDelivr and fixed ONNX files from Hugging Face. The external requests fetch runtime files, model configuration, tokenizers and weights. The visitor’s question and passages stay in the browser. Cache persistence depends on browser storage settings. [Transformers.js 3.8.1](https://github.com/huggingface/transformers.js/tree/3.8.1)

| Task | ONNX file size | Pinned model |
| --- | ---: | --- |
| Embeddings | 22,972,370 bytes, q8 | [MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/751bff37182d3f1213fa05d7196b954e230abad9/onnx) |
| Cross-encoder reranking | 23,143,499 bytes, q8 | [MS MARCO MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/tree/a09144355adeed5f58c8ed011d209bf8ee5a1fec/onnx) |
| Generation | 110,502,358-byte encoder and 164,739,632-byte merged decoder, q8 | [Flan-T5 Base](https://huggingface.co/Xenova/flan-t5-base/tree/c6a1b98a4a35e1063d88edaceef4dc52cf3b2784/onnx) |

The **Results** view displays a recorded evaluation of the Python backend on twelve synthetic questions. The elapsed time under **Source passages** measures the current search and includes any first-use model loading. The hosted edition runs its own retrieval in the browser. The Python application, CLI, ingestion, indexes and evaluation tools remain available in the repository. [Local demo](demo.md) · [Evaluation](evaluation.md)

## Preview a project-site build

From the repository root, use Node.js 22 and a local static HTTP server.

```sh
node --test web/test/*.test.mjs
node web/build.mjs
python3 -m http.server 4173 --bind 127.0.0.1 --directory dist
```

Open [the local preview](http://127.0.0.1:4173/RAGForge/). The build writes only to `dist/RAGForge/`. The Pages workflow publishes that directory from `main`. It contains the shared interface, browser code, six policies, sample questions, recorded measurements and architecture image.
