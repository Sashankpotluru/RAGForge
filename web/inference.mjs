import { search, validateAnswer, validateQuery } from "./retrieval.mjs";
import {
  AutoModelForSequenceClassification, AutoTokenizer, env, pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";

env.allowLocalModels = false;
// GitHub Pages does not set cross-origin isolation headers. Single-threaded WASM
// runs inside this dedicated worker and works without SharedArrayBuffer.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

const embedId = "Xenova/all-MiniLM-L6-v2";
const rerankId = "Xenova/ms-marco-MiniLM-L-6-v2";
const generatorId = "Xenova/flan-t5-base";
const postProgress = message => self.postMessage({ type: "progress", message });
let lastProgress = 0;
function progress(event) {
  if (event.status !== "progress" || performance.now() - lastProgress < 350) return;
  lastProgress = performance.now();
  const size = (event.loaded / 1e6).toFixed(1) + " MB";
  const percent = event.total ? " (" + Math.floor(100 * event.loaded / event.total) + "%)" : "";
  postProgress("Loading " + event.file.split("/").pop() + " — " + size + percent);
}
const commonOptions = { device: "wasm", dtype: "q8", progress_callback: progress };
let embedder, vectors, reranker, tokenizer, generator;

const models = {
  async dense(query, documents) {
    postProgress("Preparing the MiniLM embedding model…");
    embedder ??= await pipeline("feature-extraction", embedId, {
      ...commonOptions, revision: "751bff37182d3f1213fa05d7196b954e230abad9",
    });
    const options = { pooling: "mean", normalize: true };
    if (!vectors) {
      postProgress("Embedding the six source passages…");
      const output = await embedder(documents.map(d => d.title + "\n" + d.content), options);
      vectors = output.tolist();
    }
    postProgress("Comparing the question with each passage…");
    const [question] = (await embedder(query, options)).tolist();
    return vectors.map(v => v.reduce((sum, n, i) => sum + n * question[i], 0));
  },

  async rerank(query, candidates) {
    postProgress("Preparing the MiniLM reranker…");
    const options = { ...commonOptions, revision: "a09144355adeed5f58c8ed011d209bf8ee5a1fec" };
    tokenizer ??= await AutoTokenizer.from_pretrained(rerankId, options);
    reranker ??= await AutoModelForSequenceClassification.from_pretrained(rerankId, options);
    postProgress("Scoring each question and passage together…");
    const pairs = tokenizer(candidates.map(() => query), {
      text_pair: candidates.map(d => d.title + "\n" + d.content), padding: true, truncation: true,
    });
    const { logits } = await reranker(pairs);
    return Array.from(logits.data);
  },
};

async function generate(raw, hits) {
  const query = validateQuery(raw);
  if (!hits.length) throw Error("Find a matching source passage before generating an answer.");
  postProgress("Preparing Flan-T5 for local generation…");
  generator ??= await pipeline("text2text-generation", generatorId, {
    ...commonOptions, revision: "c6a1b98a4a35e1063d88edaceef4dc52cf3b2784",
  });
  const context = hits.slice(0, 4);
  const prompt = "Answer the question using the context. If it cannot be answered, say unknown.\nContext: "
    + context.map(hit => hit.quote).join("\n") + "\nQuestion: " + query;
  if (generator.tokenizer.encode(prompt).length > 512) {
    throw Error("The question and sources exceed the 512-token context. Shorten the question.");
  }
  postProgress("Generating an answer from the retrieved passages…");
  const [output] = await generator(prompt, { max_new_tokens: 40, do_sample: false });
  return validateAnswer(output.generated_text, hits, context.length);
}

self.onmessage = async ({ data }) => {
  try {
    const result = data.action === "generate"
      ? await generate(data.query, data.hits)
      : await search(data.documents, data.query, data.mode, models);
    self.postMessage({ type: "result", result });
  } catch (error) {
    self.postMessage({ type: "error", message: error.message || "Model inference failed." });
  }
};
