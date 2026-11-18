import { fetchAsset, loadCorpus } from "./assets.mjs";
import { packContext, validateAnswer } from "./generation.mjs";
import { createModels } from "./neural.mjs";
import { createIndex, dotScores, hitsFor, rank, validateQuery } from "./retrieval.mjs";
import { validateVectors } from "./vector-index.mjs";

const progress = message => self.postMessage({ type: "progress", message });
let manifest, index, vectors, neural;
async function getModels() {
  if (neural) return neural;
  progress("Downloading the neural runtime…");
  const runtime = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js");
  runtime.env.allowLocalModels = false;
  runtime.env.backends.onnx.wasm.numThreads = 1;
  runtime.env.backends.onnx.wasm.proxy = false;
  let lastProgress = 0;
  const options = { device: "wasm", progress_callback(event) {
    if (event.status !== "progress" || performance.now() - lastProgress < 350) return;
    lastProgress = performance.now();
    progress("Downloading " + event.file.split("/").pop() + " · " + (event.loaded / 1e6).toFixed(1) + " MB");
  } };
  return neural = createModels(runtime, options, progress);
}
const engine = {
  async dense(query) {
    if (!vectors) {
      progress("Loading the precomputed abstract vectors…");
      vectors = await validateVectors(await (await fetchAsset("vectors.f32")).arrayBuffer(), manifest);
    }
    const models = await getModels();
    return dotScores(vectors, await models.embeddings([query]));
  },
  async rerank(query, candidates) { return (await getModels()).rerank(query, candidates); },
};
async function search(query, mode) {
  if (!index) {
    progress("Building the abstract keyword index…");
    const data = await loadCorpus();
    manifest = data.manifest;
    index = createIndex(data.corpus.documents);
  }
  return hitsFor(index.documents, await rank(index, query, mode, engine));
}
async function generate(raw, hits) {
  const query = validateQuery(raw);
  if (!hits.length) throw Error("Retrieve evidence before generating an answer.");
  progress("Preparing Flan-T5 for local generation…");
  const generator = await (await getModels()).generation();
  const { prompt, excerpts } = packContext(query, hits, text => generator.tokenizer.encode(text));
  progress("Generating from the supplied excerpts…");
  const [output] = await generator(prompt, { max_new_tokens: 40, do_sample: false });
  return { ...validateAnswer(output.generated_text, excerpts), excerpts };
}
self.onmessage = async ({ data }) => {
  try {
    const result = data.action === "generate" ? await generate(data.query, data.hits) : await search(data.query, data.mode);
    self.postMessage({ type: "result", result });
  } catch (error) {
    self.postMessage({ type: "error", message: error.message || "Search failed." });
  }
};
