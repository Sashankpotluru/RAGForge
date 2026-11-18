import { search, validateQuery } from "./retrieval.mjs";
import { WorkerClient } from "./worker-client.mjs";

const client = new WorkerClient(() => new Worker(new URL("./inference.mjs", import.meta.url), { type: "module" }));
let documents;
async function get(file) {
  const response = await fetch(new URL("../" + file, import.meta.url));
  if (!response.ok) throw Error("Could not load the sample data. Reload to try again.");
  return response.json();
}

export async function load() {
  const [corpus, results] = await Promise.all([get("corpus.json"), get("results.json")]);
  documents = corpus.documents;
  const supportsModels = typeof Worker === "function" && typeof WebAssembly === "object";
  return {
    corpus, results, caps: {
      modes: supportsModels ? ["bm25", "dense", "hybrid", "hybrid_rerank"] : ["bm25"],
      defaultMode: "bm25", generation: supportsModels ? "Flan-T5 Base" : null,
    },
  };
}

export function cancel() { client.cancel(); }

export async function run(raw, mode, answer, progress) {
  const query = validateQuery(raw);
  const started = performance.now();
  const hits = mode === "bm25"
    ? await search(documents, query, mode)
    : await client.run({ action: "search", mode, query, documents }, progress);
  const result = { hits, search_ms: performance.now() - started };
  if (answer && hits.length) {
    const generated = performance.now();
    try {
      Object.assign(result, await client.run({ action: "generate", query, hits }, progress));
      result.generation_ms = performance.now() - generated;
    } catch (error) {
      if (error.name === "AbortError") throw error;
      result.generation_error = error.message + " You can retry generation.";
    }
  }
  return result;
}
