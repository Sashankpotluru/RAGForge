import { fetchAsset, loadCorpus } from "./assets.mjs";
import { protocol } from "./protocol.mjs";
import { validateQuery } from "./retrieval.mjs";
import { WorkerClient } from "./worker-client.mjs";

const client = new WorkerClient(() => new Worker(new URL("./inference.mjs", import.meta.url), { type: "module" }));
export async function load() {
  if (typeof Worker !== "function") throw Error("Enable browser workers to search the corpus.");
  const [{ corpus }, capacity, ...results] = await Promise.all([loadCorpus(),
    fetchAsset("backend-load.json").then(r => r.json()),
    ...["results.json", "arguana-results.json"].map(name => fetchAsset(name).then(r => r.json()))]);
  const neural = typeof WebAssembly === "object";
  return { corpus, results, capacity, caps: {
    modes: neural ? protocol.modes : ["bm25"], defaultMode: "bm25", generation: neural ? "Flan-T5 Base" : null,
  } };
}
export function cancel() { client.cancel(); }
export async function run(raw, mode, answer, progress) {
  const query = validateQuery(raw);
  const started = performance.now();
  const hits = await client.run({ action: "search", mode, query }, progress);
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
