import { fetchAsset, loadCorpus } from "./assets.mjs";
import { protocol } from "./protocol.mjs";
import { validateQuery } from "./retrieval.mjs";
import { WorkerClient } from "./worker-client.mjs";

const client = new WorkerClient(() => new Worker(new URL("./inference.mjs", import.meta.url), { type: "module" }));
function readSummary(name) {
  return fetchAsset(name).then(response => response.json());
}

export async function load() {
  if (typeof Worker !== "function") throw Error("Enable browser workers to search the corpus.");
  const [{ corpus }, results] = await Promise.all([loadCorpus(), readSummary("results.json")]);
  const neural = typeof WebAssembly === "object";
  // The workbench can start searching while these independent Results panels load.
  const arguana = readSummary("arguana-results.json").catch(() => null);
  const capacity = readSummary("backend-load.json").catch(() => null);
  return { corpus, results, arguana, capacity, caps: {
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
