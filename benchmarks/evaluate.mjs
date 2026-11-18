import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { arch, cpus, platform, release, totalmem } from "node:os";
import { join } from "node:path";
import { assets, dataset, digest, rawDir, readDataset, readJson, root } from "./dataset.mjs";
import { queryMetrics, summarize } from "./metrics.mjs";
import { neural, session } from "./runtime.mjs";
import { protocol } from "../web/protocol.mjs";
import { createIndex, dotScores, rank } from "../web/retrieval.mjs";
import { validateCorpus, validateVectors } from "../web/vector-index.mjs";

if (Number(process.versions.node.split(".")[0]) !== 22) throw Error("Use Node 22 to run this protocol.");
const bytes = async path => Uint8Array.from(await readFile(join(assets, path))).buffer;
const git = args => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
const started = performance.now();
const code = ["web/protocol.mjs", "web/retrieval.mjs", "web/neural.mjs", "web/vector-index.mjs",
  ...["dataset.mjs", "prepare.mjs", "evaluate.mjs", "metrics.mjs", "runtime.mjs", "package-lock.json"].map(f => "benchmarks/" + f)];
const provenance = {
  started_at: new Date().toISOString(), git_commit: git(["rev-parse", "HEAD"]), git_status: git(["status", "--porcelain"]),
  code_sha256: Object.fromEntries(await Promise.all(code.map(async file => [file, digest(await readFile(join(root, file)))]))),
  node: process.version, platform: platform(), os_release: release(), arch: arch(), cpu: cpus()[0].model,
  logical_cpus: cpus().length, ram_bytes: totalmem(), session,
  transformers_js: (await readJson(join(root, "benchmarks/node_modules/@huggingface/transformers/package.json"))).version,
  onnxruntime_node: (await readJson(join(root, "benchmarks/node_modules/onnxruntime-node/package.json"))).version,
};
const source = await readDataset();
const manifest = await readJson(join(assets, "index.json"));
const corpus = await validateCorpus(await bytes("corpus.json"), manifest);
if (JSON.stringify(corpus) !== JSON.stringify(source.corpus)) throw Error("Prebuilt corpus and official split disagree.");
const vectors = await validateVectors(await bytes("vectors.f32"), manifest);
const index = createIndex(corpus.documents);
const engine = {
  dense: async query => dotScores(vectors, await neural.embeddings([query])),
  rerank: neural.rerank,
};
const indexReady = performance.now();
const warmupId = corpus.exampleIds[0];
for (const mode of protocol.modes) await rank(index, source.queries.get(warmupId).text, mode, engine);
const warm = performance.now();
const modelFiles = {};
for (const model of [protocol.models.embedder, protocol.models.reranker]) {
  const filename = join(rawDir, "models", model.id, model.revision, "onnx/model_quantized.onnx");
  modelFiles[model.id] = digest(await readFile(filename));
}
const queries = [];
const expected = [...source.test.keys()].sort((a, b) => Number(a) - Number(b));
for (const id of expected) {
  const text = source.queries.get(id).text;
  const judgments = source.test.get(id);
  const modes = {};
  for (const mode of protocol.modes) {
    try {
      const start = performance.now();
      const ranking = await rank(index, text, mode, engine);
      const timing_ms = performance.now() - start;
      const ranked = ranking.map(hit => [corpus.documents[hit.row].id, hit.score]);
      modes[mode] = { ranked, metrics: queryMetrics(ranked.map(([doc]) => doc), judgments), timing_ms };
    } catch (error) {
      modes[mode] = { error: error.message };
    }
  }
  queries.push({ id, text, judgments: Object.fromEntries(judgments), modes });
  if (queries.length % 10 === 0) console.log("Evaluated", queries.length, "/", expected.length);
}
const output = join(root, "evaluations/results/scifact-v1.json");
const report = {
  protocol: { ...protocol, dataset, order: "Numeric ascending test query ID, one sequential independent call per mode in declared order",
    judgments: "Document level; positive judgments are relevant; unjudged documents have zero gain.",
    rankings: "Each tuple is [SciFact document ID, raw score]. In hybrid_rerank, first 50 use cross-encoder logits; tail retains RRF scores.",
    ties: "Ascending official corpus row", bm25_tokens: "Lowercase Unicode letter/number runs minus web/retrieval.mjs stop list; unique query terms",
    neural_input: "Title, newline, abstract. Embeddings use the first 256 tokenizer positions including special tokens; masked mean pooling and L2 normalization. Cross-encoder uses the first 256 positions of the query/document pair.",
    intervals: "Query bootstrap, 2,000 shared resamples with replacement; 95% percentile intervals with linear interpolation, paired delta versus BM25.",
    timing: "Warm Node CPU execution of shared search and model code. Independent full search per mode. Excludes startup, corpus/index prep, warmup, metrics and serialization.",
    limitations: "Single scientific-document dataset; sparse judgments; model pretraining exposure unknown. Relevance can indicate support or refutation. Generation and Python service performance are outside this evaluation." },
  provenance: { ...provenance, model_file_sha256: modelFiles },
  preparation: { manifest, corpus_and_index_load_ms: indexReady - started, model_load_and_warmup_ms: warm - indexReady,
    warmup_train_query_id: warmupId, evaluation_elapsed_ms: performance.now() - warm },
  queries,
};
try {
  report.summary = summarize(queries, expected, corpus.documents.length);
} catch (error) {
  await writeFile(join(rawDir, "failed-evaluation.json"), JSON.stringify(report));
  throw error;
}
await writeFile(output, JSON.stringify(report) + "\n");
await writeFile(join(assets, "summary.json"), JSON.stringify({ ...report.summary, run_at: provenance.started_at,
  hardware: provenance.cpu, node: provenance.node, report_sha256: digest(await readFile(output)) }, null, 2) + "\n");
console.log("Saved", output);
