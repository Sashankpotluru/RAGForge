import { datasets, paths } from "../benchmarks/dataset.mjs";
import { protocol } from "./protocol.mjs";
import { sha256, validateCorpus, validateVectors } from "./vector-index.mjs";
import { copyFile, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const demo = join(root, "src/ragforge/demo");

// Hash of all official test query IDs sorted lexicographically and joined by LF.
const testIds = {
  scifact: "26cc483d5ed08c9559e5f99948281accbb9aac17b8c2ea6e1e857286d9ebd684",
  arguana: "354220cb3aa85edc770a16b95f2afb14380048a83ad4b0ae6e55f8d9c4637a1f",
};

export async function validateRelease(dataset, manifest, reportBytes, summary, audit) {
  const reportHash = await sha256(reportBytes);
  if (reportHash !== summary.report_sha256 || audit.report_sha256 !== reportHash ||
      audit.tool !== "pytrec-eval-terrier" || audit.version !== "0.5.11" || audit.status !== "passed" ||
      audit.queries !== dataset.test || audit.comparisons !== dataset.test * protocol.modes.length * 5 ||
      !Number.isFinite(audit.max_absolute_error) || audit.max_absolute_error > 1e-12 ||
      audit.test_qrels_sha256 !== dataset.files["qrels/test.tsv"]) throw Error("Evaluation digest or audit mismatch.");
  const evidence = JSON.parse(reportBytes);
  const measured = evidence.summary;
  if (!isDeepStrictEqual(evidence.protocol.dataset, dataset)) throw Error("Dataset source registry differs from measured release.");
  if (evidence.protocol.version !== dataset.run || manifest.version !== dataset.index ||
      manifest.dataset_sha256 !== dataset.sha256 || manifest.rows !== dataset.documents ||
      measured.dataset !== "BEIR " + dataset.name || measured.documents !== dataset.documents ||
      measured.queries !== dataset.test || measured.runs.map(r => r.mode).join() !== protocol.modes.join() ||
      evidence.queries.length !== dataset.test ||
      await sha256(new TextEncoder().encode(evidence.queries.map(q => q.id).sort().join("\n"))) !== testIds[dataset.key]) {
    throw Error("Incomplete release for " + dataset.name);
  }
  // A fresh preparation can change wall time while reproducing identical index bytes.
  const { preprocessing_ms: measuredTiming, ...measuredIndex } = evidence.preparation.manifest;
  const { preprocessing_ms: localTiming, ...releaseIndex } = manifest;
  if (!isDeepStrictEqual(measuredIndex, releaseIndex)) throw Error("Index metadata differs from measured release.");
  const aggregates = measured.runs.map(({ examples, ...aggregate }) => aggregate);
  const expected = { ...measured, runs: aggregates, run: dataset.run, run_at: evidence.provenance.started_at,
    hardware: evidence.provenance.cpu, node: evidence.provenance.node, report_sha256: reportHash };
  if (JSON.stringify(summary) !== JSON.stringify(expected)) throw Error("Public summary differs from measured report.");
}

export async function buildSite(output = join(root, "dist/RAGForge")) {
  await mkdir(dirname(output), { recursive: true });
  const staging = await mkdtemp(join(dirname(output), ".pages-"));
  try {
    const assets = join(staging, "assets");
    await mkdir(assets);
    for (const name of ["workbench.js", "workbench.css", "architecture-evaluation.png"]) {
      await copyFile(join(demo, "static", name), join(assets, name));
    }
    for (const name of ["retrieval.mjs", "worker-client.mjs", "browser.mjs", "inference.mjs", "assets.mjs", "protocol.mjs", "neural.mjs", "vector-index.mjs", "generation.mjs"]) {
      await copyFile(join(root, "web", name), join(assets, name));
    }
    const html = await readFile(join(demo, "static/index.html"), "utf8");
    if (!html.includes('data-runtime="server"')) throw Error("Missing runtime marker.");
    await writeFile(join(staging, "index.html"), html.replace('data-runtime="server"', 'data-runtime="browser"'));
    const data = join(root, "benchmarks/data/scifact-v1");
    const manifest = JSON.parse(await readFile(join(data, "index.json"), "utf8"));
    const corpus = await validateCorpus(await readFile(join(data, "corpus.json")), manifest);
    const vectors = Uint8Array.from(await readFile(join(data, "vectors.f32"))).buffer;
    await validateVectors(vectors, manifest);
    if (corpus.documents.length !== datasets.scifact.documents) throw Error("Incomplete SciFact corpus.");
    for (const name of ["corpus.json", "index.json", "vectors.f32"]) await copyFile(join(data, name), join(staging, name));
    for (const dataset of Object.values(datasets)) {
      const { indexDir, runDir } = paths(dataset);
      const results = join(root, "evaluations/results");
      const summaryFile = join(runDir, "summary.json");
      const summary = JSON.parse(await readFile(summaryFile));
      await validateRelease(dataset, JSON.parse(await readFile(join(indexDir, "index.json"))),
        await readFile(join(results, dataset.run + ".json")), summary,
        JSON.parse(await readFile(join(results, dataset.run + "-metric-audit.json"))));
      const publicName = dataset.key === "scifact" ? "results.json" : dataset.key + "-results.json";
      await copyFile(summaryFile, join(staging, publicName));
    }
    await rm(output, { recursive: true, force: true });
    await rename(staging, output);
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await buildSite();
  console.log("Built dist/RAGForge/");
}
