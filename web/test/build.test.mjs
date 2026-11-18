import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { datasets, paths, root } from "../../benchmarks/dataset.mjs";
import { buildSite, validateRelease } from "../build.mjs";

test("a clean Pages artifact contains only its public allowlist, including after rebuild", async t => {
  const temp = await mkdtemp(join(tmpdir(), "ragforge-pages-"));
  t.after(() => rm(temp, { recursive: true, force: true }));
  const site = join(temp, "RAGForge");
  await buildSite(site);
  await writeFile(join(site, "secret.env"), "test");
  await buildSite(site);
  const files = (await readdir(site, { recursive: true })).sort();
  assert.deepEqual(files, [
    "assets", "assets/architecture-evaluation.png", "assets/browser.mjs", "assets/inference.mjs",
    "assets/retrieval.mjs", "assets/workbench.css", "assets/workbench.js", "assets/worker-client.mjs",
    "assets/assets.mjs", "assets/neural.mjs", "assets/protocol.mjs", "assets/vector-index.mjs",
    "arguana-results.json", "assets/generation.mjs", "corpus.json", "index.html", "index.json", "results.json", "vectors.f32",
  ].sort());
  const html = await readFile(join(site, "index.html"), "utf8");
  assert.match(html, /data-runtime="browser"/);
  assert.doesNotMatch(html, /(?:href|src)="\//);
  const corpus = JSON.parse(await readFile(join(site, "corpus.json"), "utf8"));
  assert.equal(corpus.documents.length, 5183);
  assert.equal(corpus.questions.length, 6);
  assert.equal(corpus.cases, undefined);
  for (const [file, run] of [["results.json", "scifact-v2"], ["arguana-results.json", "arguana-v1"]]) {
    const summary = JSON.parse(await readFile(join(site, file)));
    assert.equal(summary.run, run);
    assert.equal(summary.runs.length, 4);
    assert.ok(summary.runs.every(row => !("examples" in row) && !("ranked" in row)));
  }
  assert.match(html, /<select id="benchmark-dataset"/);
  assert.match(html, /id="backend-capacity" hidden/);
});

test("release binding rejects a mismatched summary, manifest, report or audit", async () => {
  const dataset = datasets.scifact;
  const { indexDir, runDir } = paths(dataset);
  const manifest = JSON.parse(await readFile(join(indexDir, "index.json")));
  const summary = JSON.parse(await readFile(join(runDir, "summary.json")));
  const base = join(root, "evaluations/results", dataset.run);
  const report = await readFile(base + ".json");
  const audit = JSON.parse(await readFile(base + "-metric-audit.json"));
  await assert.rejects(validateRelease(dataset, manifest, report, { ...summary, documents: 0 }, audit), /summary/i);
  await assert.rejects(validateRelease(dataset, { ...manifest, row_ids_sha256: "0" }, report, summary, audit), /Index/);
  const alteredSource = { ...dataset, files: { ...dataset.files, "queries.jsonl": "0".repeat(64) } };
  await assert.rejects(validateRelease(alteredSource, manifest, report, summary, audit), /source/i);
  await assert.rejects(validateRelease(dataset, { ...manifest, dimension: 1 }, report, summary, audit), /Index/);
  await assert.rejects(validateRelease(dataset, { ...manifest, model: { ...manifest.model, revision: "wrong" } },
    report, summary, audit), /Index/);
  await assert.rejects(validateRelease(dataset, manifest, Buffer.from("changed"), summary, audit), /digest/);
  await assert.rejects(validateRelease(dataset, manifest, report, summary, { ...audit, comparisons: 0 }), /audit/);
});
