import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { datasets, paths, root } from "../../benchmarks/dataset.mjs";
import { validateLoad } from "../validate-load.mjs";
import { buildSite, reportRevision, validateRelease } from "../build.mjs";

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
    "arguana-results.json", "backend-load.json", "assets/generation.mjs", "corpus.json", "index.html", "index.json", "results.json", "vectors.f32",
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
  for (const [key, run] of [["scifact", "scifact-v2"], ["arguana", "arguana-v1"], ["load", "backend-load-v1"]]) {
    const path = "evaluations/results/" + run + ".json";
    const revision = reportRevision(path, await readFile(join(root, path)));
    assert.ok(html.includes('data-' + key + '-revision="' + revision + '"'));
  }
  assert.match(html, /<select id="benchmark-dataset"/);
  assert.match(html, /id="backend-capacity" hidden/);
});

test("report links require the exact report bytes at the recorded Git revision", async t => {
  const cwd = await mkdtemp(join(tmpdir(), "ragforge-revision-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  const path = "report.json";
  const bytes = Buffer.from('{"run": 1}');
  await writeFile(join(cwd, path), bytes);
  assert.equal(reportRevision(path, bytes, cwd), "");
  const git = (...args) => execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
  git("init", "-q");
  git("add", path);
  git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
    "commit", "-qm", "Record report");
  assert.equal(reportRevision(path, bytes, cwd), git("rev-parse", "HEAD"));
  const rerun = Buffer.from('{"run": 2}');
  await writeFile(join(cwd, path), rerun);
  assert.equal(reportRevision(path, rerun, cwd), "");
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

test("load release rejects raw timing, count, failure or aggregate drift", async () => {
  const bytes = await readFile(join(root, "evaluations/results/backend-load-v1.json"));
  const summary = JSON.parse(await readFile(join(root, "benchmarks/data/backend-load-v1/summary.json")));
  for (const change of [
    report => { report.model.revision = "unexpected"; },
    report => { report.workload.document_ids.reverse(); },
    report => { report.trials[0].requests[0].finish_ns++; },
    report => { report.trials[0].requests[0].failure = "timeout"; },
    report => { report.trials[0].summary.requests++; },
    report => { report.scenarios[0].latency_ms.p95++; },
  ]) {
    const altered = JSON.parse(bytes);
    change(altered);
    const modified = Buffer.from(JSON.stringify(altered));
    const hash = createHash("sha256").update(modified).digest("hex");
    await assert.rejects(validateLoad(modified, { ...summary, report_sha256: hash }), /Invalid backend load/);
  }
  await assert.rejects(validateLoad(bytes, { ...summary, private_rows: [] }), /aggregate mismatch/);
  await assert.rejects(validateLoad(bytes, { ...summary, report_sha256: "0".repeat(64) }), /digest/);
});

test("a closed-loop worker cannot schedule its next request before its previous finish", async () => {
  const report = JSON.parse(await readFile(join(root, "evaluations/results/backend-load-v1.json")));
  const summary = JSON.parse(await readFile(join(root, "benchmarks/data/backend-load-v1/summary.json")));
  const trial = report.trials.find(trial => trial.clients === 4);
  const requests = trial.requests.filter(row => row.worker === 2);
  requests[1].scheduled_ns = requests[0].finish_ns - 1;
  const bytes = Buffer.from(JSON.stringify(report));
  const report_sha256 = createHash("sha256").update(bytes).digest("hex");
  await assert.rejects(validateLoad(bytes, { ...summary, report_sha256 }), /overlapping requests on one worker/);
});
