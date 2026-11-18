import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildSite } from "../build.mjs";

test("a clean Pages artifact contains only its public allowlist, including after rebuild", async t => {
  const temp = await mkdtemp(join(tmpdir(), "ragforge-pages-"));
  t.after(() => rm(temp, { recursive: true, force: true }));
  const site = join(temp, "RAGForge");
  await buildSite(site);
  await writeFile(join(site, "secret.env"), "test");
  await buildSite(site);
  const files = (await readdir(site, { recursive: true })).sort();
  assert.deepEqual(files, [
    "assets", "assets/architecture-pages.png", "assets/browser.mjs", "assets/inference.mjs",
    "assets/retrieval.mjs", "assets/workbench.css", "assets/workbench.js", "assets/worker-client.mjs",
    "corpus.json", "index.html", "results.json",
  ].sort());
  const html = await readFile(join(site, "index.html"), "utf8");
  assert.match(html, /data-runtime="browser"/);
  assert.doesNotMatch(html, /(?:href|src)="\//);
  const corpus = JSON.parse(await readFile(join(site, "corpus.json"), "utf8"));
  assert.equal(corpus.documents.length, 6);
  assert.equal(corpus.questions.length, 12);
  assert.equal(corpus.cases, undefined);
});
