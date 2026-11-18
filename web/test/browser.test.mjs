import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { Worker } from "node:worker_threads";
import { root } from "../../benchmarks/dataset.mjs";
import { load } from "../browser.mjs";
import { createIndex, rank } from "../retrieval.mjs";

test("SciFact starts and searches while optional reports are pending or unavailable", async t => {
  globalThis.Worker = Worker;
  t.after(() => { delete globalThis.Worker; });
  const assets = new Map([
    ["index.json", "scifact-v1/index.json"], ["corpus.json", "scifact-v1/corpus.json"],
    ["results.json", "scifact-v2/summary.json"],
  ]);
  const delayed = Promise.withResolvers();
  let missing;
  t.mock.method(globalThis, "fetch", async url => {
    const name = basename(url.pathname);
    if (name === missing || name === "arguana-results.json") return new Response(null, { status: 404 });
    if (name === "backend-load.json") return delayed.promise;
    assert.ok(assets.has(name), "Unexpected download: " + name);
    return new Response(await readFile(join(root, "benchmarks/data", assets.get(name))));
  });
  const workspace = await load();
  assert.equal(workspace.results.run, "scifact-v2");
  assert.equal(await workspace.arguana, null);
  assert.equal(workspace.caps.defaultMode, "bm25");
  const hits = await rank(createIndex(workspace.corpus.documents), workspace.corpus.questions[0], "bm25");
  assert.ok(hits.length > 0);
  delayed.reject(Error("Offline"));
  assert.equal(await workspace.capacity, null);

  for (missing of assets.keys()) {
    await assert.rejects(load(), error => error.message.includes(missing) && error.message.includes("404"));
  }
});
