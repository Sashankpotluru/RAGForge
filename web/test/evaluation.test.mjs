import assert from "node:assert/strict";
import test from "node:test";
import { parseQrels } from "../../benchmarks/dataset.mjs";
import { queryMetrics, summarize } from "../../benchmarks/metrics.mjs";
import { protocol } from "../protocol.mjs";
import { sha256, validateCorpus, validateVectors } from "../vector-index.mjs";

test("graded nDCG normalizes against all judgments and repeated hits earn no extra gain", () => {
  const qrels = new Map([["x", 2], ["y", 1], ["z", 1], ["n", 0]]);
  const actual = queryMetrics(["n", "y", "y", "x"], qrels);
  const expected = (1 / Math.log2(3) + 3 / Math.log2(5)) / (3 + 1 / Math.log2(3) + 0.5);
  assert.equal(actual["ndcg@10"], expected);
  assert.equal(actual["hit_rate@1"], 0);
  assert.equal(actual["mrr@10"], 0.5);
  assert.equal(actual["recall@10"], 2 / 3);
  assert.equal(actual["recall@100"], 2 / 3);
  assert.equal(queryMetrics(Array(10).fill("a").concat("x"), qrels)["recall@10"], 0);
  assert.equal(queryMetrics(Array(99).fill("a").concat("x"), qrels)["recall@100"], 1 / 3);
  assert.equal(queryMetrics(Array(100).fill("a").concat("x"), qrels)["recall@100"], 0);
  assert.throws(() => queryMetrics([], new Map()));
});

test("qrels reject duplicate pairs, missing references, malformed rows and grades", () => {
  const parse = text => parseQrels("query-id\tcorpus-id\tscore\n" + text, new Set(["q"]), new Set(["d"]));
  for (const row of ["q\td\t1\nq\td\t1", "z\td\t1", "q\tz\t1", "q\td\t", "q\td\tone", "q\td\t1\textra"]) {
    assert.throws(() => parse(row));
  }
  assert.equal(parse("q\td\t1").get("q").get("d"), 1);
});

test("all-query means, paired bootstrap intervals and missing-query/error refusal", () => {
  const ids = ["1", "2", "3"];
  const queries = ids.map((id, i) => {
    const modes = Object.fromEntries(protocol.modes.map(mode => {
      const positive = i === 1 || (mode !== "bm25" && i === 2);
      return [mode, { timing_ms: 2, metrics: queryMetrics(positive ? ["d"] : [], new Map([["d", 1]])) }];
    }));
    return { id, text: id, modes };
  });
  const result = summarize(queries, ids, 1);
  assert.deepEqual(result, summarize(queries, ids, 1));
  assert.equal(result.runs[0].metrics["ndcg@10"].mean, 1 / 3);
  assert.deepEqual(result.runs[0].paired_ndcg_delta.ci95, [0, 0]);
  assert.equal(result.runs[1].paired_ndcg_delta.mean, 1 / 3);
  assert.equal(result.runs[1].examples.largest_gain.id, "3");
  assert.equal(result.runs[1].examples.largest_loss, null);
  assert.throws(() => summarize(queries.slice(1), ids, 1), /Missing/);
  assert.throws(() => summarize([queries[0], queries[0], queries[2]], ids, 1), /duplicate/);
  queries[2].modes.hybrid = { error: "inference error" };
  assert.throws(() => summarize(queries, ids, 1), /Incomplete/);
});

test("validates model, binary length, checksums, membership and row order", async () => {
  const enc = new TextEncoder();
  const bytes = enc.encode(JSON.stringify({ documents: [{ id: "3" }, { id: "2" }] }));
  const binary = new Float32Array(protocol.dimension * 2).buffer;
  const manifest = { version: protocol.version, rows: 2, dimension: protocol.dimension, encoding: "float32-le",
    dtype: "q8", model: protocol.models.embedder, max_tokens: protocol.embeddingTokens,
    corpus_sha256: await sha256(bytes), row_ids_sha256: await sha256(enc.encode("3\n2")),
    vector_bytes: binary.byteLength, vector_sha256: await sha256(binary) };
  assert.equal((await validateCorpus(bytes, manifest)).documents.length, 2);
  assert.equal((await validateVectors(binary, manifest)).length, 768);
  for (const changed of [{ rows: 3 }, { corpus_sha256: "" }, { row_ids_sha256: await sha256(enc.encode("2\n3")) }]) {
    await assert.rejects(validateCorpus(bytes, { ...manifest, ...changed }));
  }
  for (const changed of [{ rows: 3 }, { vector_sha256: "" }, { model: { revision: "wrong" } }, { vector_bytes: 1 }]) {
    await assert.rejects(validateVectors(binary, { ...manifest, ...changed }));
  }
});

test("only exact query/document exceptions survive missing-qrel validation and count against recall", () => {
  const header = "query-id\tcorpus-id\tscore\n";
  const parse = (line, pairs) => parseQrels(header + line, new Set(["a", "b"]), new Set(["z"]), pairs);
  const judgments = parse("a\tx\t1", [["a", "x"]]).get("a");
  assert.equal(judgments.get("x"), 1);
  assert.equal(queryMetrics(["z"], judgments)["recall@100"], 0);
  assert.throws(() => parse("b\tx\t1", [["a", "x"]]));
  assert.throws(() => parse("a\ty\t1", [["a", "x"]]));
  assert.throws(() => parse("a\tz\t1", [["a", "x"]]));
});
