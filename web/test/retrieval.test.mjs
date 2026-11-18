import test from "node:test";
import assert from "node:assert/strict";
import { bm25, fuse, rankScores, search, cite, validateQuery, validateAnswer } from "../retrieval.mjs";

const docs = [
  { title: "Backups", source: "b", content: "Backups run every six hours." },
  { title: "Access", source: "a", content: "Credentials expire every four hours." },
  { title: "Audit", source: "c", content: "Audit events are recorded daily." },
];

test("BM25 ranks the rare matching term first and omits zero-overlap passages", () => {
  assert.deepEqual(bm25(docs, "When do credentials expire?"), [1]);
  assert.deepEqual(bm25(docs, "galaxies"), []);
  assert.equal(bm25(docs, "backups hours")[0], 0);
});

test("RRF credits each source once per ranking and breaks ties consistently", () => {
  assert.deepEqual(fuse([2, 2, 0], [0, 2]), [0, 2]);
  assert.deepEqual(fuse([1, 2], [2, 0]), [2, 1, 0]);
  assert.deepEqual(rankScores([0.7, -1, 0.7]), [0, 2, 1]);
});

test("question boundaries use trimmed Unicode characters", () => {
  assert.equal(validateQuery("  backups? "), "backups?");
  assert.equal(validateQuery("🌿".repeat(500)).length, 1000);
  for (const value of ["   ", "a".repeat(501)]) assert.throws(() => validateQuery(value));
});

test("a citation needs whole words in exactly one source and rejects invented markers", () => {
  const hits = [{ citation_id: "C1", quote: docs[0].content }, { citation_id: "C2", quote: docs[1].content }];
  assert.equal(cite("Six hours.", hits), "Six hours. [C1]");
  for (const text of ["every", "every si", "never expires", "six hours [C9]"]) {
    assert.throws(() => cite(text, hits));
  }
  assert.throws(() => cite("six hours", [...hits, { citation_id: "C3", quote: "Wait six hours." }]));
});

test("lexical runs independently, dense and reranking honor their model scores", async () => {
  const lexical = await search(docs, "backups", "bm25");
  assert.deepEqual(lexical.map(hit => hit.source), ["b"]);
  const neural = {
    dense: async () => [0.1, 0.9, 0.2],
    rerank: async (_q, candidates) => candidates.map(d => d.source === "c" ? 8 : -8),
  };
  const dense = await search(docs, "backups", "dense", neural);
  assert.equal(dense[0].source, "a");
  const reranked = await search(docs, "backups", "hybrid_rerank", neural);
  assert.deepEqual([reranked[0].source, reranked[0].citation_id], ["c", "C1"]);
});

test("generation accepts only evidence in the model's context and preserves abstention", () => {
  const hits = [
    { citation_id: "C1", quote: "Database backups run every six hours." },
    { citation_id: "C2", quote: "Temporary credentials expire after four hours." },
  ];
  assert.deepEqual(validateAnswer("every six hours", hits, 1), { answer: "every six hours [C1]" });
  for (const text of ["after four hours", "once a week", "unanswerable"]) {
    assert.ok(validateAnswer(text, hits, 1).generation_error);
  }
  const duplicate = [...hits, { citation_id: "C3", quote: "Make copies every six hours." }];
  assert.ok(validateAnswer("every six hours", duplicate, 1).generation_error);
});

test("failed models or malformed model results never fall back to another mode", async () => {
  const offline = async () => { throw Error("offline"); };
  await assert.rejects(search(docs, "backups", "hybrid", { dense: offline }), /offline/);
  await assert.rejects(search(docs, "backups", "hybrid_rerank", {
    dense: async () => [0, 1, 2], rerank: offline,
  }), /offline/);
  for (const scores of [[0.9], [1, NaN, 0]]) {
    await assert.rejects(search(docs, "backups", "dense", { dense: async () => scores }));
  }
  await assert.rejects(search(docs, "backups", "unknown"));
});
