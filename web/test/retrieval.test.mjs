import test from "node:test";
import assert from "node:assert/strict";
import { createIndex, dotScores, hitsFor, rank, validateQuery } from "../retrieval.mjs";
import { cite, packContext, validateAnswer } from "../generation.mjs";

const docs = [
  { id: "1", title: "Backups", source: "b", content: "Backups run every six hours." },
  { id: "9", title: "Access", source: "a", content: "Credentials expire every four hours." },
  { id: "4", title: "Audit", source: "c", content: "Audit events are recorded daily." },
];
const index = createIndex(docs);
test("BM25 indexes postings once and omits documents with no query terms", async () => {
  const keyword = async query => (await rank(index, query, "bm25")).map(hit => hit.row);
  assert.deepEqual(await keyword("When do credentials expire?"), [1]);
  assert.deepEqual(await keyword("galaxies"), []);
  assert.equal((await keyword("backups hours"))[0], 0);
});

test("dot products use every row and return ties in stable corpus order", async () => {
  const v = new Float32Array(3 * 384);
  v[0] = v[768] = 1;
  v[385] = 1;
  const query = new Float32Array(384);
  query[0] = 1;
  assert.deepEqual(Array.from(dotScores(v, query)), [1, 0, 1]);
  const dense = await rank(index, "backups", "dense", { dense: () => dotScores(v, query) });
  assert.deepEqual(dense.map(hit => hit.row), [0, 2, 1]);
  assert.throws(() => dotScores(v, new Float32Array(10)));
});

test("RRF top-100 and reranked top-50 preserve the original tail; browser displays ten", async () => {
  const documents = Array.from({ length: 125 }, (_, i) => ({ ...docs[0], id: String(i) }));
  const large = createIndex(documents);
  const models = { dense: () => documents.map((_, i) => i),
    rerank: async (_q, candidates) => {
      assert.equal(candidates.length, 50);
      return candidates.map((_, i) => i);
    } };
  const fused = await rank(large, "backups", "hybrid", models);
  const reranked = await rank(large, "backups", "hybrid_rerank", models);
  assert.equal(fused.length, 100);
  assert.deepEqual(reranked.map(r => r.row), [...fused.slice(0, 50).reverse(), ...fused.slice(50)].map(r => r.row));
  assert.equal(new Set(reranked.map(r => r.row)).size, 100);
  assert.deepEqual(reranked.slice(50).map(r => r.score), fused.slice(50).map(r => r.score));
  assert.equal(hitsFor(documents, reranked).length, 10);
  assert.equal(hitsFor(documents, reranked)[0].document_id, documents[reranked[0].row].id);
  const small = await rank(index, "backups", "hybrid", { dense: () => [0, 9, 4] });
  assert.deepEqual(small.map(r => r.row), [0, 1, 2]);
  assert.equal(small[0].score, 1 / 61 + 1 / 63);
});

test("question boundaries count trimmed Unicode characters", () => {
  assert.equal(validateQuery(" backups? "), "backups?");
  assert.equal(validateQuery("🌿".repeat(500)).length, 1000);
  for (const value of ["   ", "a".repeat(501)]) assert.throws(() => validateQuery(value));
});

test("citation requires an entire phrase in exactly one supplied excerpt", () => {
  const hits = [{ citation_id: "C1", quote: docs[0].content }, { citation_id: "C2", quote: docs[1].content }];
  assert.equal(cite("Six hours.", hits), "Six hours. [C1]");
  for (const text of ["every", "every si", "never expires", "six hours [C9]", "unknown"]) {
    assert.ok(validateAnswer(text, hits).generation_error);
  }
  assert.throws(() => cite("six hours", [...hits, { citation_id: "C3", quote: "Wait six hours." }]));
});

test("generation validates only the displayed word-boundary substrings sent to the model", () => {
  const hits = Array.from({ length: 5 }, (_, i) => ({ citation_id: "C" + (i + 1),
    quote: ["First source", "Next source", "Third source", "Fourth source", "Hidden source"][i]
      + " token".repeat(60) + " omitted ending" }));
  const encode = text => text.split(/\s+/);
  const { prompt, excerpts } = packContext("What appears?", hits, encode, 65);
  assert.ok(encode(prompt).length <= 65);
  assert.equal(excerpts.length, 4);
  for (const [i, excerpt] of excerpts.entries()) {
    assert.ok(hits[i].quote.startsWith(excerpt.quote));
    assert.ok(prompt.includes(excerpt.quote));
    assert.ok(!excerpt.quote.includes("ending"));
  }
  assert.equal(cite("First source", excerpts), "First source [C1]");
  assert.throws(() => cite("omitted ending", excerpts));
  assert.throws(() => cite("Hidden source", excerpts));
  assert.throws(() => packContext("long ".repeat(100), hits, encode, 25));
});

test("failures in the requested model propagate without changing mode", async () => {
  const offline = async () => { throw Error("offline"); };
  await assert.rejects(rank(index, "backups", "hybrid", { dense: offline }), /offline/);
  await assert.rejects(rank(index, "backups", "hybrid_rerank", { dense: () => [0, 1, 2], rerank: offline }), /offline/);
  for (const scores of [[0.9], [1, NaN, 0]]) await assert.rejects(rank(index, "backups", "dense", { dense: () => scores }));
  await assert.rejects(rank(index, "backups", "unknown"));
});

test("each method removes the self-ID before the top-100 cutoff and reranking", async () => {
  const documents = Array.from({ length: 101 }, (_, i) => ({ id: "d" + i, title: "topic", content: "" }));
  const index = createIndex(documents);
  const models = { dense: () => documents.map(() => 1), rerank: (_query, head) => {
    assert.equal(head.length, 50);
    assert.ok(head.every(doc => doc.id !== "d0"));
    return head.map(() => 1);
  } };
  const longQuery = "topic ".repeat(100);
  for (const mode of ["bm25", "dense", "hybrid", "hybrid_rerank"]) {
    const hits = await rank(index, longQuery, mode, models, { excludeDocumentId: "d0", maxQueryChars: 5500 });
    assert.deepEqual(hits.map(hit => hit.row), Array.from({ length: 100 }, (_, i) => i + 1));
  }
  await assert.rejects(rank(index, longQuery, "bm25", models), /500/);
});
