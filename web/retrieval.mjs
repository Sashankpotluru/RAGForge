import { protocol } from "./protocol.mjs";

const stopWords = new Set("a an and are as at be by do does for from how in is it of on or the to what when which who with".split(" "));
function terms(text) {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(word => !stopWords.has(word));
}

export function validateQuery(raw, maxChars = 500) {
  const query = raw.trim();
  if (!query || [...query].length > maxChars) throw Error("Enter a search query of 1–" + maxChars + " characters.");
  return query;
}

export function createIndex(documents) {
  const postings = new Map();
  const lengths = [];
  for (const [row, doc] of documents.entries()) {
    const words = terms(doc.title + " " + doc.content);
    lengths.push(words.length);
    const counts = new Map();
    for (const word of words) counts.set(word, (counts.get(word) ?? 0) + 1);
    for (const [word, count] of counts) {
      if (!postings.has(word)) postings.set(word, []);
      postings.get(word).push([row, count]);
    }
  }
  const meanLength = lengths.reduce((sum, n) => sum + n, 0) / lengths.length || 1;
  const { k1, b } = protocol.bm25;
  const normalizers = lengths.map(length => k1 * (1 - b + b * length / meanLength));
  return { documents, postings, normalizers };
}

function byScore(a, b) { return b.score - a.score || a.row - b.row; }

function keyword(index, query, excludedId) {
  const scores = new Map();
  for (const term of new Set(terms(query))) {
    const posting = index.postings.get(term);
    if (!posting) continue;
    const idf = Math.log1p((index.documents.length - posting.length + 0.5) / (posting.length + 0.5));
    for (const [row, count] of posting) {
      if (index.documents[row].id === excludedId) continue;
      const gain = idf * count * (protocol.bm25.k1 + 1) / (count + index.normalizers[row]);
      scores.set(row, (scores.get(row) ?? 0) + gain);
    }
  }
  return [...scores].map(([row, score]) => ({ row, score })).sort(byScore).slice(0, protocol.candidateDepth);
}

function fusion(rankings) {
  const scores = new Map();
  for (const ranking of rankings) {
    [...new Set(ranking)].slice(0, protocol.candidateDepth).forEach((row, i) => {
      scores.set(row, (scores.get(row) ?? 0) + 1 / (protocol.rrfK + i + 1));
    });
  }
  return [...scores].map(([row, score]) => ({ row, score })).sort(byScore).slice(0, protocol.candidateDepth);
}

function checkedScores(scores, count) {
  if (scores.length !== count || !scores.every(Number.isFinite)) throw Error("Invalid model scores.");
  return Array.from(scores, (score, row) => ({ row, score }));
}

export function dotScores(vectors, query) {
  if (query.length !== protocol.dimension || vectors.length % query.length !== 0) throw Error("Embedding dimensions disagree.");
  const scores = new Float64Array(vectors.length / query.length);
  for (let row = 0; row < scores.length; row++) {
    let score = 0;
    const offset = row * query.length;
    for (let j = 0; j < query.length; j++) score += vectors[offset + j] * query[j];
    scores[row] = score;
  }
  return scores;
}

export async function rank(index, raw, mode, models, { excludeDocumentId, maxQueryChars } = {}) {
  const query = validateQuery(raw, maxQueryChars);
  if (mode === "bm25") return keyword(index, query, excludeDocumentId);
  if (!protocol.modes.includes(mode)) throw Error("Unknown retrieval mode.");
  const dense = checkedScores(await models.dense(query), index.documents.length)
    .filter(hit => index.documents[hit.row].id !== excludeDocumentId).sort(byScore).slice(0, protocol.candidateDepth);
  if (mode === "dense") return dense;
  const hybrid = fusion([dense.map(r => r.row), keyword(index, query, excludeDocumentId).map(r => r.row)]);
  if (mode === "hybrid") return hybrid;
  const head = hybrid.slice(0, protocol.rerankDepth);
  const scores = await models.rerank(query, head.map(hit => index.documents[hit.row]));
  const reranked = checkedScores(scores, head.length).map(({ row, score }) =>
    ({ row: head[row].row, score, stage: "cross_encoder" })).sort(byScore);
  return [...reranked, ...hybrid.slice(head.length).map(hit => ({ ...hit, stage: "rrf_tail" }))];
}

export function hitsFor(documents, ranking) {
  return ranking.slice(0, protocol.displayDepth).map(({ row }, i) => {
    const doc = documents[row];
    return { citation_id: "C" + (i + 1), title: doc.title, source: doc.source, quote: doc.content, document_id: doc.id };
  });
}
