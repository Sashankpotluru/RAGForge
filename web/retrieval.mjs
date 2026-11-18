const stopWords = new Set("a an and are as at be by do does for from how in is it of on or the to what when which who with".split(" "));

function terms(text) {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(word => !stopWords.has(word));
}

export function validateQuery(raw) {
  const query = raw.trim();
  if (!query || [...query].length > 500) throw Error("Enter a question of 1–500 characters.");
  return query;
}

export function rankScores(scores) {
  return scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
}

export function bm25(documents, query) {
  const passages = documents.map(d => terms(d.title + " " + d.content));
  const meanLength = passages.reduce((sum, words) => sum + words.length, 0) / passages.length;
  const scores = passages.map(() => 0);
  for (const word of new Set(terms(query))) {
    const counts = passages.map(p => p.filter(token => token === word).length);
    const df = counts.filter(n => n > 0).length;
    const idf = Math.log1p((passages.length - df + 0.5) / (df + 0.5));
    for (const [i, count] of counts.entries()) {
      if (!count) continue;
      const normalizer = 1.2 * (0.25 + 0.75 * passages[i].length / meanLength);
      scores[i] += idf * count * 2.2 / (count + normalizer);
    }
  }
  return rankScores(scores).filter(i => scores[i] > 0);
}

export function fuse(...rankings) {
  const scores = new Map();
  for (const ranking of rankings) {
    const unique = [...new Set(ranking)];
    unique.forEach((id, i) => scores.set(id, (scores.get(id) ?? 0) + 1 / (60 + i + 1)));
  }
  return [...scores.keys()].sort((a, b) => scores.get(b) - scores.get(a) || a - b);
}

function checkedRanks(scores, count) {
  if (scores.length !== count || !scores.every(Number.isFinite)) {
    throw Error("The model returned invalid passage scores. Retry this search.");
  }
  return rankScores(scores);
}

export async function search(documents, raw, mode, models) {
  const query = validateQuery(raw);
  let ranked;
  if (mode === "bm25") {
    ranked = bm25(documents, query);
  } else {
    if (!["dense", "hybrid", "hybrid_rerank"].includes(mode)) throw Error("Unknown retrieval mode.");
    const scores = await models.dense(query, documents);
    ranked = checkedRanks(scores, documents.length);
    if (mode !== "dense") ranked = fuse(ranked, bm25(documents, query));
    if (mode === "hybrid_rerank") {
      const candidates = ranked.map(i => documents[i]);
      const scores = await models.rerank(query, candidates);
      ranked = checkedRanks(scores, candidates.length).map(i => ranked[i]);
    }
  }
  return ranked.map((i, rank) => ({
    citation_id: "C" + (rank + 1), title: documents[i].title,
    source: documents[i].source, quote: documents[i].content,
  }));
}

export function cite(raw, hits, contextCount = hits.length) {
  const text = raw.trim();
  const phrase = text.replace(/[.!?]+$/u, "");
  if (phrase.split(/\s+/u).length >= 2 && !/\[C\d+\]/iu.test(text)) {
    const escaped = phrase.replace(/[.*+?^{}$()|[\]\\]/g, "\\$&");
    const pattern = new RegExp("(?<![\\p{L}\\p{N}_])" + escaped + "(?![\\p{L}\\p{N}_])", "iu");
    const matches = hits.filter(hit => pattern.test(hit.quote));
    if (matches.length === 1 && hits.indexOf(matches[0]) < contextCount) {
      return text + " [" + matches[0].citation_id + "]";
    }
  }
  throw Error("The generated words could not be matched to a single supplied source passage.");
}

export function validateAnswer(text, hits, contextCount) {
  if (/^(unknown|unanswerable|none)[.!]?$/i.test(text.trim())) {
    return { generation_error: "The model did not find an answer in these sources." };
  }
  try {
    return { answer: cite(text, hits, contextCount) };
  } catch (error) {
    return { generation_error: error.message };
  }
}
