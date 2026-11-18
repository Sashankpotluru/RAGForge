import { protocol } from "../web/protocol.mjs";

export function queryMetrics(ids, judgments) {
  const relevant = [...judgments.values()].filter(n => n > 0).sort((a, b) => b - a);
  if (!relevant.length) throw Error("A query has no positive judgments.");
  const seen = new Set();
  const gains = ids.slice(0, 100).map(id => {
    const grade = seen.has(id) ? 0 : judgments.get(id) ?? 0;
    seen.add(id);
    return grade;
  });
  const dcg = grades => grades.slice(0, 10).reduce((sum, grade, i) => sum + (2 ** grade - 1) / Math.log2(i + 2), 0);
  const first = gains.slice(0, 10).findIndex(grade => grade > 0);
  const recall = k => gains.slice(0, k).filter(grade => grade > 0).length / relevant.length;
  return { "hit_rate@1": Number(gains[0] > 0), "mrr@10": first < 0 ? 0 : 1 / (first + 1),
    "ndcg@10": dcg(gains) / dcg(relevant), "recall@10": recall(10), "recall@100": recall(100) };
}
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
function percentile(sorted, p) {
  const offset = (sorted.length - 1) * p;
  const lower = Math.floor(offset), upper = Math.ceil(offset);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (offset - lower);
}
function resamples(n) {
  let state = protocol.seed;
  const samples = [];
  for (let i = 0; i < protocol.bootstrapSamples; i++) {
    const rows = new Uint32Array(n);
    for (let j = 0; j < n; j++) {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      rows[j] = Math.floor((state / 2 ** 32) * n);
    }
    samples.push(rows);
  }
  return samples;
}
function interval(values, samples) {
  const means = samples.map(sample => sample.reduce((a, i) => a + values[i], 0) / sample.length).sort((a, b) => a - b);
  return { mean: mean(values), ci95: [percentile(means, 0.025), percentile(means, 0.975)] };
}

export function summarize(queries, expectedIds, documentCount) {
  const seen = new Set(queries.map(q => q.id));
  if (queries.length !== expectedIds.length || seen.size !== queries.length || expectedIds.some(id => !seen.has(id))) {
    throw Error("Missing or duplicate evaluation queries.");
  }
  const samples = resamples(queries.length);
  const runs = protocol.modes.map(mode => {
    const entries = queries.map(q => q.modes[mode]);
    if (entries.some(entry => !entry || entry.error)) throw Error("Incomplete evaluation for " + mode);
    const metrics = Object.fromEntries(Object.keys(entries[0].metrics).map(metric =>
      [metric, interval(entries.map(entry => entry.metrics[metric]), samples)]));
    const deltas = entries.map((entry, i) => entry.metrics["ndcg@10"] - queries[i].modes.bm25.metrics["ndcg@10"]);
    const examples = queries.map((q, i) => ({ id: q.id, query: q.text, delta: deltas[i] }));
    examples.sort((a, b) => a.delta - b.delta || Number(a.id) - Number(b.id));
    const times = entries.map(entry => entry.timing_ms).sort((a, b) => a - b);
    return { mode, metrics, timing_ms: { median: percentile(times, 0.5), p95: percentile(times, 0.95) },
      paired_ndcg_delta: interval(deltas, samples),
      examples: { largest_loss: examples.find(row => row.delta < 0) ?? null,
        largest_gain: examples.filter(row => row.delta > 0).sort((a, b) => b.delta - a.delta || Number(a.id) - Number(b.id))[0] ?? null } };
  });
  return { dataset: "BEIR SciFact", documents: documentCount, queries: queries.length, runs };
}
