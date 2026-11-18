import { isDeepStrictEqual } from "node:util";
import { sha256 } from "./vector-index.mjs";

const order = [1, 4, 8, 16, 8, 16, 1, 4, 16, 8, 4, 1];
function requireValue(condition, message) {
  if (!condition) throw Error("Invalid backend load release: " + message);
}

function summarize(rows, elapsed) {
  const times = rows.filter(row => row.failure === null).map(row => row.latency_ms).sort((a, b) => a - b);
  const percentile = p => times.length ? times[Math.ceil(times.length * p / 100) - 1] : null;
  return {
    requests: rows.length, successes: times.length, failures: rows.length - times.length,
    elapsed_s: elapsed, successful_requests_per_s: times.length / elapsed,
    latency_ms: { p50: percentile(50), p95: percentile(95), p99: percentile(99) },
  };
}

function checkMetrics(expected, actual) {
  // Python and JavaScript can round the sum of three floating-point intervals differently.
  for (const key of ["elapsed_s", "successful_requests_per_s"]) {
    requireValue(Number.isFinite(actual[key]) && Math.abs(expected[key] - actual[key]) < 1e-10, key);
  }
  for (const key of ["requests", "successes", "failures", "latency_ms"]) {
    requireValue(isDeepStrictEqual(actual[key], expected[key]), key);
  }
}

export async function validateLoad(reportBytes, summary) {
  requireValue(summary.report_sha256 === await sha256(reportBytes), "report digest mismatch");
  const report = JSON.parse(reportBytes);
  requireValue(report.schema_version === 1 && typeof report.scope === "string", "schema");
  requireValue(isDeepStrictEqual(report.protocol.clients, order) && report.trials.length === 12 &&
    report.protocol.scheduled_seconds === 30 && report.protocol.timeout_s === 30, "trial protocol");
  requireValue(isDeepStrictEqual(report.search, { collection: "scifact-load-v1", mode: "hybrid",
    candidate_limit: 100, result_limit: 10 }) && report.execution.uvicorn_workers === 1 &&
    report.execution.device === "cpu" && report.execution.torch_intraop === 6 && report.execution.torch_interop === 1 &&
    report.qdrant.version === "qdrant 1.20.0" && report.qdrant.telemetry === false &&
    report.model.name === "sentence-transformers-testing/stsb-bert-tiny-safetensors" &&
    report.model.revision === "f3cb857cba53019a20df283396bcca179cf051a4" &&
    report.model.weights_sha256 === "b57380d8465cb456819716ab92ba12933f8e9142ae5f930ba18ca830e9333af2", "service profile");
  requireValue(report.protocol.warmup_s.length === order.length &&
    report.protocol.warmup_s.every(t => Number.isFinite(t) && t > 0), "warmup");
  requireValue(report.preparation.documents === 1000 && report.preparation.chunks > 0 &&
    report.preparation.qdrant_points === report.preparation.chunks &&
    new Set(report.workload.document_ids).size === 1000 && report.workload.document_ids.length === 1000 &&
    await sha256(new TextEncoder().encode(report.workload.document_ids.join("\n"))) === report.workload.document_ids_sha256, "index preparation");
  const queryIds = new Set(report.workload.queries.map(([id]) => id));
  requireValue(queryIds.size === 32 && report.workload.queries.length === 32, "query workload");
  for (const [i, trial] of report.trials.entries()) {
    requireValue(trial.trial === i + 1 && trial.clients === order[i] && Number.isFinite(trial.elapsed_s) &&
      trial.elapsed_s >= 30 && trial.requests.length >= trial.clients, "incomplete trial");
    const counts = {}, statuses = {};
    const workers = Array(trial.clients).fill(0);
    const lastFinish = Array(trial.clients).fill(0);
    for (const row of trial.requests) {
      requireValue(Number.isInteger(row.worker) && row.worker >= 0 && row.worker < trial.clients &&
        row.sequence === workers[row.worker]++ && queryIds.has(row.query_id), "request sequence");
      const slot = (row.worker + row.sequence * trial.clients) % report.workload.queries.length;
      requireValue(row.query_id === report.workload.queries[slot][0], "query schedule");
      requireValue([row.scheduled_ns, row.start_ns, row.finish_ns].every(Number.isSafeInteger) &&
        row.scheduled_ns >= 0 && row.scheduled_ns < 30e9 && row.start_ns >= row.scheduled_ns &&
        row.finish_ns >= row.start_ns && row.finish_ns / 1e9 <= trial.elapsed_s &&
        row.latency_ms === (row.finish_ns - row.start_ns) / 1e6, "request interval");
      requireValue(row.scheduled_ns >= lastFinish[row.worker], "overlapping requests on one worker");
      lastFinish[row.worker] = row.finish_ns;
      const status = row.status ?? "no_response";
      statuses[status] = (statuses[status] ?? 0) + 1;
      if (row.failure === null) {
        requireValue(row.status === 200 && typeof row.request_id === "string" && row.request_id.length > 0, "successful response");
      } else {
        requireValue(typeof row.failure === "string" && row.failure.length > 0, "failure classification");
        counts[row.failure] = (counts[row.failure] ?? 0) + 1;
      }
    }
    requireValue(workers.every(n => n > 0), "missing client");
    const expected = summarize(trial.requests, trial.elapsed_s);
    checkMetrics(expected, trial.summary);
    requireValue(isDeepStrictEqual(counts, trial.summary.failure_counts) &&
      isDeepStrictEqual(statuses, trial.summary.status_counts) &&
      trial.summary.error_rate === expected.failures / expected.requests, "trial failures");
  }
  requireValue(report.scenarios.length === 4, "scenario count");
  const scenarios = [1, 4, 8, 16].map((clients, i) => {
    const subset = report.trials.filter(t => t.clients === clients);
    const expected = summarize(subset.flatMap(t => t.requests), subset.reduce((s, t) => s + t.elapsed_s, 0));
    const actual = report.scenarios[i];
    checkMetrics(expected, actual);
    // Construct a strict public allowlist, preserving exact validated Python float values.
    return { clients, trials: subset.length, ...expected, elapsed_s: actual.elapsed_s,
      successful_requests_per_s: actual.successful_requests_per_s };
  });
  requireValue(isDeepStrictEqual(scenarios, report.scenarios) &&
    isDeepStrictEqual(summary, { schema_version: 1, report_sha256: summary.report_sha256,
      scope: report.scope, scenarios }), "public aggregate mismatch");
}
