const byId = id => document.getElementById(id);
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const browser = document.documentElement.dataset.runtime === "browser";
const labels = { bm25: "Keyword (BM25)", dense: "Dense", hybrid: "Hybrid", hybrid_rerank: "Hybrid + reranking" };
const notes = {
  bm25: "Ranks abstracts by matching search terms. Runs on this device.",
  dense: "Compares meanings with MiniLM. First use downloads about 24 MB of model files, 8 MB of document vectors and a 23 MB runtime.",
  hybrid: "Combines MiniLM and keyword rankings. First use downloads about 24 MB of model files, 8 MB of document vectors and a 23 MB runtime.",
  hybrid_rerank: "Rescores hybrid results with a cross-encoder. First use downloads about 48 MB of model files, 8 MB of document vectors and a 23 MB runtime.",
};
const views = ["workbench", "corpus", "results", "architecture"];
function navigate(moveFocus = false) {
  const hash = location.hash.slice(1);
  if (hash === "main") return;
  const target = views.includes(hash) ? hash : "workbench";
  for (const id of views) byId(id).hidden = id !== target;
  for (const link of document.querySelectorAll("nav a")) {
    if (link.hash === "#" + target) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  const heading = byId(target).querySelector("h1");
  document.title = heading.textContent + " | RAGForge";
  if (moveFocus) {
    const anchor = /^C\d+$/.test(hash) && byId(hash);
    const destination = anchor || heading;
    destination.tabIndex = -1;
    destination.focus({ preventScroll: true });
    destination.scrollIntoView({ block: "nearest" });
  }
}
window.addEventListener("hashchange", () => navigate(true));
navigate();

async function api(path, options) {
  const response = await fetch("./api/" + path, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw Error(data.detail || "Request failed. Please retry.");
  }
  return response.json();
}
const server = {
  async load() {
    const [caps, corpus, results] = await Promise.all(["capabilities", "corpus", "results"].map(path => api(path)));
    return { caps, corpus, results };
  },
  run(query, mode, answer) {
    return api(answer ? "answer" : "search", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, mode }),
    });
  },
};
let transport, caps, busy = false, canGenerate = false;
const setStatus = text => { byId("status").textContent = text; };

function refreshControls() {
  byId("mode").disabled = byId("search").disabled = busy || !caps;
  byId("answer").disabled = busy || !canGenerate;
  byId("cancel").hidden = !browser || !busy;
  byId("question").readOnly = busy;
  for (const sample of byId("samples").children) sample.disabled = busy;
}
function describeMode() {
  byId("model").textContent = browser ? notes[byId("mode").value] : "Python backend embedding model: " + caps.embedding + ".";
}

function passageLink(id, label) {
  const link = element("a", label, "cite");
  link.href = "#" + id;
  link.setAttribute("aria-label", "Go to source passage " + id);
  return link;
}
function renderEvidence(data) {
  const generated = byId("generated");
  if (data.answer) {
    const paragraph = element("p");
    for (const part of data.answer.split(/(\[C\d+\])/g)) {
      if (/^\[C\d+\]$/.test(part)) paragraph.append(passageLink(part.slice(1, -1), part));
      else paragraph.append(document.createTextNode(part));
    }
    generated.append(element("h3", "Cited answer"), paragraph,
      element("small", "Generation took " + formatTime(data.generation_ms) + "."));
  } else if (data.generation_error) {
    generated.append(element("h3", "No cited answer"), element("p", data.generation_error));
  }
  for (const hit of data.hits) {
    const block = element("div", undefined, "passage");
    block.id = hit.citation_id;
    block.tabIndex = -1;
    const content = element("div");
    content.append(element("h3", hit.title), element("p", hit.quote), element("small", hit.source));
    const excerpt = data.excerpts?.find(item => item.citation_id === hit.citation_id);
    if (excerpt && excerpt.quote !== hit.quote) {
      const detail = element("details");
      detail.append(element("summary", "Excerpt supplied to Flan-T5"), element("p", excerpt.quote));
      content.append(detail);
    }
    block.append(passageLink(hit.citation_id, hit.citation_id), content);
    byId("hits").append(block);
  }
}
function formatTime(ms) {
  if (ms >= 1000) return (ms / 1000).toFixed(1) + " s";
  if (ms < 1) return "<1 ms";
  return Math.round(ms) + " ms";
}
async function run(answer) {
  if (busy || !caps) return;
  const query = byId("question").value.trim();
  if (!query || [...query].length > 500) {
    byId("question").focus();
    setStatus("Enter a question of 1–500 characters.");
    return;
  }
  const mode = byId("mode").value;
  busy = true;
  refreshControls();
  byId("examples").open = false;
  byId("hits").replaceChildren();
  byId("generated").replaceChildren();
  setStatus(browser && (mode !== "bm25" || answer) ? "Starting the model worker…" : "Retrieving source passages…");
  try {
    const data = await transport.run(query, mode, answer, setStatus);
    renderEvidence(data);
    if (data.hits.length) {
      const count = data.hits.length + (data.hits.length === 1 ? " passage" : " passages");
      setStatus(labels[mode] + " found " + count + " for “" + query + "” in " + formatTime(data.search_ms) + (browser ? " on this device." : " on the server."));
    } else {
      setStatus(labels[mode] + " found no passages for “" + query + "”. Try a different search term.");
    }
  } catch (error) {
    setStatus(error.name === "AbortError" ? error.message : "Search failed. " + error.message + " Retry when ready.");
  } finally {
    busy = false;
    refreshControls();
  }
}
byId("query").addEventListener("submit", e => { e.preventDefault(); run(false); });
byId("answer").addEventListener("click", () => run(true));
byId("cancel").addEventListener("click", () => transport.cancel());
byId("mode").addEventListener("change", describeMode);
byId("examples").addEventListener("keydown", e => {
  if (e.key === "Escape") {
    byId("examples").open = false;
    byId("examples").querySelector("summary").focus();
  }
});

function renderCorpus(corpus) {
  for (const question of corpus.questions) {
    const button = element("button", question);
    button.type = "button";
    button.onclick = () => {
      byId("question").value = question;
      byId("examples").open = false;
      byId("question").focus();
    };
    byId("samples").append(button);
  }
  const total = corpus.documents.length.toLocaleString();
  const scientific = Boolean(corpus.attribution);
  document.querySelector("#workbench .intro").textContent = scientific
    ? "Retrieve evidence from " + total + " SciFact scientific abstracts. A relevant paper may support or refute a claim."
    : "Search " + total + " sample policies and inspect the passages behind each answer.";
  byId("query-label").textContent = scientific ? "Claim or search query" : "Your question";
  byId("example-label").textContent = scientific ? "Example claims" : "Sample questions";
  byId("question").value = corpus.questions[0];
  document.querySelector("#corpus h1").textContent = scientific ? corpus.name + " corpus" : "Policy corpus";
  document.querySelector("#corpus .intro").textContent = total + (scientific ? " scientific abstracts" : " synthetic policies")
    + ". Filter by title, text or source ID.";
  if (scientific) {
    const credit = corpus.attribution;
    byId("attribution").append("SciFact by Wadden et al., distributed by BEIR. Abstracts use ODC-By 1.0. Claims and evidence annotations use CC BY 4.0. ",
      link("Dataset paper", credit.paper), " · ", link("Source and licensing", credit.terms));
  }
  const pageSize = 12;
  let matches = corpus.documents, page = 0;
  function displayPage() {
    const start = page * pageSize;
    byId("docs").replaceChildren();
    for (const doc of matches.slice(start, start + pageSize)) {
      const item = element("details");
      item.append(element("summary", doc.title), element("p", doc.content), element("small", doc.source));
      byId("docs").append(item);
    }
    byId("page-count").textContent = matches.length
      ? (start + 1) + "–" + Math.min(start + pageSize, matches.length) + " of " + matches.length.toLocaleString()
      : "No matching documents";
    byId("previous").disabled = page === 0;
    byId("next").disabled = start + pageSize >= matches.length;
  }
  byId("previous").onclick = () => { page--; displayPage(); };
  byId("next").onclick = () => { page++; displayPage(); };
  byId("filter").oninput = () => {
    const term = byId("filter").value.trim().toLowerCase();
    matches = corpus.documents.filter(doc => [doc.title, doc.content, doc.source].some(value => value.toLowerCase().includes(term)));
    page = 0;
    displayPage();
  };
  displayPage();
}
function link(text, href) {
  const anchor = element("a", text);
  anchor.href = href;
  return anchor;
}
function renderBenchmark(results) {
  document.querySelector("#results .intro").textContent = results.runs.length + " retrieval methods on " + results.queries
    + " official SciFact test claims and " + results.documents.toLocaleString() + " abstracts.";
  const table = element("table");
  table.createCaption().textContent = "Mean document retrieval quality with 95% query-bootstrap intervals";
  const head = table.createTHead().insertRow();
  const columns = [["nDCG @10", "ndcg@10"], ["Recall @10", "recall@10"], ["Recall @100", "recall@100"],
    ["MRR @10", "mrr@10"], ["Hit rate @1", "hit_rate@1"]];
  for (const label of ["Method", ...columns.map(([label]) => label), "Median Node time"]) {
    const cell = element("th", label);
    cell.scope = "col";
    head.append(cell);
  }
  const rows = table.createTBody();
  for (const run of results.runs) {
    const row = rows.insertRow();
    const name = element("th", labels[run.mode]);
    name.scope = "row";
    row.append(name);
    for (const [, key] of columns) {
      const metric = run.metrics[key];
      const ci = metric.ci95.map(value => value.toFixed(3)).join("–");
      row.insertCell().append(metric.mean.toFixed(3), element("small", ci, "interval"));
    }
    row.insertCell().textContent = run.timing_ms.median.toFixed(1) + " ms";
  }
  const comparisons = element("details", undefined, "comparisons");
  comparisons.append(element("summary", "Paired nDCG @10 differences from BM25"));
  const changes = element("ul");
  const alternatives = results.runs.filter(run => run.mode !== "bm25");
  for (const run of alternatives) {
    const delta = run.paired_ndcg_delta;
    const signed = value => (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(3);
    changes.append(element("li", labels[run.mode] + " " + signed(delta.mean) + " (95% interval "
      + delta.ci95.map(signed).join(" to ") + ")"));
  }
  comparisons.append(changes);
  const overlap = alternatives.filter(run => run.paired_ndcg_delta.ci95[0] <= 0 && run.paired_ndcg_delta.ci95[1] >= 0);
  const pairedNote = element("p", overlap.length + " of " + alternatives.length
    + " paired nDCG intervals relative to BM25 include zero.", "note");
  const data = element("p", undefined, "note");
  data.append(link("Full per-query results", "https://raw.githubusercontent.com/Sashankpotluru/RAGForge/main/evaluations/results/scifact-v1.json"),
    " · ", link("Methodology and paired comparisons", "https://github.com/Sashankpotluru/RAGForge/blob/main/docs/scifact-evaluation.md"));
  byId("metrics").append(table, pairedNote, comparisons, element("p", "Measured on " + results.hardware + " using Node " + results.node
    + " after model warmup. Browser downloads, initialization and answer generation are excluded.", "note"), data);
  byId("metric-guide").textContent = "nDCG rewards relevant papers near the top. Recall measures the fraction of judged relevant papers retrieved. MRR measures how early the first relevant paper appears. Hit rate counts top-ranked relevant papers. Intervals summarize variation across these test claims.";
}
function renderResults(results, corpus) {
  if (browser) return renderBenchmark(results);
  document.querySelector("#results .intro").textContent = "Recorded Python backend measurements for " + corpus.questions.length + " fixture questions across " + corpus.documents.length + " synthetic policies.";
  const table = element("table");
  table.createCaption().textContent = "Python backend fixture evaluation";
  const head = table.createTHead().insertRow();
  for (const label of ["Mode", "Hit rate @1", "MRR @10", "Median time"]) {
    const cell = element("th", label);
    cell.scope = "col";
    head.append(cell);
  }
  const rows = table.createTBody();
  for (const r of results.runs) {
    const row = rows.insertRow();
    const name = element("th", labels[r.mode]);
    name.scope = "row";
    row.append(name);
    for (const value of [(100 * r.metrics["hit_rate@1"]).toFixed(1) + "%", r.metrics["mrr@10"].toFixed(3), r.latency_ms.p50.toFixed(1) + " ms"]) {
      row.insertCell().textContent = value;
    }
  }
  byId("metrics").append(table, element("p",
    "Recorded " + results.created_at.slice(0, 10) + " on " + results.hardware.machine + " with Python " + results.hardware.python + ".", "note"));
}
async function start() {
  transport = browser ? await import("./browser.mjs") : server;
  const data = await transport.load();
  caps = data.caps;
  for (const mode of caps.modes) byId("mode").add(new Option(labels[mode], mode));
  byId("mode").value = caps.defaultMode || "hybrid";
  describeMode();
  renderCorpus(data.corpus);
  renderResults(data.results, data.corpus);
  canGenerate = Boolean(caps.generation);
  if (browser) {
    byId("gen-note").textContent = canGenerate
      ? "Flan-T5 answers from the retrieved passages on this device. First use downloads about 278 MB for the model, plus a shared 23 MB runtime."
      : "Local models require a browser with WebAssembly and worker support.";
  } else {
    byId("gen-note").textContent = canGenerate ? "This server runs " + caps.generation + " locally." : "Generation is disabled on this server.";
  }
  refreshControls();
  await run(false);
}
start().catch(error => setStatus("Could not open the workspace. " + error.message));
