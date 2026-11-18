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
  bm25: "Matches words in each policy. Ready to use on this device.",
  dense: "Compares meanings with MiniLM. First use downloads about 24 MB of model files and a 23 MB runtime.",
  hybrid: "Combines MiniLM and keyword rankings. First use downloads about 24 MB of model files and a 23 MB runtime.",
  hybrid_rerank: "Rescores hybrid results with a cross-encoder. First use downloads about 48 MB of model files and a 23 MB runtime.",
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
      setStatus(labels[mode] + " found no passages for “" + query + "”. Try a policy term such as backups or credentials.");
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
  for (const doc of corpus.documents) {
    const item = element("details");
    item.append(element("summary", doc.title), element("p", doc.content), element("small", doc.source));
    byId("docs").append(item);
  }
}
function renderResults(results) {
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
  renderResults(data.results);
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
