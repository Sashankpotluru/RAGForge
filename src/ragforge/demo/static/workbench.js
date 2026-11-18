"use strict";
const byId = (id) => document.getElementById(id);
const element = (tag, value) => {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = value;
  return node;
};
const views = ["workbench", "corpus", "results", "architecture"];
const labels = {
  dense: "Dense",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + reranking",
};
function navigate() {
  const hash = location.hash.slice(1);
  const view = views.includes(hash) ? hash : "workbench";
  for (const id of views) byId(id).hidden = id !== view;
  for (const link of document.querySelectorAll("nav a")) {
    if (link.hash === "#" + view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}
window.addEventListener("hashchange", navigate);
navigate();
const get = async (path) => {
  const response = await fetch("/api/" + path);
  if (!response.ok)
    throw Error("Sample data is unavailable. Reload to try again.");
  return response.json();
};
let hasGenerator = false;
async function run(answer) {
  const query = byId("question").value.trim();
  if (!query) {
    byId("question").focus();
    return;
  }
  byId("search").disabled = byId("answer").disabled = true;
  byId("status").textContent = answer
    ? "Retrieving passages and generating an answer…"
    : "Retrieving passages…";
  byId("hits").replaceChildren();
  byId("generated").replaceChildren();
  try {
    const response = await fetch("/api/" + (answer ? "answer" : "search"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode: byId("mode").value }),
    });
    const data = await response.json();
    if (!response.ok)
      throw Error(data.detail || "Request failed. Please retry.");
    byId("status").textContent =
      data.hits.length + " passages retrieved in " + data.search_ms + " ms.";
    if (answer) {
      const output = byId("generated");
      output.append(
        element("h3", "Generated answer"),
        element("p", data.answer || data.generation_error),
      );
      if (data.answer) {
        for (const id of new Set(data.answer.match(/\bC\d+\b/g) || [])) {
          const link = element("a", "[" + id + "] ");
          link.href = "#" + id;
          output.append(link);
        }
        output.append(
          element("small", " Local inference · " + data.generation_ms + " ms"),
        );
      }
    }
    for (const hit of data.hits) {
      const block = element("div");
      block.className = "passage";
      block.id = hit.citation_id;
      block.append(
        element("h3", "[" + hit.citation_id + "] " + hit.title),
        element("p", hit.quote),
        element("small", hit.source),
      );
      byId("hits").append(block);
    }
  } catch (error) {
    byId("status").textContent = error.message;
  } finally {
    byId("search").disabled = false;
    byId("answer").disabled = !hasGenerator;
  }
}
byId("query").addEventListener("submit", (e) => {
  e.preventDefault();
  run(false);
});
byId("answer").addEventListener("click", () => run(true));
async function start() {
  const [caps, corpus, results] = await Promise.all(
    ["capabilities", "corpus", "results"].map(get),
  );
  for (const mode of caps.modes)
    byId("mode").add(new Option(labels[mode], mode));
  byId("mode").value = "hybrid";
  hasGenerator = Boolean(caps.generation);
  byId("model").textContent =
    "Embedding: " +
    caps.embedding +
    ". Generation: " +
    (caps.generation || "disabled") +
    ".";
  for (const question of corpus.questions.slice(0, 4)) {
    const button = element("button", question);
    button.type = "button";
    button.onclick = () => {
      byId("question").value = question;
      byId("question").focus();
    };
    byId("samples").append(button);
  }
  for (const doc of corpus.documents) {
    const item = element("details");
    item.append(
      element("summary", doc.title),
      element("p", doc.content),
      element("small", doc.source),
    );
    byId("docs").append(item);
  }
  const table = element("table");
  const head = table.createTHead().insertRow();
  for (const label of ["Mode", "Hit rate @1", "MRR @10", "Median latency"])
    head.append(element("th", label));
  const rows = table.createTBody();
  for (const r of results.runs) {
    const row = rows.insertRow();
    for (const value of [
      labels[r.mode],
      (100 * r.metrics["hit_rate@1"]).toFixed(1) + "%",
      r.metrics["mrr@10"].toFixed(3),
      r.latency_ms.p50.toFixed(1) + " ms",
    ])
      row.insertCell().textContent = value;
  }
  byId("metrics").append(
    table,
    element(
      "p",
      "Recorded " +
        results.created_at.slice(0, 10) +
        " on " +
        results.hardware.machine +
        ".",
    ),
  );
  await run(false);
}
start().catch((e) => {
  byId("status").textContent = e.message;
});
