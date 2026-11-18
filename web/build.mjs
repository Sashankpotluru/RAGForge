import { validateCorpus, validateVectors } from "./vector-index.mjs";
import { copyFile, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const demo = join(root, "src/ragforge/demo");

export async function buildSite(output = join(root, "dist/RAGForge")) {
  await mkdir(dirname(output), { recursive: true });
  const staging = await mkdtemp(join(dirname(output), ".pages-"));
  try {
    const assets = join(staging, "assets");
    await mkdir(assets);
    for (const name of ["workbench.js", "workbench.css", "architecture-scifact.png"]) {
      await copyFile(join(demo, "static", name), join(assets, name));
    }
    for (const name of ["retrieval.mjs", "worker-client.mjs", "browser.mjs", "inference.mjs", "assets.mjs", "protocol.mjs", "neural.mjs", "vector-index.mjs", "generation.mjs"]) {
      await copyFile(join(root, "web", name), join(assets, name));
    }
    const html = await readFile(join(demo, "static/index.html"), "utf8");
    if (!html.includes('data-runtime="server"')) throw Error("Missing runtime marker.");
    await writeFile(join(staging, "index.html"), html.replace('data-runtime="server"', 'data-runtime="browser"'));
    const data = join(root, "benchmarks/data/scifact-v1");
    const manifest = JSON.parse(await readFile(join(data, "index.json"), "utf8"));
    const corpus = await validateCorpus(await readFile(join(data, "corpus.json")), manifest);
    const vectors = Uint8Array.from(await readFile(join(data, "vectors.f32"))).buffer;
    await validateVectors(vectors, manifest);
    const summary = JSON.parse(await readFile(join(data, "summary.json"), "utf8"));
    if (corpus.documents.length !== 5183 || summary.documents !== manifest.rows || summary.queries !== 300 ||
      summary.runs.map(run => run.mode).join() !== "bm25,dense,hybrid,hybrid_rerank") throw Error("Incomplete SciFact release.");
    for (const name of ["corpus.json", "index.json", "vectors.f32"]) await copyFile(join(data, name), join(staging, name));
    await copyFile(join(data, "summary.json"), join(staging, "results.json"));
    await rm(output, { recursive: true, force: true });
    await rename(staging, output);
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await buildSite();
  console.log("Built dist/RAGForge/");
}
