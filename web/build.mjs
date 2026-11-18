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
    for (const name of ["workbench.js", "workbench.css", "architecture.png"]) {
      await copyFile(join(demo, "static", name), join(assets, name));
    }
    for (const name of ["retrieval.mjs", "worker-client.mjs", "browser.mjs", "inference.mjs"]) {
      await copyFile(join(root, "web", name), join(assets, name));
    }
    const html = await readFile(join(demo, "static/index.html"), "utf8");
    if (!html.includes('data-runtime="server"')) throw Error("Missing runtime marker.");
    await writeFile(join(staging, "index.html"), html.replace('data-runtime="server"', 'data-runtime="browser"'));
    const fixture = JSON.parse(await readFile(join(root, "src/ragforge/evaluation/fixture.json"), "utf8"));
    const documents = fixture.documents.map(({ title, source, content }) => ({ title, source, content }));
    const questions = fixture.cases.map(entry => entry.question);
    await writeFile(join(staging, "corpus.json"), JSON.stringify({ documents, questions }));
    await copyFile(join(demo, "measurements.json"), join(staging, "results.json"));
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
