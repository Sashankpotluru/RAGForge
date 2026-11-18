import { validateCorpus } from "./vector-index.mjs";

export async function fetchAsset(name) {
  const response = await fetch(new URL("../" + name, import.meta.url));
  if (!response.ok) throw Error("Could not load " + name + " (HTTP " + response.status + "). Retry the download.");
  return response;
}
export async function loadCorpus() {
  const [manifest, bytes] = await Promise.all([
    fetchAsset("index.json").then(r => r.json()), fetchAsset("corpus.json").then(r => r.arrayBuffer()),
  ]);
  return { manifest, corpus: await validateCorpus(bytes, manifest) };
}
