import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const root = fileURLToPath(new URL("../", import.meta.url));
export const rawDir = join(root, "data/scifact");
export const assets = join(root, "benchmarks/data/scifact-v1");
export const dataset = {
  url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
  md5: "5f7d1de60b170fc8027bb7898e2efca1",
  sha256: "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
  files: {
    "corpus.jsonl": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
    "queries.jsonl": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
    "qrels/train.tsv": "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
    "qrels/test.tsv": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
  },
};
export const digest = (bytes, algo = "sha256") => createHash(algo).update(bytes).digest("hex");
export const readJson = async path => JSON.parse(await readFile(path, "utf8"));

export async function fetchArchive() {
  await mkdir(rawDir, { recursive: true });
  const path = join(rawDir, "scifact.zip");
  let bytes;
  try { bytes = await readFile(path); } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const response = await fetch(dataset.url);
    if (!response.ok) throw Error("Dataset download failed: HTTP " + response.status);
    bytes = Buffer.from(await response.arrayBuffer());
  }
  if (digest(bytes) !== dataset.sha256 || digest(bytes, "md5") !== dataset.md5) throw Error("SciFact archive checksum mismatch.");
  await writeFile(path, bytes);
  const { unzipSync } = await import("fflate");
  const members = unzipSync(bytes, { filter: file =>
    Object.hasOwn(dataset.files, file.name.slice(8)) && file.name.startsWith("scifact/") });
  for (const [name, hash] of Object.entries(dataset.files)) {
    const member = members["scifact/" + name];
    if (!member || digest(member) !== hash) throw Error("SciFact member checksum mismatch: " + name);
    await mkdir(dirname(join(rawDir, name)), { recursive: true });
    await writeFile(join(rawDir, name), member);
  }
}

export function parseQrels(text, queryMap, docMap) {
  const [header, ...lines] = text.trimEnd().split(/\r?\n/);
  if (header !== "query-id\tcorpus-id\tscore") throw Error("Invalid qrels header.");
  const qrels = new Map();
  for (const line of lines) {
    const [query, doc, score, ...extra] = line.split("\t");
    if (extra.length || !queryMap.has(query) || !docMap.has(doc) || !/^[0-9]+$/.test(score)) throw Error("Invalid qrel: " + line);
    if (!qrels.has(query)) qrels.set(query, new Map());
    if (qrels.get(query).has(doc)) throw Error("Duplicate qrel: " + line);
    qrels.get(query).set(doc, Number(score));
  }
  return qrels;
}

export async function readDataset() {
  const files = {};
  for (const [name, hash] of Object.entries(dataset.files)) {
    const bytes = await readFile(join(rawDir, name));
    if (digest(bytes) !== hash) throw Error("Checksum mismatch: " + name);
    files[name] = bytes.toString("utf8");
  }
  function jsonl(name) {
    const rows = files[name].trimEnd().split("\n").map(line => JSON.parse(line));
    const map = new Map(rows.map(row => [row._id, row]));
    if (rows.length !== map.size || rows.some(row => !row._id || typeof row.text !== "string")) throw Error("Invalid IDs/text in " + name);
    return map;
  }
  const docs = jsonl("corpus.jsonl");
  const queries = jsonl("queries.jsonl");
  const train = parseQrels(files["qrels/train.tsv"], queries, docs);
  const test = parseQrels(files["qrels/test.tsv"], queries, docs);
  if (docs.size !== 5183 || queries.size !== 1109 || train.size !== 809 || test.size !== 300 ||
    [...train.keys()].some(id => test.has(id))) throw Error("Unexpected SciFact counts or overlapping splits.");
  const documents = [...docs.values()].map(row => ({
    id: row._id, title: row.title, content: row.text, source: "SciFact " + row._id,
  }));
  const exampleIds = [...train.keys()].sort((a, b) => Number(a) - Number(b)).slice(0, 6);
  const attribution = {
    authors: "David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, Hannaneh Hajishirzi",
    paper: "https://aclanthology.org/2020.emnlp-main.609/",
    source: "https://github.com/allenai/scifact", distribution: dataset.url,
    terms: "https://github.com/allenai/scifact/blob/master/LICENSE.md",
    claims_and_evidence: "CC BY 4.0", abstracts: "ODC-By 1.0",
    adaptation: "BEIR title/abstract records retain original IDs and text. RAGForge reformats fields, selects six train claims, and computes document embeddings.",
  };
  const corpus = { name: "SciFact", attribution, documents, questions: exampleIds.map(id => queries.get(id).text), exampleIds };
  return { corpus, queries, train, test };
}
