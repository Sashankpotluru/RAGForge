import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const root = fileURLToPath(new URL("../", import.meta.url));
const scifact = {
  key: "scifact", name: "SciFact", run: "scifact-v2", index: "scifact-v1",
  documents: 5183, queries: 1109, train: 809, test: 300,
  maxQueryChars: 500, excludeSelf: false,
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
const arguana = {
  key: "arguana", name: "ArguAna", run: "arguana-v1", index: "arguana-v1",
  documents: 8674, queries: 1406, train: 0, test: 1406,
  maxQueryChars: 5500, excludeSelf: true,
  warmup: "Public libraries should extend their opening hours so more residents can use them.",
  url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
  md5: "8ad3e3c2a5867cdced806d6503f29b99",
  sha256: "cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
  files: {
    "corpus.jsonl": "0627982cec4ee896df6dc98604ef6b499c83fd87a19eaaeb3f23b88bbef6cbce",
    "queries.jsonl": "f8b7c903f95d9a2d291de3915a8fac25085e9db679f19f59fea4145bfd1df618",
    "qrels/test.tsv": "0c47b481fa8b47fb9ca5e74bc8a017bdea14518c0e4a5d21345291180b3aca76",
  },
  // Exact query/document exceptions in the pinned distribution.
  missingQrels: [
    ["test-free-speech-debate-yfsdfkhbwu-con03a", "test-free-speech-debate-yfsdfkhbwu-con03b"],
    ["test-education-ufsdfkhbwu-con03a", "test-education-ufsdfkhbwu-con03b"],
    ["test-politics-dhwem-pro06a", "test-politics-dhwem-pro06b"],
    ["test-science-sghwbdgmo-con03a", "test-science-sghwbdgmo-con03b"],
    ["test-society-asfhwapg-con04a", "test-society-asfhwapg-con04b"],
  ],
};
export const datasets = { scifact, arguana };
export const modelCache = join(root, "data/scifact/models");
export function paths(dataset) {
  const rawDir = join(root, "data", dataset.key);
  const indexDir = join(root, "benchmarks/data", dataset.index);
  const assets = dataset.key === "scifact" ? indexDir : join(rawDir, "prepared");
  const runDir = join(root, "benchmarks/data", dataset.run);
  return { rawDir, assets, indexDir, runDir };
}
export function fromArgs(args) {
  const key = args[0] ?? "scifact";
  if (args.length > 1 || !Object.hasOwn(datasets, key)) throw Error("Expected dataset scifact or arguana.");
  return datasets[key];
}
export const digest = (bytes, algo = "sha256") => createHash(algo).update(bytes).digest("hex");
export const readJson = async path => JSON.parse(await readFile(path, "utf8"));

export async function fetchArchive(dataset) {
  const { rawDir } = paths(dataset);
  await mkdir(rawDir, { recursive: true });
  const path = join(rawDir, dataset.key + ".zip");
  let bytes;
  try { bytes = await readFile(path); } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const response = await fetch(dataset.url);
    if (!response.ok) throw Error("Dataset download failed: HTTP " + response.status);
    bytes = Buffer.from(await response.arrayBuffer());
  }
  if (digest(bytes) !== dataset.sha256 || digest(bytes, "md5") !== dataset.md5) throw Error("Dataset archive checksum mismatch.");
  await writeFile(path, bytes);
  const { unzipSync } = await import("fflate");
  const members = unzipSync(bytes, { filter: file =>
    Object.hasOwn(dataset.files, file.name.slice(dataset.key.length + 1)) && file.name.startsWith(dataset.key + "/") });
  for (const [name, hash] of Object.entries(dataset.files)) {
    const member = members[dataset.key + "/" + name];
    if (!member || digest(member) !== hash) throw Error("Dataset member checksum mismatch: " + name);
    await mkdir(dirname(join(rawDir, name)), { recursive: true });
    await writeFile(join(rawDir, name), member);
  }
}

export function parseQrels(text, queryMap, docMap, missingPairs = []) {
  const allowed = new Set(missingPairs.map(pair => JSON.stringify(pair)));
  const [header, ...lines] = text.trimEnd().split(/\r?\n/);
  if (header !== "query-id\tcorpus-id\tscore") throw Error("Invalid qrels header.");
  const qrels = new Map();
  for (const line of lines) {
    const [query, doc, score, ...extra] = line.split("\t");
    if (extra.length || !queryMap.has(query) || (!docMap.has(doc) && !allowed.delete(JSON.stringify([query, doc]))) || !/^[0-9]+$/.test(score)) throw Error("Invalid qrel: " + line);
    if (!qrels.has(query)) qrels.set(query, new Map());
    if (qrels.get(query).has(doc)) throw Error("Duplicate qrel: " + line);
    qrels.get(query).set(doc, Number(score));
  }
  if (allowed.size) throw Error("Expected missing qrel was not observed.");
  return qrels;
}

export async function readDataset(dataset) {
  const { rawDir } = paths(dataset);
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
  const train = dataset.train ? parseQrels(files["qrels/train.tsv"], queries, docs) : new Map();
  const test = parseQrels(files["qrels/test.tsv"], queries, docs, dataset.missingQrels);
  if (docs.size !== dataset.documents || queries.size !== dataset.queries || train.size !== dataset.train || test.size !== dataset.test ||
    [...train.keys()].some(id => test.has(id))) throw Error("Unexpected dataset counts or overlapping splits.");
  const documents = [...docs.values()].map(row => ({
    id: row._id, title: row.title, content: row.text, source: dataset.name + " " + row._id,
  }));
  const exampleIds = [...train.keys()].sort((a, b) => Number(a) - Number(b)).slice(0, 6);
  const attribution = dataset.key === "scifact" ? {
    authors: "David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, Hannaneh Hajishirzi",
    paper: "https://aclanthology.org/2020.emnlp-main.609/",
    source: "https://github.com/allenai/scifact", distribution: dataset.url,
    terms: "https://github.com/allenai/scifact/blob/master/LICENSE.md",
    claims_and_evidence: "CC BY 4.0", abstracts: "ODC-By 1.0",
    adaptation: "BEIR title/abstract records retain original IDs and text. RAGForge reformats fields, selects six train claims, and computes document embeddings.",
  } : {
    authors: "Henning Wachsmuth, Shahbaz Syed, Benno Stein",
    source: "https://webis.de/data/arguana-counterargs.html",
    original: "https://zenodo.org/records/3973258", original_license: "CC BY 4.0",
    distribution: dataset.url, mirror_license: "CC BY-SA 4.0",
    adaptation: "BEIR argument records retain the original IDs and text. RAGForge reformats fields and computes document embeddings.",
  };
  const corpus = { name: dataset.name, attribution, documents, questions: exampleIds.map(id => queries.get(id).text), exampleIds };
  const missingRelevant = [...test].flatMap(([query, judgments]) =>
    [...judgments].filter(([id, score]) => score > 0 && !docs.has(id)).map(([document]) => ({ query, document })));
  const coverage = { missing_relevant: missingRelevant,
    test_ids_in_corpus: [...test.keys()].filter(id => docs.has(id)).length,
    max_test_query_chars: Math.max(...[...test.keys()].map(id => [...queries.get(id).text].length)) };
  return { corpus, queries, train, test, coverage };
}
