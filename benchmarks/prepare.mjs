import { mkdir, writeFile } from "node:fs/promises";
import { endianness } from "node:os";
import { join } from "node:path";
import { assets, dataset, digest, fetchArchive, readDataset } from "./dataset.mjs";
import { neural } from "./runtime.mjs";
import { passage, protocol } from "../web/protocol.mjs";

const started = performance.now();
await fetchArchive();
const { corpus } = await readDataset();
await mkdir(assets, { recursive: true });
const corpusBytes = Buffer.from(JSON.stringify(corpus));
const docs = corpus.documents;
if (endianness() !== "LE") throw Error("The index writer requires a little-endian host.");
const vectorData = new Float32Array(docs.length * protocol.dimension);
const embeddingStart = performance.now();
for (let row = 0; row < docs.length; row += protocol.embeddingBatch) {
  const texts = docs.slice(row, row + protocol.embeddingBatch).map(passage);
  const batch = await neural.embeddings(texts);
  for (let i = 0; i < texts.length; i++) {
    const vector = batch.subarray(i * protocol.dimension, (i + 1) * protocol.dimension);
    if (!(Math.abs(Math.hypot(...vector) - 1) < 1e-4)) throw Error("Non-unit embedding at row " + (row + i));
  }
  vectorData.set(batch, row * protocol.dimension);
  if (row % 256 === 0) console.log("Embedded", row + texts.length, "/", docs.length);
}
const bytes = Buffer.from(vectorData.buffer);
const manifest = {
  version: protocol.version, rows: docs.length, dimension: protocol.dimension, encoding: "float32-le",
  model: protocol.models.embedder, dtype: protocol.dtype, max_tokens: protocol.embeddingTokens,
  input: "title + newline + abstract", pooling: "attention-masked mean, L2-normalized",
  row_order: "official corpus.jsonl order; hash of UTF-8 IDs joined with LF, no final LF",
  row_ids_sha256: digest(docs.map(doc => doc.id).join("\n")), corpus_sha256: digest(corpusBytes),
  vector_bytes: bytes.length, vector_sha256: digest(bytes), dataset_sha256: dataset.sha256,
  preprocessing_ms: { load_and_embed: performance.now() - embeddingStart, total_to_serialization: performance.now() - started },
};
await writeFile(join(assets, "corpus.json"), corpusBytes);
await writeFile(join(assets, "vectors.f32"), bytes);
await writeFile(join(assets, "index.json"), JSON.stringify(manifest, null, 2) + "\n");
console.log("Prepared SciFact index", manifest.vector_sha256);
