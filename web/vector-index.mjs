import { protocol } from "./protocol.mjs";

export async function sha256(bytes) {
  const value = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(value), n => n.toString(16).padStart(2, "0")).join("");
}
export async function validateCorpus(corpusBytes, manifest) {
  if (await sha256(corpusBytes) !== manifest.corpus_sha256) throw Error("Corpus checksum mismatch. Reload the workspace.");
  const corpus = JSON.parse(new TextDecoder().decode(corpusBytes));
  const ids = corpus.documents.map(doc => doc.id);
  if (manifest.version !== protocol.version || manifest.rows !== ids.length || new Set(ids).size !== ids.length ||
      await sha256(new TextEncoder().encode(ids.join("\n"))) !== manifest.row_ids_sha256) {
    throw Error("Document ID order differs from the vector index.");
  }
  return corpus;
}
export async function validateVectors(bytes, manifest) {
  if (manifest.dimension !== protocol.dimension || manifest.dtype !== protocol.dtype ||
      manifest.model?.id !== protocol.models.embedder.id || manifest.model?.revision !== protocol.models.embedder.revision ||
      manifest.max_tokens !== protocol.embeddingTokens || manifest.encoding !== "float32-le" ||
      new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1 ||
      bytes.byteLength !== manifest.rows * protocol.dimension * 4 || bytes.byteLength !== manifest.vector_bytes ||
      await sha256(bytes) !== manifest.vector_sha256) {
    throw Error("Vector index is incomplete or incompatible. Reload and retry.");
  }
  return new Float32Array(bytes);
}
