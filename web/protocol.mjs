// Fixed before evaluation. Row order is the order of the official corpus JSONL.
export const models = {
  embedder: { id: "Xenova/all-MiniLM-L6-v2", revision: "751bff37182d3f1213fa05d7196b954e230abad9" },
  reranker: { id: "Xenova/ms-marco-MiniLM-L-6-v2", revision: "a09144355adeed5f58c8ed011d209bf8ee5a1fec" },
  generator: { id: "Xenova/flan-t5-base", revision: "c6a1b98a4a35e1063d88edaceef4dc52cf3b2784" },
};
export const protocol = {
  version: "scifact-v1", modes: ["bm25", "dense", "hybrid", "hybrid_rerank"],
  bm25: { k1: 1.2, b: 0.75 }, candidateDepth: 100, rrfK: 60,
  rerankDepth: 50, displayDepth: 10, dtype: "q8", dimension: 384,
  embeddingTokens: 256, pairTokens: 256, embeddingBatch: 8, rerankBatch: 8,
  bootstrapSamples: 2000, seed: 20261118, models,
};
export const passage = doc => doc.title + "\n" + doc.content;
