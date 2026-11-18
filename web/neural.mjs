import { models, passage, protocol } from "./protocol.mjs";

export function createModels(runtime, options, progress = () => {}) {
  let embedder, pairTokenizer, crossEncoder, generator;
  const config = model => ({ ...options, dtype: protocol.dtype, revision: model.revision });
  return {
    async embeddings(texts) {
      if (!embedder) {
        progress("Loading the MiniLM embedding model…");
        embedder = await runtime.pipeline("feature-extraction", models.embedder.id, config(models.embedder));
        embedder.tokenizer.model_max_length = protocol.embeddingTokens;
      }
      const output = await embedder(texts, { pooling: "mean", normalize: true });
      if (output.dims[0] !== texts.length || output.dims[1] !== protocol.dimension) {
        throw Error("Unexpected MiniLM embedding dimensions.");
      }
      return output.data;
    },
    async rerank(query, docs) {
      if (!crossEncoder) {
        progress("Loading the MiniLM cross-encoder…");
        pairTokenizer = await runtime.AutoTokenizer.from_pretrained(models.reranker.id, config(models.reranker));
        crossEncoder = await runtime.AutoModelForSequenceClassification.from_pretrained(models.reranker.id, config(models.reranker));
      }
      const scores = [];
      for (let start = 0; start < docs.length; start += protocol.rerankBatch) {
        const batch = docs.slice(start, start + protocol.rerankBatch);
        progress("Rescoring abstracts " + (start + 1) + "–" + (start + batch.length) + " of " + docs.length + "…");
        const inputs = pairTokenizer(batch.map(() => query), {
          text_pair: batch.map(passage), padding: true, truncation: true, max_length: protocol.pairTokens,
        });
        scores.push(...(await crossEncoder(inputs)).logits.data);
      }
      return scores;
    },
    async generation() {
      generator ??= await runtime.pipeline("text2text-generation", models.generator.id, config(models.generator));
      return generator;
    },
  };
}
