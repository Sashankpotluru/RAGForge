import { models, passage, protocol } from "./protocol.mjs";

// The pinned BERT model expects [CLS] query [SEP] passage [SEP]. Preserve both
// separators and trim the longer segment first; an equal-length tie trims the passage.
export function packPair(queryIds, passageIds, cls, sep, limit = protocol.pairTokens) {
  if (!queryIds.length || !passageIds.length || limit < 5) throw Error("Both reranker segments need content tokens.");
  const budget = limit - 3;
  const queryLength = Math.min(queryIds.length, Math.max(Math.ceil(budget / 2), budget - passageIds.length));
  const passageLength = Math.min(passageIds.length, budget - queryLength);
  return { input_ids: [cls, ...queryIds.slice(0, queryLength), sep, ...passageIds.slice(0, passageLength), sep],
    token_type_ids: [...Array(queryLength + 2).fill(0), ...Array(passageLength + 1).fill(1)] };
}

export function pairInputs(runtime, pairs, padId) {
  const width = Math.max(...pairs.map(pair => pair.input_ids.length));
  const dims = [pairs.length, width];
  const ids = new BigInt64Array(pairs.length * width).fill(BigInt(padId));
  const segments = new BigInt64Array(ids.length);
  const mask = new BigInt64Array(ids.length);
  pairs.forEach((pair, i) => {
    const offset = i * width;
    ids.set(pair.input_ids.map(BigInt), offset);
    segments.set(pair.token_type_ids.map(BigInt), offset);
    mask.fill(1n, offset, offset + pair.input_ids.length);
  });
  const tensor = values => new runtime.Tensor("int64", values, dims);
  return { input_ids: tensor(ids), attention_mask: tensor(mask), token_type_ids: tensor(segments) };
}

export function checkPairTokenizer(tokenizer) {
  const encode = text => tokenizer.encode(text, { add_special_tokens: false });
  const probe = tokenizer("hello", { text_pair: "world", return_tensor: false });
  const expected = packPair(encode("hello"), encode("world"), 101, 102);
  if (tokenizer.pad_token_id !== 0 || tokenizer.sep_token_id !== 102 || tokenizer.padding_side !== "right" ||
      probe.input_ids.join() !== expected.input_ids.join() || probe.token_type_ids?.join() !== expected.token_type_ids.join() ||
      !probe.attention_mask?.every(mask => mask === 1)) throw Error("Unsupported reranker tokenizer: expected the pinned BERT pair layout.");
}

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
        checkPairTokenizer(pairTokenizer);
        crossEncoder = await runtime.AutoModelForSequenceClassification.from_pretrained(models.reranker.id, config(models.reranker));
      }
      const encode = text => pairTokenizer.encode(text, { add_special_tokens: false });
      const queryIds = encode(query);
      const scores = [];
      for (let start = 0; start < docs.length; start += protocol.rerankBatch) {
        const batch = docs.slice(start, start + protocol.rerankBatch);
        progress("Rescoring abstracts " + (start + 1) + "–" + (start + batch.length) + " of " + docs.length + "…");
        const pairs = batch.map(doc => packPair(queryIds, encode(passage(doc)), 101, 102));
        const inputs = pairInputs(runtime, pairs, pairTokenizer.pad_token_id);
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
