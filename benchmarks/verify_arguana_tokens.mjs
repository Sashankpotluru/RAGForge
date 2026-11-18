// Inspect the original and corrected pair encodings. No model inference or rankings.
import { AutoTokenizer, env, Tensor } from "@huggingface/transformers";
import { join } from "node:path";
import { datasets, modelCache, readDataset } from "./dataset.mjs";
import { checkPairTokenizer, packPair, pairInputs } from "../web/neural.mjs";
import { protocol } from "../web/protocol.mjs";
import assert from "node:assert/strict";

const { reranker } = protocol.models;
env.allowRemoteModels = false;
const tokenizer = await AutoTokenizer.from_pretrained(join(modelCache, reranker.id, reranker.revision));
checkPairTokenizer(tokenizer);
const { queries } = await readDataset(datasets.arguana);
const encode = text => tokenizer.encode(text, { add_special_tokens: false });
const candidates = ["Different candidate passage about books.", "Astronomy and nebulae."];
const documentIds = candidates.map(encode);
const options = { padding: false, truncation: true, max_length: protocol.pairTokens, return_tensor: false };
const shortText = "Short query.";
const shortNative = tokenizer(candidates.map(() => shortText),
  { ...options, text_pair: candidates, padding: true, return_tensor: true });
const shortCorrected = pairInputs({ Tensor }, documentIds.map(ids => packPair(encode(shortText), ids, 101, 102)), 0);
for (const key of ["input_ids", "attention_mask", "token_type_ids"]) {
  assert.deepEqual(shortNative[key].dims, shortCorrected[key].dims);
  assert.deepEqual(shortNative[key].data, shortCorrected[key].data);
}
let maxQueryTokens = 0, originalNoCandidate = 0, correctedNoCandidate = 0;
let example;
for (const [id, query] of queries) {
  const queryIds = encode(query.text);
  const queryLength = queryIds.length + 2;
  maxQueryTokens = Math.max(maxQueryTokens, queryLength);
  const corrected = documentIds.map(ids => packPair(queryIds, ids, 101, 102));
  if (corrected.some(pair => pair.token_type_ids.filter(x => x === 1).length <= 1)) correctedNoCandidate++;
  assert.notDeepEqual(corrected[0].input_ids, corrected[1].input_ids);
  const pair = tokenizer(query.text, { ...options, text_pair: candidates[0] });
  if (pair.token_type_ids.includes(1)) continue;
  const other = tokenizer(query.text, { ...options, text_pair: candidates[1] });
  assert.deepEqual(pair.input_ids, other.input_ids);
  originalNoCandidate++;
  example ??= { query_id: id, query_tokens_including_specials: queryLength, original_pair_positions: pair.input_ids.length };
}
assert.equal(correctedNoCandidate, 0);
console.log(JSON.stringify({ model: reranker, pair_limit: protocol.pairTokens, queries: queries.size,
  max_query_tokens_including_specials: maxQueryTokens, original_queries_with_zero_candidate_positions: originalNoCandidate,
  corrected_queries_with_zero_candidate_content_tokens: correctedNoCandidate,
  method: "Compare two fixed candidate texts with distinct prefixes for every official query. Also compare short batched inputs, masks and type IDs against the native tokenizer.",
  example }, null, 2));
