export function cite(raw, hits) {
  const text = raw.trim();
  const phrase = text.replace(/[.!?]+$/u, "");
  if (phrase.split(/\s+/u).length >= 2 && !/\[C\d+\]/iu.test(text)) {
    const escaped = phrase.replace(/[.*+?^{}$()|[\]\\]/g, "\\$&");
    const pattern = new RegExp("(?<![\\p{L}\\p{N}_])" + escaped + "(?![\\p{L}\\p{N}_])", "iu");
    const matches = hits.filter(hit => pattern.test(hit.quote));
    if (matches.length === 1) {
      return text + " [" + matches[0].citation_id + "]";
    }
  }
  throw Error("The generated words could not be matched to a single supplied source passage.");
}

export function validateAnswer(text, hits) {
  if (/^(unknown|unanswerable|none)[.!]?$/i.test(text.trim())) {
    return { generation_error: "The model did not find an answer in these sources." };
  }
  try {
    return { answer: cite(text, hits) };
  } catch (error) {
    return { generation_error: error.message };
  }
}

// Preserve exact substrings, including whitespace, so the UI can display exactly
// what was in the model's prompt. Give each remaining source a share of the budget.
export function packContext(query, hits, encode, limit = 512) {
  const promptFor = excerpts => "Answer the question using the context. If it cannot be answered, say unknown.\nContext: "
    + excerpts.map(item => item.quote).join("\n") + "\nQuestion: " + query;
  const tokens = excerpts => encode(promptFor(excerpts)).length;
  const selected = hits.slice(0, 4);
  const excerpts = [];
  for (const [i, hit] of selected.entries()) {
    const used = tokens(excerpts);
    const target = used + Math.floor((limit - used) / (selected.length - i));
    const ends = [...hit.quote.matchAll(/\S+/gu)].map(match => match.index + match[0].length);
    let low = 0, high = ends.length;
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      const candidate = { citation_id: hit.citation_id, quote: hit.quote.slice(0, ends[mid - 1]) };
      if (tokens([...excerpts, candidate]) <= target) low = mid;
      else high = mid - 1;
    }
    if (low > 0) excerpts.push({ citation_id: hit.citation_id, quote: hit.quote.slice(0, ends[low - 1]) });
  }
  if (!excerpts.length || tokens(excerpts) > limit) throw Error("No source excerpt fits alongside this query.");
  return { prompt: promptFor(excerpts), excerpts };
}
