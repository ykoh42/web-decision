/** Internal validation for the experimental dynamic DeBERTa probe, not a supported public API. */
const MAX_SEQUENCE_LENGTH = 512,
  VOCAB_SIZE = 128100,
  PAIRS_PER_CHUNK = 3;
const invalid = (message) => {
  throw new RangeError(message);
};
const integer = (v, min, max, name) => {
  if (!Number.isSafeInteger(v) || v < min || v > max) invalid(`${name} must be an integer in [${min}, ${max}]`);
  return v;
};
function sequence(value, name) {
  if (!Array.isArray(value) && !(ArrayBuffer.isView(value) && !(value instanceof DataView)))
    invalid(`${name} must be a numeric array`);
  return Array.from(value);
}

/** Return an immutable snapshot after validation; inputs are unpadded and one-dimensional. */
export function validateDebertaTensors(request) {
  if (!request || typeof request !== 'object') invalid('request must be an object');
  const groups = sequence(request.groups, 'groups');
  if (groups.length < 1) invalid('groups count is invalid');
  for (const value of groups) integer(value, 1, MAX_SEQUENCE_LENGTH, 'group size');
  const total = groups.reduce((a, b) => a + b, 0);
  const inputs = request.inputs;
  if (!inputs || typeof inputs !== 'object') invalid('inputs missing');
  const ids = sequence(inputs.input_ids, 'input_ids'),
    mask = sequence(inputs.attention_mask, 'attention_mask'),
    seg = sequence(inputs.seg, 'seg');
  const n = ids.length;
  if (n < 1 || n > MAX_SEQUENCE_LENGTH || mask.length !== n || seg.length !== n)
    invalid('inconsistent sequence length');
  for (const value of ids) integer(value, 0, VOCAB_SIZE - 1, 'token id');
  for (const value of mask) if (value !== 1) invalid('only unpadded attention masks are supported');
  const slots = total + groups.length;
  if (!Number.isSafeInteger(slots) || slots > n) invalid('question and option segments cannot exceed sequence length');
  const counts = new Array(slots).fill(0);
  for (const value of seg) {
    integer(value, -1, slots - 1, 'segment');
    if (value >= 0) counts[value]++;
  }
  if (counts.some((count) => count === 0)) invalid('question or option segment has no tokens');
  const q = sequence(inputs.pair_q, 'pair_q'),
    o = sequence(inputs.pair_opt, 'pair_opt');
  if (q.length !== total || o.length !== total) invalid('pair arrays do not match groups');
  let index = 0;
  for (let group = 0; group < groups.length; group++)
    for (let choice = 0; choice < groups[group]; choice++) {
      if (q[index] !== total + group || o[index] !== index) invalid('pair slot mapping is invalid');
      index++;
    }
  const chunks = [];
  for (let start = 0; start < total; start += PAIRS_PER_CHUNK) {
    const count = Math.min(PAIRS_PER_CHUNK, total - start),
      cq = q.slice(start, start + count),
      co = o.slice(start, start + count);
    while (cq.length < PAIRS_PER_CHUNK) {
      cq.push(cq.at(-1));
      co.push(co.at(-1));
    }
    chunks.push(Object.freeze({start, count, pair_q: Object.freeze(cq), pair_opt: Object.freeze(co)}));
  }
  return Object.freeze({
    groups: Object.freeze(groups),
    realTokens: n,
    pairs: total,
    inputs: Object.freeze({
      input_ids: Object.freeze(ids),
      attention_mask: Object.freeze(mask),
      seg: Object.freeze(seg),
      pair_q: Object.freeze(q),
      pair_opt: Object.freeze(o)
    }),
    chunks: Object.freeze(chunks)
  });
}
