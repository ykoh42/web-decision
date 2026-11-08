/** Internal validation for the experimental dynamic DeBERTa probe, not a supported public API. */
const MAX_SEQUENCE_LENGTH = 512,
  VOCAB_SIZE = 128100,
  PAIRS_PER_CHUNK = 3;

export interface DebertaChunk {
  readonly start: number;
  readonly count: number;
  readonly pair_q: readonly number[];
  readonly pair_opt: readonly number[];
}

export interface ValidatedDebertaTensors {
  readonly groups: readonly number[];
  readonly realTokens: number;
  readonly pairs: number;
  readonly inputs: Readonly<{
    input_ids: readonly number[];
    attention_mask: readonly number[];
    seg: readonly number[];
    pair_q: readonly number[];
    pair_opt: readonly number[];
  }>;
  readonly chunks: readonly DebertaChunk[];
}

const invalid = (message: string): never => {
  throw new RangeError(message);
};

function integer(value: unknown, min: number, max: number, name: string): number {
  const number = value as number;
  if (!Number.isSafeInteger(number) || number < min || number > max)
    invalid(`${name} must be an integer in [${min}, ${max}]`);
  return number;
}

function sequence(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value) && !(ArrayBuffer.isView(value) && !(value instanceof DataView)))
    invalid(`${name} must be a numeric array`);
  if (Array.isArray(value)) return Array.from(value as unknown[]);
  return Array.from(value as ArrayLike<unknown>);
}

function object(value: unknown, message: string): Record<string, unknown> {
  if (!value || typeof value !== 'object') invalid(message);
  return value as Record<string, unknown>;
}

function validateGroups(request: unknown): {
  groups: number[];
  total: number;
  inputs: Record<string, unknown>;
} {
  const value = object(request, 'request must be an object');
  const rawGroups = sequence(value.groups, 'groups');
  if (rawGroups.length < 1) invalid('groups count is invalid');
  const groups = rawGroups.map((item) => integer(item, 1, MAX_SEQUENCE_LENGTH, 'group size'));
  const total = groups.reduce((sum, size) => sum + size, 0);
  const inputs = object(value.inputs, 'inputs missing');
  return {groups, total, inputs};
}

function validateInputArrays(inputs: Record<string, unknown>): {
  ids: unknown[];
  mask: unknown[];
  seg: unknown[];
} {
  const ids = sequence(inputs.input_ids, 'input_ids');
  const mask = sequence(inputs.attention_mask, 'attention_mask');
  const seg = sequence(inputs.seg, 'seg');
  const n = ids.length;
  if (n < 1 || n > MAX_SEQUENCE_LENGTH || mask.length !== n || seg.length !== n)
    invalid('inconsistent sequence length');
  return {ids, mask, seg};
}

function validateTokenIds(ids: readonly unknown[]): number[] {
  return ids.map((value) => integer(value, 0, VOCAB_SIZE - 1, 'token id'));
}

function validateAttentionMask(mask: readonly unknown[]): number[] {
  for (const value of mask) if (value !== 1) invalid('only unpadded attention masks are supported');
  return mask as number[];
}

function validateSegments(seg: readonly unknown[], slots: number, n: number): number[] {
  if (!Number.isSafeInteger(slots) || slots > n) invalid('question and option segments cannot exceed sequence length');
  const segments = seg.map((value) => integer(value, -1, slots - 1, 'segment'));
  const counts = Array.from({length: slots}, () => 0);
  for (const value of segments) if (value >= 0) counts[value] = counts[value]! + 1;
  if (counts.some((count) => count === 0)) invalid('question or option segment has no tokens');
  return segments;
}

function validatePairArrays(inputs: Record<string, unknown>, total: number): {q: number[]; o: number[]} {
  const q = sequence(inputs.pair_q, 'pair_q');
  const o = sequence(inputs.pair_opt, 'pair_opt');
  if (q.length !== total || o.length !== total) invalid('pair arrays do not match groups');
  return {q: q as number[], o: o as number[]};
}

function validatePairMapping(
  groups: readonly number[],
  q: readonly number[],
  o: readonly number[],
  total: number
): void {
  let index = 0;
  for (let group = 0; group < groups.length; group++) {
    for (let choice = 0; choice < groups[group]!; choice++) {
      if (q[index] !== total + group || o[index] !== index) invalid('pair slot mapping is invalid');
      index++;
    }
  }
}

function createChunks(q: readonly number[], o: readonly number[], total: number): DebertaChunk[] {
  const chunks: DebertaChunk[] = [];
  for (let start = 0; start < total; start += PAIRS_PER_CHUNK) {
    const count = Math.min(PAIRS_PER_CHUNK, total - start);
    const pairQuestions = q.slice(start, start + count);
    const pairOptions = o.slice(start, start + count);
    while (pairQuestions.length < PAIRS_PER_CHUNK) {
      pairQuestions.push(pairQuestions.at(-1)!);
      pairOptions.push(pairOptions.at(-1)!);
    }
    chunks.push(
      Object.freeze({
        start,
        count,
        pair_q: Object.freeze(pairQuestions),
        pair_opt: Object.freeze(pairOptions)
      })
    );
  }
  return chunks;
}

/** Return an immutable snapshot after validation; inputs are unpadded and one-dimensional. */
export function validateDebertaTensors(request: unknown): ValidatedDebertaTensors {
  const {groups, total, inputs} = validateGroups(request);
  const {ids, mask, seg} = validateInputArrays(inputs);
  const inputIds = validateTokenIds(ids);
  const attentionMask = validateAttentionMask(mask);
  const slots = total + groups.length;
  const segments = validateSegments(seg, slots, inputIds.length);
  const {q, o} = validatePairArrays(inputs, total);
  validatePairMapping(groups, q, o, total);
  const chunks = createChunks(q, o, total);
  return Object.freeze({
    groups: Object.freeze(groups),
    realTokens: inputIds.length,
    pairs: total,
    inputs: Object.freeze({
      input_ids: Object.freeze(inputIds),
      attention_mask: Object.freeze(attentionMask),
      seg: Object.freeze(segments),
      pair_q: Object.freeze(q),
      pair_opt: Object.freeze(o)
    }),
    chunks: Object.freeze(chunks)
  });
}
