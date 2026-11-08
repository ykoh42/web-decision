/** Text-choice protocol for the pinned Elephant-1 checkpoint. */
// Source and license attribution: ELEPHANT-NOTICE.
const SYSTEM =
  'Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.';
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const EXCLUDED = new Set([
  'BQ',
  'BZ',
  'CJ',
  'CQ',
  'CZ',
  'DQ',
  'DZ',
  'EJ',
  'EY',
  'FJ',
  'FQ',
  'FV',
  'FZ',
  'GJ',
  'GK',
  'GQ',
  'GZ',
  'HJ',
  'IY',
  'JF',
  'JG',
  'JH',
  'JL',
  'JN',
  'JQ'
]);
const SPACE = new Set([0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000]);

export const SLOT_LABELS = Object.freeze([
  ...LETTERS,
  ...[...LETTERS]
    .flatMap((a) => [...LETTERS].map((b) => a + b))
    .filter((s) => !EXCLUDED.has(s))
    .slice(0, 230)
]);

function pythonWhitespace(code) {
  return (
    (code >= 9 && code <= 13) || (code >= 28 && code <= 32) || (code >= 0x2000 && code <= 0x200a) || SPACE.has(code)
  );
}

function pythonStrip(value) {
  let first = 0,
    last = value.length;
  while (first < last && pythonWhitespace(value.charCodeAt(first))) first++;
  while (last > first && pythonWhitespace(value.charCodeAt(last - 1))) last--;
  return value.slice(first, last);
}

function safeText(value, reserved) {
  if (typeof value !== 'string') throw new Error('Text must be a string');
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new Error('Unpaired UTF-16 surrogate');
    } else if (code >= 0xdc00 && code <= 0xdfff) throw new Error('Unpaired UTF-16 surrogate');
  }
  if (reserved.some((token) => value.includes(token))) throw new Error('Reserved token in input');
  if (value.includes('QEV_PREFIX_MARK')) throw new Error('Reserved prefix marker in input');
  return value;
}

function sameIds(a, b) {
  return a.length === b.length && a.every((id, i) => id === b[i]);
}

/** Verify every upstream slot label is one unique, exactly round-tripping token. */
export function verifySlots(tokenizer) {
  const ids = SLOT_LABELS.map((label) => {
    const encoded = tokenizer.encode(label, {add_special_tokens: false}).ids;
    if (encoded.length !== 1 || tokenizer.decode(encoded, {skip_special_tokens: false}) !== label)
      throw new Error(`Invalid answer slot ${label}`);
    return encoded[0];
  });
  if (new Set(ids).size !== ids.length) throw new Error('Answer slots collide');
  return ids;
}

function suffixFor(question, reserved) {
  if (!question || !Array.isArray(question.options) || question.options.length < 2 || question.options.length > 255)
    throw new Error('Each question needs 2 to 255 options');
  const instructions = pythonStrip(safeText(question.instructions, reserved));
  if (!instructions) throw new Error('Question needs instructions');
  const options = question.options.map((option) => safeText(option, reserved));
  if (options.some((option) => !pythonStrip(option))) throw new Error('Empty option');
  if (new Set(options).size !== options.length) throw new Error('Duplicate option');
  const labels = SLOT_LABELS.slice(0, options.length);
  const lines = options.map((option, i) => `${labels[i]}. ${option}`);
  const answer = labels.length <= 26 ? 'letter' : 'label';
  return {
    options,
    suffix: `Question: ${instructions}\n\nOptions:\n${lines.join('\n')}\n\nAnswer with one ${answer}: ${labels.join(', ')}.`
  };
}

/** Reproduce upstream rendering and admit only an independently verifiable token boundary.
 * The original fast tokenizer exposes character offsets; JS does not, so inputs crossing
 * the boundary are rejected rather than inventing a block boundary. */
export function packElephantRequest(tokenizer, request, config) {
  if (!request || !Array.isArray(request.questions) || !request.questions.length)
    throw new Error('Provide at least one question');
  if (!Array.isArray(config.addedTokens) || config.addedTokens.some((x) => typeof x !== 'string' || !x))
    throw new Error('Invalid reserved-token list');
  if (!Number.isSafeInteger(config.maxBranchTokens) || config.maxBranchTokens < 1)
    throw new Error('Invalid token limits');
  const state = safeText(request.state, config.addedTokens);
  const start = `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n<state>\n${state}\n</state>\n\n`;
  const end = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';
  const prefixIds = tokenizer.encode(start, {add_special_tokens: false}).ids;
  const branches = request.questions.map((question) => {
    const {options, suffix} = suffixFor(question, config.addedTokens);
    const prompt = start + suffix + end;
    const ids = tokenizer.encode(prompt, {add_special_tokens: false}).ids;
    const suffixIds = tokenizer.encode(suffix + end, {add_special_tokens: false}).ids;
    if (!sameIds(ids, [...prefixIds, ...suffixIds])) throw new Error('Unsafe token boundary');
    if (ids.length > config.maxBranchTokens) throw new Error('Branch exceeds token limit');
    return {options, suffix, prompt, ids, suffixIds};
  });
  const tokenCount = prefixIds.length + branches.reduce((sum, branch) => sum + branch.suffixIds.length, 0);
  return {prefixIds, branches, tokenCount};
}

/** Full-attention layer mask, with right-padding keys excluded. Linear layers stay causal. */
export function blockMask(prefixLength, length, width) {
  if (
    ![prefixLength, length, width].every(Number.isSafeInteger) ||
    prefixLength < 0 ||
    prefixLength > length ||
    length > width ||
    width < 1
  )
    throw new Error('Invalid block sizes');
  const ids = Array.from({length: width}, (_, i) => (i < prefixLength ? 0 : i < length ? 1 : -1));
  return {
    ids,
    allowed: ids.map((qId, q) => ids.map((kId, k) => k < length && (k <= q || (qId >= 0 && qId === kId))))
  };
}

/** Apply upstream's unit-temperature softmax to exactly the offered options. */
export function unpackElephantScores(rows, packed) {
  if (!Array.isArray(rows) || rows.length !== packed.branches.length) throw new Error('Invalid score rows');
  return {
    tokenCount: packed.tokenCount,
    answers: packed.branches.map((branch, i) => {
      const logits = Array.from(rows[i] ?? []);
      if (logits.length !== branch.options.length || logits.some((value) => !Number.isFinite(value)))
        throw new Error('Invalid logits');
      const max = Math.max(...logits),
        weights = logits.map((value) => Math.exp(value - max));
      const total = weights.reduce((a, b) => a + b, 0),
        probabilities = weights.map((value) => value / total);
      const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
      return {
        selected: branch.options[selectedIndex],
        selectedIndex,
        probabilities: branch.options.map((option, j) => ({option, probability: probabilities[j], logit: logits[j]}))
      };
    })
  };
}
