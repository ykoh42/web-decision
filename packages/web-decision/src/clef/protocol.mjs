// Text-choice subset of Cloudflare/clef-flash joint_schema_model.py (Apache-2.0),
// pinned revision fb4417d31c1d0f9e7e77f1141b40aaede7491e00.
export const CLEF_MAX_OPTIONS = 64;

const SYSTEM =
  "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const VOCAB = 248320,
  SOURCE_LIMIT = 512;
const fail = (message) => {
  throw new Error(message);
};

function wellFormed(value) {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
    } else if (code >= 0xdc00 && code <= 0xdfff) return false;
  }
  return true;
}
function encode(tokenizer, text) {
  const result = tokenizer.encode(text, {add_special_tokens: false});
  if (!result?.ids || typeof result.ids[Symbol.iterator] !== 'function') return fail('Invalid tokenizer output');
  const ids = Array.from(result.ids);
  if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= VOCAB)) return fail('Invalid tokenizer ID');
  return ids;
}

/** Encode all choice questions together. Numeric IDs are adapter-assigned and source-sorted. */
export function packChoiceRequest(tokenizer, request, controlIds, {maxInputTokens = SOURCE_LIMIT} = {}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    !request ||
    typeof request !== 'object' ||
    Array.isArray(request) ||
    typeof request.state !== 'string' ||
    !Array.isArray(request.questions) ||
    !request.questions.length ||
    request.questions.length > 8 ||
    request.questions.reduce((sum, q) => sum + (Array.isArray(q?.options) ? q.options.length : 0), 0) > 64 ||
    request.images !== undefined ||
    request.videos !== undefined ||
    request.media_kwargs !== undefined ||
    request.model !== undefined ||
    !Array.isArray(controlIds) ||
    !controlIds.length ||
    new Set(controlIds).size !== controlIds.length ||
    controlIds.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= VOCAB) ||
    !Number.isSafeInteger(maxInputTokens) ||
    maxInputTokens < 1 ||
    maxInputTokens > SOURCE_LIMIT
  )
    return fail('Invalid choice request or limits');
  const controls = new Set(controlIds);
  const text = (value) => {
    if (typeof value !== 'string') return fail('Invalid user text');
    if (!wellFormed(value)) return fail('Invalid user text');
    const ids = encode(tokenizer, value);
    if (ids.some((id) => controls.has(id))) return fail('User text contains an added token');
    return ids;
  };
  const state = text(request.state),
    questions = [];
  const schema = encode(tokenizer, '\n\nSCHEMA FIELDS:\n');
  const append = (target, more) => {
    if (target.length + more.length > maxInputTokens) return fail('Input exceeds token limit');
    for (const id of more) target.push(id);
  };
  for (const [i, question] of request.questions.entries()) {
    if (
      !question ||
      typeof question !== 'object' ||
      Array.isArray(question) ||
      (question.type !== undefined && question.type !== 'choice') ||
      question.criteria !== undefined ||
      question.id !== undefined ||
      typeof question.instructions !== 'string' ||
      !question.instructions.trim() ||
      !Array.isArray(question.options) ||
      question.options.length < 2 ||
      question.options.length > CLEF_MAX_OPTIONS ||
      question.options.some((option) => typeof option !== 'string' || !option.trim()) ||
      new Set(question.options).size !== question.options.length
    )
      return fail('Unsupported or invalid choice question');
    const questionId = `q${i}`,
      instructions = question.instructions,
      original = [...question.options];
    text(instructions);
    for (const option of original) text(option);
    append(schema, encode(tokenizer, `\nFIELD ${i + 1}\nID: ${questionId}\nTYPE: choice\nINSTRUCTION: `));
    const start = schema.length;
    append(schema, encode(tokenizer, instructions));
    const end = schema.length;
    append(schema, encode(tokenizer, '\nALLOWED OPTIONS:\n'));
    const sorted = original
      .map((description, index) => ({id: String(index), description}))
      .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    const optionSpans = [];
    for (const [j, option] of sorted.entries()) {
      append(schema, encode(tokenizer, `OPTION ${j + 1}: `));
      const begin = schema.length;
      append(schema, encode(tokenizer, JSON.stringify({description: option.description, option_id: option.id})));
      optionSpans.push([begin, schema.length]);
      append(schema, encode(tokenizer, '\n'));
    }
    append(schema, encode(tokenizer, 'END FIELD\n'));
    questions.push({
      question_id: questionId,
      question_type: 1,
      question_span: [start, end],
      option_spans: optionSpans,
      option_ids: sorted.map((option) => option.id),
      options: original
    });
  }
  const prefix = encode(tokenizer, `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\nSTATE:\n`);
  const suffix = encode(
    tokenizer,
    '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:'
  );
  const offset = prefix.length + state.length;
  if (offset + schema.length + suffix.length > maxInputTokens) return fail('Input exceeds token limit');
  for (const q of questions) {
    q.question_span = q.question_span.map((value) => value + offset);
    q.option_spans = q.option_spans.map((span) => span.map((value) => value + offset));
  }
  const packed = {ids: [...prefix, ...state, ...schema, ...suffix], questions, maxInputTokens};
  validatePacked(packed);
  return structuredClone(packed);
}

export function validatePacked(packed) {
  const n = packed?.ids?.length,
    questions = packed?.questions;
  if (
    !Number.isSafeInteger(packed?.maxInputTokens) ||
    packed.maxInputTokens < 1 ||
    packed.maxInputTokens > SOURCE_LIMIT ||
    !Array.isArray(packed.ids) ||
    n < 4 ||
    n > packed.maxInputTokens ||
    packed.ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= VOCAB) ||
    !Array.isArray(questions) ||
    !questions.length ||
    questions.length > 8
  )
    return fail('Invalid packed tokens');
  const spanOk = (span) =>
    Array.isArray(span) &&
    span.length === 2 &&
    span.every(Number.isSafeInteger) &&
    0 <= span[0] &&
    span[0] < span[1] &&
    span[1] <= n;
  let last = 0;
  for (const [i, q] of questions.entries()) {
    const count = q?.options?.length;
    if (
      q?.question_id !== `q${i}` ||
      q.question_type !== 1 ||
      !Array.isArray(q.options) ||
      count < 2 ||
      count > 64 ||
      q.options.some((option) => typeof option !== 'string' || !option.trim()) ||
      new Set(q.options).size !== count ||
      !spanOk(q.question_span) ||
      q.question_span[0] < last ||
      !Array.isArray(q.option_ids) ||
      !same(q.option_ids, Array.from({length: count}, (_, j) => String(j)).sort()) ||
      !Array.isArray(q.option_spans) ||
      q.option_spans.length !== count
    )
      return fail('Invalid packed question');
    let previous = q.question_span[1];
    for (const span of q.option_spans) {
      if (!spanOk(span) || span[0] < previous) return fail('Invalid option span');
      previous = span[1];
    }
    last = previous;
  }
  if (questions.reduce((sum, q) => sum + q.options.length, 0) > 64) return fail('Too many total options');
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Exact decimal-place rounding of a nonnegative FP32 value, ties to even. */
function roundProbability(value) {
  if (!Number.isFinite(value) || value < 0 || value > 1 || Math.fround(value) !== value)
    return fail('Expected an FP32 probability');
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value, false);
  const bits = view.getUint32(0, false),
    exponent = (bits >>> 23) & 255,
    fraction = bits & 0x7fffff;
  const numerator = BigInt(exponent === 0 ? fraction : fraction + 0x800000) * 10000n;
  const power = exponent === 0 ? -149 : exponent - 150;
  let rounded;
  if (power >= 0) rounded = numerator << BigInt(power);
  else {
    const denominator = 1n << BigInt(-power),
      whole = numerator / denominator,
      remainder = numerator % denominator;
    rounded = whole + (2n * remainder > denominator || (2n * remainder === denominator && whole % 2n === 1n) ? 1n : 0n);
  }
  return Number(rounded) / 10000;
}

/** Consume actual FP32 backend probabilities; no local softmax approximation is used. */
export function unpackChoiceScores(packed, rows, probabilityRows) {
  validatePacked(packed);
  if (
    !Array.isArray(rows) ||
    !Array.isArray(probabilityRows) ||
    rows.length !== packed.questions.length ||
    probabilityRows.length !== rows.length
  )
    return fail('Missing joint FP32 score or probability rows');
  const answers = packed.questions.map((q, i) => {
    const logits = rows[i],
      probs = probabilityRows[i];
    if (
      !(logits instanceof Float32Array) ||
      !(probs instanceof Float32Array) ||
      logits.length !== q.options.length ||
      probs.length !== logits.length ||
      logits.some((value) => !Number.isFinite(value)) ||
      probs.some((value) => !Number.isFinite(value) || value < 0 || value > 1)
    )
      return fail('Invalid joint FP32 result');
    const sum = probs.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 1e-5) return fail('Invalid probability sum');
    // Sanity check only: the double values are never used for source choice.
    const max = Math.max(...logits),
      weights = Array.from(logits, (value) => Math.exp(value - max));
    const weightSum = weights.reduce((a, b) => a + b, 0);
    if (probs.some((value, j) => Math.abs(value - weights[j] / weightSum) > 1e-5))
      return fail('Probabilities do not match logits');
    const byId = new Map(q.option_ids.map((id, j) => [id, j]));
    const output = q.options.map((option, j) => {
      const index = byId.get(String(j));
      if (index === undefined) return fail('Missing option ID');
      return {option, logit: logits[index], probability: probs[index]};
    });
    const selectedIndex = output.reduce((best, item, j) => (item.probability > output[best].probability ? j : best), 0);
    const rounded = Object.fromEntries(output.map((item, j) => [String(j), roundProbability(item.probability)]));
    return {
      selected: q.options[selectedIndex],
      selectedIndex,
      probabilities: output,
      sourceAnswer: {
        type: 'choice',
        choice: String(selectedIndex),
        confidence: roundProbability(output[selectedIndex].probability),
        probabilities: rounded
      }
    };
  });
  return {tokenCount: packed.ids.length, answers};
}
