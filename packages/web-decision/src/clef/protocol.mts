// Text-choice subset of Cloudflare/clef-flash joint_schema_model.py (Apache-2.0),
// pinned revision fb4417d31c1d0f9e7e77f1141b40aaede7491e00.
export const CLEF_MAX_OPTIONS = 64;

const SYSTEM =
  "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const VOCAB = 248320,
  SOURCE_LIMIT = 512;
const fail = (message: string): never => {
  throw new Error(message);
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function wellFormed(value: string): boolean {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
    } else if (code >= 0xdc00 && code <= 0xdfff) return false;
  }
  return true;
}
function encode(tokenizer: ClefTokenizer, text: string): number[] {
  const result = tokenizer.encode(text, {add_special_tokens: false});
  if (!result?.ids || typeof result.ids[Symbol.iterator] !== 'function') return fail('Invalid tokenizer output');
  const ids = Array.from(result.ids);
  if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= VOCAB)) return fail('Invalid tokenizer ID');
  return ids;
}

/** Encode all choice questions together. Numeric IDs are adapter-assigned and source-sorted. */
export function packChoiceRequest(
  tokenizer: ClefTokenizer,
  request: unknown,
  controlIds: unknown,
  {maxInputTokens = SOURCE_LIMIT}: {maxInputTokens?: number} = {}
): PackedClefChoice {
  const requestRecord = isRecord(request) ? request : null;
  const requestQuestions = requestRecord?.questions;
  if (
    typeof tokenizer?.encode !== 'function' ||
    !requestRecord ||
    typeof requestRecord.state !== 'string' ||
    !Array.isArray(requestQuestions) ||
    !requestQuestions.length ||
    requestQuestions.length > 8 ||
    requestQuestions.reduce(
      (sum: number, q: unknown) => sum + (isRecord(q) && Array.isArray(q.options) ? q.options.length : 0),
      0
    ) > 64 ||
    requestRecord.images !== undefined ||
    requestRecord.videos !== undefined ||
    requestRecord.media_kwargs !== undefined ||
    requestRecord.model !== undefined ||
    !Array.isArray(controlIds) ||
    !controlIds.length ||
    new Set(controlIds).size !== controlIds.length ||
    controlIds.some((id: unknown) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= VOCAB) ||
    !Number.isSafeInteger(maxInputTokens) ||
    maxInputTokens < 1 ||
    maxInputTokens > SOURCE_LIMIT
  )
    return fail('Invalid choice request or limits');
  const validControlIds = controlIds as number[];
  const controls = new Set(validControlIds);
  const text = (value: unknown): number[] => {
    if (typeof value !== 'string') return fail('Invalid user text');
    if (!wellFormed(value)) return fail('Invalid user text');
    const ids = encode(tokenizer, value);
    if (ids.some((id) => controls.has(id))) return fail('User text contains an added token');
    return ids;
  };
  const state = text(requestRecord.state),
    questions: PackedClefChoice['questions'] = [];
  const schema = encode(tokenizer, '\n\nSCHEMA FIELDS:\n');
  const append = (target: number[], more: number[]): void => {
    if (target.length + more.length > maxInputTokens) return fail('Input exceeds token limit');
    for (const id of more) target.push(id);
  };
  for (const [i, questionValue] of requestQuestions.entries()) {
    const question = isRecord(questionValue) ? questionValue : null;
    const options = question?.options;
    if (
      !question ||
      (question.type !== undefined && question.type !== 'choice') ||
      question.criteria !== undefined ||
      question.id !== undefined ||
      typeof question.instructions !== 'string' ||
      !question.instructions.trim() ||
      !Array.isArray(options) ||
      options.length < 2 ||
      options.length > CLEF_MAX_OPTIONS ||
      options.some((option: unknown) => typeof option !== 'string' || !option.trim()) ||
      new Set(options).size !== options.length
    )
      return fail('Unsupported or invalid choice question');
    const questionId = `q${i}`,
      instructions = question.instructions as string,
      original = [...(options as string[])];
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
    const optionSpans: Array<[number, number]> = [];
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
    q.question_span = [q.question_span[0] + offset, q.question_span[1] + offset];
    q.option_spans = q.option_spans.map(([start, end]) => [start + offset, end + offset]);
  }
  const packed = {ids: [...prefix, ...state, ...schema, ...suffix], questions, maxInputTokens};
  validatePacked(packed);
  return structuredClone(packed);
}

export function validatePacked(packed: unknown): asserts packed is PackedClefChoice {
  const record = isRecord(packed) ? packed : null;
  const ids = record?.ids,
    questions = record?.questions,
    maxInputTokens = record?.maxInputTokens;
  if (
    !record ||
    !Number.isSafeInteger(maxInputTokens) ||
    (maxInputTokens as number) < 1 ||
    (maxInputTokens as number) > SOURCE_LIMIT ||
    !Array.isArray(ids) ||
    ids.length < 4 ||
    ids.length > (maxInputTokens as number) ||
    ids.some((id: unknown) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= VOCAB) ||
    !Array.isArray(questions) ||
    !questions.length ||
    questions.length > 8
  )
    return fail('Invalid packed tokens');
  const tokenIds = ids as number[];
  const typedQuestions = questions as unknown[];
  const n = tokenIds.length;
  const spanOk = (span: unknown): span is [number, number] =>
    Array.isArray(span) &&
    span.length === 2 &&
    span.every((value: unknown) => Number.isSafeInteger(value)) &&
    0 <= span[0] &&
    span[0] < span[1] &&
    span[1] <= n;
  let last = 0;
  for (const [i, qValue] of typedQuestions.entries()) {
    const q = isRecord(qValue) ? qValue : null;
    const options = q?.options;
    const optionIds = q?.option_ids;
    const questionSpan = q?.question_span;
    const optionSpans = q?.option_spans;
    const count = Array.isArray(options) ? options.length : undefined;
    if (
      !q ||
      q?.question_id !== `q${i}` ||
      q.question_type !== 1 ||
      !Array.isArray(options) ||
      count === undefined ||
      count < 2 ||
      count > 64 ||
      options.some((option: unknown) => typeof option !== 'string' || !option.trim()) ||
      new Set(options).size !== count ||
      !spanOk(questionSpan) ||
      questionSpan[0] < last ||
      !Array.isArray(optionIds) ||
      !same(optionIds, Array.from({length: count}, (_, j) => String(j)).sort()) ||
      !Array.isArray(optionSpans) ||
      optionSpans.length !== count
    )
      return fail('Invalid packed question');
    let previous = questionSpan[1];
    for (const span of optionSpans) {
      if (!spanOk(span) || span[0] < previous) return fail('Invalid option span');
      previous = span[1];
    }
    last = previous;
  }
  if (
    typedQuestions.reduce(
      (sum: number, question) =>
        sum + (isRecord(question) && Array.isArray(question.options) ? question.options.length : 0),
      0
    ) > 64
  )
    return fail('Too many total options');
}
const same = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);

/** Exact decimal-place rounding of a nonnegative FP32 value, ties to even. */
function roundProbability(value: number): number {
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
export function unpackChoiceScores(
  packed: unknown,
  rows: unknown,
  probabilityRows: unknown
): {
  tokenCount: number;
  answers: Array<{
    selected: string;
    selectedIndex: number;
    probabilities: Array<{option: string; logit: number; probability: number}>;
    sourceAnswer: {type: 'choice'; choice: string; confidence: number; probabilities: Record<string, number>};
  }>;
} {
  validatePacked(packed);
  if (
    !Array.isArray(rows) ||
    !Array.isArray(probabilityRows) ||
    rows.length !== packed.questions.length ||
    probabilityRows.length !== rows.length
  )
    return fail('Missing joint FP32 score or probability rows');
  const scoreRows = rows as unknown[];
  const probabilityValues = probabilityRows as unknown[];
  const answers = packed.questions.map((q, i) => {
    const logits = scoreRows[i],
      probs = probabilityValues[i];
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
    if (probs.some((value, j) => Math.abs(value - (weights[j] ?? 0) / weightSum) > 1e-5))
      return fail('Probabilities do not match logits');
    const byId = new Map<string, number>(q.option_ids.map((id, j) => [id, j]));
    const output = q.options.map((option, j) => {
      const index = byId.get(String(j));
      if (index === undefined) return fail('Missing option ID');
      return {option, logit: logits[index]!, probability: probs[index]!};
    });
    const selectedIndex = output.reduce(
      (best, item, j) => (item.probability > (output[best]?.probability ?? -1) ? j : best),
      0
    );
    const rounded = Object.fromEntries(output.map((item, j) => [String(j), roundProbability(item.probability)]));
    return {
      selected: q.options[selectedIndex] ?? fail('Missing selected option'),
      selectedIndex,
      probabilities: output,
      sourceAnswer: {
        type: 'choice' as const,
        choice: String(selectedIndex),
        confidence: roundProbability(output[selectedIndex]?.probability ?? fail('Missing selected probability')),
        probabilities: rounded
      }
    };
  });
  return {tokenCount: packed.ids.length, answers};
}
import type {ClefTokenizer, PackedClefChoice} from './types.mjs';
