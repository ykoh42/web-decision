import {WebDecisionError} from '../kev.js';

export const STRANDS_GEMMA4_MAX_LENGTH = 4096;

export type Gemma4QuestionType = 'choice' | 'score' | 'noul';

export interface Gemma4Option {
  readonly label: string;
  readonly description: string;
}

export interface Gemma4PackedRow {
  readonly name: string;
  readonly type: Gemma4QuestionType;
  readonly options: readonly string[];
  readonly descriptions: readonly string[];
  readonly inputIds: readonly number[];
  readonly optionIndices: readonly number[];
  readonly answerIndex: number;
  readonly stateText: string;
  readonly questionText: string;
}

export interface Gemma4PackedRequest {
  readonly rows: readonly Gemma4PackedRow[];
  readonly tokenCount: number;
}

export interface Gemma4DecisionConfig {
  readonly temperatureByKind: Readonly<Record<Gemma4QuestionType, number>>;
  readonly ordinalSmoothing: number;
}

export interface Gemma4ChoiceAnswer {
  readonly type: 'choice';
  readonly choice: string;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
}

export interface Gemma4ScoreAnswer {
  readonly type: 'score';
  readonly score: number;
  readonly legend: Readonly<Record<string, string>>;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
}

export interface Gemma4NoulAnswer {
  readonly type: 'noul';
  readonly noul: number;
}

export type Gemma4Answer = Gemma4ChoiceAnswer | Gemma4ScoreAnswer | Gemma4NoulAnswer;

export interface Gemma4InferenceResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, Gemma4Answer>>;
}

export interface Gemma4DecisionResult {
  readonly tokenCount: number;
  readonly answers: readonly {
    readonly selected: string;
    readonly selectedIndex: number;
    readonly probabilities: readonly {readonly option: string; readonly probability: number; readonly logit: number}[];
  }[];
}

type Gemma4Entry = [string, unknown];

interface Gemma4QuestionContent {
  readonly type: Gemma4QuestionType;
  readonly instructions: string;
  readonly options: Gemma4Option[];
}

interface Gemma4QuestionSnapshot extends Gemma4QuestionContent {
  readonly name: string;
}

interface Gemma4RenderedQuestion {
  readonly type: Gemma4QuestionType;
  readonly options: Gemma4Option[];
  readonly text: string;
  readonly spans: Array<[number, number]>;
}

interface Gemma4RequestSnapshot {
  readonly stateText: string;
  readonly questions: Gemma4QuestionSnapshot[];
}

type Gemma4TokenSpan = [number, number];

interface Gemma4TokenizerContract {
  encode(text: string, options: {readonly add_special_tokens: boolean}): unknown;
  decode(
    ids: number[],
    options: {readonly skip_special_tokens: false; readonly clean_up_tokenization_spaces: false}
  ): unknown;
}

interface Gemma4AnswerRow {
  readonly type: Gemma4QuestionType;
  readonly options: readonly string[];
  readonly descriptions?: readonly string[] | undefined;
}

const whitespace =
  '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const trimStart = new RegExp(`^${whitespace}+`, 'u');
const trimEnd = new RegExp(`${whitespace}+$`, 'u');
const collapseWhitespace = new RegExp(`${whitespace}+`, 'gu');
const invalid = (message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError('INVALID_INPUT', message, cause ? {cause} : undefined);
const invalidTokenizer = (message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError('INVALID_ASSET', message, cause ? {cause} : undefined);
const invalidLogits = (): WebDecisionError =>
  new WebDecisionError('INFERENCE_ERROR', 'Strands Gemma 4 returned invalid pointer logits.');

function pythonStrip(value: string): string {
  return value.replace(trimStart, '').replace(trimEnd, '');
}

function assertUnicode(value: string, label: string): void {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw invalid(`${label} contains an unpaired surrogate.`);
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw invalid(`${label} contains an unpaired surrogate.`);
    }
  }
}

function objectEntries(value: unknown, label: string, allowMap = false): Gemma4Entry[] {
  if (allowMap && value instanceof Map) {
    if (Reflect.ownKeys(value).length) throw invalid(`${label} maps cannot have extra properties.`);
    let entries: Gemma4Entry[];
    try {
      entries = Array.from(Map.prototype.entries.call(value)) as Gemma4Entry[];
    } catch (cause) {
      throw invalid(`${label} must be a valid string-keyed map.`, cause);
    }
    if (entries.some(([key]) => typeof key !== 'string')) throw invalid(`${label} map keys must be strings.`);
    return entries;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && Object.getPrototypeOf(prototype) !== null)
    throw invalid(`${label} must be a plain object.`);
  const entries: Gemma4Entry[] = [];
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} must contain enumerable string data properties.`);
    entries.push([key, descriptor.value]);
  }
  return entries;
}

function arrayValues(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw invalid(`${label} must be an array.`);
  const values: unknown[] = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} must be a dense data array.`);
    values.push(descriptor.value);
  }
  for (const key of Reflect.ownKeys(value))
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/u.test(key) || Number(key) >= value.length))
      throw invalid(`${label} cannot have extra properties.`);
  return values;
}

// The JSON wire decides whether Python parses a number as an int or a float.
function pythonJsonNumber(value: number): string {
  if (!Number.isFinite(value)) throw invalid('Content numbers must be finite.');
  const wire = JSON.stringify(value) as string;
  if (!/[.e]/u.test(wire)) return wire;
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e16)) {
    const [mantissa, exponent] = value.toExponential().split('e') as [string, string];
    const power = Number(exponent);
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`;
  }
  return wire;
}

function renderJson(value: unknown, depth = 0, ancestors: Set<object> = new Set()): string {
  if (value === null || typeof value === 'boolean') return String(value);
  if (typeof value === 'string') {
    assertUnicode(value, 'Content');
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    return pythonJsonNumber(value);
  }
  if (!value || typeof value !== 'object') throw invalid('Content must be JSON data.');
  if (ancestors.has(value)) throw invalid('Content cannot contain a cycle.');

  ancestors.add(value);
  const indent = ' '.repeat((depth + 1) * 2);
  const closing = ' '.repeat(depth * 2);
  let rendered: string;
  if (Array.isArray(value)) {
    const values = arrayValues(value, 'Content arrays');
    rendered = values.length
      ? `[\n${values.map((item) => `${indent}${renderJson(item, depth + 1, ancestors)}`).join(',\n')}\n${closing}]`
      : '[]';
  } else {
    const entries = objectEntries(value, 'Content objects');
    rendered = entries.length
      ? `{\n${entries
          .map(([key, item]) => {
            assertUnicode(key, 'Content keys');
            return `${indent}${JSON.stringify(key)}: ${renderJson(item, depth + 1, ancestors)}`;
          })
          .join(',\n')}\n${closing}}`
      : '{}';
  }
  ancestors.delete(value);
  return rendered;
}

function renderContent(value: unknown, label: string, allowNull = false): string {
  if (value === null && allowNull) return '';
  if (typeof value === 'string') {
    assertUnicode(value, label);
    return pythonStrip(value);
  }
  if (Array.isArray(value) || (value && typeof value === 'object')) return renderJson(value);
  throw invalid(`${label} must be text or JSON object data.`);
}

function snapshotChoice(fields: Record<string, unknown>, name: string, instructions: string): Gemma4QuestionContent {
  const entries: Gemma4Entry[] = Array.isArray(fields.criteria)
    ? arrayValues(fields.criteria, `Question ${name} criteria`).map((label) => [label, null] as Gemma4Entry)
    : objectEntries(fields.criteria, `Question ${name} criteria`, true);
  if (entries.length < 2) throw invalid(`Question ${name} choice requires at least 2 options.`);
  const labels = new Set<string>();
  const options: Gemma4Option[] = entries.map(([label, description]) => {
    if (typeof label !== 'string' || labels.has(label))
      throw invalid(`Question ${name} choice labels must be distinct strings.`);
    assertUnicode(label, 'Option labels');
    labels.add(label);
    return {label, description: renderContent(description, 'Option descriptions', true)};
  });
  return {type: 'choice', instructions, options};
}

function snapshotScore(fields: Record<string, unknown>, name: string, instructions: string): Gemma4QuestionContent {
  const levels = arrayValues(fields.criteria, `Question ${name} criteria`);
  if (levels.length < 2 || levels.length > 10) throw invalid(`Question ${name} score requires 2 to 10 levels.`);
  const options: Gemma4Option[] = levels.map((description, index) => {
    if (typeof description !== 'string') throw invalid(`Question ${name} score level ${index} must be text.`);
    assertUnicode(description, 'Score levels');
    return {label: String(index), description: pythonStrip(description)};
  });
  return {type: 'score', instructions, options};
}

function snapshotNoul(fields: Record<string, unknown>, name: string, instructions: string): Gemma4QuestionContent {
  const criteria =
    fields.criteria === undefined || fields.criteria === null
      ? []
      : objectEntries(fields.criteria, `Question ${name} criteria`, true);
  if (criteria.some(([key]) => key !== 'true' && key !== 'false'))
    throw invalid(`Question ${name} noul criteria can only describe false and true.`);
  const descriptions = Object.fromEntries(criteria);
  const falseDescription = Object.hasOwn(descriptions, 'false')
    ? descriptions.false
    : 'the statement does not hold for this state';
  const trueDescription = Object.hasOwn(descriptions, 'true')
    ? descriptions.true
    : 'the statement holds for this state';
  return {
    type: 'noul',
    instructions,
    options: [
      {label: 'false', description: renderContent(falseDescription, 'Noul descriptions', true)},
      {label: 'true', description: renderContent(trueDescription, 'Noul descriptions', true)}
    ]
  };
}

const questionSnapshots: Readonly<
  Record<
    Gemma4QuestionType,
    (fields: Record<string, unknown>, name: string, instructions: string) => Gemma4QuestionContent
  >
> = {choice: snapshotChoice, score: snapshotScore, noul: snapshotNoul};

function snapshotQuestion(question: unknown, name: string): Gemma4QuestionContent {
  const fields = Object.fromEntries(objectEntries(question, `Question ${name}`)) as Record<string, unknown>;
  const unsupported = Object.keys(fields).some((key) => !['type', 'instructions', 'criteria'].includes(key));
  if (unsupported) throw invalid(`Question ${name} has an unsupported field.`);
  if (!Object.hasOwn(questionSnapshots, fields.type as PropertyKey))
    throw invalid(`Question ${name} type must be choice, score, or noul.`);
  const snapshot = questionSnapshots[fields.type as Gemma4QuestionType];
  const instructions = renderContent(fields.instructions, `Question ${name} instructions`);
  return snapshot(fields, name, instructions);
}

function renderOptionBlock(options: readonly Gemma4Option[]): {text: string; spans: Array<[number, number]>} {
  const lines: string[] = [];
  const spans: Array<[number, number]> = [];
  let cursor = 0;
  for (let index = 0; index < options.length; index++) {
    const {label, description} = options[index] as Gemma4Option;
    const compact = description.replace(collapseWhitespace, ' ').replace(/^ +| +$/gu, '');
    const line = `${index + 1}. ${label}${compact ? ` — ${compact}` : ''}`;
    lines.push(line);
    spans.push([cursor, cursor + line.length]);
    cursor += line.length + 1;
  }
  return {text: lines.join('\n'), spans};
}

function renderQuestion(question: Gemma4QuestionContent): Gemma4RenderedQuestion {
  const headers = {
    choice: 'Select exactly one option.',
    score: 'Rate the state against the ordered levels below (lowest first).',
    noul: 'Decide whether the statement is true of the state.'
  };
  const block = renderOptionBlock(question.options);
  const prefix = `<question type="${question.type}">\n${headers[question.type]}\n${question.instructions}\n<options>\n`;
  const text = `${prefix}${block.text}\n</options>\n</question>\n<answer>`;
  return {
    type: question.type,
    options: question.options,
    text,
    spans: block.spans.map(([start, end]): Gemma4TokenSpan => [prefix.length + start, prefix.length + end])
  };
}

function snapshotRequest(state: unknown, questions: unknown): Gemma4RequestSnapshot {
  const stateText = renderContent(state, 'State');
  const entries = objectEntries(questions, 'Questions', true);
  if (!entries.length) throw invalid('Provide at least one named question.');
  const snapshots: Gemma4QuestionSnapshot[] = entries.map(([name, question]) => ({
    name,
    ...snapshotQuestion(question, name)
  }));
  return {stateText: `<state>\n${stateText}\n</state>\n`, questions: snapshots};
}

function tokenIds(encoded: unknown): number[] {
  const ids = (encoded as {readonly ids?: unknown} | null | undefined)?.ids;
  if (!Array.isArray(ids)) throw invalidTokenizer('Strands Gemma 4 tokenizer returned invalid token IDs.');
  const copy: number[] = [];
  for (let index = 0; index < ids.length; index++) {
    if (!Object.hasOwn(ids, index) || !Number.isSafeInteger(ids[index]) || ids[index] < 0 || ids[index] >= 262144)
      throw invalidTokenizer('Strands Gemma 4 tokenizer returned invalid token IDs.');
    copy.push(ids[index]);
  }
  return copy;
}

function tokenOffsets(tokenizer: unknown, text: string, ids: readonly number[]): Gemma4TokenSpan[] {
  if (typeof (tokenizer as Gemma4TokenizerContract | null | undefined)?.decode !== 'function')
    throw invalidTokenizer('Strands Gemma 4 tokenizer cannot recover token offsets.');
  const spans: Gemma4TokenSpan[] = [];
  const pieces: string[] = [];
  let cursor = 0;
  for (const id of ids) {
    let piece;
    try {
      piece = (tokenizer as Gemma4TokenizerContract).decode([id], {
        skip_special_tokens: false,
        clean_up_tokenization_spaces: false
      });
    } catch (cause) {
      throw invalidTokenizer('Strands Gemma 4 tokenizer could not decode token offsets.', cause);
    }
    if (typeof piece !== 'string' || !piece.length)
      throw invalidTokenizer('Strands Gemma 4 tokenizer returned an empty token piece.');
    pieces.push(piece);
    spans.push([cursor, cursor + piece.length]);
    cursor += piece.length;
  }
  if (pieces.join('') !== text)
    throw invalidTokenizer('Strands Gemma 4 tokenizer cannot align decoded tokens with the exact prompt text.');
  return spans;
}

function encode(tokenizer: unknown, text: string, addSpecialTokens: boolean): number[] {
  if (typeof (tokenizer as Gemma4TokenizerContract | null | undefined)?.encode !== 'function')
    throw invalidTokenizer('Strands Gemma 4 tokenizer is unavailable.');
  try {
    return tokenIds((tokenizer as Gemma4TokenizerContract).encode(text, {add_special_tokens: addSpecialTokens}));
  } catch (cause) {
    if (cause instanceof WebDecisionError) throw cause;
    throw invalidTokenizer('Strands Gemma 4 tokenizer failed.', cause);
  }
}

function optionIndices(
  question: Gemma4RenderedQuestion,
  offsets: readonly Gemma4TokenSpan[],
  prefixLength: number
): number[] {
  return question.spans.map(([start, end]) => {
    let last = -1;
    for (let index = 0; index < offsets.length; index++) {
      const [left, right] = offsets[index] as Gemma4TokenSpan;
      if (right > left && left >= start && right <= end) last = index;
    }
    if (last < 0) throw invalidTokenizer('A Strands Gemma 4 option has no token in its rendered span.');
    return prefixLength + last;
  });
}

/** Encode the upstream state/question split, preserving every token and refusing truncation. */
export function packStrandsGemma4Request(
  tokenizer: unknown,
  state: unknown,
  questions: unknown,
  maxLength: number = STRANDS_GEMMA4_MAX_LENGTH
): Gemma4PackedRequest {
  if (!Number.isSafeInteger(maxLength) || maxLength < 1) throw invalid('Invalid Strands Gemma 4 context length.');
  const request = snapshotRequest(state, questions);
  const stateIds = encode(tokenizer, request.stateText, true);
  const rows = request.questions.map((source) => {
    const question = renderQuestion(source);
    const ids = encode(tokenizer, question.text, false);
    const offsets = tokenOffsets(tokenizer, question.text, ids);
    if (stateIds.length + ids.length > maxLength)
      throw invalid(`Question ${source.name} exceeds the ${maxLength}-token context window.`);
    const inputIds = stateIds.concat(ids);
    return Object.freeze({
      name: source.name,
      type: source.type,
      options: Object.freeze(source.options.map(({label}) => label)),
      descriptions: Object.freeze(source.options.map(({description}) => description)),
      inputIds: Object.freeze(inputIds),
      optionIndices: Object.freeze(optionIndices(question, offsets, stateIds.length)),
      answerIndex: inputIds.length - 1,
      stateText: request.stateText,
      questionText: question.text
    });
  });
  return Object.freeze({
    rows: Object.freeze(rows),
    tokenCount: rows.reduce((total, row) => total + row.inputIds.length, 0)
  });
}

function roundPython(value: number, places = 4): number {
  if (!Number.isFinite(value)) return value;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, Math.abs(value));
  const bits = view.getBigUint64(0);
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  const mantissa = exponentBits === 0 ? fraction : fraction | (1n << 52n);
  const exponent = exponentBits === 0 ? -1074 : exponentBits - 1023 - 52;
  const scaled = mantissa * 10n ** BigInt(places);
  let rounded: bigint;
  if (exponent >= 0) rounded = scaled << BigInt(exponent);
  else {
    const divisor = 1n << BigInt(-exponent);
    const quotient = scaled / divisor;
    const remainder = scaled % divisor;
    const twice = remainder * 2n;
    rounded = twice > divisor || (twice === divisor && quotient % 2n === 1n) ? quotient + 1n : quotient;
  }
  const result = Number(rounded) / 10 ** places;
  return value < 0 ? -result : result;
}

function softmax(logits: readonly number[], temperature: number): number[] {
  const scaled = logits.map((value) => value / temperature);
  if (scaled.some((value) => !Number.isFinite(value))) throw invalidLogits();
  const maximum = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - maximum));
  const total = exps.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) throw invalidLogits();
  return exps.map((value) => value / total);
}

function confidence(probabilities: readonly number[]): number {
  if (probabilities.length <= 1) return 1;
  const maximum = Math.max(...probabilities);
  return Math.max(0, Math.min(1, (probabilities.length * maximum - 1) / (probabilities.length - 1)));
}

function scoreConfidence(probabilities: readonly number[], smoothing: number): number {
  const count = probabilities.length;
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  if (count <= 1) return 1;
  if (!Number.isFinite(smoothing) || smoothing < 0) throw invalid('Invalid Strands Gemma 4 ordinal smoothing.');
  if (total <= 0) return 0;
  const normalized = probabilities.map((value) => value / total);
  const mean = normalized.reduce((sum, value, index) => sum + index * value, 0);
  const variance = normalized.reduce((sum, value, index) => sum + value * (index - mean) ** 2, 0);
  const sigma = Math.sqrt(variance);
  const maximum = (count - 1) / 2;
  const floor = Math.sqrt(smoothing);
  if (maximum <= floor) return sigma <= floor ? 1 : 0;
  return Math.max(0, Math.min(1, (maximum - sigma) / (maximum - floor)));
}

function validateLogits(logits: unknown, count: number): number[] {
  if (
    (!Array.isArray(logits) && !(ArrayBuffer.isView(logits) && !(logits instanceof DataView))) ||
    (logits as ArrayLike<unknown>).length !== count
  )
    throw invalidLogits();
  const values: number[] = [];
  for (let index = 0; index < count; index++) {
    if (Array.isArray(logits) && !Object.hasOwn(logits, index)) throw invalidLogits();
    const value = (logits as ArrayLike<unknown>)[index];
    if (!Number.isFinite(value)) throw invalidLogits();
    values.push(value as number);
  }
  return values;
}

/** Apply the checkpoint's per-question temperature and native Strands response rules. */
export function finishStrandsGemma4Answer(
  row: Gemma4AnswerRow,
  logits: unknown,
  config: Gemma4DecisionConfig
): Gemma4Answer {
  const count = row?.options?.length;
  if (
    !Number.isSafeInteger(count) ||
    (count as number) < 2 ||
    (row.type === 'score' && (count as number) > 10) ||
    (row.type === 'noul' && count !== 2) ||
    !['choice', 'score', 'noul'].includes(row.type) ||
    (row.type === 'score' && (!Array.isArray(row.descriptions) || row.descriptions.length !== count))
  )
    throw invalidLogits();
  const temperature = config?.temperatureByKind?.[row.type];
  if (!Number.isFinite(temperature) || temperature <= 0) throw invalid('Invalid Strands Gemma 4 temperature.');
  const probabilities = softmax(validateLogits(logits, count as number), temperature);
  if (row.type === 'noul') return {type: 'noul', noul: roundPython(probabilities[1] as number)};
  const probabilityRecord = Object.fromEntries(
    row.options.map((label, index) => [label, roundPython(probabilities[index] as number)])
  );
  if (row.type === 'choice') {
    const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
    return {
      type: 'choice',
      choice: row.options[selectedIndex] as string,
      probabilities: probabilityRecord,
      confidence: roundPython(confidence(probabilities))
    };
  }
  const expected = probabilities.reduce((sum, probability, index) => sum + index * probability, 0);
  const legend = Object.fromEntries(
    row.options.map((label, index) => [String(index), row.descriptions![index] as string])
  );
  return {
    type: 'score',
    score: roundPython(expected),
    legend,
    probabilities: probabilityRecord,
    confidence: roundPython(scoreConfidence(probabilities, config.ordinalSmoothing))
  };
}

/** Restore question names and model identity after one graph result per typed prompt. */
export function unpackStrandsGemma4Scores(
  packed: Gemma4PackedRequest,
  scoreRows: readonly ArrayLike<number>[],
  config: Gemma4DecisionConfig,
  model: string
): Gemma4InferenceResult {
  if (
    !Array.isArray(packed?.rows) ||
    !Array.isArray(scoreRows) ||
    scoreRows.length !== packed.rows.length ||
    !Number.isSafeInteger(packed.tokenCount) ||
    packed.tokenCount < 0 ||
    typeof model !== 'string' ||
    !model
  )
    throw invalidLogits();
  const answers: Array<[string, Gemma4Answer]> = [];
  const names = new Set<string>();
  for (let index = 0; index < packed.rows.length; index++) {
    if (!Object.hasOwn(packed.rows, index) || !Object.hasOwn(scoreRows, index)) throw invalidLogits();
    const row = packed.rows[index] as Gemma4AnswerRow & {readonly name: string};
    if (typeof row?.name !== 'string' || names.has(row.name)) throw invalidLogits();
    names.add(row.name);
    answers.push([row.name, finishStrandsGemma4Answer(row, scoreRows[index] as ArrayLike<number>, config)]);
  }
  return {model, tokenCount: packed.tokenCount, answers: Object.fromEntries(answers)};
}
