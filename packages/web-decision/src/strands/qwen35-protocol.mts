import {readKananaFloat, type KananaFloat} from '../kanana/json-float.ts';
import {WebDecisionError} from '../kev.js';

/**
 * Internal typed protocol used by the Qwen session and model loader for the
 * pinned Strands Decider revision. The module is not exported directly.
 */

// The pinned checkpoint config and full graph export both cap context at 4096 tokens.
export const QWEN35_MAX_LENGTH = 4096;
// The pinned schema.py validators define these option-count limits.
export const QWEN35_MAX_CHOICE_OPTIONS = 255;
export const QWEN35_MIN_SCORE_LEVELS = 2;
export const QWEN35_MAX_SCORE_LEVELS = 10;
// VisionEngineConfig defaults; callers can override these engine settings.
export const QWEN35_MAX_IMAGES = 4;
// EngineConfig defaults; maxBatch is configurable and does not cap request size.
export const QWEN35_MAX_BATCH = 32;
// The pinned R13 graph is bounded to three rows; split larger source batches.
export const QWEN35_GRAPH_MAX_BATCH = 3;
export const QWEN35_MAX_QUESTION_FRACTION = 0.75;
export const QWEN35_ORDINAL_SMOOTHING = 0.1;
export const QWEN35_TEMPERATURE_BY_KIND = Object.freeze({
  choice: 0.8629556148274228,
  noul: 1.0704394404346953,
  score: 1.834409163182587
});
const QWEN35_MASKED_LOGIT = -10_000;

/**
 * Build a JSON record whose enumerable keys follow Python dict insertion order.
 * JavaScript normally reorders integer-like keys numerically, so an ownKeys
 * proxy is needed for source-compatible serialization without extra fields.
 */
export function qwen35OrderedRecord<Value>(
  entries: readonly (readonly [key: string, value: Value])[]
): Readonly<Record<string, Value>> {
  const record: Record<string, Value> = {};
  const keys: string[] = [];
  for (const [key, value] of entries) {
    if (Object.hasOwn(record, key)) throw new TypeError(`Duplicate Qwen record key: ${key}`);
    Object.defineProperty(record, key, {value, enumerable: true, writable: true, configurable: true});
    keys.push(key);
  }
  const orderedRecord = new Proxy(record, {ownKeys: () => keys});
  return Object.freeze(orderedRecord);
}

export type Qwen35BatchOptions = Readonly<{
  maxBatch?: number;
  maxQuestionFraction?: number;
  maxBatchTokens?: number | null;
}>;

export type Qwen35ResolvedBatchOptions = Readonly<{
  maxBatch: number;
  maxQuestionFraction: number;
  maxBatchTokens: number | null;
}>;

export type Qwen35RequestOptions = Qwen35BatchOptions & Readonly<{maxImages?: number; maxLength?: number}>;

export type Qwen35QuestionType = 'noul' | 'choice' | 'score';

/** `bigint` preserves exact Python integer content where JavaScript Number cannot. */
export type Qwen35JsonValue =
  | null
  | boolean
  | number
  | KananaFloat
  | bigint
  | string
  | readonly Qwen35JsonValue[]
  | ReadonlyMap<string, Qwen35JsonValue>
  | Qwen35JsonObject;
export type Qwen35Content =
  | string
  | readonly Qwen35JsonValue[]
  | ReadonlyMap<string, Qwen35JsonValue>
  | Qwen35JsonObject;
export interface Qwen35JsonObject {
  readonly [key: string]: Qwen35JsonValue;
}

class OrderedContentMap implements ReadonlyMap<string, Qwen35JsonValue> {
  readonly #entries: readonly (readonly [string, Qwen35JsonValue])[];

  constructor(entries: readonly (readonly [string, Qwen35JsonValue])[]) {
    this.#entries = Object.freeze(entries.map(([key, value]) => Object.freeze([key, value] as const)));
    Object.freeze(this);
  }

  get size(): number {
    return this.#entries.length;
  }

  get(key: string): Qwen35JsonValue | undefined {
    return this.#entries.find(([entryKey]) => entryKey === key)?.[1];
  }

  has(key: string): boolean {
    return this.#entries.some(([entryKey]) => entryKey === key);
  }

  forEach(
    callbackfn: (value: Qwen35JsonValue, key: string, map: ReadonlyMap<string, Qwen35JsonValue>) => void,
    thisArg?: unknown
  ): void {
    for (const [key, value] of this.#entries) callbackfn.call(thisArg, value, key, this);
  }

  *entries(): MapIterator<[string, Qwen35JsonValue]> {
    for (const [key, value] of this.#entries) yield [key, value];
  }

  *keys(): MapIterator<string> {
    for (const [key] of this.#entries) yield key;
  }

  *values(): MapIterator<Qwen35JsonValue> {
    for (const [, value] of this.#entries) yield value;
  }

  [Symbol.iterator](): MapIterator<[string, Qwen35JsonValue]> {
    return this.entries();
  }

  get [Symbol.toStringTag](): string {
    return 'Map';
  }
}

export interface Qwen35Option {
  readonly label: string;
  /** Deep-copied source criteria value, before the Strands content renderer flattens it. */
  readonly sourceDescription: Qwen35Content | null;
  /** Stripped source description, retained for answer legends. */
  readonly description: string;
}

export interface Qwen35QuestionSnapshot {
  readonly name: string;
  readonly type: Qwen35QuestionType;
  readonly instructions: Qwen35Content;
  readonly instructionText: string;
  readonly options: readonly Qwen35Option[];
}

export interface Qwen35RequestSnapshot {
  readonly model: string;
  readonly state: Qwen35Content;
  readonly stateText: string;
  readonly questions: readonly Qwen35QuestionSnapshot[];
  /** Kept for the later vision path. This module does not decode or embed images. */
  readonly images: readonly string[];
}

export interface Qwen35RenderedQuestion {
  readonly type: Qwen35QuestionType;
  readonly questionText: string;
  /** Character spans in questionText, one per option line. */
  readonly optionSpans: readonly (readonly [number, number])[];
}

export interface Qwen35PackedRow {
  readonly name: string;
  readonly type: Qwen35QuestionType;
  readonly options: readonly Qwen35Option[];
  readonly inputIds: readonly number[];
  readonly optionIndices: readonly number[];
  readonly answerIndex: number;
  readonly stateText: string;
  readonly questionText: string;
}

export interface Qwen35PackedRequest extends Qwen35RequestSnapshot {
  /** Source StrandsDeciderConfig.max_length used to fit this request. */
  readonly maxLength: number;
  readonly rows: readonly Qwen35PackedRow[];
  /** Consecutive row groups produced by EngineConfig max_batch/max_batch_tokens. */
  readonly batchSizes: readonly number[];
  readonly tokenCount: number;
}

export interface Qwen35NoulAnswer {
  readonly type: 'noul';
  readonly noul: number;
}

export interface Qwen35ChoiceAnswer {
  readonly type: 'choice';
  readonly choice: string;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
}

export interface Qwen35ScoreAnswer {
  readonly type: 'score';
  readonly score: number;
  readonly legend: Readonly<Record<string, string>>;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
}

export type Qwen35Answer = Qwen35NoulAnswer | Qwen35ChoiceAnswer | Qwen35ScoreAnswer;

export interface Qwen35NamedAnswerRow {
  readonly name: string;
  readonly answer: Qwen35Answer;
}

interface Qwen35SourceObject {
  readonly [key: string]: unknown;
}

interface Qwen35Encoded {
  readonly ids?: unknown;
  readonly tokens?: unknown;
}

interface Qwen35Tokenizer {
  readonly decoder?:
    | {
        readonly byte_decoder?: Readonly<Record<string, string | number>> | null | undefined;
      }
    | null
    | undefined;
  get_added_tokens_decoder?: (() => {has(value: number): boolean} | null | undefined) | undefined;
  encode(text: string, options: {readonly add_special_tokens: boolean}): unknown;
}

interface Qwen35Tokenized {
  readonly ids: readonly number[];
  readonly tokens: readonly string[];
}

type Qwen35TokenSpan = readonly [number, number];

const PYTHON_WHITESPACE =
  '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const trimStart = new RegExp(`^${PYTHON_WHITESPACE}+`, 'u');
const trimEnd = new RegExp(`${PYTHON_WHITESPACE}+$`, 'u');
const collapseWhitespace = new RegExp(`${PYTHON_WHITESPACE}+`, 'gu');
const encoder = new TextEncoder();
const noulDefaults = Object.freeze({
  false: 'the statement does not hold for this state',
  true: 'the statement holds for this state'
});

function invalid(message: string): never {
  throw new TypeError(message);
}

function invalidRequest(message: string): never {
  throw new WebDecisionError('INVALID_INPUT', message);
}

export function resolveQwen35BatchOptions(options: Qwen35BatchOptions = {}): Qwen35ResolvedBatchOptions {
  const maxBatch = options.maxBatch === undefined ? QWEN35_MAX_BATCH : options.maxBatch;
  if (typeof maxBatch !== 'number' || !Number.isInteger(maxBatch) || maxBatch <= 0)
    throw new RangeError('Qwen maxBatch must be a positive integer.');

  const maxQuestionFraction =
    options.maxQuestionFraction === undefined ? QWEN35_MAX_QUESTION_FRACTION : options.maxQuestionFraction;
  if (typeof maxQuestionFraction !== 'number' || !Number.isFinite(maxQuestionFraction))
    throw new RangeError('Qwen maxQuestionFraction must be a finite number.');

  const maxBatchTokens = options.maxBatchTokens === undefined ? null : options.maxBatchTokens;
  if (maxBatchTokens !== null && (typeof maxBatchTokens !== 'number' || !Number.isInteger(maxBatchTokens)))
    throw new RangeError('Qwen maxBatchTokens must be an integer or null.');

  return {maxBatch, maxQuestionFraction, maxBatchTokens};
}

export function resolveQwen35MaxLength(maxLength = QWEN35_MAX_LENGTH): number {
  if (!Number.isSafeInteger(maxLength) || maxLength < 1)
    throw new RangeError('Qwen maxLength must be a positive safe integer.');
  if (maxLength > QWEN35_MAX_LENGTH)
    throw new RangeError(`Qwen maxLength cannot exceed the pinned ${QWEN35_MAX_LENGTH}-token context.`);
  return maxLength;
}

/** Split one already-fitted max_batch group using Strands Engine._chunks token budgeting. */
export function splitQwen35BatchRanges(
  questionLengths: readonly number[],
  prefixLength: number,
  maxBatchTokens: number | null
): readonly (readonly [start: number, end: number])[] {
  const ranges: Array<readonly [number, number]> = [];
  let start = 0;
  let longest = 0;
  for (let index = 0; index < questionLengths.length; index += 1) {
    const questionLength = questionLengths[index]!;
    if (maxBatchTokens !== null && maxBatchTokens !== 0 && index > start) {
      const rows = index - start + 1;
      const paddedWidth = prefixLength + Math.max(longest, questionLength);
      if (rows * paddedWidth > maxBatchTokens) {
        ranges.push([start, index]);
        start = index;
        longest = 0;
      }
    }
    longest = Math.max(longest, questionLength);
  }
  if (start < questionLengths.length) ranges.push([start, questionLengths.length]);
  return ranges;
}

function strip(value: string): string {
  return value.replace(trimStart, '').replace(trimEnd, '');
}

function assertUnicode(value: string, label: string): void {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) invalid(`${label} contains an unpaired surrogate.`);
    } else if (code >= 0xdc00 && code <= 0xdfff) invalid(`${label} contains an unpaired surrogate.`);
  }
}

function ownEntries(value: unknown, label: string, allowMap = false): Array<[string, unknown]> {
  if (allowMap && value instanceof Map) {
    if (Reflect.ownKeys(value).length) invalid(`${label} maps cannot have extra properties.`);
    const entries = Array.from(Map.prototype.entries.call(value)) as unknown[];
    const result: Array<[string, unknown]> = [];
    for (const entry of entries) {
      if (!Array.isArray(entry) || entry.length !== 2 || typeof entry[0] !== 'string')
        invalid(`${label} map keys must be strings.`);
      result.push([entry[0] as string, entry[1]]);
    }
    return result;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) invalid(`${label} must be a plain object.`);
  const entries: Array<[string, unknown]> = [];
  // Keep ECMAScript own-key order for records; integer-index keys enumerate numerically.
  // A JavaScript object cannot recover a JSON parser's original member order for those keys.
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    entries.push([key, descriptor.value]);
  }
  return entries;
}

function arrayValues(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) invalid(`${label} must be an array.`);
  const values: unknown[] = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value')) invalid(`${label} must be a dense data array.`);
    values.push(descriptor.value);
  }
  for (const key of Reflect.ownKeys(value))
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/u.test(key) || Number(key) >= value.length))
      invalid(`${label} cannot have extra properties.`);
  return values;
}

function snapshotContainer(value: object, label: string, ancestors: Set<object>): Qwen35JsonValue {
  if (value instanceof OrderedContentMap) {
    return value;
  }
  if (value instanceof Map) {
    const entries = ownEntries(value, `${label} dictionaries`, true).map(([key, item]) => {
      assertUnicode(key, `${label} keys`);
      return [key, snapshotContent(item, label, ancestors)] as const;
    });
    return new OrderedContentMap(entries);
  }
  if (Array.isArray(value)) {
    return Object.freeze(arrayValues(value, `${label} arrays`).map((item) => snapshotContent(item, label, ancestors)));
  }
  const entries = ownEntries(value, `${label} objects`);
  const object: Record<string, Qwen35JsonValue> = Object.create(null) as Record<string, Qwen35JsonValue>;
  for (const [key, item] of entries) {
    assertUnicode(key, `${label} keys`);
    object[key] = snapshotContent(item, label, ancestors);
  }
  return Object.freeze(object);
}

function snapshotContent(value: unknown, label: string, ancestors = new Set<object>()): Qwen35JsonValue {
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    assertUnicode(value, label);
    return value;
  }
  if (typeof value === 'number') {
    const wire = JSON.stringify(value) as string;
    // Exponent-form wire numbers are Python floats; integer-form tokens may
    // already have lost source precision in JavaScript's Number representation.
    if (Number.isInteger(value) && !Number.isSafeInteger(value) && !/[eE]/u.test(wire))
      invalid(`${label} contains an unsafe integer Number; use bigint to preserve its exact value.`);
    return value;
  }
  if (typeof value === 'bigint') return value;
  if (readKananaFloat(value) !== undefined) return value as KananaFloat;
  if (!value || typeof value !== 'object') invalid(`${label} must be text or JSON data.`);
  if (ancestors.has(value)) invalid(`${label} cannot contain a cycle.`);
  ancestors.add(value);
  const result = snapshotContainer(value, label, ancestors);
  ancestors.delete(value);
  return result;
}

function isStructuredContent(value: Qwen35JsonValue): value is Exclude<Qwen35Content, string> {
  return value !== null && typeof value === 'object' && readKananaFloat(value) === undefined;
}

function snapshotSourceContent(value: unknown, label: string): Qwen35Content;
function snapshotSourceContent(value: unknown, label: string, allowNull: true): Qwen35Content | null;
function snapshotSourceContent(value: unknown, label: string, allowNull = false): Qwen35Content | null {
  const result = snapshotContent(value, label);
  if (result === null && allowNull) return null;
  if (typeof result === 'string' || isStructuredContent(result)) return result;
  invalid(`${label} must be text or JSON data.`);
}

function pythonJsonNumber(value: number): string {
  if (Number.isNaN(value)) return 'NaN';
  if (value === Number.POSITIVE_INFINITY) return 'Infinity';
  if (value === Number.NEGATIVE_INFINITY) return '-Infinity';
  if (Object.is(value, -0)) return '-0.0';
  const wire = JSON.stringify(value) as string;
  // Map JS finite integer-form Numbers to Python ints; other Numbers retain float form.
  // BigInt is the exact-integer path for values beyond Number's safe range.
  if (!wire.includes('.') && !wire.includes('e') && !wire.includes('E')) return wire;
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e16)) {
    const [mantissa, exponent] = value.toExponential().split('e') as [string, string];
    const power = Number(exponent);
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`;
  }
  return wire;
}

function pythonJsonFloat(value: number): string {
  if (Object.is(value, -0)) return '-0.0';
  if (value === 0) return '0.0';
  const magnitude = Math.abs(value);
  if (magnitude < 1e-4 || magnitude >= 1e16) {
    const [mantissa, exponent] = value.toExponential().split('e') as [string, string];
    const power = Number(exponent);
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`;
  }
  const wire = JSON.stringify(value) as string;
  return wire.includes('.') ? wire : `${wire}.0`;
}

function renderJson(value: Qwen35JsonValue, depth = 0): string {
  if (value === null || typeof value === 'boolean') return String(value);
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'number') return pythonJsonNumber(value);
  const float = readKananaFloat(value);
  if (float !== undefined) return pythonJsonFloat(float);
  const indent = ' '.repeat((depth + 1) * 2);
  const closing = ' '.repeat(depth * 2);
  if (Array.isArray(value)) {
    return value.length
      ? `[\n${value.map((item) => `${indent}${renderJson(item, depth + 1)}`).join(',\n')}\n${closing}]`
      : '[]';
  }
  const entries =
    value instanceof OrderedContentMap ? Array.from(value.entries()) : Object.entries(value as Qwen35JsonObject);
  return entries.length
    ? `{\n${entries.map(([key, item]) => `${indent}${JSON.stringify(key)}: ${renderJson(item, depth + 1)}`).join(',\n')}\n${closing}}`
    : '{}';
}

function renderContent(value: Qwen35Content | null, label: string, allowNull = false): string {
  if (value === null) {
    if (allowNull) return '';
    invalid(`${label} must be text or JSON data.`);
  }
  if (typeof value === 'string') return strip(value);
  if (isStructuredContent(value)) return renderJson(value);
  invalid(`${label} must be text or JSON data.`);
}

function inferredQuestionType(fields: Qwen35SourceObject, name: string): Qwen35QuestionType {
  const criteria = fields.criteria;
  // Match the pinned Pydantic union: Noul is tried first, then criteria shape selects Choice or Score.
  if (criteria == null) return 'noul';
  if (Array.isArray(criteria)) return 'score';
  const entries = ownEntries(criteria, `Question ${name} criteria`, true);
  return entries.every(([key]) => key === 'false' || key === 'true') ? 'noul' : 'choice';
}

function snapshotQuestion(question: unknown, name: string): Qwen35QuestionSnapshot {
  const fields = Object.fromEntries(ownEntries(question, `Question ${name}`)) as Qwen35SourceObject;
  const type = fields.type === undefined ? inferredQuestionType(fields, name) : fields.type;
  if (type !== 'noul' && type !== 'choice' && type !== 'score')
    invalid(`Question ${name} type must be noul, choice, or score.`);
  const instructionValue = snapshotSourceContent(fields.instructions, `Question ${name} instructions`);
  const instructionText = renderContent(instructionValue, `Question ${name} instructions`);
  let options: Qwen35Option[];
  if (type === 'choice') {
    const entries = ownEntries(fields.criteria, `Question ${name} criteria`, true);
    if (entries.length < 2 || entries.length > QWEN35_MAX_CHOICE_OPTIONS)
      invalidRequest(`Question ${name} choice requires 2 to ${QWEN35_MAX_CHOICE_OPTIONS} options.`);
    options = entries.map(([label, description]) => {
      assertUnicode(label, 'Choice labels');
      const captured = snapshotSourceContent(description, 'Choice descriptions', true);
      return Object.freeze({
        label,
        sourceDescription: captured,
        description: renderContent(captured, 'Choice descriptions', true)
      });
    });
  } else if (type === 'score') {
    const levels = arrayValues(fields.criteria, `Question ${name} criteria`);
    if (levels.length < QWEN35_MIN_SCORE_LEVELS || levels.length > QWEN35_MAX_SCORE_LEVELS)
      invalidRequest(
        `Question ${name} score requires ${QWEN35_MIN_SCORE_LEVELS} to ${QWEN35_MAX_SCORE_LEVELS} levels.`
      );
    options = levels.map((level, index) => {
      if (typeof level !== 'string') invalid(`Question ${name} score level ${index} must be text.`);
      assertUnicode(level, 'Score levels');
      const captured = snapshotSourceContent(level, 'Score levels');
      return Object.freeze({label: String(index), sourceDescription: captured, description: strip(level)});
    });
  } else {
    const entries = fields.criteria == null ? [] : ownEntries(fields.criteria, `Question ${name} criteria`, true);
    if (entries.some(([key]) => key !== 'false' && key !== 'true'))
      invalid(`Question ${name} noul criteria keys must be a subset of false and true.`);
    const descriptions = new Map(entries);
    options = (['false', 'true'] as const).map((label) => {
      const description = descriptions.has(label) ? descriptions.get(label) : noulDefaults[label];
      const captured = snapshotSourceContent(description, 'Noul descriptions', true);
      return Object.freeze({
        label,
        sourceDescription: captured,
        description: renderContent(captured, 'Noul descriptions', true)
      });
    });
  }
  return Object.freeze({name, type, instructions: instructionValue, instructionText, options: Object.freeze(options)});
}

function renderOptionBlock(options: readonly Qwen35Option[]): {
  readonly text: string;
  readonly spans: Array<readonly [number, number]>;
} {
  const lines: string[] = [];
  const spans: Array<readonly [number, number]> = [];
  let cursor = 0;
  for (let index = 0; index < options.length; index++) {
    const option = options[index]!;
    const description = option.description.replace(collapseWhitespace, ' ').replace(/^ +| +$/gu, '');
    const line = `${index + 1}. ${option.label}${description ? ` — ${description}` : ''}`;
    lines.push(line);
    spans.push(Object.freeze([cursor, cursor + line.length] as const));
    cursor += line.length + 1;
  }
  return {text: lines.join('\n'), spans};
}

export function renderQwen35Question(question: Qwen35QuestionSnapshot): Qwen35RenderedQuestion {
  const headers = {
    choice: 'Select exactly one option.',
    score: 'Rate the state against the ordered levels below (lowest first).',
    noul: 'Decide whether the statement is true of the state.'
  };
  const options = renderOptionBlock(question.options);
  const prefix = `<question type="${question.type}">\n${headers[question.type]}\n${question.instructionText}\n<options>\n`;
  return Object.freeze({
    type: question.type,
    questionText: `${prefix}${options.text}\n</options>\n</question>\n<answer>`,
    optionSpans: Object.freeze(
      options.spans.map(([start, end]) => Object.freeze([prefix.length + start, prefix.length + end] as const))
    )
  });
}

function configuredMaxImages(options: Qwen35RequestOptions): number {
  const maxImages = options.maxImages ?? QWEN35_MAX_IMAGES;
  if (!Number.isSafeInteger(maxImages) || maxImages < 0)
    throw new RangeError('Qwen maxImages must be a non-negative safe integer.');
  return maxImages;
}

export function snapshotQwen35Request(request: unknown, options: Qwen35RequestOptions = {}): Qwen35RequestSnapshot {
  const fields = Object.fromEntries(ownEntries(request, 'Request')) as Qwen35SourceObject;
  const state = snapshotSourceContent(fields.state, 'State');
  const questionEntries = ownEntries(fields.questions, 'Questions', true);
  if (!questionEntries.length) invalid('Provide at least one named question.');
  const questions = questionEntries.map(([name, question]) => snapshotQuestion(question, name));
  const model = fields.model === undefined ? 'strands-decider-latest' : fields.model;
  if (typeof model !== 'string') invalid('Request model must be text.');
  const imageValues = fields.images === undefined ? [] : arrayValues(fields.images, 'Images');
  const maxImages = configuredMaxImages(options);
  if (imageValues.length > maxImages) invalidRequest(`Request accepts at most ${maxImages} images.`);
  const images = imageValues.map((image) => {
    if (typeof image !== 'string') invalid('Images must be base64 or data-URI strings.');
    return image;
  });
  return Object.freeze({
    model,
    state,
    stateText: `<state>\n${renderContent(state, 'State')}\n</state>\n`,
    questions: Object.freeze(questions),
    images: Object.freeze(images)
  });
}

export function renderQwen35State(state: Qwen35Content): string {
  return `<state>\n${renderContent(state, 'State')}\n</state>\n`;
}

function snapshotTokenized(encoded: unknown): Qwen35Tokenized {
  const source = encoded as Qwen35Encoded | null | undefined;
  const idsSource = source?.ids;
  const tokensSource = source?.tokens;
  if (!Array.isArray(idsSource) || !Array.isArray(tokensSource) || idsSource.length !== tokensSource.length)
    invalid('Qwen tokenizer must return matching indexed IDs and token strings.');
  const ids: number[] = [];
  const tokens: string[] = [];
  for (let index = 0; index < idsSource.length; index++) {
    if (!Object.hasOwn(idsSource, index) || !Object.hasOwn(tokensSource, index))
      invalid('Qwen tokenizer returned sparse IDs or tokens.');
    const id = idsSource[index];
    const token = tokensSource[index];
    if (!Number.isSafeInteger(id) || (id as number) < 0 || typeof token !== 'string')
      invalid('Qwen tokenizer returned invalid token data.');
    ids.push(id as number);
    tokens.push(token as string);
  }
  return {ids: Object.freeze(ids), tokens: Object.freeze(tokens)};
}

function encode(tokenizer: unknown, text: string, addSpecialTokens: boolean): Qwen35Tokenized {
  if (typeof (tokenizer as Qwen35Tokenizer | null | undefined)?.encode !== 'function')
    invalid('Qwen tokenizer is unavailable.');
  return snapshotTokenized((tokenizer as Qwen35Tokenizer).encode(text, {add_special_tokens: addSpecialTokens}));
}

function recoverTokenByteSpans(tokenizer: unknown, text: string, encoded: Qwen35Tokenized): Qwen35TokenSpan[] {
  const runtime = tokenizer as Qwen35Tokenizer | null | undefined;
  const decoder = runtime?.decoder?.byte_decoder;
  const added = runtime?.get_added_tokens_decoder?.();
  if (!decoder || !added) invalid('Expected a ByteLevel tokenizer with an added-token table.');
  const expected = encoder.encode(text.normalize('NFC'));
  const spans: Qwen35TokenSpan[] = [];
  let cursor = 0;
  for (let index = 0; index < encoded.ids.length; index++) {
    const token = encoded.tokens[index]!;
    const bytes = added.has(encoded.ids[index]!)
      ? encoder.encode(token)
      : Uint8Array.from(
          Array.from(token, (character) => {
            const byte = decoder[character];
            if (byte === undefined) invalid('Tokenizer returned a non-ByteLevel token.');
            return Number(byte);
          })
        );
    const start = cursor;
    for (const byte of bytes) {
      if (cursor >= expected.length || expected[cursor] !== byte)
        invalid('Tokenizer bytes do not match NFC prompt text.');
      cursor++;
    }
    spans.push(Object.freeze([start, cursor] as const));
  }
  if (cursor !== expected.length) invalid('Tokenizer did not cover the full prompt text.');
  return spans;
}

function optionIndices(
  rendered: Qwen35RenderedQuestion,
  tokenSpans: readonly Qwen35TokenSpan[],
  prefixLength: number
): number[] {
  return rendered.optionSpans.map(([start, end]) => {
    const prefix = rendered.questionText.slice(0, start).normalize('NFC');
    const option = rendered.questionText.slice(start, end).normalize('NFC');
    const suffix = rendered.questionText.slice(end).normalize('NFC');
    if (prefix + option + suffix !== rendered.questionText.normalize('NFC'))
      invalid('Unicode normalization crossed an option boundary.');
    const byteStart = encoder.encode(prefix).length;
    const byteEnd = byteStart + encoder.encode(option).length;
    let last = -1;
    for (let index = 0; index < tokenSpans.length; index++) {
      const [left, right] = tokenSpans[index]!;
      if (right > left && left >= byteStart && right <= byteEnd) last = index;
    }
    if (last < 0) invalid('An option has no surviving token for the pointer head.');
    return prefixLength + last;
  });
}

export function packQwen35Request(
  tokenizer: unknown,
  request: unknown,
  options: Qwen35RequestOptions = {}
): Qwen35PackedRequest {
  const snapshot = snapshotQwen35Request(request, options);
  return packQwen35RequestSnapshot(tokenizer, snapshot, options);
}

export function packQwen35RequestSnapshot(
  tokenizer: unknown,
  snapshot: Qwen35RequestSnapshot,
  options: Qwen35RequestOptions = {}
): Qwen35PackedRequest {
  const batchOptions = resolveQwen35BatchOptions(options);
  const maxLength = resolveQwen35MaxLength(options.maxLength);
  const encodedQuestions = snapshot.questions.map((question) => {
    const rendered = renderQwen35Question(question);
    const encoded = encode(tokenizer, rendered.questionText, false);
    return {question, rendered, encoded, spans: recoverTokenByteSpans(tokenizer, rendered.questionText, encoded)};
  });
  const state = encode(tokenizer, snapshot.stateText, true).ids;
  const rows: Qwen35PackedRow[] = [];
  const batchSizes: number[] = [];
  let tokenCount = 0;
  for (let start = 0; start < encodedQuestions.length; start += batchOptions.maxBatch) {
    const chunk = encodedQuestions.slice(start, start + batchOptions.maxBatch);
    const longest = chunk.reduce((length, item) => Math.max(length, item.encoded.ids.length), 0);
    const reserve = Math.min(longest, Math.max(1, Math.trunc(maxLength * batchOptions.maxQuestionFraction)));
    const stateBudget = Math.max(1, maxLength - reserve);
    const stateIds = state.slice(0, stateBudget);
    const fittedRows: Qwen35PackedRow[] = [];
    for (const item of chunk) {
      const cut = Math.max(0, item.encoded.ids.length - reserve);
      const questionIds = item.encoded.ids.slice(cut);
      const questionSpans = item.spans.slice(cut);
      const inputIds = [...stateIds, ...questionIds];
      if (inputIds.length > maxLength) invalid(`Fitted request exceeds the ${maxLength}-token context.`);
      fittedRows.push(
        Object.freeze({
          name: item.question.name,
          type: item.question.type,
          options: item.question.options,
          inputIds: Object.freeze(inputIds),
          optionIndices: Object.freeze(optionIndices(item.rendered, questionSpans, stateIds.length)),
          answerIndex: inputIds.length - 1,
          stateText: snapshot.stateText,
          questionText: item.rendered.questionText
        })
      );
    }
    const ranges = splitQwen35BatchRanges(
      fittedRows.map((row) => row.inputIds.length - stateIds.length),
      stateIds.length,
      batchOptions.maxBatchTokens
    );
    for (const [rangeStart, rangeEnd] of ranges) {
      const batchRows = fittedRows.slice(rangeStart, rangeEnd);
      rows.push(...batchRows);
      batchSizes.push(batchRows.length);
      tokenCount += stateIds.length;
      tokenCount += batchRows.reduce((sum, row) => sum + row.inputIds.length - stateIds.length, 0);
    }
  }
  return Object.freeze({
    ...snapshot,
    maxLength,
    rows: Object.freeze(rows),
    batchSizes: Object.freeze(batchSizes),
    tokenCount
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

function float32SoftmaxFromLogits(logits: readonly number[], temperature: number, slotCount: number): number[] {
  // Match Strands' inference path: its pointer head emits float32, the per-kind
  // temperature is materialized as float32, inactive batch slots are masked with -1e4,
  // and PyTorch applies float32 log_softmax before exponentiating its result.
  const f32 = Math.fround;
  const temperature32 = Math.max(f32(temperature), f32(1e-6));
  const scaled = logits.map((value, index) =>
    index < slotCount ? f32(f32(value) / temperature32) : QWEN35_MASKED_LOGIT
  );
  const maximum = Math.max(...scaled);
  const centered = scaled.map((value) => f32(value - maximum));
  const exps = centered.map((value) => f32(Math.exp(value)));
  const total = exps.reduce((sum, value) => f32(sum + value), f32(0));
  // PyTorch log_softmax propagates NaN across the row, including rows whose
  // temperature scaling overflows a finite float32 logit to +Infinity.
  if (Number.isNaN(total)) return Array.from({length: slotCount}, () => Number.NaN);
  if (!Number.isFinite(total) || total <= 0) invalid('Qwen returned invalid pointer logits.');
  const logTotal = f32(Math.log(total));
  const logProbabilities = centered.slice(0, slotCount).map((value) => f32(value - logTotal));
  return logProbabilities.map((value) => f32(Math.exp(value)));
}

function confidence(probabilities: readonly number[]): number {
  const count = probabilities.length;
  if (count <= 1) return 1;
  const maximum = Math.max(...probabilities);
  return clampPythonConfidence((count * maximum - 1) / (count - 1));
}

function clampPythonConfidence(value: number): number {
  // Strands uses max(0.0, min(1.0, value)); Python's ordered min returns its
  // first argument when `value` is NaN, so that expression yields 1.0.
  if (Number.isNaN(value)) return 1;
  return Math.max(0, Math.min(1, value));
}

function scoreConfidence(probabilities: readonly number[]): number {
  const count = probabilities.length;
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  if (count <= 1) return 1;
  if (total <= 0) return 0;
  const normalized = probabilities.map((value) => value / total);
  const mean = normalized.reduce((sum, value, index) => sum + index * value, 0);
  const variance = normalized.reduce((sum, value, index) => sum + value * (index - mean) ** 2, 0);
  const sigma = Math.sqrt(variance);
  const maximum = (count - 1) / 2;
  const floor = Math.sqrt(QWEN35_ORDINAL_SMOOTHING);
  if (maximum <= floor) return sigma <= floor ? 1 : 0;
  return clampPythonConfidence((maximum - sigma) / (maximum - floor));
}

function checkedLogits(logits: unknown, count: number): number[] {
  const length =
    !Array.isArray(logits) && !(ArrayBuffer.isView(logits) && !(logits instanceof DataView))
      ? -1
      : (logits as ArrayLike<unknown>).length;
  if (!Number.isSafeInteger(length) || length < count || length > QWEN35_MAX_CHOICE_OPTIONS)
    invalid(`Qwen pointer logits must contain ${count} to ${QWEN35_MAX_CHOICE_OPTIONS} values.`);
  const values: number[] = [];
  for (let index = 0; index < length; index++) {
    if (Array.isArray(logits) && !Object.hasOwn(logits, index)) invalid('Qwen returned sparse pointer logits.');
    const value = (logits as ArrayLike<unknown>)[index];
    if (typeof value !== 'number') invalid('Qwen returned non-numeric pointer logits.');
    values.push(value);
  }
  return values;
}

function checkedMaskedProbabilities(probabilities: unknown, count: number): number[] {
  if (
    (!Array.isArray(probabilities) && !(ArrayBuffer.isView(probabilities) && !(probabilities instanceof DataView))) ||
    (probabilities as ArrayLike<unknown>).length < count
  )
    invalid('Qwen returned fewer masked probabilities than the question has options.');
  const values: number[] = [];
  for (let index = 0; index < (probabilities as ArrayLike<unknown>).length; index++) {
    if (Array.isArray(probabilities) && !Object.hasOwn(probabilities, index))
      invalid('Qwen returned sparse masked probabilities.');
    const value = (probabilities as ArrayLike<unknown>)[index];
    if (typeof value !== 'number') invalid('Qwen returned a non-numeric masked probability.');
    if (index >= count && value !== 0) invalid('Qwen graph probabilities must be zero in masked option columns.');
    if (index < count) values.push(value);
  }
  return values;
}

function decodeQwen35Answer(question: Qwen35QuestionSnapshot, probabilities: readonly number[]): Qwen35Answer {
  const count = question.options.length;
  if (
    count < 2 ||
    (question.type === 'choice' && count > QWEN35_MAX_CHOICE_OPTIONS) ||
    (question.type === 'score' && (count < QWEN35_MIN_SCORE_LEVELS || count > QWEN35_MAX_SCORE_LEVELS)) ||
    (question.type === 'noul' && count !== 2)
  )
    invalid('Qwen question options do not match the typed protocol.');
  if (question.type === 'noul') return Object.freeze({type: 'noul', noul: roundPython(probabilities[1]!)});
  // Python dict output keeps integer-like option labels in source order. Keep
  // that order in the record itself so the public schema needs no helper field.
  const records = qwen35OrderedRecord(
    question.options.map(({label}, index) => [label, roundPython(probabilities[index]!)] as const)
  );
  if (question.type === 'choice') {
    // Python's max-by-key retains the first label on ties and when every key is
    // NaN, as it does for a NaN row from PyTorch log_softmax.
    let selectedIndex = 0;
    for (let index = 1; index < probabilities.length; index += 1)
      if (probabilities[index]! > probabilities[selectedIndex]!) selectedIndex = index;
    return Object.freeze({
      type: 'choice',
      choice: question.options[selectedIndex]!.label,
      probabilities: records,
      confidence: roundPython(confidence(probabilities))
    });
  }
  const score = probabilities.reduce((sum, probability, index) => sum + index * probability, 0);
  const legend = Object.fromEntries(question.options.map(({description}, index) => [String(index), description]));
  return Object.freeze({
    type: 'score',
    score: roundPython(score),
    legend: Object.freeze(legend),
    probabilities: records,
    confidence: roundPython(scoreConfidence(probabilities))
  });
}

/** Decode probabilities from the pinned graph's second, already masked output. */
export function finishQwen35Answer(question: Qwen35QuestionSnapshot, probabilities: unknown): Qwen35Answer {
  return decodeQwen35Answer(question, checkedMaskedProbabilities(probabilities, question.options.length));
}

/** Raw-logit postprocessor for CPU/source reference probes; graph adapters must use masked probabilities. */
export function finishQwen35AnswerFromLogits(question: Qwen35QuestionSnapshot, logits: unknown): Qwen35Answer {
  const slotCount = question.options.length;
  const probabilities = float32SoftmaxFromLogits(
    checkedLogits(logits, slotCount),
    QWEN35_TEMPERATURE_BY_KIND[question.type],
    slotCount
  );
  return decodeQwen35Answer(question, probabilities);
}

function readOutputChunk(value: unknown, chunkIndex: number, maxBatch: number, remainingRows: number): unknown[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > maxBatch)
    invalid(`Qwen output chunk ${chunkIndex} must contain 1 to ${maxBatch} rows.`);
  if (value.length > remainingRows) invalid('Qwen returned the wrong number of question output rows.');
  const rows: unknown[] = [];
  for (let rowIndex = 0; rowIndex < value.length; rowIndex += 1) {
    if (!Object.hasOwn(value, rowIndex)) invalid(`Qwen output chunk ${chunkIndex} has a sparse row.`);
    rows.push(value[rowIndex]);
  }
  return rows;
}

function readQuestionOutputRows(chunks: unknown, maxBatch: number, questionCount: number): unknown[] {
  if (!Array.isArray(chunks) || !chunks.length) invalid('Qwen returned no question output chunks.');
  const rows: unknown[] = [];
  for (let index = 0; index < chunks.length; index += 1) {
    if (!Object.hasOwn(chunks, index)) invalid('Qwen returned sparse question output chunks.');
    const chunk = readOutputChunk(chunks[index], index, maxBatch, questionCount - rows.length);
    for (const row of chunk) rows.push(row);
  }
  if (rows.length !== questionCount) invalid('Qwen returned the wrong number of question output rows.');
  return rows;
}

function finishQwen35AnswerChunks(
  request: Pick<Qwen35RequestSnapshot, 'questions'>,
  chunkRows: unknown,
  options: Qwen35BatchOptions,
  decode: (question: Qwen35QuestionSnapshot, row: unknown) => Qwen35Answer
): readonly Qwen35NamedAnswerRow[] {
  const questions = request.questions;
  const {maxBatch} = resolveQwen35BatchOptions(options);
  if (!questions.length) invalid('Qwen request has no questions.');
  const rows = readQuestionOutputRows(chunkRows, maxBatch, questions.length);
  return Object.freeze(
    rows.map((row, index) => {
      const question = questions[index]!;
      return Object.freeze({name: question.name, answer: decode(question, row)});
    })
  );
}

/** Decode post-temperature, n_slots-masked graph probability rows by question order. */
export function finishQwen35Answers(
  request: Pick<Qwen35RequestSnapshot, 'questions'>,
  chunkProbabilities: unknown,
  options: Qwen35BatchOptions = {}
): readonly Qwen35NamedAnswerRow[] {
  return finishQwen35AnswerChunks(request, chunkProbabilities, options, finishQwen35Answer);
}

/** Raw-logit postprocessor for CPU/source reference probes. */
export function finishQwen35AnswersFromLogits(
  request: Pick<Qwen35RequestSnapshot, 'questions'>,
  chunkLogits: unknown,
  options: Qwen35BatchOptions = {}
): readonly Qwen35NamedAnswerRow[] {
  return finishQwen35AnswerChunks(request, chunkLogits, options, finishQwen35AnswerFromLogits);
}
