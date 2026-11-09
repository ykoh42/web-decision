import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {CRIVO_MAX_SEQUENCE_TOKENS} from './contract.mjs';
import type {CRIVO_MODEL_FP32, CRIVO_MODEL_INT8} from './contract.mjs';

export interface CrivoTokenizer {
  readonly special: {
    readonly cls: number;
    readonly sep: number;
    readonly pad: number;
    readonly opt: number;
  };
  encode(text: string): number[];
}

export interface CrivoPreparedRow {
  readonly inputIds: number[];
  readonly optionPositions: number[];
  readonly qtype: number;
}

export type CrivoSystemOneRow = CrivoPreparedRow &
  (
    | {readonly id: string; readonly type: 'choice'; readonly labels: string[]; readonly optionTexts: string[]}
    | {readonly id: string; readonly type: 'noul'; readonly labels: null; readonly optionTexts: string[]}
    | {readonly id: string; readonly type: 'score'; readonly labels: null; readonly optionTexts: string[]}
  );

export interface PreparedCrivoSystemOne {
  readonly rows: CrivoSystemOneRow[];
  readonly tokenCount: number;
}

export interface PreparedCrivoDecision {
  readonly rows: (CrivoPreparedRow & {readonly labels: string[]})[];
  readonly tokenCount: number;
  readonly questions: {readonly id: string; readonly labels: string[]}[];
}

export interface CrivoAbortSignalLike {
  readonly aborted?: boolean;
  readonly reason?: unknown;
  throwIfAborted(): void;
  addEventListener: AbortSignal['addEventListener'];
  removeEventListener: AbortSignal['removeEventListener'];
}

export type CrivoModelName = typeof CRIVO_MODEL_FP32 | typeof CRIVO_MODEL_INT8;

export type CrivoSystemOneResult = {
  readonly model: CrivoModelName;
  readonly tokenCount: number;
  readonly answers: Readonly<
    Record<
      string,
      | {readonly type: 'choice'; readonly choice: string; readonly probabilities: Readonly<Record<string, number>>}
      | {readonly type: 'noul'; readonly noul: number}
      | {readonly type: 'score'; readonly score: number; readonly probabilities: Readonly<Record<string, number>>}
    >
  >;
};

export interface CrivoDecisionResult {
  readonly tokenCount: number;
  readonly answers: readonly {
    readonly selected: string;
    readonly selectedIndex: number;
    readonly probabilities: readonly {readonly option: string; readonly probability: number; readonly logit: number}[];
  }[];
}

type CrivoQuestionType = keyof typeof QUESTION_TYPES;
type CrivoRecordEntry = [string, unknown];

interface CrivoQuestion {
  readonly type?: unknown;
  readonly instructions?: unknown;
  readonly criteria?: unknown;
}

interface CrivoOptionTexts {
  readonly labels: string[] | null;
  readonly texts: string[];
}

interface CrivoCheckedQuestion {
  readonly id: string;
  readonly question: CrivoQuestion;
  readonly labels: string[] | null;
  readonly texts: string[];
}

interface CrivoInputQuestion {
  readonly id: string;
  readonly question: {readonly type: 'choice'; readonly instructions: string};
  readonly labels: string[];
  readonly texts: string[];
}

interface CrivoValidatedRequest {
  readonly state: unknown;
  readonly questions: Array<{readonly instructions: unknown; readonly options: unknown}>;
}

const QUESTION_TYPES = Object.freeze({noul: 0, choice: 1, score: 2});
const QUESTION_FIELDS = new Set(['type', 'instructions', 'criteria']);
const fail = (message: string): never => {
  throw new WebDecisionError('INVALID_INPUT', message);
};

function recordEntries(value: unknown, label: string): CrivoRecordEntry[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(label + ' must be an object.');
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) fail(label + ' must be a plain object.');
  const result: CrivoRecordEntry[] = [];
  for (const key of Reflect.ownKeys(value as object)) {
    const descriptor = Object.getOwnPropertyDescriptor(value as object, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      fail(label + ' must contain enumerable string data properties.');
    result.push([key as string, (descriptor as PropertyDescriptor).value]);
  }
  return result;
}

function arrayValues(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) fail(label + ' must be an array.');
  const values: unknown[] = [];
  for (let index = 0; index < (value as unknown[]).length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value as object, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      fail(label + ' cannot contain holes or accessors.');
    values.push((descriptor as PropertyDescriptor).value);
  }
  for (const key of Reflect.ownKeys(value as object)) {
    if (key === 'length') continue;
    if (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= (value as unknown[]).length)
      fail(label + ' cannot have extra properties.');
  }
  return values;
}

function compareCodePoints(left: string, right: string): number {
  const a = Array.from(left, (character) => character.codePointAt(0));
  const b = Array.from(right, (character) => character.codePointAt(0));
  const length = Math.min(a.length, b.length);
  for (let index = 0; index < length; index++) {
    if (a[index] !== b[index]) return (a[index] as number) - (b[index] as number);
  }
  return a.length - b.length;
}

function jsonValue(value: unknown, ancestors: Set<object> = new Set()): string {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value) as string;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail('Crivo JSON numbers must be finite.');
    return JSON.stringify(value) as string;
  }
  if (!value || typeof value !== 'object') fail('Crivo values must be JSON values.');
  if (ancestors.has(value as object)) fail('Crivo JSON values cannot contain cycles.');
  ancestors.add(value as object);
  const result = Array.isArray(value)
    ? '[' +
      arrayValues(value, 'Crivo JSON arrays')
        .map((item) => jsonValue(item, ancestors))
        .join(',') +
      ']'
    : '{' +
      recordEntries(value, 'Crivo JSON objects')
        .sort(([left], [right]) => compareCodePoints(left, right))
        .map(([key, item]) => JSON.stringify(key) + ':' + jsonValue(item, ancestors))
        .join(',') +
      '}';
  ancestors.delete(value as object);
  return result;
}

function render(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value.trim();
  return jsonValue(value);
}

function encodeText(tokenizer: CrivoTokenizer, value: unknown): number[] {
  const text = render(value);
  return tokenizer.encode(text);
}

function validateOptionCount(count: number, label: string): void {
  if (count < 2) fail(label + (label === 'Crivo Choice' ? ' needs at least 2 options.' : ' needs at least 2 levels.'));
}

function choiceOptionTexts(question: CrivoQuestion): CrivoOptionTexts {
  const entries = recordEntries(question.criteria, 'Crivo choice criteria');
  validateOptionCount(entries.length, 'Crivo Choice');
  const labels: string[] = [];
  const texts: string[] = [];
  for (const [key, value] of entries) {
    jsonValue(value);
    const description = render(value);
    labels.push(key);
    texts.push(description && description !== key ? key + ': ' + description : String(key));
  }
  return {labels, texts};
}

function noulOptionTexts(question: CrivoQuestion): CrivoOptionTexts {
  const descriptions = question.criteria == null ? [] : recordEntries(question.criteria, 'Crivo Noul criteria');
  if (descriptions.some(([key]) => key !== 'false' && key !== 'true'))
    fail('Crivo Noul criteria can only describe false and true.');
  const values = Object.fromEntries(descriptions) as Record<string, unknown>;
  for (const value of Object.values(values)) jsonValue(value);
  const falseText = render(values.false);
  const trueText = render(values.true);
  return {
    labels: null,
    texts: [falseText ? 'false: ' + falseText : 'No.', trueText ? 'true: ' + trueText : 'Yes.']
  };
}

function scoreOptionTexts(question: CrivoQuestion): CrivoOptionTexts {
  const criteria = Array.isArray(question.criteria)
    ? arrayValues(question.criteria, 'Crivo Score criteria')
    : recordEntries(question.criteria, 'Crivo Score criteria').map(([, value]) => value);
  validateOptionCount(criteria.length, 'Crivo Score');
  for (const value of criteria) jsonValue(value);
  return {labels: null, texts: criteria.map((value, index) => index + ': ' + render(value))};
}

const OPTION_TEXT_BUILDERS: Readonly<Record<CrivoQuestionType, (question: CrivoQuestion) => CrivoOptionTexts>> =
  Object.freeze({
    choice: choiceOptionTexts,
    noul: noulOptionTexts,
    score: scoreOptionTexts
  });

function optionTexts(question: CrivoQuestion): CrivoOptionTexts {
  const build = OPTION_TEXT_BUILDERS[question.type as CrivoQuestionType];
  if (!build) fail('Crivo question type must be choice, noul, or score.');
  return build(question);
}

function checkedQuestion(id: string, value: unknown): CrivoCheckedQuestion {
  if (!id) fail('Crivo question names cannot be empty.');
  const entries = recordEntries(value, 'Crivo question ' + id);
  if (entries.some(([key]) => !QUESTION_FIELDS.has(key))) fail('Crivo question has an unsupported field.');
  const question = Object.fromEntries(entries) as Record<string, unknown>;
  if (typeof question.type !== 'string' || !Object.hasOwn(QUESTION_TYPES, question.type))
    fail('Crivo question type must be choice, noul, or score.');
  if (
    question.instructions !== undefined &&
    question.instructions !== null &&
    typeof question.instructions !== 'string' &&
    typeof question.instructions !== 'boolean' &&
    typeof question.instructions !== 'number' &&
    typeof question.instructions !== 'object'
  )
    fail('Crivo instructions must be a JSON value.');
  jsonValue(question.instructions ?? null);
  const options = optionTexts(question as CrivoQuestion);
  return {id, question, ...options};
}

function encodeQuestion(
  state: unknown,
  item: CrivoCheckedQuestion | CrivoInputQuestion,
  tokenizer: CrivoTokenizer
): CrivoSystemOneRow {
  const {special} = tokenizer;
  const instructions = encodeText(tokenizer, item.question.instructions);
  const stateIds = encodeText(tokenizer, state);
  const inputIds = [special.cls, ...instructions, special.sep, ...stateIds, special.sep];
  const optionPositions: number[] = [];
  for (const text of item.texts) {
    optionPositions.push(inputIds.length);
    inputIds.push(special.opt, ...tokenizer.encode(text));
  }
  if (inputIds.length > CRIVO_MAX_SEQUENCE_TOKENS)
    fail(
      'Crivo question ' +
        item.id +
        ' needs ' +
        inputIds.length +
        ' tokens; the limit is ' +
        CRIVO_MAX_SEQUENCE_TOKENS +
        '.'
    );
  return {
    id: item.id,
    type: item.question.type as CrivoQuestionType,
    labels: item.labels,
    optionTexts: item.texts,
    inputIds,
    optionPositions,
    qtype: QUESTION_TYPES[item.question.type as CrivoQuestionType]
  } as CrivoSystemOneRow;
}

function stateText(state: unknown): string {
  if (typeof state === 'string') return state.trim();
  return render(state);
}

function prepareRows(
  state: unknown,
  questions: Array<CrivoCheckedQuestion | CrivoInputQuestion>,
  tokenizer: CrivoTokenizer
): {rows: CrivoSystemOneRow[]; tokenCount: number} {
  const text = stateText(state);
  const rows = questions.map((question) => encodeQuestion(text, question, tokenizer));
  return {rows, tokenCount: rows.reduce((total, row) => total + row.inputIds.length, 0)};
}

export function prepareCrivoSystemOne(
  state: unknown,
  questions: unknown,
  tokenizer: CrivoTokenizer | null | undefined
): PreparedCrivoSystemOne {
  if (typeof tokenizer?.encode !== 'function' || !tokenizer.special)
    throw new TypeError('Crivo tokenizer is required.');
  const entries = recordEntries(questions, 'Crivo questions');
  if (entries.length < 1) fail('Crivo needs at least one question.');
  const checked = entries.map(([id, question]) => checkedQuestion(id, question));
  return prepareRows(state, checked, tokenizer);
}

export function prepareCrivoDecision(
  request: unknown,
  tokenizer: CrivoTokenizer | null | undefined
): PreparedCrivoDecision {
  if (typeof tokenizer?.encode !== 'function' || !tokenizer.special)
    throw new TypeError('Crivo tokenizer is required.');
  const snapshot = validateRequestFields(request, Infinity, Infinity, Infinity, Infinity, Infinity, {
    dataPropertiesOnly: true,
    allowInheritedDataProperties: true
  }) as CrivoValidatedRequest;
  if (typeof snapshot.state !== 'string' || snapshot.questions.length < 1)
    fail('Crivo decisions need a string state and at least one question.');
  const questions: CrivoInputQuestion[] = snapshot.questions.map((question, index) => {
    if (
      typeof question.instructions !== 'string' ||
      !Array.isArray(question.options) ||
      question.options.length < 2 ||
      (question.options as unknown[]).some((option: unknown) => typeof option !== 'string')
    )
      fail('Crivo decision questions need text instructions and at least 2 string options.');
    return {
      id: String(index),
      question: {type: 'choice', instructions: question.instructions as string},
      labels: [...(question.options as string[])],
      texts: [...(question.options as string[])]
    };
  });
  return {
    ...(prepareRows(snapshot.state, questions, tokenizer) as {
      rows: Array<CrivoPreparedRow & {readonly labels: string[]}>;
      tokenCount: number;
    }),
    questions: questions.map(({id, labels}) => ({id, labels}))
  };
}

function probabilities(logits: unknown): number[] {
  if (!Array.isArray(logits) && !ArrayBuffer.isView(logits))
    throw new WebDecisionError('INFERENCE_ERROR', 'Crivo logits are invalid.');
  const values = Array.from(logits as ArrayLike<number>);
  if (values.length < 2 || values.some((value) => !Number.isFinite(value)))
    throw new WebDecisionError('INFERENCE_ERROR', 'Crivo logits are not finite.');
  const max = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function bestIndex(values: number[]): number {
  let best = 0;
  for (let index = 1; index < values.length; index++) if (values[index]! > values[best]!) best = index;
  return best;
}

export function finishCrivoSystemOne(
  prepared: PreparedCrivoSystemOne,
  logits: readonly Float32Array[],
  model: CrivoModelName
): CrivoSystemOneResult {
  if (!Array.isArray(logits) || logits.length !== prepared.rows.length)
    throw new WebDecisionError('INFERENCE_ERROR', 'Crivo returned an unexpected number of question rows.');
  const answers = prepared.rows.map((row, index) => {
    const values = probabilities(logits[index]);
    if (values.length !== row.optionPositions.length)
      throw new WebDecisionError('INFERENCE_ERROR', 'Crivo returned an unexpected option count.');
    if (row.type === 'choice') {
      const choiceProbabilities = Object.fromEntries(row.labels.map((label, i) => [label, values[i]!])) as Record<
        string,
        number
      >;
      return [row.id, {type: 'choice', choice: row.labels[bestIndex(values)]!, probabilities: choiceProbabilities}];
    }
    if (row.type === 'noul') return [row.id, {type: 'noul', noul: values[1]!}];
    return [
      row.id,
      {
        type: 'score',
        score: values.reduce((total, probability, level) => total + level * probability, 0),
        probabilities: Object.fromEntries(values.map((probability, level) => [String(level), probability]))
      }
    ];
  });
  return {
    model,
    tokenCount: prepared.tokenCount,
    answers: Object.fromEntries(answers) as CrivoSystemOneResult['answers']
  };
}

export function finishCrivoDecision(
  prepared: PreparedCrivoDecision,
  logits: readonly Float32Array[]
): CrivoDecisionResult {
  if (!Array.isArray(logits) || logits.length !== prepared.rows.length)
    throw new WebDecisionError('INFERENCE_ERROR', 'Crivo returned an unexpected number of question rows.');
  return {
    tokenCount: prepared.tokenCount,
    answers: prepared.rows.map((row, index) => {
      const values = probabilities(logits[index]);
      if (values.length !== row.optionPositions.length)
        throw new WebDecisionError('INFERENCE_ERROR', 'Crivo returned an unexpected option count.');
      const selectedIndex = bestIndex(values);
      return {
        selected: row.labels[selectedIndex]!,
        selectedIndex,
        probabilities: row.labels.map((option, optionIndex) => ({
          option,
          probability: values[optionIndex]!,
          logit: (logits[index] as Float32Array)[optionIndex]!
        }))
      };
    })
  };
}

export function validateCrivoCallOptions(options: unknown): CrivoAbortSignalLike | null | undefined {
  validateDecisionFields(options);
  if (options == null) return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(options as object, 'signal');
  if (descriptor && !Object.hasOwn(descriptor, 'value')) fail('Crivo does not accept signal accessors.');
  const signal = descriptor?.value as CrivoAbortSignalLike | null | undefined;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    fail('Crivo signal must be an AbortSignal.');
  return signal;
}
