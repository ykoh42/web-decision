import {WebDecisionError} from '../kev.js';
import {D1_OMNI_MAX_TOKENS, D1_OMNI_MODEL} from './contract.mjs';
import type {D1OmniTokenizer} from './tokenizer.mjs';

interface D1OmniQuestionValues {
  readonly type?: unknown;
  readonly instructions?: unknown;
  readonly criteria?: unknown;
}

type D1OmniOptionEntry = [string, unknown];

interface D1OmniQuestionDraft {
  readonly id: string;
  readonly type: 'choice' | 'score' | 'noul';
  readonly qtype: number;
  readonly labels: string[];
  readonly options: string[];
  readonly legend?: string[];
}

interface D1OmniPreparedQuestion extends D1OmniQuestionDraft {
  readonly instructions: string;
}

interface D1OmniEncodedParts {
  readonly ids: number[];
  readonly markerPositions: number[];
  readonly truncated: boolean;
}

interface D1OmniEncodedQuestion {
  readonly ids: number[];
  readonly markerPositions: number[];
  readonly truncated: boolean;
  readonly question: D1OmniPreparedQuestion;
}

export interface D1OmniPreparedSystemOne {
  readonly model: string;
  readonly rows: D1OmniEncodedQuestion[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

interface D1OmniAnswerBase {
  readonly confidence: number;
  readonly probabilities: Record<string, number>;
}

interface D1OmniChoiceAnswer extends D1OmniAnswerBase {
  readonly type: 'choice';
  readonly choice: string;
}

interface D1OmniScoreAnswer extends D1OmniAnswerBase {
  readonly type: 'score';
  readonly score: number;
  readonly expected: number;
  readonly mostLikelyLevel: number;
  readonly legend: Record<string, string>;
}

interface D1OmniNoulAnswer extends D1OmniAnswerBase {
  readonly type: 'noul';
  readonly noul: number;
}

type D1OmniAnswer = D1OmniChoiceAnswer | D1OmniScoreAnswer | D1OmniNoulAnswer;

export interface D1OmniSystemOneResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly outputTokenCount: number;
  readonly truncated?: true;
  readonly answers: Record<string, D1OmniAnswer>;
}

const QUESTION_FIELDS = new Set(['type', 'instructions', 'criteria']);
const QUESTION_TYPES = Object.freeze({choice: 0, score: 1, noul: 2});
const TOKENS = Object.freeze({state: 17, question: 18, option: 19, optionEnd: 20, decide: 21, mask: 16, bos: 1});
const SPECIAL_TEXT = /<\|([A-Za-z0-9_]+)\|>/g;
const OPTION_TOKEN_BUDGET = 24;

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const inference = (message: string): WebDecisionError => new WebDecisionError('INFERENCE_ERROR', message);

function ownEntries(value: unknown, id: string, label: string): D1OmniOptionEntry[] {
  if (Array.isArray(value)) {
    return (value as unknown[]).map((item, index) => {
      if (!Object.hasOwn(value, index)) throw invalid(`Question ${id} ${label} cannot contain holes.`);
      const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
      if (!descriptor || !Object.hasOwn(descriptor, 'value'))
        throw invalid(`Question ${id} ${label} cannot contain accessors.`);
      return [String(index), descriptor.value] as D1OmniOptionEntry;
    });
  }
  if (
    !value ||
    typeof value !== 'object' ||
    (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)
  )
    throw invalid(`Question ${id} ${label} must be a plain object.`);
  return Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`Question ${id} ${label} cannot contain accessors or symbol keys.`);
    return [key, descriptor.value] as D1OmniOptionEntry;
  });
}

function serializeArray(value: unknown[], seen: Set<object>): string {
  const items: string[] = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid('d1-omni arrays cannot contain holes or accessors.');
    items.push(serializeJson(descriptor.value, seen));
  }
  return `[${items.join(', ')}]`;
}

function serializeObject(value: object, seen: Set<object>): string {
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null)
    throw invalid('d1-omni state and criteria must contain plain JSON values.');
  const entries = Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid('d1-omni objects cannot contain accessors or symbol keys.');
    return `${JSON.stringify(key)}: ${serializeJson(descriptor.value, seen)}`;
  });
  return `{${entries.join(', ')}}`;
}

function serializeJson(value: unknown, seen: Set<object> = new Set()): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value) as string;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw invalid('d1-omni JSON values must contain finite numbers.');
    return JSON.stringify(value) as string;
  }
  if (!value || typeof value !== 'object') throw invalid('d1-omni state and criteria must contain JSON values.');
  if (seen.has(value)) throw invalid('d1-omni state and criteria cannot contain cycles.');
  seen.add(value);
  const result = Array.isArray(value) ? serializeArray(value, seen) : serializeObject(value, seen);
  seen.delete(value);
  return result;
}

function renderValue(value: unknown): string {
  return typeof value === 'string' ? value : serializeJson(value);
}

function escapeText(value: string): string {
  return value.replace(SPECIAL_TEXT, '<¦$1¦>');
}

function tokenizerIds(tokenizer: D1OmniTokenizer, text: string): number[] {
  const ids = tokenizer.encode(escapeText(text));
  if (!Array.isArray(ids) || ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 65536))
    throw new WebDecisionError('INVALID_ASSET', 'd1-omni tokenizer returned invalid or out-of-range token IDs.');
  return ids;
}

function snapshotQuestions(questions: unknown): Array<{id: string; value: D1OmniQuestionValues}> {
  if (!questions || typeof questions !== 'object' || Array.isArray(questions))
    throw invalid('d1-omni questions must be a named object.');
  const ids = Reflect.ownKeys(questions) as string[];
  if (ids.length < 1 || ids.some((id) => typeof id !== 'string'))
    throw invalid('d1-omni needs at least one named question with string IDs.');
  return ids.map((id) => {
    const descriptor = Object.getOwnPropertyDescriptor(questions, id);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw invalid(`Question ${id} cannot be an accessor.`);
    const question = descriptor.value;
    if (!question || typeof question !== 'object' || Array.isArray(question))
      throw invalid(`Question ${id} must be an object.`);
    const values: Record<string, unknown> = Object.create(null);
    for (const key of Reflect.ownKeys(question)) {
      const field = Object.getOwnPropertyDescriptor(question, key);
      if (typeof key !== 'string' || !QUESTION_FIELDS.has(key) || !field || !Object.hasOwn(field, 'value'))
        throw invalid(`Question ${id} has an unsupported field.`);
      values[key] = field.value;
    }
    if (!Object.hasOwn(QUESTION_TYPES, values.type as PropertyKey))
      throw invalid(`Question ${id} type must be choice, score, or noul.`);
    if (typeof values.instructions !== 'string' || !values.instructions.trim())
      throw invalid(`Question ${id} needs non-empty string instructions.`);
    return {id, value: values as D1OmniQuestionValues};
  });
}

function prepareChoice(id: string, question: D1OmniQuestionValues): D1OmniQuestionDraft {
  const criteria: D1OmniOptionEntry[] = Array.isArray(question.criteria)
    ? Array.from({length: (question.criteria as unknown[]).length}, (_, index) => {
        const descriptor = Object.getOwnPropertyDescriptor(question.criteria as object, String(index));
        if (!descriptor || !Object.hasOwn(descriptor, 'value'))
          throw invalid(`Question ${id} criteria cannot contain holes or accessors.`);
        if (typeof descriptor.value !== 'string') throw invalid(`Question ${id} choice labels must be strings.`);
        return [descriptor.value, null] as D1OmniOptionEntry;
      })
    : ownEntries(question.criteria, id, 'criteria');
  if (criteria.length < 2) throw invalid(`Question ${id} needs at least 2 choice options.`);
  const labels = criteria.map(([label]) => label);
  if (labels.some((label) => !label.trim()) || new Set(labels).size !== labels.length)
    throw invalid(`Question ${id} choice labels must be distinct non-empty strings.`);
  return {
    id,
    type: 'choice',
    qtype: QUESTION_TYPES.choice,
    labels,
    options: criteria.map(([label, description]) =>
      description == null || description === '' ? label : `${label}: ${renderValue(description)}`
    )
  };
}

function prepareScore(id: string, question: D1OmniQuestionValues): D1OmniQuestionDraft {
  if (
    !Array.isArray(question.criteria) ||
    (question.criteria as unknown[]).length < 2 ||
    (question.criteria as unknown[]).length > 10
  )
    throw invalid(`Question ${id} score needs 2 to 10 ordered levels.`);
  return {
    id,
    type: 'score',
    qtype: QUESTION_TYPES.score,
    labels: (question.criteria as unknown[]).map((_, index) => String(index)),
    legend: (question.criteria as unknown[]).map(renderValue),
    options: (question.criteria as unknown[]).map((criterion, index) => `level ${index}: ${renderValue(criterion)}`)
  };
}

function prepareNoul(id: string, question: D1OmniQuestionValues): D1OmniQuestionDraft {
  if (question.criteria != null && (typeof question.criteria !== 'object' || Array.isArray(question.criteria)))
    throw invalid(`Question ${id} noul criteria must be an object.`);
  const criteria = question.criteria == null ? [] : ownEntries(question.criteria, id, 'noul criteria');
  const values = new Map(criteria);
  for (const [key] of criteria)
    if (!['false', 'true', 'no', 'yes'].includes(key))
      throw invalid(`Question ${id} noul criteria accepts true and false only.`);
  const falseValue = values.has('false') ? values.get('false') : values.get('no');
  const trueValue = values.has('true') ? values.get('true') : values.get('yes');
  return {
    id,
    type: 'noul',
    qtype: QUESTION_TYPES.noul,
    labels: ['false', 'true'],
    options: [
      `false: ${falseValue == null || falseValue === '' ? 'no, the statement does not hold' : renderValue(falseValue)}`,
      `true: ${trueValue == null || trueValue === '' ? 'yes, the statement holds' : renderValue(trueValue)}`
    ]
  };
}

function prepareQuestion(id: string, question: D1OmniQuestionValues): D1OmniPreparedQuestion {
  const prepared =
    question.type === 'choice'
      ? prepareChoice(id, question)
      : question.type === 'score'
        ? prepareScore(id, question)
        : prepareNoul(id, question);
  return {...prepared, instructions: question.instructions as string};
}

function encodeQuestion(
  stateIds: number[],
  question: D1OmniPreparedQuestion,
  tokenizer: D1OmniTokenizer,
  maxTokens: number
): D1OmniEncodedParts {
  const optionCount = question.options.length;
  const optionBudget = Math.max(96, Math.min(optionCount * OPTION_TOKEN_BUDGET + 32, Math.floor(maxTokens / 2)));
  const perOption = Math.max(2, Math.floor((optionBudget - 3 * optionCount) / optionCount));
  const instructionIds = [TOKENS.question, ...tokenizerIds(tokenizer, question.instructions)];
  const headLimit = Math.max(16, optionBudget);
  let truncated = instructionIds.length > headLimit;
  const head = instructionIds.slice(0, headLimit);
  const markerPositions: number[] = [];
  for (const option of question.options) {
    markerPositions.push(head.length + 1);
    const optionIds = tokenizerIds(tokenizer, ` ${option}`);
    truncated ||= optionIds.length > perOption;
    head.push(TOKENS.option, TOKENS.mask, ...optionIds.slice(0, perOption), TOKENS.optionEnd);
  }
  head.push(TOKENS.decide);
  const room = Math.max(0, maxTokens - head.length - 2);
  const retainedState = [TOKENS.state, ...stateIds.slice(0, room)];
  const ids = [TOKENS.bos, ...retainedState, ...head].slice(0, maxTokens);
  const positions = markerPositions.map((position) => position + 1 + retainedState.length);
  if (positions.some((position) => position >= ids.length || ids[position] !== TOKENS.mask))
    throw invalid(`Question ${question.id} options do not fit in the ${maxTokens}-token context.`);
  truncated ||= retainedState.length < 1 + stateIds.length;
  return {ids, markerPositions: positions, truncated};
}

export function prepareD1OmniSystemOne(
  state: unknown,
  questions: unknown,
  tokenizer: D1OmniTokenizer | null | undefined,
  maxTokens = D1_OMNI_MAX_TOKENS
): D1OmniPreparedSystemOne {
  if (!tokenizer || typeof tokenizer.encode !== 'function' || !Number.isSafeInteger(maxTokens) || maxTokens < 64)
    throw new TypeError('d1-omni tokenizer and valid context size are required.');
  const stateText = state === null ? '' : typeof state === 'string' ? state : serializeJson(state);
  const stateIds = tokenizerIds(tokenizer, stateText);
  const preparedQuestions = snapshotQuestions(questions).map(({id, value}) => prepareQuestion(id, value));
  const rows = preparedQuestions.map((question) => ({
    ...encodeQuestion(stateIds, question, tokenizer, maxTokens),
    question
  }));
  const tokenCount = rows.reduce((sum, row) => sum + row.ids.length, 0);
  return {model: D1_OMNI_MODEL, rows, tokenCount, truncated: rows.some((row) => row.truncated)};
}

function optionGroup(count: number): string {
  if (count <= 2) return '2';
  if (count <= 5) return '3-5';
  if (count <= 10) return '6-10';
  return '11+';
}

function temperature(question: D1OmniPreparedQuestion, temperatures: Readonly<Record<string, number>>): number {
  return temperatures[`${question.type}:${optionGroup(question.options.length)}`] ?? temperatures[question.type] ?? 1;
}

function probabilities(
  logits: unknown,
  question: D1OmniPreparedQuestion,
  temperatures: Readonly<Record<string, number>>
): number[] {
  if (
    !Array.isArray(logits) ||
    logits.length !== question.options.length ||
    logits.some((value: unknown) => !Number.isFinite(value as number))
  )
    throw inference('d1-omni returned invalid option logits.');
  const scale = temperature(question, temperatures);
  const scaled = (logits as number[]).map((value) => value / scale);
  const max = Math.max(...scaled);
  const weights = scaled.map((value) => Math.exp(value - max));
  const total = weights.reduce((sum, value) => sum + value, 0);
  const values = weights.map((value) => value / total);
  return question.type === 'noul' ? values.reverse() : values;
}

function answer(question: D1OmniPreparedQuestion, values: number[]): D1OmniAnswer {
  const best = values.indexOf(Math.max(...values));
  const confidence = values[best]!;
  if (question.type === 'choice') {
    return {
      type: 'choice',
      choice: question.labels[best]!,
      confidence,
      probabilities: Object.fromEntries(question.labels.map((label, index) => [label, values[index]!])) as Record<
        string,
        number
      >
    };
  }
  if (question.type === 'score') {
    const expected = values.reduce((sum, value, index) => sum + index * value, 0);
    return {
      type: 'score',
      score: expected,
      expected,
      mostLikelyLevel: best,
      confidence,
      legend: Object.fromEntries(question.labels.map((label, index) => [label, question.legend![index]!])) as Record<
        string,
        string
      >,
      probabilities: Object.fromEntries(question.labels.map((label, index) => [label, values[index]!])) as Record<
        string,
        number
      >
    };
  }
  return {
    type: 'noul',
    noul: values[0]!,
    confidence,
    probabilities: {true: values[0]!, false: values[1]!}
  };
}

export function finishD1OmniSystemOne(
  prepared: D1OmniPreparedSystemOne,
  rows: unknown,
  temperatures: Readonly<Record<string, number>>
): D1OmniSystemOneResult {
  if (!Array.isArray(rows) || rows.length !== prepared.rows.length)
    throw inference('d1-omni returned an unexpected number of question rows.');
  const answers: Record<string, D1OmniAnswer> = Object.create(null);
  for (let index = 0; index < (rows as unknown[]).length; index++) {
    const item = prepared.rows[index]!;
    answers[item.question.id] = answer(
      item.question,
      probabilities((rows as unknown[])[index], item.question, temperatures)
    );
  }
  return {
    model: prepared.model,
    tokenCount: prepared.tokenCount,
    outputTokenCount: 0,
    ...(prepared.truncated ? {truncated: true} : {}),
    answers
  };
}
