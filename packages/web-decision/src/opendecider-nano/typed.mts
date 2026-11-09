import {WebDecisionError} from '../kev.js';
import {OPENDECIDER_NANO_MODEL} from './contract.mjs';

const invalid = (message: string): never => {
  throw new WebDecisionError('INVALID_INPUT', message);
};

type OpenDeciderQuestionType = 'choice' | 'score' | 'noul';

interface OpenDeciderOption {
  readonly name: string;
  readonly rendered: string;
  readonly label?: string | null;
}

interface OpenDeciderQuestionMetadata {
  readonly name: string;
  readonly type: OpenDeciderQuestionType;
  readonly options: OpenDeciderOption[];
}

export interface PreparedOpenDeciderNanoSystemOne {
  readonly request: {
    readonly state: string;
    readonly questions: readonly {readonly instructions: string; readonly options: readonly string[]}[];
  };
  readonly metadata: readonly OpenDeciderQuestionMetadata[];
}

interface OpenDeciderAnswerInput {
  readonly selectedIndex: number;
  readonly probabilities: readonly {readonly probability: number}[];
}

interface OpenDeciderPackedQuestion {
  readonly truncated?: boolean;
}

interface OpenDeciderPackedRequest {
  readonly questions: readonly OpenDeciderPackedQuestion[];
}

interface OpenDeciderInferenceResult {
  readonly tokenCount: number;
  readonly truncated?: boolean;
  readonly answers: readonly OpenDeciderAnswerInput[];
}

interface OpenDeciderChoiceAnswer {
  readonly type: 'choice';
  readonly choice: string;
  readonly probabilities: Record<string, number>;
  readonly confidence: number;
}

interface OpenDeciderScoreAnswer {
  readonly type: 'score';
  readonly score: number;
  readonly expected: number;
  readonly legend: Record<string, string>;
  readonly probabilities: Record<string, number>;
  readonly confidence: number;
}

interface OpenDeciderNoulAnswer {
  readonly type: 'noul';
  readonly noul: number;
  readonly probabilities: {readonly true: number; readonly false: number};
  readonly confidence: number;
}

type OpenDeciderAnswer = OpenDeciderChoiceAnswer | OpenDeciderScoreAnswer | OpenDeciderNoulAnswer;
type OpenDeciderFinalAnswer = OpenDeciderAnswer | (OpenDeciderAnswer & {readonly truncated: true});

export interface OpenDeciderNanoSystemOneResult {
  readonly model: typeof OPENDECIDER_NANO_MODEL;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Record<string, OpenDeciderFinalAnswer>;
}

function recordEntries(value: unknown, label: string): [string, unknown][] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  const record = value as object;
  const prototype = Object.getPrototypeOf(record);
  if (prototype !== null && Object.getPrototypeOf(prototype) !== null) invalid(`${label} must be a plain object.`);
  const entries: [string, unknown][] = [];
  for (const key of Reflect.ownKeys(record)) {
    const descriptor = Object.getOwnPropertyDescriptor(record, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    entries.push([key as string, (descriptor as PropertyDescriptor).value]);
  }
  return entries;
}

function arrayValues(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) invalid(`${label} must be an array.`);
  const array = value as unknown[];
  const values: unknown[] = [];
  for (let index = 0; index < array.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(array, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value')) invalid(`${label} must be a dense data array.`);
    values.push((descriptor as PropertyDescriptor).value);
  }
  for (const key of Reflect.ownKeys(array))
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= array.length))
      invalid(`${label} cannot have extra properties.`);
  return values;
}

function serializeStateValue(value: unknown, ancestors: Set<object> = new Set()): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value)!;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) invalid('State numbers must be finite.');
    return JSON.stringify(value)!;
  }
  if (!value || typeof value !== 'object') invalid('State must be a string or a JSON value.');
  const objectValue = value as object;
  if (ancestors.has(objectValue)) invalid('State cannot contain a cycle.');

  ancestors.add(objectValue);
  let serialized: string;
  if (Array.isArray(value)) {
    serialized = `[${arrayValues(value, 'State arrays')
      .map((item) => serializeStateValue(item, ancestors))
      .join(', ')}]`;
  } else {
    serialized = `{${recordEntries(objectValue, 'State objects')
      .map(([key, item]) => `${JSON.stringify(key)}: ${serializeStateValue(item, ancestors)}`)
      .join(', ')}}`;
  }
  ancestors.delete(objectValue);
  return serialized;
}

function serializeState(state: unknown): string {
  return typeof state === 'string' ? state : serializeStateValue(state);
}

function assertText(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim()) invalid(`${label} must be a non-empty string.`);
  return value as string;
}

function renderOption(name: string, description: unknown): string {
  if (description !== null && typeof description !== 'string') invalid('Option descriptions must be strings or null.');
  return description && description !== name ? `${name}: ${description}` : name;
}

function choiceOptions(criteria: unknown): OpenDeciderOption[] {
  let entries: [unknown, unknown][];
  if (Array.isArray(criteria)) {
    entries = arrayValues(criteria, 'Choice criteria').map((name): [unknown, unknown] => [name, null]);
  } else {
    entries = recordEntries(criteria, 'Choice criteria');
  }
  if (entries.length < 2) invalid('Choice criteria must contain at least 2 options.');
  const names = new Set<string>();
  const options = entries.map(([name, description]) => {
    const optionName = assertText(name, 'Choice option names');
    if (names.has(optionName)) invalid('Choice option names must be unique.');
    names.add(optionName);
    return {name: optionName, rendered: renderOption(optionName, description)};
  });
  if (new Set(options.map(({rendered}) => rendered)).size !== options.length)
    invalid('Choice criteria render to duplicate model options.');
  return options;
}

function scoreOptions(criteria: unknown): OpenDeciderOption[] {
  const levels = arrayValues(criteria, 'Score criteria');
  if (levels.length < 2) invalid('Score criteria must contain at least 2 levels.');
  return levels.map((description, index) => {
    const label = assertText(description, 'Score levels');
    const name = String(index);
    return {name, label, rendered: renderOption(name, label)};
  });
}

function noulOptions(criteria: unknown): OpenDeciderOption[] {
  let descriptions: Record<string, unknown> = {true: 'Yes', false: 'No'};
  if (criteria !== undefined) {
    const entries = recordEntries(criteria, 'Noul criteria');
    if (entries.some(([key]) => key !== 'true' && key !== 'false'))
      invalid('Noul criteria can only describe true and false.');
    descriptions = {...descriptions, ...Object.fromEntries(entries)};
  }
  return [
    {name: 'yes', rendered: renderOption('yes', descriptions.true)},
    {name: 'no', rendered: renderOption('no', descriptions.false)}
  ];
}

export function prepareOpenDeciderNanoSystemOne(state: unknown, questions: unknown): PreparedOpenDeciderNanoSystemOne {
  const entries = recordEntries(questions, 'Questions');
  if (entries.length < 1) invalid('Provide at least 1 named question.');

  const metadata: OpenDeciderQuestionMetadata[] = [];
  const rows = entries.map(([name, question]) => {
    if (!question || typeof question !== 'object' || Array.isArray(question))
      invalid(`Question ${name} must be an object.`);
    const fields = Object.fromEntries(recordEntries(question, `Question ${name}`));
    if (Object.keys(fields).some((key) => !['type', 'instructions', 'criteria'].includes(key)))
      invalid(`Question ${name} has an unsupported field.`);
    const instructions = assertText(fields.instructions, `Question ${name} instructions`);
    let type: OpenDeciderQuestionType;
    let options: OpenDeciderOption[];
    if (fields.type === 'choice') {
      type = 'choice';
      options = choiceOptions(fields.criteria);
    } else if (fields.type === 'score') {
      type = 'score';
      options = scoreOptions(fields.criteria);
    } else if (fields.type === 'noul') {
      type = 'noul';
      options = noulOptions(fields.criteria);
    } else {
      invalid(`Question ${name} type must be choice, score, or noul.`);
    }
    metadata.push({name, type: type!, options: options!});
    return {instructions, options: options!.map(({rendered}) => rendered)};
  });

  return {request: {state: serializeState(state), questions: rows}, metadata};
}

function probabilityRecord(
  options: readonly OpenDeciderOption[],
  answer: OpenDeciderAnswerInput
): Record<string, number> {
  return Object.fromEntries(options.map(({name}, index) => [name, answer.probabilities[index]!.probability]));
}

function finishAnswer(
  type: OpenDeciderQuestionType,
  options: readonly OpenDeciderOption[],
  answer: OpenDeciderAnswerInput
): OpenDeciderAnswer {
  const probabilities = probabilityRecord(options, answer);
  const confidence = Math.max(...Object.values(probabilities));
  if (type === 'choice') return {type, choice: options[answer.selectedIndex]!.name, probabilities, confidence};
  if (type === 'score')
    return {
      type,
      score: answer.selectedIndex,
      expected: options.reduce((sum, option, index) => sum + index * probabilities[option.name]!, 0),
      legend: Object.fromEntries(options.map(({name, label}, index) => [String(index), label ?? name])),
      probabilities,
      confidence
    };
  const yes = probabilities.yes!;
  return {type, noul: yes, probabilities: {true: yes, false: 1 - yes}, confidence: Math.max(yes, 1 - yes)};
}

export function finishOpenDeciderNanoSystemOne(
  prepared: PreparedOpenDeciderNanoSystemOne,
  result: OpenDeciderInferenceResult,
  packed: OpenDeciderPackedRequest
): OpenDeciderNanoSystemOneResult {
  const answers = Object.fromEntries(
    prepared.metadata.map(({name, type, options}, index) => {
      const answer = finishAnswer(type, options, result.answers[index]!);
      return [name, packed.questions[index]!.truncated ? {...answer, truncated: true} : answer];
    })
  );
  return {
    model: OPENDECIDER_NANO_MODEL,
    tokenCount: result.tokenCount,
    ...(result.truncated ? {truncated: true} : {}),
    answers
  };
}
