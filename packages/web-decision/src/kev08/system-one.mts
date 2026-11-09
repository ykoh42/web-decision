import {WebDecisionError} from '../kev.js';

type QuestionFields = Record<string, unknown>;
type TypedMetadata =
  | {name: string; type: 'choice'; keys: string[]}
  | {name: string; type: 'score'; legend: string[]}
  | {name: string; type: 'noul'};
interface PreparedQuestion {
  metadata: TypedMetadata;
  request: {instructions: string; options: string[]};
  allowDuplicateOptions?: number;
}
interface TypedPrepared {
  metadata: TypedMetadata[];
}
interface TypedAnswer {
  selectedIndex: number;
  probabilities: {probability: number}[];
}
interface TypedResult {
  tokenCount: number;
  answers: TypedAnswer[];
}

const MODEL_ID = 'kev-0.8b-f16';
const MAX_OPTIONS = 255;

function invalid(message: string): never {
  throw new WebDecisionError('INVALID_INPUT', message);
}

function recordEntries(value: unknown, label: string): [string, unknown][] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) invalid(`${label} must be a plain object.`);
  const entries: [string, unknown][] = [];
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
  const values = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value')) invalid(`${label} must be a dense data array.`);
    values.push(descriptor.value);
  }
  for (const key of Reflect.ownKeys(value))
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= value.length))
      invalid(`${label} cannot have extra properties.`);
  return values;
}

function render(value: unknown, indent = 0, ancestors = new Set<object>()): string {
  if (value === null) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) invalid('JSON numbers must be finite.');
    return String(value);
  }
  if (!value || typeof value !== 'object') invalid('Values must be JSON values.');
  if (ancestors.has(value)) invalid('JSON values cannot contain cycles.');
  if (indent > 256) invalid('JSON values are nested too deeply.');

  ancestors.add(value);
  const pad = '  '.repeat(indent);
  let text;
  if (Array.isArray(value)) {
    text = arrayValues(value, 'JSON arrays')
      .map((item) => `${pad}- ${render(item, indent + 1, ancestors).trimStart()}`)
      .join('\n');
  } else {
    text = recordEntries(value, 'JSON objects')
      .map(([key, item]) =>
        item && typeof item === 'object'
          ? `${pad}${key}:\n${render(item, indent + 1, ancestors)}`
          : `${pad}${key}: ${render(item, indent + 1, ancestors)}`
      )
      .join('\n');
  }
  ancestors.delete(value);
  return text;
}

function questionFields(value: unknown, name: string): QuestionFields {
  const entries = recordEntries(value, `Question ${name}`);
  const fields = Object.fromEntries(entries);
  if (entries.some(([key]) => !['type', 'instructions', 'criteria'].includes(key)))
    invalid(`Question ${name} has an unsupported field.`);
  if (Object.hasOwn(fields, 'instructions') && fields.instructions === undefined)
    invalid(`Question ${name} instructions must be a JSON value.`);
  return fields;
}

function instructions(fields: QuestionFields) {
  return Object.hasOwn(fields, 'instructions') ? render(fields.instructions) : '';
}

function optionText(name: string, description: unknown) {
  if (description === null || description === '') return name;
  return `${name}: ${render(description)}`;
}

function prepareChoice(name: string, fields: QuestionFields): PreparedQuestion {
  const criteria = recordEntries(fields.criteria, `Question ${name} criteria`);
  if (criteria.length < 1 || criteria.length > MAX_OPTIONS)
    invalid(`Question ${name} needs 1 to ${MAX_OPTIONS} Choice options.`);
  const keys = criteria.map(([key]) => key);
  const options = criteria.map(([key, description]) => optionText(key, description));
  return {
    metadata: {name, type: 'choice', keys},
    request: {instructions: instructions(fields), options}
  };
}

function prepareScore(name: string, fields: QuestionFields, index: number): PreparedQuestion {
  const values = arrayValues(fields.criteria, `Question ${name} criteria`);
  if (values.length < 1 || values.length > MAX_OPTIONS)
    invalid(`Question ${name} needs 1 to ${MAX_OPTIONS} Score levels.`);
  const legend = values.map((value) => render(value));
  return {
    metadata: {name, type: 'score', legend},
    request: {instructions: instructions(fields), options: legend},
    allowDuplicateOptions: index
  };
}

function prepareNoul(name: string, fields: QuestionFields): PreparedQuestion {
  const criteria = fields.criteria == null ? [] : recordEntries(fields.criteria, `Question ${name} criteria`);
  const descriptions = Object.fromEntries(criteria);
  const falseDescription = Object.hasOwn(descriptions, 'false') ? descriptions.false : null;
  const trueDescription = Object.hasOwn(descriptions, 'true') ? descriptions.true : null;
  return {
    metadata: {name, type: 'noul'},
    request: {
      instructions: instructions(fields),
      options: [optionText('no', falseDescription), optionText('yes', trueDescription)]
    }
  };
}

function prepareQuestion(name: string, value: unknown, index: number): PreparedQuestion {
  const fields = questionFields(value, name);
  if (fields.type === 'choice') return prepareChoice(name, fields);
  if (fields.type === 'score') return prepareScore(name, fields, index);
  if (fields.type === 'noul') return prepareNoul(name, fields);
  invalid(`Question ${name} type must be choice, score, or noul.`);
}

export function prepareKev08SystemOne(state: unknown, questions: unknown) {
  const entries = recordEntries(questions, 'Questions');
  if (!entries.length) invalid('Provide at least one named question.');
  const stateText = render(state);
  const metadata: TypedMetadata[] = [];
  const requestQuestions: {instructions: string; options: string[]}[] = [];
  const allowDuplicateOptionsFor = new Set<number>();
  for (const [name, value] of entries) {
    const prepared = prepareQuestion(name, value, requestQuestions.length);
    metadata.push(prepared.metadata);
    requestQuestions.push(prepared.request);
    if (prepared.allowDuplicateOptions !== undefined) allowDuplicateOptionsFor.add(prepared.allowDuplicateOptions);
  }
  return {
    metadata,
    request: {state: stateText, questions: requestQuestions},
    packing: {
      allowEmptyInstructions: true,
      allowSingleOption: true,
      allowEmptyOptions: true,
      allowDuplicateOptionsFor
    }
  };
}

function rounded(value: number) {
  return Number(value.toFixed(4));
}

function choiceConfidence(probabilities: number[]) {
  if (probabilities.length <= 1) return 1;
  const maximum = Math.max(...probabilities);
  return (maximum - 1 / probabilities.length) / (1 - 1 / probabilities.length);
}

function scoreConfidence(probabilities: number[]) {
  if (probabilities.length <= 1) return 1;
  let mode = 0;
  for (let index = 1; index < probabilities.length; index++)
    if (probabilities[index]! > probabilities[mode]!) mode = index;
  const distance = probabilities.reduce((sum, probability, index) => sum + probability * Math.abs(index - mode), 0);
  return 1 - distance / (probabilities.length - 1);
}

function typedAnswer(metadata: TypedMetadata, answer: TypedAnswer) {
  const probabilities = answer.probabilities.map(({probability}) => probability);
  if (metadata.type === 'choice') {
    return {
      type: 'choice',
      choice: metadata.keys[answer.selectedIndex],
      confidence: rounded(choiceConfidence(probabilities)),
      probabilities: Object.fromEntries(metadata.keys.map((key, index) => [key, rounded(probabilities[index]!)]))
    };
  }
  if (metadata.type === 'score') {
    return {
      type: 'score',
      score: rounded(probabilities.reduce((sum, probability, index) => sum + index * probability, 0)),
      legend: Object.fromEntries(metadata.legend.map((label, index) => [String(index), label])),
      probabilities: Object.fromEntries(
        probabilities.map((probability, index) => [String(index), rounded(probability)])
      ),
      confidence: rounded(scoreConfidence(probabilities))
    };
  }
  return {type: 'noul', noul: rounded(probabilities[1]!)};
}

export function finishKev08SystemOne(prepared: TypedPrepared, result: TypedResult) {
  if (!Array.isArray(result?.answers) || result.answers.length !== prepared.metadata.length)
    throw new WebDecisionError('INFERENCE_ERROR', 'Kev 0.8B returned an unexpected typed answer.');
  const answers = Object.fromEntries(
    prepared.metadata.map((metadata, index) => {
      if (!Object.hasOwn(result.answers, index))
        throw new WebDecisionError('INFERENCE_ERROR', 'Kev 0.8B returned an unexpected typed answer.');
      return [metadata.name, typedAnswer(metadata, result.answers[index]!)];
    })
  );
  return {model: MODEL_ID, tokenCount: result.tokenCount, answers};
}
