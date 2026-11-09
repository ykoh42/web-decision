import {WebDecisionError} from '../kev.js';
import type {DecimaInferenceKind} from './protocol.mjs';

type DecimaQuestionType = DecimaInferenceKind;

interface DecimaQuestionCandidate {
  readonly type?: unknown;
  readonly instructions?: unknown;
  readonly criteria?: unknown;
  readonly ordinal?: unknown;
  readonly label_blind?: unknown;
}

interface PreparedDecimaQuestion {
  readonly id: string;
  readonly type: DecimaQuestionType;
  readonly outputType: DecimaQuestionType;
  readonly instructions: string;
  readonly labels: string[];
  readonly choices: string[];
}

interface PreparedDecimaSystemOne {
  readonly state: string;
  readonly questions: readonly PreparedDecimaQuestion[];
  readonly language?: string | undefined;
}

interface DecimaInferenceAnswer {
  readonly probabilities: readonly number[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

interface DecimaChoiceAnswer {
  readonly type: 'choice';
  readonly choice: string;
  readonly confidence: number;
  readonly probabilities: Record<string, number>;
}

interface DecimaScoreAnswer {
  readonly type: 'score';
  readonly score: number;
  readonly confidence: number;
  readonly legend: Record<string, string>;
  readonly probabilities: Record<string, number>;
}

interface DecimaNoulAnswer {
  readonly type: 'noul';
  readonly noul: number;
}

interface DecimaRankAnswer {
  readonly type: 'rank';
  readonly probabilities: Record<string, number>;
  readonly order: string[];
}

type DecimaSystemOneAnswer = DecimaChoiceAnswer | DecimaScoreAnswer | DecimaNoulAnswer | DecimaRankAnswer;

export interface DecimaSystemOneResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Record<string, DecimaSystemOneAnswer>;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const MAX_LEVELS = 10;
const QUESTION_FIELDS = new Set(['type', 'instructions', 'criteria', 'ordinal', 'label_blind']);

function text(value: unknown): string {
  if (value == null) return '';
  return typeof value === 'string' ? value : (JSON.stringify(value) as string);
}

function objectEntries(value: object): Array<[string, unknown]> {
  return Reflect.ownKeys(value).map((key): [string, unknown] => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid('Question criteria must contain plain text values.');
    return [key, descriptor.value];
  });
}

function questionEntries(questions: unknown): Array<[string, DecimaQuestionCandidate]> {
  if (!questions || typeof questions !== 'object' || Array.isArray(questions))
    throw invalid('Decima questions must be an object.');
  const entries = Reflect.ownKeys(questions);
  if (entries.length < 1 || entries.some((key) => typeof key !== 'string'))
    throw invalid('Decima needs at least one named question.');
  return entries.map((id): [string, DecimaQuestionCandidate] => {
    const descriptor = Object.getOwnPropertyDescriptor(questions, id);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw invalid('Question accessors are not supported.');
    const question = descriptor.value;
    if (!question || typeof question !== 'object' || Array.isArray(question))
      throw invalid(`Question ${id as string} must be an object.`);
    const snapshot: Record<string, unknown> = Object.create(null);
    for (const key of Reflect.ownKeys(question)) {
      const field = Object.getOwnPropertyDescriptor(question, key);
      if (typeof key !== 'string' || !QUESTION_FIELDS.has(key) || !field || !Object.hasOwn(field, 'value'))
        throw invalid(`Question ${id as string} has an unsupported field.`);
      snapshot[key] = field.value;
    }
    return [id as string, snapshot as DecimaQuestionCandidate];
  });
}

function criteriaEntries(
  criteria: unknown,
  id: string,
  maximum: number,
  preserveList = false
): {readonly labels: string[]; readonly descriptions: string[]} {
  let entries: Array<[unknown, unknown]> = [];
  if (Array.isArray(criteria)) {
    entries = (criteria as unknown[]).map((label): [unknown, unknown] => [text(label), null]);
  } else if (criteria && typeof criteria === 'object' && !Array.isArray(criteria)) entries = objectEntries(criteria);
  if (entries.length < 2 || entries.length > maximum) {
    const range = Number.isFinite(maximum) ? `2 to ${maximum}` : 'at least 2';
    throw invalid(`Question ${id} needs ${range} criteria.`);
  }
  const labels = entries.map(([label]) => String(label));
  if ((!preserveList && new Set(labels).size !== labels.length) || labels.some((label) => !label.trim()))
    throw invalid(`Question ${id} needs distinct non-empty labels.`);
  return {labels, descriptions: entries.map(([, description]) => text(description))};
}

function choiceSpec(
  id: string,
  question: DecimaQuestionCandidate,
  kind: DecimaQuestionType,
  outputType: DecimaQuestionType = kind === 'score' ? 'score' : kind
): PreparedDecimaQuestion {
  const maximum = kind === 'score' ? MAX_LEVELS : Infinity;
  const {labels, descriptions} = criteriaEntries(question.criteria, id, maximum, kind === 'score');
  const labelBlind = question.label_blind === true;
  const choices = labels.map((label, index) => {
    const description = descriptions[index]!;
    if (!description.trim()) return label;
    return labelBlind ? description : `${label}: ${description}`;
  });
  if (outputType !== 'score' && new Set(choices).size !== choices.length)
    throw invalid(`Question ${id} has duplicate rendered criteria.`);
  return {
    id,
    type: kind,
    outputType,
    instructions: question.instructions == null ? id : text(question.instructions),
    labels,
    choices
  };
}

function prepareChoice(id: string, question: DecimaQuestionCandidate): PreparedDecimaQuestion {
  if (question.ordinal !== undefined && typeof question.ordinal !== 'boolean')
    throw invalid(`Question ${id} ordinal must be boolean.`);
  if (question.label_blind !== undefined && typeof question.label_blind !== 'boolean')
    throw invalid(`Question ${id} label_blind must be boolean.`);
  const kind = question.ordinal ? 'score' : 'choice';
  return choiceSpec(id, question, kind, 'choice');
}

function prepareScore(id: string, question: DecimaQuestionCandidate): PreparedDecimaQuestion {
  if (!Array.isArray(question.criteria)) throw invalid(`Question ${id} score criteria must be an array.`);
  if (question.criteria.length < 2 || question.criteria.length > MAX_LEVELS)
    throw invalid(`Question ${id} score needs 2 to ${MAX_LEVELS} ordered levels.`);
  return choiceSpec(id, question, 'score', 'score');
}

function prepareNoul(id: string, question: DecimaQuestionCandidate): PreparedDecimaQuestion {
  const instructions = question.instructions == null ? id : text(question.instructions);
  const criteria = question.criteria;
  if (criteria != null && (typeof criteria !== 'object' || Array.isArray(criteria)))
    throw invalid(`Question ${id} yes/no criteria must be an object.`);
  const values = criteria == null ? [] : objectEntries(criteria);
  if (values.some(([key]) => key !== 'true' && key !== 'false'))
    throw invalid(`Question ${id} yes/no criteria accepts true and false only.`);
  const criteriaRecord = Object.fromEntries(values);
  const trueText = text(criteriaRecord.true).trim();
  const falseText = text(criteriaRecord.false).trim();
  const expanded =
    trueText || falseText
      ? `${instructions}\nTrue if: ${trueText || '—'}\nFalse if: ${falseText || '—'}`
      : instructions;
  return {
    id,
    type: 'noul',
    outputType: 'noul',
    instructions: expanded,
    labels: ['yes', 'no'],
    choices: ['yes', 'no']
  };
}

function prepareOne(
  id: string,
  question: DecimaQuestionCandidate,
  supportedTypes: readonly DecimaQuestionType[]
): PreparedDecimaQuestion {
  if (!supportedTypes.includes(question.type as DecimaQuestionType))
    throw invalid(`Question ${id} type must be ${supportedTypes.join(', ')}.`);
  switch (question.type as DecimaQuestionType) {
    case 'choice':
      return prepareChoice(id, question);
    case 'score':
      return prepareScore(id, question);
    case 'noul':
      return prepareNoul(id, question);
    case 'rank':
      return choiceSpec(id, question, 'rank', 'rank');
  }
  throw invalid(`Question ${id} type must be choice, score, noul, or rank.`);
}

export function prepareDecimaSystemOne(
  state: unknown,
  questions: unknown,
  language = 'en',
  supportedTypes: DecimaQuestionType[] = ['choice', 'score', 'noul', 'rank']
): PreparedDecimaSystemOne {
  if (typeof state !== 'string' && (state === null || typeof state !== 'object'))
    throw invalid('Decima state must be a string, object, or array.');
  if (
    !Array.isArray(supportedTypes) ||
    supportedTypes.some((type) => !['choice', 'score', 'noul', 'rank'].includes(type))
  )
    throw new TypeError('Decima supported question types are required.');
  const stateText = (typeof state === 'string' ? state : JSON.stringify(state)) as string;
  const prepared = questionEntries(questions).map(([id, question]) => prepareOne(id, question, supportedTypes));
  return {state: stateText, questions: prepared, language};
}

function rounded(value: number): number {
  return Math.round(value * 10000) / 10000;
}

function confidence(probabilities: readonly number[]): number {
  const maximum = probabilities.reduce((current, value) => Math.max(current, value), -Infinity);
  return Math.max(0, Math.min(1, (probabilities.length * maximum - 1) / (probabilities.length - 1)));
}

function answerFor(question: PreparedDecimaQuestion, probabilities: readonly number[]): DecimaSystemOneAnswer {
  const values = Object.fromEntries(
    question.labels.map((label, index) => [label, rounded(probabilities[index]!)] as const)
  );
  if (question.outputType === 'noul') return {type: 'noul', noul: rounded(probabilities[0]!)};
  if (question.outputType === 'rank') {
    const order = question.labels
      .map((label, index) => ({label, index}))
      .sort((left, right) => probabilities[right.index]! - probabilities[left.index]!)
      .map(({label}) => label);
    return {type: 'rank', probabilities: values, order};
  }
  if (question.outputType === 'score') {
    const expected = probabilities.reduce((sum, probability, index) => sum + index * probability, 0);
    return {
      type: 'score',
      score: rounded(expected),
      confidence: rounded(confidence(probabilities)),
      legend: Object.fromEntries(question.labels.map((label, index) => [String(index), label] as const)),
      probabilities: Object.fromEntries(
        question.labels.map((_, index) => [String(index), rounded(probabilities[index]!)] as const)
      )
    };
  }
  const maximum = probabilities.reduce((current, value) => Math.max(current, value), -Infinity);
  const index = probabilities.indexOf(maximum);
  return {
    type: 'choice',
    choice: question.labels[index]!,
    confidence: rounded(confidence(probabilities)),
    probabilities: values
  };
}

export function finishDecimaSystemOne(
  prepared: PreparedDecimaSystemOne,
  answers: readonly DecimaInferenceAnswer[],
  model = 'experimental-decima-small-int8'
): DecimaSystemOneResult {
  const result = Object.fromEntries(
    prepared.questions.map((question, index) => [question.id, answerFor(question, answers[index]!.probabilities)])
  );
  const tokenCount = answers.reduce((sum, answer) => sum + answer.tokenCount, 0);
  return {
    model,
    tokenCount,
    ...(answers.some((answer) => answer.truncated) ? {truncated: true} : {}),
    answers: result
  };
}
