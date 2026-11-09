import {WebDecisionError} from '../kev.js';
import {ETTIN_MAX_OPTIONS, ETTIN_MAX_SCORE_LEVELS, ETTIN_MAX_SEQUENCE_TOKENS} from './contract.mjs';
import type {EttinMarkerIds} from './contract.mjs';

export interface EttinTokenizer {
  readonly special: EttinMarkerIds;
  encode(text: string): number[];
}

export interface EttinRunnerResult {
  readonly choiceLogits: number[];
  readonly noulLogits: number[];
  readonly scoreLogits: number[];
}

export interface PackedEttinQuestion {
  readonly type: 'choice' | 'noul' | 'score';
  readonly inputIds: number[];
  readonly optionPositions: number[];
  readonly optionMask: boolean[];
  readonly optionCount: number;
  readonly scoreCount: number;
  readonly labels: string[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

export interface PreparedEttinDecisionQuestion extends PackedEttinQuestion {
  readonly type: 'choice';
  readonly options: readonly unknown[];
}

export type PreparedEttinTypedQuestion =
  | (PackedEttinQuestion & {readonly type: 'choice'; readonly id: string})
  | (PackedEttinQuestion & {readonly type: 'noul'; readonly id: string})
  | (PackedEttinQuestion & {readonly type: 'score'; readonly id: string; readonly legend: readonly string[]});

export interface PreparedEttinRequest<Question extends PackedEttinQuestion = PackedEttinQuestion> {
  readonly questions: Question[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

export type EttinSystemOneAnswer =
  | {
      readonly type: 'choice';
      readonly choice: string;
      readonly confidence: number;
      readonly probabilities: Record<string, number>;
      readonly truncated?: true;
    }
  | {
      readonly type: 'noul';
      readonly noul: number;
      readonly probabilities: {readonly false: number; readonly true: number};
      readonly truncated?: true;
    }
  | {
      readonly type: 'score';
      readonly score: number;
      readonly expected: number;
      readonly legend: Record<string, string>;
      readonly confidence: number;
      readonly probabilities: Record<string, number>;
      readonly truncated?: true;
    };

const invalid = (message: string) => new WebDecisionError('INVALID_INPUT', message);

function entries(value: unknown, label: string): [string, unknown][] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`${label} must be an object.`);
  return Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} must contain plain data properties.`);
    return [key, descriptor.value];
  });
}

function nonEmptyText(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim()) throw invalid(`${label} must be a non-empty string.`);
  return value;
}

function trimToSequenceBudget(
  stateIds: number[],
  questionIds: number[],
  suffixIds: number[]
): {stateIds: number[]; questionIds: number[]; truncated: boolean} {
  const budget = ETTIN_MAX_SEQUENCE_TOKENS - 3 - suffixIds.length;
  if (budget < 1) throw invalid('Decider Ettin options leave no room for state or question tokens.');

  const total = stateIds.length + questionIds.length;
  if (total <= budget) return {stateIds, questionIds, truncated: false};

  const stateLength = Math.min(stateIds.length, Math.floor((budget * stateIds.length) / total));
  const questionLength = Math.min(questionIds.length, budget - stateLength);
  const keptState = stateIds.slice(0, stateLength);
  let keptQuestion = questionIds.slice(0, questionLength);
  while (keptState.length + keptQuestion.length > budget) keptQuestion = keptQuestion.slice(0, -1);
  return {stateIds: keptState, questionIds: keptQuestion, truncated: true};
}

interface EttinValue {
  readonly label: string;
  readonly text: string;
  readonly index?: number;
}

function prepareQuestion<Type extends PackedEttinQuestion['type']>(
  state: string,
  instructions: string,
  type: Type,
  values: readonly EttinValue[],
  tokenizer: EttinTokenizer
): PackedEttinQuestion & {readonly type: Type} {
  const stateText = state.trim();
  const questionText = instructions.trim();

  const optionRows: number[][] =
    type === 'choice'
      ? values.map(({text}, index) => [tokenizer.special.options[index]!, ...tokenizer.encode(text.trim())])
      : [];
  const suffixIds: number[] =
    type === 'choice'
      ? optionRows.flat()
      : type === 'score'
        ? tokenizer.encode(`SCALE ${values.map(({text}, index) => `${index}=${text.trim()}`).join(' ')}`)
        : [];

  const fit = trimToSequenceBudget(tokenizer.encode(stateText), tokenizer.encode(questionText), suffixIds);
  const inputIds = [
    tokenizer.special.decision,
    tokenizer.special.state,
    ...fit.stateIds,
    tokenizer.special.question,
    ...fit.questionIds
  ];
  const optionPositions = Array(ETTIN_MAX_OPTIONS).fill(0);
  const optionMask = Array(ETTIN_MAX_OPTIONS).fill(false);

  if (type === 'choice') {
    for (let index = 0; index < optionRows.length; index++) {
      optionPositions[index] = inputIds.length;
      optionMask[index] = true;
      inputIds.push(...optionRows[index]!);
    }
  } else {
    inputIds.push(...suffixIds);
  }

  if (inputIds.length > ETTIN_MAX_SEQUENCE_TOKENS)
    throw new TypeError('Decider Ettin sequence exceeded its validated token budget.');

  return {
    type,
    inputIds,
    optionPositions,
    optionMask,
    optionCount: type === 'choice' ? values.length : 0,
    scoreCount: type === 'score' ? values.length : 0,
    labels: values.map(({label}) => label),
    tokenCount: inputIds.length,
    truncated: fit.truncated
  };
}

function choiceValues(options: unknown, label: string): EttinValue[] {
  if (!Array.isArray(options) || options.length < 2 || options.length > ETTIN_MAX_OPTIONS)
    throw invalid(`${label} needs 2 to ${ETTIN_MAX_OPTIONS} options.`);
  return (options as unknown[]).map((option) => {
    const text = nonEmptyText(option, `${label} option`);
    return {label: option as string, text};
  });
}

export function prepareEttinDecision(
  tokenizer: EttinTokenizer,
  request: unknown
): PreparedEttinRequest<PreparedEttinDecisionQuestion> {
  type DecisionRequestInput = {readonly state?: unknown; readonly questions?: unknown};
  if (
    typeof (request as DecisionRequestInput | null | undefined)?.state !== 'string' ||
    !Array.isArray((request as DecisionRequestInput | null | undefined)?.questions)
  )
    throw invalid('Decider Ettin requires a string state and a question list.');
  if (((request as DecisionRequestInput).questions as unknown[]).length < 1)
    throw invalid('Decider Ettin requires at least one question.');

  const questions = ((request as DecisionRequestInput).questions as unknown[]).map(
    (question, index): PreparedEttinDecisionQuestion => {
      const fields = entries(question, `Question ${index}`);
      if (fields.some(([key]) => key !== 'instructions' && key !== 'options'))
        throw invalid(`Question ${index} has an unsupported field.`);
      const values = Object.fromEntries(fields) as Record<string, unknown>;
      if (typeof values.instructions !== 'string' || !Array.isArray(values.options))
        throw invalid(`Question ${index} needs text instructions and an options list.`);
      const options = choiceValues(values.options, `Question ${index}`);
      return {
        ...prepareQuestion(
          (request as DecisionRequestInput).state as string,
          values.instructions,
          'choice',
          options,
          tokenizer
        ),
        options: values.options as unknown[]
      };
    }
  );

  return {
    questions,
    tokenCount: questions.reduce((total, question) => total + question.tokenCount, 0),
    truncated: questions.some((question) => question.truncated)
  };
}

interface TypedQuestionInput {
  readonly type: 'choice' | 'noul' | 'score';
  readonly instructions: string;
  readonly criteria: unknown;
}

function typedQuestionEntries(questions: unknown): [string, TypedQuestionInput][] {
  const questionRows = entries(questions, 'Decider Ettin questions');
  if (questionRows.length < 1) throw invalid('Decider Ettin requires at least one named question.');
  return questionRows.map(([id, value]) => {
    const fields = entries(value, `Question ${id}`);
    if (fields.some(([key]) => !['type', 'instructions', 'criteria'].includes(key)))
      throw invalid(`Question ${id} has an unsupported field.`);
    const question = Object.fromEntries(fields) as Record<string, unknown>;
    if (!['choice', 'noul', 'score'].includes(question.type as string))
      throw invalid(`Question ${id} type must be choice, noul or score.`);
    if (typeof question.instructions !== 'string') throw invalid(`Question ${id} instructions must be text.`);
    return [id, question as unknown as TypedQuestionInput];
  });
}

function typedChoiceValues(criteria: unknown, id: string): EttinValue[] {
  let rows: [string, unknown][];
  if (Array.isArray(criteria)) {
    rows = [];
    for (let index = 0; index < criteria.length; index++) {
      if (!Object.hasOwn(criteria, index)) throw invalid(`Question ${id} criteria cannot contain holes.`);
      const label = nonEmptyText((criteria as unknown[])[index], `Question ${id} criterion`);
      rows.push([label, null]);
    }
  } else {
    rows = entries(criteria, `Question ${id} criteria`);
  }

  if (rows.length < 2 || rows.length > ETTIN_MAX_OPTIONS)
    throw invalid(`Question ${id} needs 2 to ${ETTIN_MAX_OPTIONS} choice criteria.`);

  const labels = rows.map(([label]) => nonEmptyText(label, `Question ${id} criterion name`));
  if (new Set(labels).size !== labels.length) throw invalid(`Question ${id} criterion names must be distinct.`);

  return rows.map(([label, description], index) => {
    const criterion = label as string;
    if (description != null && typeof description !== 'string')
      throw invalid(`Question ${id} criterion ${criterion} must have text or null.`);
    const text = (description as string | null | undefined)?.trim() ? `${criterion}: ${description}` : criterion;
    return {label: criterion, text, index};
  });
}

function typedScoreValues(criteria: unknown, id: string): EttinValue[] {
  if (!Array.isArray(criteria) || criteria.length < 2 || criteria.length > ETTIN_MAX_SCORE_LEVELS)
    throw invalid(`Question ${id} needs 2 to ${ETTIN_MAX_SCORE_LEVELS} ordered score levels.`);
  return (criteria as unknown[]).map((label, index) => ({
    label: nonEmptyText(label, `Question ${id} score level`),
    text: label as string,
    index
  }));
}

function validateNoulCriteria(criteria: unknown, id: string): void {
  if (criteria == null) return;
  const rows = entries(criteria, `Question ${id} criteria`);
  if (rows.some(([key, value]) => !['true', 'false'].includes(key) || (value != null && value !== '')))
    throw invalid(`Question ${id} uses custom Noul criteria, which Decider Ettin does not support.`);
}

export function prepareEttinSystemOne(
  state: unknown,
  questions: unknown,
  tokenizer: EttinTokenizer
): PreparedEttinRequest<PreparedEttinTypedQuestion> {
  if (typeof state !== 'string') throw invalid('Decider Ettin state must be text.');
  const preparedQuestions = typedQuestionEntries(questions).map(([id, question]): PreparedEttinTypedQuestion => {
    if (question.type === 'choice') {
      const values = typedChoiceValues(question.criteria, id);
      return {
        id,
        ...prepareQuestion(state, question.instructions, 'choice', values, tokenizer)
      };
    }
    if (question.type === 'score') {
      const values = typedScoreValues(question.criteria, id);
      return {
        id,
        ...prepareQuestion(state, question.instructions, 'score', values, tokenizer),
        legend: values.map(({label}) => label)
      };
    }
    validateNoulCriteria(question.criteria, id);
    return {id, ...prepareQuestion(state, question.instructions, 'noul', [], tokenizer)};
  });

  return {
    questions: preparedQuestions,
    tokenCount: preparedQuestions.reduce((total, question) => total + question.tokenCount, 0),
    truncated: preparedQuestions.some((question) => question.truncated)
  };
}

function softmax(values: readonly number[]): number[] {
  if (!values.length || values.some((value) => !Number.isFinite(value)))
    throw new WebDecisionError('INFERENCE_ERROR', 'Decider Ettin returned invalid logits.');
  const maximum = Math.max(...values);
  const weights = values.map((value) => (value - maximum < -80 ? 0 : Math.exp(value - maximum)));
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (total <= 0) throw new WebDecisionError('INFERENCE_ERROR', 'Decider Ettin returned invalid logits.');
  return weights.map((value) => value / total);
}

function rounded(value: number): number {
  return Number(value.toFixed(4));
}

function confidence(probabilities: readonly number[]): number {
  const best = Math.max(...probabilities);
  if (probabilities.length <= 1) return 1;
  return rounded((best - 1 / probabilities.length) / (1 - 1 / probabilities.length));
}

function choiceResult(question: PreparedEttinDecisionQuestion, row: EttinRunnerResult) {
  const logits = row.choiceLogits.slice(0, question.optionCount);
  const probabilities = softmax(logits);
  const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
  return {
    selected: question.options[selectedIndex],
    selectedIndex,
    probabilities: question.options.map((option, index) => ({
      option,
      probability: rounded(probabilities[index]!),
      logit: logits[index]!
    }))
  };
}

export function finishEttinDecision(
  prepared: PreparedEttinRequest<PreparedEttinDecisionQuestion>,
  rows: readonly EttinRunnerResult[]
) {
  if (rows.length !== prepared.questions.length) throw new TypeError('Decider Ettin result rows are incomplete.');
  return {
    tokenCount: prepared.tokenCount,
    ...(prepared.truncated ? {truncated: true} : {}),
    answers: prepared.questions.map((question, index) => choiceResult(question, rows[index]!))
  };
}

function typedResult(question: PreparedEttinTypedQuestion, row: EttinRunnerResult): EttinSystemOneAnswer {
  if (question.type === 'noul') {
    const probabilities = softmax(row.noulLogits);
    return {
      type: 'noul',
      noul: rounded(probabilities[1]!),
      probabilities: {false: rounded(probabilities[0]!), true: rounded(probabilities[1]!)}
    };
  }

  const logits =
    question.type === 'choice'
      ? row.choiceLogits.slice(0, question.optionCount)
      : row.scoreLogits.slice(0, question.scoreCount);
  const probabilities = softmax(logits);
  const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
  const distribution = Object.fromEntries(
    question.labels.map((label, index) => [String(index), rounded(probabilities[index]!)])
  ) as Record<string, number>;
  if (question.type === 'choice') {
    return {
      type: 'choice',
      choice: question.labels[selectedIndex]!,
      confidence: confidence(probabilities),
      probabilities: Object.fromEntries(
        question.labels.map((label, index) => [label, rounded(probabilities[index]!)])
      ) as Record<string, number>
    };
  }

  const expected = probabilities.reduce((sum, probability, index) => sum + index * probability, 0);
  return {
    type: 'score',
    score: selectedIndex,
    expected: rounded(expected),
    legend: Object.fromEntries(question.legend.map((label, index) => [String(index), label])) as Record<string, string>,
    confidence: confidence(probabilities),
    probabilities: distribution
  };
}

export function finishEttinSystemOne(
  prepared: PreparedEttinRequest<PreparedEttinTypedQuestion>,
  rows: readonly EttinRunnerResult[],
  model: string
) {
  if (rows.length !== prepared.questions.length) throw new TypeError('Decider Ettin result rows are incomplete.');
  const answers = Object.fromEntries(
    prepared.questions.map((question, index) => {
      const answer = typedResult(question, rows[index]!);
      return [question.id, question.truncated ? {...answer, truncated: true} : answer];
    })
  ) as Record<string, EttinSystemOneAnswer>;
  return {
    model,
    tokenCount: prepared.tokenCount,
    ...(prepared.truncated ? {truncated: true} : {}),
    answers
  };
}
