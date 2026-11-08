import {WebDecisionError} from '../kev.js';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const MAX_LEVELS = 10;
const QUESTION_FIELDS = new Set(['type', 'instructions', 'criteria', 'ordinal', 'label_blind']);

function text(value) {
  if (value == null) return '';
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function objectEntries(value) {
  return Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid('Question criteria must contain plain text values.');
    return [key, descriptor.value];
  });
}

function questionEntries(questions) {
  if (!questions || typeof questions !== 'object' || Array.isArray(questions))
    throw invalid('Decima questions must be an object.');
  const entries = Reflect.ownKeys(questions);
  if (entries.length < 1 || entries.some((key) => typeof key !== 'string'))
    throw invalid('Decima needs at least one named question.');
  return entries.map((id) => {
    const descriptor = Object.getOwnPropertyDescriptor(questions, id);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw invalid('Question accessors are not supported.');
    const question = descriptor.value;
    if (!question || typeof question !== 'object' || Array.isArray(question))
      throw invalid(`Question ${id} must be an object.`);
    const snapshot = Object.create(null);
    for (const key of Reflect.ownKeys(question)) {
      const field = Object.getOwnPropertyDescriptor(question, key);
      if (typeof key !== 'string' || !QUESTION_FIELDS.has(key) || !field || !Object.hasOwn(field, 'value'))
        throw invalid(`Question ${id} has an unsupported field.`);
      snapshot[key] = field.value;
    }
    return [id, snapshot];
  });
}

function criteriaEntries(criteria, id, maximum, preserveList = false) {
  let entries = [];
  if (Array.isArray(criteria)) {
    entries = criteria.map((label) => [text(label), null]);
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

function choiceSpec(id, question, kind, outputType = kind === 'score' ? 'score' : kind) {
  const maximum = kind === 'score' ? MAX_LEVELS : Infinity;
  const {labels, descriptions} = criteriaEntries(question.criteria, id, maximum, kind === 'score');
  const labelBlind = question.label_blind === true;
  const choices = labels.map((label, index) => {
    const description = descriptions[index];
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

function prepareChoice(id, question) {
  if (question.ordinal !== undefined && typeof question.ordinal !== 'boolean')
    throw invalid(`Question ${id} ordinal must be boolean.`);
  if (question.label_blind !== undefined && typeof question.label_blind !== 'boolean')
    throw invalid(`Question ${id} label_blind must be boolean.`);
  const kind = question.ordinal ? 'score' : 'choice';
  return choiceSpec(id, question, kind, 'choice');
}

function prepareScore(id, question) {
  if (!Array.isArray(question.criteria)) throw invalid(`Question ${id} score criteria must be an array.`);
  if (question.criteria.length < 2 || question.criteria.length > MAX_LEVELS)
    throw invalid(`Question ${id} score needs 2 to ${MAX_LEVELS} ordered levels.`);
  return choiceSpec(id, question, 'score', 'score');
}

function prepareNoul(id, question) {
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

function prepareOne(id, question, supportedTypes) {
  if (!supportedTypes.includes(question.type))
    throw invalid(`Question ${id} type must be ${supportedTypes.join(', ')}.`);
  switch (question.type) {
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
  state,
  questions,
  language = 'en',
  supportedTypes = ['choice', 'score', 'noul', 'rank']
) {
  if (typeof state !== 'string' && (state === null || typeof state !== 'object'))
    throw invalid('Decima state must be a string, object, or array.');
  if (
    !Array.isArray(supportedTypes) ||
    supportedTypes.some((type) => !['choice', 'score', 'noul', 'rank'].includes(type))
  )
    throw new TypeError('Decima supported question types are required.');
  const stateText = typeof state === 'string' ? state : JSON.stringify(state);
  const prepared = questionEntries(questions).map(([id, question]) => prepareOne(id, question, supportedTypes));
  return {state: stateText, questions: prepared, language};
}

function rounded(value) {
  return Math.round(value * 10000) / 10000;
}

function confidence(probabilities) {
  const maximum = probabilities.reduce((current, value) => Math.max(current, value), -Infinity);
  return Math.max(0, Math.min(1, (probabilities.length * maximum - 1) / (probabilities.length - 1)));
}

function answerFor(question, probabilities) {
  const values = Object.fromEntries(question.labels.map((label, index) => [label, rounded(probabilities[index])]));
  if (question.outputType === 'noul') return {type: 'noul', noul: rounded(probabilities[0])};
  if (question.outputType === 'rank') {
    const order = question.labels
      .map((label, index) => ({label, index}))
      .sort((left, right) => probabilities[right.index] - probabilities[left.index])
      .map(({label}) => label);
    return {type: 'rank', probabilities: values, order};
  }
  if (question.outputType === 'score') {
    const expected = probabilities.reduce((sum, probability, index) => sum + index * probability, 0);
    return {
      type: 'score',
      score: rounded(expected),
      confidence: rounded(confidence(probabilities)),
      legend: Object.fromEntries(question.labels.map((label, index) => [String(index), label])),
      probabilities: Object.fromEntries(
        question.labels.map((_, index) => [String(index), rounded(probabilities[index])])
      )
    };
  }
  const maximum = probabilities.reduce((current, value) => Math.max(current, value), -Infinity);
  const index = probabilities.indexOf(maximum);
  return {
    type: 'choice',
    choice: question.labels[index],
    confidence: rounded(confidence(probabilities)),
    probabilities: values
  };
}

export function finishDecimaSystemOne(prepared, answers, model = 'experimental-decima-small-int8') {
  const result = Object.fromEntries(
    prepared.questions.map((question, index) => [question.id, answerFor(question, answers[index].probabilities)])
  );
  const tokenCount = answers.reduce((sum, answer) => sum + answer.tokenCount, 0);
  return {
    model,
    tokenCount,
    ...(answers.some((answer) => answer.truncated) ? {truncated: true} : {}),
    answers: result
  };
}
