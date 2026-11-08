import {WebDecisionError} from '../kev.js';
import {GLINER_MAX_OPTIONS, GLINER_MAX_TOKENS, GLINER_MOBILE_MODEL} from './contract.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const inference = (message) => new WebDecisionError('INFERENCE_ERROR', message);
const questionFields = new Set(['type', 'instructions', 'criteria']);
const reservedTokens = ['[SEP_STRUCT]', '[SEP_TEXT]', '[P]', '[L]', '[DESCRIPTION]', '[PAD]'];
const hasReservedToken = (text) => reservedTokens.some((token) => text.includes(token));

function ownValue(object, key, message) {
  const descriptor = Object.getOwnPropertyDescriptor(object, key);
  if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw invalid(message);
  return descriptor.value;
}

function arrayValues(value, id, name) {
  const values = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`Question ${id} ${name} must not contain holes or accessors.`);
    values.push(descriptor.value);
  }
  return values;
}

function optionEntries(value, id) {
  if (Array.isArray(value)) {
    if (value.length < 2 || value.length > GLINER_MAX_OPTIONS)
      throw invalid(`Question ${id} needs 2 to ${GLINER_MAX_OPTIONS} string options.`);
    const labels = arrayValues(value, id, 'options');
    if (labels.some((label) => typeof label !== 'string'))
      throw invalid(`Question ${id} needs 2 to ${GLINER_MAX_OPTIONS} string options.`);
    return labels.map((label) => [label, null]);
  }
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw invalid(`Question ${id} criteria must be an option array or label-description object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null)
    throw invalid(`Question ${id} criteria must be a plain object.`);
  const keys = Reflect.ownKeys(value);
  if (keys.length < 2 || keys.length > GLINER_MAX_OPTIONS)
    throw invalid(`Question ${id} needs 2 to ${GLINER_MAX_OPTIONS} options.`);
  return keys.map((label) => {
    if (typeof label !== 'string') throw invalid(`Question ${id} labels must be strings.`);
    const description = ownValue(value, label, `Question ${id} labels cannot be accessors.`);
    if (description !== null && typeof description !== 'string')
      throw invalid(`Question ${id} descriptions must be strings or null.`);
    if (typeof description === 'string' && hasReservedToken(description))
      throw invalid(`Question ${id} descriptions cannot contain GLiNER protocol tokens.`);
    return [label, description];
  });
}

function validateText(value, id, field) {
  if (typeof value !== 'string' || !value.trim()) throw invalid(`Question ${id} needs non-empty ${field} text.`);
  if (hasReservedToken(value)) throw invalid(`Question ${id} ${field} cannot contain GLiNER protocol tokens.`);
  return value;
}

function snapshotQuestionFields(id, source) {
  if (!source || typeof source !== 'object' || Array.isArray(source))
    throw invalid(`Question ${id} must be an object.`);
  const question = Object.create(null);
  for (const key of Reflect.ownKeys(source)) {
    if (typeof key !== 'string' || !questionFields.has(key)) throw invalid(`Question ${id} has an unsupported field.`);
    question[key] = ownValue(source, key, `Question ${id} cannot contain accessors.`);
  }
  if (!['choice', 'score', 'noul'].includes(question.type))
    throw invalid(`Question ${id} type must be choice, score, or noul.`);
  return question;
}

function validateChoiceLabels(labels, id) {
  const invalidLabel = (label) => !label.trim() || hasReservedToken(label);
  if (labels.some(invalidLabel) || new Set(labels).size !== labels.length)
    throw invalid(`Question ${id} labels must be distinct non-empty strings.`);
}

function snapshotChoiceQuestion(id, instructions, criteria) {
  const entries = optionEntries(criteria, id);
  const labels = entries.map(([label]) => label);
  validateChoiceLabels(labels, id);
  return {id, type: 'choice', instructions, labels, descriptions: entries.map(([, description]) => description)};
}

function snapshotScoreQuestion(id, instructions, criteria) {
  if (!Array.isArray(criteria) || criteria.length < 2 || criteria.length > GLINER_MAX_OPTIONS)
    throw invalid(`Question ${id} score needs 2 to ${GLINER_MAX_OPTIONS} ordered levels.`);
  const legend = arrayValues(criteria, id, 'score levels').map((value) => validateText(value, id, 'score level'));
  const labels = legend.map((_, index) => String(index));
  return {id, type: 'score', instructions, labels, descriptions: legend, legend};
}

function snapshotNoulQuestion(id, instructions, rawCriteria) {
  const criteria = rawCriteria ?? Object.create(null);
  if (!criteria || typeof criteria !== 'object' || Array.isArray(criteria))
    throw invalid(`Question ${id} Noul criteria must contain true/false descriptions.`);
  const prototype = Object.getPrototypeOf(criteria);
  if (prototype !== Object.prototype && prototype !== null)
    throw invalid(`Question ${id} Noul criteria must be a plain object.`);
  const descriptions = {yes: null, no: null};
  for (const key of Reflect.ownKeys(criteria)) {
    if (key !== 'true' && key !== 'false') throw invalid(`Question ${id} Noul criteria accepts true and false only.`);
    const value = ownValue(criteria, key, `Question ${id} Noul criteria cannot contain accessors.`);
    if (value !== null && typeof value !== 'string')
      throw invalid(`Question ${id} Noul descriptions must be strings or null.`);
    if (typeof value === 'string' && hasReservedToken(value))
      throw invalid(`Question ${id} Noul descriptions cannot contain GLiNER protocol tokens.`);
    descriptions[key === 'true' ? 'yes' : 'no'] = value;
  }
  return {id, type: 'noul', instructions, labels: ['yes', 'no'], descriptions: [descriptions.yes, descriptions.no]};
}

function snapshotQuestion(id, source) {
  const question = snapshotQuestionFields(id, source);
  const instructions = validateText(question.instructions, id, 'instructions');
  if (question.type === 'choice') return snapshotChoiceQuestion(id, instructions, question.criteria);
  if (question.type === 'score') return snapshotScoreQuestion(id, instructions, question.criteria);
  return snapshotNoulQuestion(id, instructions, question.criteria);
}

function append(inputIds, values) {
  for (const value of values) {
    if (inputIds.length >= GLINER_MAX_TOKENS)
      throw invalid(`GLiNER question schemas exceed the ${GLINER_MAX_TOKENS}-token context.`);
    inputIds.push(value);
  }
}

function appendSchema(inputIds, markerPositions, question, tokenizer) {
  const {special} = tokenizer;
  append(inputIds, tokenizer.encode('('));
  append(inputIds, [special['[P]']]);
  let prompt = question.instructions;
  for (let index = 0; index < question.labels.length; index++) {
    const description = question.descriptions[index];
    if (description != null && description !== '') prompt += ` [DESCRIPTION] ${question.labels[index]}: ${description}`;
  }
  append(inputIds, tokenizer.encode(prompt));
  append(inputIds, tokenizer.encode('('));
  for (const label of question.labels) {
    markerPositions.push(inputIds.length);
    append(inputIds, [special['[L]']]);
    append(inputIds, tokenizer.encode(label));
  }
  append(inputIds, tokenizer.encode(')'));
  append(inputIds, tokenizer.encode(')'));
}

export function prepareGlinerMobileSystemOne(state, questions, tokenizer) {
  if (typeof state !== 'string') throw invalid('GLiNER state must be text.');
  if (hasReservedToken(state)) throw invalid('GLiNER state cannot contain GLiNER protocol tokens.');
  if (!questions || typeof questions !== 'object' || Array.isArray(questions))
    throw invalid('GLiNER questions must be a named object.');
  const ids = Reflect.ownKeys(questions);
  if (ids.length < 1 || ids.some((id) => typeof id !== 'string'))
    throw invalid('GLiNER needs at least one named question with a string name.');
  const preparedQuestions = ids.map((id) =>
    snapshotQuestion(id, ownValue(questions, id, `Question ${id} cannot be an accessor.`))
  );
  const inputIds = [];
  const markerPositions = [];
  for (let index = 0; index < preparedQuestions.length; index++) {
    if (index > 0) append(inputIds, [tokenizer.special['[SEP_STRUCT]']]);
    appendSchema(inputIds, markerPositions, preparedQuestions[index], tokenizer);
  }
  append(inputIds, [tokenizer.special['[SEP_TEXT]']]);
  if (inputIds.length > GLINER_MAX_TOKENS)
    throw invalid(`GLiNER question schemas exceed the ${GLINER_MAX_TOKENS}-token context.`);

  const stateIds = tokenizer.encodeState(state);
  const availableStateTokens = GLINER_MAX_TOKENS - inputIds.length;
  const retainedState = stateIds.slice(0, availableStateTokens);
  append(inputIds, retainedState);
  if (inputIds.length < 1 || inputIds.length > GLINER_MAX_TOKENS || markerPositions.length < 2)
    throw invalid('GLiNER request does not fit the pinned graph.');

  let markerOffset = 0;
  for (const question of preparedQuestions) {
    question.markerOffset = markerOffset;
    markerOffset += question.labels.length;
  }

  return {
    questions: preparedQuestions,
    inputIds,
    markerPositions,
    tokenCount: inputIds.length,
    truncated: retainedState.length < stateIds.length
  };
}

function softmax(logits) {
  if (!Array.isArray(logits) || logits.length < 2 || logits.some((value) => !Number.isFinite(value)))
    throw inference('GLiNER returned invalid classification logits.');
  const max = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - max));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(denominator) || denominator <= 0)
    throw inference('GLiNER returned invalid classification scores.');
  return exponentials.map((value) => value / denominator);
}

function answerQuestion(question, logits) {
  const probabilities = softmax(logits);
  let selectedIndex = 0;
  for (let index = 1; index < probabilities.length; index++)
    if (probabilities[index] > probabilities[selectedIndex]) selectedIndex = index;
  const confidence = probabilities[selectedIndex];
  const distribution = Object.fromEntries(question.labels.map((label, index) => [label, probabilities[index]]));

  if (question.type === 'choice')
    return {type: 'choice', choice: question.labels[selectedIndex], confidence, probabilities: distribution};
  if (question.type === 'score') {
    const expected = probabilities.reduce((sum, probability, index) => sum + probability * index, 0);
    return {
      type: 'score',
      score: expected,
      expected,
      confidence,
      legend: Object.fromEntries(question.legend.map((label, index) => [String(index), label])),
      probabilities: distribution
    };
  }
  return {
    type: 'noul',
    noul: probabilities[0],
    confidence,
    probabilities: {true: probabilities[0], false: probabilities[1]}
  };
}

export function finishGlinerMobileSystemOne(prepared, logits, model = GLINER_MOBILE_MODEL) {
  if (!Array.isArray(logits) || logits.length !== prepared.markerPositions.length)
    throw inference('GLiNER logits do not match the input label markers.');
  const answers = Object.create(null);
  for (const question of prepared.questions) {
    const start = question.markerOffset;
    const end = start + question.labels.length;
    answers[question.id] = answerQuestion(question, logits.slice(start, end));
  }
  return {
    model,
    tokenCount: prepared.tokenCount,
    ...(prepared.truncated ? {truncated: true} : {}),
    answers
  };
}
