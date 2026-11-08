import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_MODEL} from './contract.mjs';

const invalid = (message) => {
  throw new WebDecisionError('INVALID_INPUT', message);
};
const inference = (message) => new WebDecisionError('INFERENCE_ERROR', message);
const QUESTION_FIELDS = new Set(['type', 'instructions', 'criteria']);

function recordEntries(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) invalid(`${label} must be a plain object.`);
  const entries = [];
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    entries.push([key, descriptor.value]);
  }
  return entries;
}

function countRecordEntries(value, label, validateKey) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) invalid(`${label} must be a plain object.`);
  let count = 0;
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    validateKey?.(key);
    count++;
  }
  return count;
}

function arrayValues(value, label, expectedLength) {
  if (!Array.isArray(value)) invalid(`${label} must be an array.`);
  const length = value.length;
  if (!Number.isSafeInteger(length) || length < 0) invalid(`${label} must have a valid length.`);
  if (expectedLength !== undefined && length !== expectedLength) invalid(`${label} changed after preflight.`);
  const values = [];
  for (let index = 0; index < length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value')) invalid(`${label} must be a dense data array.`);
    values.push(descriptor.value);
  }
  for (const key of Reflect.ownKeys(value))
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= length))
      invalid(`${label} cannot have extra properties.`);
  return values;
}

function serializeJson(value, ancestors = new Set()) {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) invalid('SafeDecide state numbers must be finite.');
    return JSON.stringify(value);
  }
  if (!value || typeof value !== 'object') invalid('SafeDecide state must be a string or JSON value.');
  if (ancestors.has(value)) invalid('SafeDecide state cannot contain a cycle.');
  ancestors.add(value);
  const serialized = Array.isArray(value)
    ? `[${arrayValues(value, 'SafeDecide state arrays')
        .map((item) => serializeJson(item, ancestors))
        .join(', ')}]`
    : `{${recordEntries(value, 'SafeDecide state objects')
        .map(([key, item]) => `${JSON.stringify(key)}: ${serializeJson(item, ancestors)}`)
        .join(', ')}}`;
  ancestors.delete(value);
  return serialized;
}

function stateText(state) {
  if (typeof state === 'string') return state;
  if (!state || typeof state !== 'object') invalid('SafeDecide state must be a string, array, or object.');
  return serializeJson(state);
}

function text(value, label) {
  if (typeof value !== 'string') invalid(`${label} must be a string.`);
  return value;
}

function description(value, label) {
  if (value !== undefined && value !== null && typeof value !== 'string') invalid(`${label} must be a string or null.`);
  return value || null;
}

function readQuestion(id, value) {
  if (!id) invalid('SafeDecide question names cannot be empty.');
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`Question ${id} must be an object.`);
  const fields = Object.create(null);
  for (const [key, item] of recordEntries(value, `Question ${id}`)) {
    if (!QUESTION_FIELDS.has(key)) invalid(`Question ${id} has an unsupported field.`);
    fields[key] = item;
  }
  const type = fields.type ?? 'choice';
  if (!['choice', 'score', 'noul'].includes(type)) invalid(`Question ${id} type must be choice, score, or noul.`);
  if (fields.instructions !== undefined && fields.instructions !== null && typeof fields.instructions !== 'string')
    invalid(`Question ${id} instructions must be a string.`);
  const instructions = fields.instructions || id;
  text(instructions, `Question ${id} instructions`);

  return {id, type, instructions, criteria: fields.criteria};
}

function countQuestionOptions(question) {
  const {id, type, criteria} = question;
  if (type === 'choice') {
    const count = countRecordEntries(criteria, `Question ${id} choice criteria`);
    if (count < 2) invalid(`Question ${id} needs at least 2 choice options.`);
    return count;
  }
  if (type === 'score') {
    if (!Array.isArray(criteria)) invalid(`Question ${id} score criteria must be an array.`);
    const count = criteria.length;
    if (!Number.isSafeInteger(count) || count < 2) invalid(`Question ${id} needs at least 2 score levels.`);
    return count;
  }
  if (criteria !== undefined && criteria !== null) {
    countRecordEntries(criteria, `Question ${id} noul criteria`, (key) => {
      if (key !== 'false' && key !== 'true') invalid(`Question ${id} noul criteria can only describe false and true.`);
    });
  }
  return 2;
}

function snapshotQuestion({id, type, instructions, criteria, optionCount}) {
  if (type === 'choice') return snapshotChoice(id, instructions, criteria, optionCount);
  if (type === 'score') return snapshotScore(id, instructions, criteria, optionCount);
  return snapshotNoul(id, instructions, criteria);
}

function snapshotChoice(id, instructions, criteria, expectedCount) {
  const entries = recordEntries(criteria, `Question ${id} choice criteria`);
  if (entries.length < 2) invalid(`Question ${id} needs at least 2 choice options.`);
  if (entries.length !== expectedCount) invalid(`Question ${id} choice criteria changed after preflight.`);
  const options = entries.map(([name, value]) => {
    text(name, `Question ${id} option name`);
    const descriptionText = description(value, `Question ${id} option description`);
    return {name, text: descriptionText ? `${name}: ${descriptionText}` : name};
  });
  return {id, type: 'choice', instructions, options};
}

function snapshotScore(id, instructions, criteria, expectedLength) {
  const levels = arrayValues(criteria, `Question ${id} score criteria`, expectedLength);
  if (levels.length < 2) invalid(`Question ${id} needs at least 2 score levels.`);
  const options = levels.map((item, index) => {
    const name = String(index);
    const descriptionText = description(item, `Question ${id} score level`);
    return {name, text: descriptionText ? `${name}: ${descriptionText}` : name};
  });
  return {id, type: 'score', instructions, options};
}

function snapshotNoul(id, instructions, criteria) {
  let descriptions = {false: null, true: null};
  if (criteria !== undefined && criteria !== null) {
    descriptions = Object.fromEntries(recordEntries(criteria, `Question ${id} noul criteria`));
    if (Object.keys(descriptions).some((key) => key !== 'false' && key !== 'true'))
      invalid(`Question ${id} noul criteria can only describe false and true.`);
    descriptions = {
      false: description(descriptions.false, `Question ${id} false description`),
      true: description(descriptions.true, `Question ${id} true description`)
    };
  }
  return {
    id,
    type: 'noul',
    instructions,
    options: [
      {name: 'false', text: descriptions.false ? `no: ${descriptions.false}` : 'no'},
      {name: 'true', text: descriptions.true ? `yes: ${descriptions.true}` : 'yes'}
    ]
  };
}

function preflightQuestions(questions) {
  const entries = recordEntries(questions, 'SafeDecide questions');
  if (entries.length < 1) invalid('SafeDecide accepts at least one question.');
  const questionData = [];
  for (const [id, value] of entries) {
    const question = readQuestion(id, value);
    question.optionCount = countQuestionOptions(question);
    questionData.push(question);
  }
  return questionData;
}

function snapshotQuestions(questionData) {
  return questionData.map(snapshotQuestion);
}

export function prepareSafeDecideSystemOne(state, questions, tokenizer) {
  if (typeof tokenizer?.encodePair !== 'function') throw new TypeError('SafeDecide tokenizer is required.');
  const questionData = preflightQuestions(questions);
  const serializedState = stateText(state);
  text(serializedState, 'SafeDecide state');
  const entries = snapshotQuestions(questionData);
  const rows = [];
  const truncatedQuestions = [];

  for (const question of entries) {
    for (let optionIndex = 0; optionIndex < question.options.length; optionIndex++) {
      const option = question.options[optionIndex];
      text(option.text, `Question ${question.id} option text`);
      const first = `[${question.type}] option: ${option.text} || question: ${question.instructions}`;
      const encoded = tokenizer.encodePair(first, serializedState);
      rows.push({
        questionId: question.id,
        questionType: question.type,
        optionIndex,
        optionName: option.name,
        inputIds: encoded.inputIds,
        truncated: encoded.truncated
      });
      if (encoded.truncated && !truncatedQuestions.includes(question.id)) truncatedQuestions.push(question.id);
    }
  }

  return {
    questions: entries,
    rows,
    tokenCount: rows.reduce((total, row) => total + row.inputIds.length, 0),
    truncatedQuestions
  };
}

function softmax(logits, temperature) {
  const scaled = logits.map((logit) => logit / temperature);
  const maximum = scaled.reduce((current, logit) => Math.max(current, logit), Number.NEGATIVE_INFINITY);
  const exponentials = scaled.map((logit) => Math.exp(logit - maximum));
  const total = exponentials.reduce((sum, probability) => sum + probability, 0);
  if (!Number.isFinite(total) || total <= 0) throw inference('SafeDecide could not normalize option scores.');
  return exponentials.map((probability) => probability / total);
}

function finishAnswer(question, logits, temperatures, threshold) {
  const probabilities = softmax(logits, temperatures[question.type]);
  const selectedIndex = probabilities.reduce(
    (best, probability, index) => (probability > probabilities[best] ? index : best),
    0
  );
  const confidence = probabilities[selectedIndex];
  const action = confidence >= threshold ? 'automate' : 'escalate';
  const byName = Object.fromEntries(question.options.map(({name}, index) => [name, probabilities[index]]));
  if (question.type === 'choice')
    return {type: 'choice', choice: question.options[selectedIndex].name, probabilities: byName, confidence, action};
  if (question.type === 'score')
    return {
      type: 'score',
      score: probabilities.reduce((expected, probability, index) => expected + index * probability, 0),
      choice: question.options[selectedIndex].name,
      probabilities: byName,
      confidence,
      action
    };
  return {
    type: 'noul',
    noul: probabilities[1],
    probabilities: {false: probabilities[0], true: probabilities[1]},
    confidence,
    action
  };
}

export function finishSafeDecideSystemOne(prepared, logits, config, model = SAFEDECIDE_MODEL, alpha) {
  if (!Array.isArray(logits) || logits.length !== prepared.rows.length)
    throw inference('SafeDecide returned an unexpected number of option scores.');
  if (logits.some((value) => !Number.isFinite(value)))
    throw inference('SafeDecide returned a non-finite option score.');
  const rule =
    alpha === undefined ? config.selective_rule : config.selective_rules.find((item) => item.alpha === alpha);
  if (!rule) throw invalid('SafeDecide has no certified rule for this alpha level.');
  const threshold = rule.threshold === null ? Number.POSITIVE_INFINITY : rule.threshold;
  const answers = Object.create(null);

  let rowOffset = 0;
  for (const question of prepared.questions) {
    const questionLogits = logits.slice(rowOffset, rowOffset + question.options.length);
    rowOffset += question.options.length;
    answers[question.id] = finishAnswer(question, questionLogits, config.temperatures, threshold);
  }

  return {
    model,
    tokenCount: prepared.tokenCount,
    outputTokenCount: 0,
    ...(prepared.truncatedQuestions.length ? {truncated: true, truncatedQuestions: prepared.truncatedQuestions} : {}),
    guarantee: {alpha: rule.alpha, delta: rule.delta, statement: rule.statement},
    answers
  };
}
