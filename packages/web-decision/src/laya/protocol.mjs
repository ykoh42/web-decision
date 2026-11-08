import {WebDecisionError} from '../kev.js';
import {LAYA_MAX_HEAD_TOKENS, LAYA_MAX_OPTION_TOKENS, LAYA_MAX_OPTIONS, LAYA_MAX_SEQUENCE_TOKENS} from './contract.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const inference = (message) => new WebDecisionError('INFERENCE_ERROR', message);
const questionFields = new Set(['type', 'instructions', 'criteria', 'labels']);

function serializeJson(value, seen = new WeakSet()) {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw invalid('Laya JSON values must contain finite numbers.');
    return JSON.stringify(value);
  }
  if (!value || typeof value !== 'object') throw invalid('Laya state and criteria must contain plain JSON values.');
  if (seen.has(value)) throw invalid('Laya state and criteria cannot contain cycles.');
  seen.add(value);
  if (Array.isArray(value)) {
    const result = `[${value.map((item) => serializeJson(item, seen)).join(', ')}]`;
    seen.delete(value);
    return result;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null)
    throw invalid('Laya state and criteria must contain plain JSON values.');
  const entries = Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid('Laya JSON values cannot contain accessors or symbol keys.');
    return `${JSON.stringify(key)}: ${serializeJson(descriptor.value, seen)}`;
  });
  seen.delete(value);
  return `{${entries.join(', ')}}`;
}

function renderCriterion(value) {
  return typeof value === 'string' ? value : serializeJson(value);
}

function entries(value, id) {
  if (Array.isArray(value)) {
    if (value.some((item) => typeof item !== 'string')) throw invalid(`Question ${id} option labels must be strings.`);
    return value.map((item) => [item, null]);
  }
  if (
    !value ||
    typeof value !== 'object' ||
    (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)
  )
    throw invalid(`Question ${id} criteria must be an array or plain object.`);
  return Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`Question ${id} criteria cannot contain accessors or symbol keys.`);
    return [key, descriptor.value];
  });
}

function snapshotQuestions(questions) {
  if (!questions || typeof questions !== 'object' || Array.isArray(questions))
    throw invalid('Laya questions must be a named object.');
  const ids = Reflect.ownKeys(questions);
  if (ids.length < 1 || ids.some((id) => typeof id !== 'string'))
    throw invalid('Laya needs at least one named question.');
  return ids.map((id) => {
    const descriptor = Object.getOwnPropertyDescriptor(questions, id);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw invalid(`Question ${id} cannot be an accessor.`);
    const question = descriptor.value;
    if (!question || typeof question !== 'object' || Array.isArray(question))
      throw invalid(`Question ${id} must be an object.`);
    const value = Object.create(null);
    for (const key of Reflect.ownKeys(question)) {
      const field = Object.getOwnPropertyDescriptor(question, key);
      if (typeof key !== 'string' || !questionFields.has(key) || !field || !Object.hasOwn(field, 'value'))
        throw invalid(`Question ${id} has an unsupported field.`);
      value[key] = field.value;
    }
    if (!['choice', 'score', 'noul'].includes(value.type))
      throw invalid(`Question ${id} type must be choice, score, or noul.`);
    if (typeof value.instructions !== 'string' || !value.instructions.trim())
      throw invalid(`Question ${id} needs non-empty string instructions.`);
    if (value.labels !== undefined && value.type !== 'noul')
      throw invalid(`Question ${id} labels are only supported for noul.`);
    return {id, value};
  });
}

function prepareChoice(id, question) {
  const criteria = entries(question.criteria, id);
  if (criteria.length < 2 || criteria.length > LAYA_MAX_OPTIONS)
    throw invalid(`Question ${id} needs 2 to ${LAYA_MAX_OPTIONS} choice options.`);
  const labels = criteria.map(([label]) => label);
  if (labels.some((label) => !label.trim()) || new Set(labels).size !== labels.length)
    throw invalid(`Question ${id} choice labels must be distinct non-empty strings.`);
  return {
    id,
    type: 'choice',
    qtype: 0,
    labels,
    options: criteria.map(([label, description]) =>
      description == null || description === '' ? label : `${label}: ${renderCriterion(description)}`
    )
  };
}

function prepareScore(id, question) {
  if (!Array.isArray(question.criteria) || question.criteria.length < 2 || question.criteria.length > LAYA_MAX_OPTIONS)
    throw invalid(`Question ${id} score needs 2 to ${LAYA_MAX_OPTIONS} ordered levels.`);
  if (question.criteria.some((criterion) => criterion === null))
    throw invalid(`Question ${id} score levels cannot be null.`);
  return {
    id,
    type: 'score',
    qtype: 1,
    labels: question.criteria.map((_, index) => String(index)),
    legend: question.criteria.map(renderCriterion),
    options: question.criteria.map((criterion, index) => `level ${index}: ${renderCriterion(criterion)}`)
  };
}

function prepareNoulCriteria(id, value) {
  const criteria = Object.create(null);
  if (value === undefined || value === null) return criteria;
  if (typeof value !== 'object' || Array.isArray(value))
    throw invalid(`Question ${id} noul criteria must be an object.`);
  for (const [key, item] of entries(value, id)) {
    if (key !== 'false' && key !== 'true') throw invalid(`Question ${id} noul criteria accepts false and true only.`);
    criteria[key] = item;
  }
  return criteria;
}

function prepareNoulLabels(id, value) {
  const supplied = value ?? {false: 'false', true: 'true'};
  if (!supplied || typeof supplied !== 'object' || Array.isArray(supplied))
    throw invalid(`Question ${id} noul labels must contain false and true strings.`);
  const entries = Reflect.ownKeys(supplied).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(supplied, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`Question ${id} noul labels must contain false and true strings.`);
    return [key, descriptor.value];
  });
  const labels = Object.fromEntries(entries);
  if (
    entries.length !== 2 ||
    entries.some(([key]) => key !== 'false' && key !== 'true') ||
    typeof labels.false !== 'string' ||
    typeof labels.true !== 'string' ||
    !labels.false.trim() ||
    !labels.true.trim() ||
    labels.false.trim() === labels.true.trim()
  )
    throw invalid(`Question ${id} noul labels must contain distinct non-empty false and true strings.`);
  return labels;
}

function prepareNoul(id, question) {
  const criteria = prepareNoulCriteria(id, question.criteria);
  const labels = prepareNoulLabels(id, question.labels);
  const falseText = criteria.false;
  const trueText = criteria.true;
  return {
    id,
    type: 'noul',
    qtype: 2,
    labels: ['false', 'true'],
    options: [
      `${labels.false.trim()}: ${falseText == null || falseText === '' ? 'no, the statement does not hold' : renderCriterion(falseText)}`,
      `${labels.true.trim()}: ${trueText == null || trueText === '' ? 'yes, the statement holds' : renderCriterion(trueText)}`
    ]
  };
}

function prepareQuestion(id, question) {
  if (question.type === 'choice') return prepareChoice(id, question);
  if (question.type === 'score') return prepareScore(id, question);
  return prepareNoul(id, question);
}

function buildHead(question, instructions, tokenizer) {
  const mask = tokenizer.special.maskToken;
  const sanitized = String(instructions).replaceAll(mask, ' ');
  const instructionIds = tokenizer.encode(`${question.type} question: ${sanitized}`);
  let optionTruncated = false;
  const optionIds = question.options.map((option) => {
    const ids = tokenizer.encode(` ${option.replaceAll(mask, ' ')}`);
    if (ids.length > LAYA_MAX_OPTION_TOKENS) optionTruncated = true;
    return ids.slice(0, LAYA_MAX_OPTION_TOKENS);
  });
  const perOption = Math.min(LAYA_MAX_OPTION_TOKENS, Math.floor((LAYA_MAX_HEAD_TOKENS - 11) / optionIds.length) - 1);
  if (perOption < 1) throw invalid(`Question ${question.id} has too many options for the Laya head.`);
  for (let index = 0; index < optionIds.length; index++) {
    if (optionIds[index].length > perOption) optionTruncated = true;
    optionIds[index] = optionIds[index].slice(0, perOption);
  }
  const instructionBudget = LAYA_MAX_HEAD_TOKENS - 3 - optionIds.reduce((sum, ids) => sum + ids.length + 1, 0);
  const headTokens = instructionIds.slice(0, Math.max(8, instructionBudget));
  const inputIds = [tokenizer.special.clsId, ...headTokens, tokenizer.special.sepId];
  const markerPositions = [];
  for (const ids of optionIds) {
    markerPositions.push(inputIds.length);
    inputIds.push(tokenizer.special.maskId, ...ids);
  }
  inputIds.push(tokenizer.special.sepId);
  return {
    inputIds,
    markerPositions,
    instructionTruncated: headTokens.length < instructionIds.length,
    optionTruncated,
    distinctOptions: new Set(optionIds.map((ids) => ids.join(','))).size
  };
}

export function prepareLayaSystemOne(state, questions, tokenizer) {
  if (typeof state !== 'string' && (!state || typeof state !== 'object'))
    throw invalid('Laya state must be a string, plain object, or array.');
  const stateText = typeof state === 'string' ? state : serializeJson(state);
  const entries = snapshotQuestions(questions).map(({id, value}) => ({
    ...prepareQuestion(id, value),
    instructions: value.instructions
  }));
  const stateIds = tokenizer.encode(stateText.replaceAll(tokenizer.special.maskToken, ' '));
  const rows = entries.map((question) => {
    const head = buildHead(question, question.instructions, tokenizer);
    if (head.markerPositions.length !== question.options.length)
      throw invalid(`Question ${question.id} does not fit the Laya option head.`);
    const stateRoom = Math.max(0, LAYA_MAX_SEQUENCE_TOKENS - head.inputIds.length - 1);
    const retainedState = Array.isArray(state)
      ? stateIds.slice(Math.max(0, stateIds.length - stateRoom))
      : stateIds.slice(0, stateRoom);
    const inputIds = [...head.inputIds, ...retainedState, tokenizer.special.sepId].slice(0, LAYA_MAX_SEQUENCE_TOKENS);
    const markerPositions = head.markerPositions.filter((position) => position < LAYA_MAX_SEQUENCE_TOKENS);
    if (markerPositions.length !== question.options.length)
      throw invalid(`Question ${question.id} has option markers beyond the Laya context limit.`);
    return {
      ...question,
      inputIds,
      markerPositions,
      tokenCount: inputIds.length,
      stateTruncated: retainedState.length < stateIds.length,
      instructionTruncated: head.instructionTruncated,
      optionTruncated: head.optionTruncated,
      distinctOptions: head.distinctOptions
    };
  });
  return {rows, tokenCount: rows.reduce((sum, row) => sum + row.tokenCount, 0)};
}

function softmax(values) {
  if (!Array.isArray(values) || values.length < 2 || values.some((value) => !Number.isFinite(value)))
    throw inference('Laya returned invalid logits.');
  const maximum = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - maximum));
  const total = exp.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) throw inference('Laya returned invalid probabilities.');
  return exp.map((value) => value / total);
}

function rounded(value) {
  return Math.round(value * 10000) / 10000;
}

function entropyConfidence(probabilities) {
  const entropy = -probabilities.reduce(
    (sum, probability) => sum + probability * Math.log(Math.max(probability, 1e-12)),
    0
  );
  return rounded(Math.max(0, Math.min(1, 1 - entropy / Math.log(probabilities.length))));
}

export function finishLayaSystemOne(prepared, raw, temperatures, model) {
  if (
    !Array.isArray(raw?.rows) ||
    raw.rows.length !== prepared.rows.length ||
    raw.rows.some(
      (item, index) =>
        !Array.isArray(item?.logits) ||
        item.logits.length !== prepared.rows[index].options.length ||
        !Array.isArray(item?.actLogits) ||
        item.actLogits.length !== 2
    )
  )
    throw inference('Laya returned an unexpected output shape.');

  const answers = Object.create(null);
  for (let index = 0; index < prepared.rows.length; index++) {
    const question = prepared.rows[index];
    const output = raw.rows[index];
    const temperature =
      temperatures.byOptions[`${question.type}:${bucket(question.options.length)}`] ??
      temperatures.byType[question.qtype];
    const probabilities = softmax(output.logits.map((logit) => logit / temperature));
    const actionProbabilities = softmax(output.actLogits);
    const confidence =
      question.type === 'noul' ? rounded(Math.max(...probabilities)) : entropyConfidence(probabilities);
    const common = {
      type: question.type,
      confidence,
      answerConfidence: rounded(Math.max(...probabilities)),
      actionProbability: rounded(actionProbabilities[0]),
      actionLogits: [...output.actLogits]
    };
    if (question.type === 'choice') {
      const values = Object.fromEntries(
        question.labels.map((label, option) => [label, rounded(probabilities[option])])
      );
      const choiceIndex = probabilities.indexOf(Math.max(...probabilities));
      answers[question.id] = {...common, type: 'choice', choice: question.labels[choiceIndex], probabilities: values};
    } else if (question.type === 'score') {
      const expected = probabilities.reduce((sum, probability, level) => sum + probability * level, 0);
      answers[question.id] = {
        ...common,
        type: 'score',
        score: rounded(expected),
        expected: rounded(expected),
        legend: Object.fromEntries(question.legend.map((label, option) => [String(option), label])),
        probabilities: Object.fromEntries(
          probabilities.map((probability, option) => [String(option), rounded(probability)])
        )
      };
    } else {
      answers[question.id] = {
        ...common,
        type: 'noul',
        noul: rounded(probabilities[1]),
        probabilities: {false: rounded(probabilities[0]), true: rounded(probabilities[1])}
      };
    }
  }
  const truncatedQuestions = prepared.rows
    .filter((row) => row.stateTruncated || row.instructionTruncated || row.optionTruncated)
    .map((row) => row.id);
  const collapsedOptions = Object.fromEntries(
    prepared.rows
      .filter((row) => row.distinctOptions < row.options.length)
      .map((row) => [row.id, {total: row.options.length, distinct: row.distinctOptions}])
  );
  return {
    model,
    tokenCount: prepared.tokenCount,
    ...(truncatedQuestions.length ? {truncated: true, truncatedQuestions} : {}),
    ...(Object.keys(collapsedOptions).length ? {collapsedOptions} : {}),
    answers
  };
}

function bucket(count) {
  return count <= 2 ? '2' : count <= 5 ? '3-5' : count <= 10 ? '6-10' : '11+';
}

export function validateLayaDecisionRequest(request) {
  if (!request || typeof request !== 'object' || Array.isArray(request))
    throw invalid('A Laya decision request is required.');
  if (typeof request.state !== 'string') throw invalid('Laya decide() state must be a string.');
  if (!Array.isArray(request.questions) || request.questions.length < 1)
    throw invalid('Laya needs at least one generic question.');
  return request;
}
