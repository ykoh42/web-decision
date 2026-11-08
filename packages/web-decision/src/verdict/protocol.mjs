import {WebDecisionError} from '../kev.js';
import {
  VERDICT_ABSTENTION_DESCRIPTION,
  VERDICT_ABSTENTION_ID,
  VERDICT_GLOBAL_TEMPERATURE,
  VERDICT_MAX_CANDIDATES,
  VERDICT_MAX_TOKENS,
  VERDICT_MODEL
} from './contract.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const inference = (message) => new WebDecisionError('INFERENCE_ERROR', message);
const RESERVED_IDS = new Set([VERDICT_ABSTENTION_ID, 'insufficient_evidence']);

function recordEntries(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`${label} must be a plain object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) throw invalid(`${label} must be a plain object.`);
  return Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} cannot contain symbols, non-enumerable fields, or accessors.`);
    return [key, descriptor.value];
  });
}

function arrayValues(value, label) {
  if (!Array.isArray(value)) throw invalid(`${label} must be an array.`);
  const result = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} cannot contain holes or accessors.`);
    result.push(descriptor.value);
  }
  for (const key of Reflect.ownKeys(value)) {
    if (key === 'length') continue;
    if (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= value.length)
      throw invalid(`${label} cannot have extra properties.`);
  }
  return result;
}

function checkedText(value, label) {
  if (typeof value !== 'string' || value.length === 0) throw invalid(`${label} must be a non-empty string.`);
  return value;
}

function checkedId(id, seen, label) {
  checkedText(id, `${label} ID`);
  if (RESERVED_IDS.has(id) || seen.has(id))
    throw invalid(`${label} IDs must be unique and cannot use a reserved abstention ID.`);
  seen.add(id);
}

function checkedChoiceQuestion(id, values) {
  if (Object.hasOwn(values, 'proposition') || Object.hasOwn(values, 'semantics'))
    throw invalid(`Verdict Choice ${id} has fields from another question type.`);
  const instructions = checkedText(values.instructions, `Verdict Choice ${id} instructions`);
  const entries = recordEntries(values.criteria, `Verdict Choice ${id} criteria`);
  if (entries.length < 2 || entries.length > VERDICT_MAX_CANDIDATES)
    throw invalid(`Verdict Choice ${id} needs 2 to ${VERDICT_MAX_CANDIDATES} options.`);
  const seen = new Set();
  const options = entries.map(([optionId, description]) => {
    checkedId(optionId, seen, `Verdict Choice ${id}`);
    return {id: optionId, description: checkedText(description, `Verdict Choice ${id} description`)};
  });
  return {id, type: 'choice', instructions, options};
}

function checkedScoreQuestion(id, values) {
  if (Object.hasOwn(values, 'proposition') || Object.hasOwn(values, 'semantics'))
    throw invalid(`Verdict Score ${id} has fields from another question type.`);
  const instructions = checkedText(values.instructions, `Verdict Score ${id} instructions`);
  const rawLevels = arrayValues(values.criteria, `Verdict Score ${id} criteria`);
  if (rawLevels.length < 2 || rawLevels.length > VERDICT_MAX_CANDIDATES)
    throw invalid(`Verdict Score ${id} needs 2 to ${VERDICT_MAX_CANDIDATES} ordered levels.`);
  const seen = new Set();
  let previousValue = -Infinity;
  const levels = rawLevels.map((level, index) => {
    const fields = Object.fromEntries(recordEntries(level, `Verdict Score ${id} level ${index}`));
    if (Object.keys(fields).some((key) => !['id', 'description', 'value'].includes(key)))
      throw invalid(`Verdict Score ${id} level ${index} has an unsupported field.`);
    checkedId(fields.id, seen, `Verdict Score ${id}`);
    const description = checkedText(fields.description, `Verdict Score ${id} description`);
    if (!Number.isFinite(fields.value) || fields.value <= previousValue)
      throw invalid(`Verdict Score ${id} values must be finite and strictly increasing.`);
    previousValue = fields.value;
    return {id: fields.id, description, value: fields.value};
  });
  return {id, type: 'score', instructions, levels};
}

function checkedNoulQuestion(id, values) {
  if (Object.hasOwn(values, 'instructions') || Object.hasOwn(values, 'criteria'))
    throw invalid(`Verdict Noul ${id} has fields from another question type.`);
  if (values.semantics !== 'conditional_on_sufficient_evidence_v2')
    throw invalid(`Verdict Noul ${id} must declare conditional_on_sufficient_evidence_v2 semantics.`);
  return {
    id,
    type: 'noul',
    proposition: checkedText(values.proposition, `Verdict Noul ${id} proposition`),
    semantics: values.semantics
  };
}

function checkedQuestion(id, question) {
  const entries = recordEntries(question, `Verdict question ${id}`);
  const values = Object.create(null);
  for (const [key, value] of entries) {
    if (!['type', 'instructions', 'criteria', 'proposition', 'semantics'].includes(key))
      throw invalid(`Verdict question ${id} has an unsupported field.`);
    values[key] = value;
  }
  if (values.type === 'choice') return checkedChoiceQuestion(id, values);
  if (values.type === 'score') return checkedScoreQuestion(id, values);
  if (values.type === 'noul') return checkedNoulQuestion(id, values);
  throw invalid(`Verdict question ${id} has an unsupported type.`);
}

function promptFor(state, question) {
  if (question.type === 'choice' || question.type === 'score')
    return `Question: ${question.instructions}\n\nContext:\n${state}`;
  return `Context:\n${state}\n\nEvaluate proposition: ${question.proposition}`;
}

function labelsFor(question) {
  if (question.type === 'choice') return question.options.map((option) => `It is ${option.description}`);
  if (question.type === 'score') return question.levels.map((level) => `${level.description} (Value: ${level.value})`);
  return [`true: ${question.proposition}`, `false: not ${question.proposition}`];
}

function candidateIdsFor(question) {
  if (question.type === 'choice') return question.options.map(({id}) => id);
  if (question.type === 'score') return question.levels.map(({id}) => id);
  return ['true', 'false'];
}

function encodeRow(state, question, tokenizer) {
  const labels = [...labelsFor(question), VERDICT_ABSTENTION_DESCRIPTION];
  const candidateIds = [...candidateIdsFor(question), VERDICT_ABSTENTION_ID];
  if (candidateIds.length < 3 || candidateIds.length > 25)
    throw invalid(`Verdict question ${question.id} exceeds the model's candidate capacity.`);
  const text = `${labels.map((label) => `<<LABEL>>${label}`).join('')}<<SEP>>${promptFor(state, question)}`;
  const encoded = tokenizer.encode(text);
  if (
    !encoded ||
    !Array.isArray(encoded.ids) ||
    encoded.ids.length < 2 ||
    encoded.ids.length > VERDICT_MAX_TOKENS ||
    encoded.ids.some((token) => !Number.isSafeInteger(token) || token < 0)
  )
    throw new WebDecisionError('INVALID_ASSET', 'Verdict tokenizer returned an invalid encoded row.');
  return {
    question,
    labels,
    ids: encoded.ids,
    candidateIds,
    truncated: encoded.truncated === true
  };
}

function snapshotQuestions(questions) {
  const entries = recordEntries(questions, 'Verdict questions');
  if (entries.length < 1) throw invalid('Verdict needs at least one named question.');
  return entries.map(([id, value]) => {
    checkedText(id, 'Verdict question name');
    return checkedQuestion(id, value);
  });
}

export function prepareVerdictSystemOne(state, questions, tokenizer) {
  if (typeof state !== 'string') throw invalid('Verdict context must be a string.');
  if (typeof tokenizer?.encode !== 'function') throw new TypeError('Verdict tokenizer is required.');
  const rows = snapshotQuestions(questions).map((question) => encodeRow(state, question, tokenizer));
  const truncatedQuestions = rows.filter((row) => row.truncated).map((row) => row.question.id);
  return {
    model: VERDICT_MODEL,
    rows,
    tokenCount: rows.reduce((total, row) => total + row.ids.length, 0),
    truncatedQuestions
  };
}

function probabilities(logits, temperature) {
  if (!Array.isArray(logits) || logits.length < 3 || logits.some((value) => !Number.isFinite(value)))
    throw inference('Verdict returned invalid candidate logits.');
  if (!Number.isFinite(temperature) || temperature <= 0) throw inference('Verdict calibration temperature is invalid.');
  const scaled = logits.map((value) => value / temperature);
  const maximum = Math.max(...scaled);
  const exponents = scaled.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

function concentration(values) {
  const entropy = -values.reduce((sum, value) => sum + (value > 0 ? value * Math.log(value) : 0), 0);
  return Math.max(0, Math.min(1, 1 - entropy / Math.log(values.length)));
}

function calibrationFor(calibrator, candidateCount) {
  const perCardinality = calibrator?.per_k?.[String(candidateCount)];
  const temperature = perCardinality ?? calibrator?.temperature ?? VERDICT_GLOBAL_TEMPERATURE;
  return {
    temperature,
    source: perCardinality === undefined ? 'global' : 'per-cardinality'
  };
}

function answerFor(question, ids, logits, values, calibration) {
  const probabilitiesById = Object.fromEntries(ids.map((id, index) => [id, values[index]]));
  const logitsById = Object.fromEntries(ids.map((id, index) => [id, logits[index]]));
  const selectedIndex = values.indexOf(Math.max(...values));
  const selectedId = ids[selectedIndex];
  const isAbstention = selectedId === VERDICT_ABSTENTION_ID;
  const common = {
    isAbstention,
    probabilities: probabilitiesById,
    logits: logitsById,
    calibrationStatus: 'calibrated_for_scope',
    calibrationTemperature: calibration.temperature,
    calibrationSource: calibration.source
  };

  if (question.type === 'choice')
    return {
      type: 'choice',
      choice: selectedId,
      selectedProbability: values[selectedIndex],
      concentration: concentration(values),
      ...common
    };

  if (question.type === 'score') {
    const substantiveMass = question.levels.reduce((sum, level) => sum + probabilitiesById[level.id], 0);
    const selectedLevel = question.levels.find((level) => level.id === selectedId);
    const expectedScore =
      !isAbstention && substantiveMass > 0
        ? question.levels.reduce((sum, level) => sum + level.value * (probabilitiesById[level.id] / substantiveMass), 0)
        : null;
    return {
      type: 'score',
      selectedLevelId: selectedId,
      selectedValue: selectedLevel?.value ?? null,
      expectedScore,
      abstentionProbability: probabilitiesById[VERDICT_ABSTENTION_ID],
      ...common
    };
  }

  const substantiveMass = probabilitiesById.true + probabilitiesById.false;
  return {
    type: 'noul',
    selectedOutcome: selectedId,
    pTrueGivenSufficientEvidence:
      !isAbstention && substantiveMass > 0 ? probabilitiesById.true / substantiveMass : null,
    pInsufficientEvidence: probabilitiesById[VERDICT_ABSTENTION_ID],
    ...common
  };
}

export function finishVerdictSystemOne(prepared, batchLogits, calibrator) {
  if (!Array.isArray(batchLogits) || batchLogits.length !== prepared?.rows?.length)
    throw inference('Verdict returned an unexpected number of output rows.');
  const answers = Object.create(null);
  for (let index = 0; index < prepared.rows.length; index++) {
    const row = prepared.rows[index];
    const logits = batchLogits[index];
    if (!Array.isArray(logits) || logits.length !== row.candidateIds.length)
      throw inference(`Verdict returned an invalid logit count for question ${row.question.id}.`);
    const calibration = calibrationFor(calibrator, row.candidateIds.length);
    const values = probabilities(logits, calibration.temperature);
    answers[row.question.id] = answerFor(row.question, row.candidateIds, logits, values, calibration);
  }
  return {
    model: prepared.model,
    tokenCount: prepared.tokenCount,
    outputTokenCount: 0,
    forwardPassCount: 1,
    ...(prepared.truncatedQuestions.length ? {truncated: true, truncatedQuestions: prepared.truncatedQuestions} : {}),
    answers
  };
}
