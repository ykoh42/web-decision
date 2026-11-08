import {WebDecisionError} from '../kev.js';
import {createSkejjaPrompt} from './tokenizer.mjs';
import {SKEJJA_MAX_CHOICES, SKEJJA_MAX_SCORE_LEVELS, SKEJJA_INT8_PROFILE} from './contract.mjs';

const QUESTION_FIELDS = new Set(['type', 'instructions', 'criteria']);
const NOUL_PATTERN = /^Does the user want to (.*)\?$/;
const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);

function serializeJson(value, stack = new Set()) {
  if (value === null) return 'null';
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'boolean') return String(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (!value || typeof value !== 'object' || stack.has(value))
    throw invalid('Skejja values must be valid JSON values.');

  stack.add(value);
  try {
    if (Array.isArray(value)) {
      const rows = [];
      for (let index = 0; index < value.length; index++) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
        if (!descriptor || !Object.hasOwn(descriptor, 'value'))
          throw invalid('Skejja arrays must contain plain values.');
        rows.push(serializeJson(descriptor.value, stack));
      }
      return `[${rows.join(', ')}]`;
    }
    return `{${objectEntries(value, 'Skejja values')
      .map(([key, item]) => `${JSON.stringify(key)}: ${serializeJson(item, stack)}`)
      .join(', ')}}`;
  } finally {
    stack.delete(value);
  }
}

export function asSkejjaText(value) {
  if (value == null) return '';
  return typeof value === 'string' ? value : serializeJson(value);
}

function objectEntries(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`${label} must be an object.`);
  return Reflect.ownKeys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} must contain plain data properties.`);
    return [key, descriptor.value];
  });
}

function snapshotQuestion(value, id) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`Question ${id} must be an object.`);
  const question = Object.create(null);
  for (const [key, item] of objectEntries(value, `Question ${id}`)) {
    if (!QUESTION_FIELDS.has(key)) throw invalid(`Question ${id} has an unsupported field.`);
    question[key] = item;
  }
  return question;
}

function readQuestions(questions) {
  const entries = objectEntries(questions, 'Skejja questions');
  if (entries.length < 1) throw invalid('Skejja needs at least one question.');
  return entries.map(([id, value]) => [id, snapshotQuestion(value, id)]);
}

function questionBody(instructions, state) {
  return instructions ? `${instructions}\n\n${state}` : state;
}

function choiceQuestion(id, instructions, criteria, state, tokenizer) {
  const entries = objectEntries(criteria, `Question ${id} criteria`);
  if (entries.length < 2 || entries.length > SKEJJA_MAX_CHOICES)
    throw invalid(`Question ${id} needs 2 to ${SKEJJA_MAX_CHOICES} choice options.`);
  if (entries.some(([name]) => !name)) throw invalid(`Question ${id} has an empty option name.`);
  const names = entries.map(([name]) => name);
  const labels = entries.map(([name, description]) => asSkejjaText(description) || name.replaceAll('_', ' '));
  const prompt = createSkejjaPrompt(tokenizer, labels, questionBody(instructions, state));
  return {id, type: 'choice', names, labels, ...prompt};
}

function scoreQuestion(id, instructions, criteria, state, tokenizer) {
  if (!Array.isArray(criteria) || criteria.length < 2 || criteria.length > SKEJJA_MAX_SCORE_LEVELS)
    throw invalid(`Question ${id} score criteria need 2 to ${SKEJJA_MAX_SCORE_LEVELS} ordered levels.`);
  const values = [];
  for (let index = 0; index < criteria.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(criteria, String(index));
    if (!descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`Question ${id} score levels must be plain values.`);
    values.push(descriptor.value);
  }
  const labels = values.map(asSkejjaText);
  const prompt = createSkejjaPrompt(tokenizer, labels, questionBody(instructions, state));
  return {id, type: 'score', labels, values, ...prompt};
}

function noulQuestion(id, instructions, criteria, state, tokenizer, profile) {
  const hasCriteria = criteria != null;
  let label;
  if (hasCriteria) {
    const entries = objectEntries(criteria, `Question ${id} criteria`);
    const values = new Map(entries);
    if (!values.has('true')) throw invalid(`Question ${id} Noul criteria needs a true value.`);
    label = asSkejjaText(values.get('true'));
  } else {
    if (!instructions.trim()) throw invalid(`Question ${id} needs instructions when Noul criteria are omitted.`);
    label = instructions;
  }

  const match = label.trim().match(NOUL_PATTERN);
  const inferenceLabel = match ? match[1] : label;
  const head = hasCriteria ? instructions : '';
  const prompt = createSkejjaPrompt(tokenizer, [inferenceLabel], questionBody(head, state));
  return {
    id,
    type: 'noul',
    labels: [inferenceLabel],
    label,
    calibrationKey: JSON.stringify([profile.model, label, head]),
    ...prompt
  };
}

export function normalizeSkejjaNoulQuestion(question) {
  const input = snapshotQuestion(question, 'calibration');
  if (input.type !== 'noul') throw invalid('Skejja calibrates only Noul questions.');
  const instructions = asSkejjaText(input.instructions);
  const criteria = input.criteria;
  if (criteria == null) {
    if (!instructions.trim()) throw invalid('Skejja Noul calibration needs a question in instructions.');
    return {type: 'noul', instructions, criteria: null};
  }
  const entries = objectEntries(criteria, 'Question calibration criteria');
  const values = new Map(entries);
  if (!values.has('true')) throw invalid('Skejja Noul calibration criteria needs a true value.');
  return {type: 'noul', instructions, criteria: {true: asSkejjaText(values.get('true'))}};
}

function prepareQuestion(id, question, state, tokenizer, profile) {
  const type = question.type;
  if (!['choice', 'noul', 'score'].includes(type)) throw invalid(`Question ${id} type must be choice, noul or score.`);
  const instructions = asSkejjaText(question.instructions);
  const criteria = question.criteria;
  if (type === 'choice') return choiceQuestion(id, instructions, criteria, state, tokenizer);
  if (type === 'score') return scoreQuestion(id, instructions, criteria, state, tokenizer);
  return noulQuestion(id, instructions, criteria, state, tokenizer, profile);
}

function prepareQuestions(questions, state, tokenizer, profile) {
  const rows = readQuestions(questions).map(([id, question]) =>
    prepareQuestion(id, question, state, tokenizer, profile)
  );
  return {
    questions: rows,
    tokenCount: rows.reduce((total, row) => total + row.tokenCount, 0),
    truncated: rows.some((row) => row.truncated)
  };
}

export function prepareSkejjaSystemOne(state, questions, tokenizer, profile = SKEJJA_INT8_PROFILE) {
  const stateText = asSkejjaText(state);
  return prepareQuestions(questions, stateText, tokenizer, profile);
}

export function prepareSkejjaDecision(request, tokenizer) {
  const requestFields = new Map(objectEntries(request, 'Skejja request'));
  if ([...requestFields.keys()].some((key) => key !== 'state' && key !== 'questions'))
    throw invalid('Skejja request has an unsupported field.');
  const state = requestFields.get('state');
  const rawQuestions = requestFields.get('questions');
  if (typeof state !== 'string') throw invalid('Skejja state must be text.');
  if (!Array.isArray(rawQuestions) || rawQuestions.length < 1)
    throw invalid('Skejja request needs at least one question.');
  const questions = rawQuestions.map((question, index) => {
    if (!question || typeof question !== 'object' || Array.isArray(question))
      throw invalid(`Question ${index} must be an object.`);
    const entries = objectEntries(question, `Question ${index}`);
    if (entries.some(([key]) => key !== 'instructions' && key !== 'options'))
      throw invalid(`Question ${index} has an unsupported field.`);
    const fields = new Map(entries);
    const instructions = fields.get('instructions');
    const optionsInput = fields.get('options');
    if (typeof instructions !== 'string') throw invalid(`Question ${index} instructions must be text.`);
    if (!Array.isArray(optionsInput) || optionsInput.length < 2 || optionsInput.length > SKEJJA_MAX_CHOICES)
      throw invalid(`Question ${index} needs 2 to ${SKEJJA_MAX_CHOICES} options.`);
    const options = optionsInput.map((option, optionIndex) => {
      const descriptor = Object.getOwnPropertyDescriptor(optionsInput, String(optionIndex));
      if (!descriptor || !Object.hasOwn(descriptor, 'value') || typeof option !== 'string' || !option.trim())
        throw invalid(`Question ${index} has an empty or non-text option.`);
      return option;
    });
    const prompt = createSkejjaPrompt(tokenizer, options, questionBody(instructions, state));
    return {id: String(index), type: 'choice', options, labels: options, ...prompt};
  });
  return {
    questions,
    tokenCount: questions.reduce((total, question) => total + question.tokenCount, 0),
    truncated: questions.some((question) => question.truncated)
  };
}

function softmax(logits, temperature) {
  if (!Number.isFinite(temperature) || temperature <= 0) throw new TypeError('Invalid Skejja calibration temperature.');
  const scaled = logits.map((value) => value / temperature);
  const max = Math.max(...scaled);
  const exponents = scaled.map((value) => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

function bestIndex(probabilities) {
  return probabilities.indexOf(Math.max(...probabilities));
}

function choiceConfidence(probabilities) {
  const count = probabilities.length;
  return (Math.max(...probabilities) - 1 / count) / (1 - 1 / count);
}

function scoreConfidence(probabilities) {
  const count = probabilities.length;
  const mode = bestIndex(probabilities);
  const meanUniformDistance =
    Array.from({length: count}, (_, index) => Math.abs(index - (count - 1) / 2)).reduce(
      (sum, value) => sum + value,
      0
    ) / count;
  const expectedDistance = probabilities.reduce(
    (sum, probability, index) => sum + probability * Math.abs(index - mode),
    0
  );
  return Math.max(0, 1 - expectedDistance / meanUniformDistance);
}

function resultByOption(names, probabilities) {
  return Object.fromEntries(names.map((name, index) => [name, probabilities[index]]));
}

function finishTypedAnswer(question, logits, profile, calibration) {
  if (question.type === 'noul') {
    const coefficients = calibration.get(question.calibrationKey) ?? [profile.plattA, profile.plattB];
    const scaled = coefficients[0] * logits[0] + coefficients[1];
    const probability = scaled >= 0 ? 1 / (1 + Math.exp(-scaled)) : Math.exp(scaled) / (1 + Math.exp(scaled));
    return {type: 'noul', noul: probability};
  }

  const probabilities = softmax(logits, profile.temperature);
  if (question.type === 'choice') {
    return {
      type: 'choice',
      choice: question.names[bestIndex(probabilities)],
      probabilities: resultByOption(question.names, probabilities),
      confidence: choiceConfidence(probabilities)
    };
  }

  const expected = probabilities.reduce((sum, probability, index) => sum + probability * index, 0);
  return {
    type: 'score',
    score: expected,
    probabilities: resultByOption(
      question.values.map((_, index) => String(index)),
      probabilities
    ),
    legend: Object.fromEntries(question.values.map((value, index) => [String(index), value])),
    confidence: scoreConfidence(probabilities)
  };
}

export function finishSkejjaSystemOne(prepared, rows, profile, calibration) {
  if (rows.length !== prepared.questions.length) throw new TypeError('Skejja result rows are incomplete.');
  const answers = Object.fromEntries(
    rows.map((row, index) => {
      const question = prepared.questions[index];
      return [question.id, finishTypedAnswer(question, row, profile, calibration)];
    })
  );
  return {
    model: profile.model,
    tokenCount: prepared.tokenCount,
    ...(prepared.truncated ? {truncated: true} : {}),
    answers
  };
}

export function finishSkejjaDecision(prepared, rows, profile) {
  if (rows.length !== prepared.questions.length) throw new TypeError('Skejja result rows are incomplete.');
  const answers = prepared.questions.map((question, index) => {
    const logits = rows[index];
    const probabilities = softmax(logits, profile.temperature);
    const selectedIndex = bestIndex(probabilities);
    return {
      selected: question.options[selectedIndex],
      selectedIndex,
      probabilities: question.options.map((option, optionIndex) => ({
        option,
        probability: probabilities[optionIndex],
        logit: logits[optionIndex]
      }))
    };
  });
  return {
    tokenCount: prepared.tokenCount,
    ...(prepared.truncated ? {truncated: true} : {}),
    answers
  };
}
