import {WebDecisionError} from '../kev.js';
import {OPENDECIDER_NANO_MODEL} from './contract.mjs';

const invalid = (message) => {
  throw new WebDecisionError('INVALID_INPUT', message);
};

function recordEntries(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && Object.getPrototypeOf(prototype) !== null) invalid(`${label} must be a plain object.`);
  const entries = [];
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    entries.push([key, descriptor.value]);
  }
  return entries;
}

function arrayValues(value, label) {
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

function serializeStateValue(value, ancestors = new Set()) {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) invalid('State numbers must be finite.');
    return JSON.stringify(value);
  }
  if (!value || typeof value !== 'object') invalid('State must be a string or a JSON value.');
  if (ancestors.has(value)) invalid('State cannot contain a cycle.');

  ancestors.add(value);
  let serialized;
  if (Array.isArray(value)) {
    serialized = `[${arrayValues(value, 'State arrays')
      .map((item) => serializeStateValue(item, ancestors))
      .join(', ')}]`;
  } else {
    serialized = `{${recordEntries(value, 'State objects')
      .map(([key, item]) => `${JSON.stringify(key)}: ${serializeStateValue(item, ancestors)}`)
      .join(', ')}}`;
  }
  ancestors.delete(value);
  return serialized;
}

function serializeState(state) {
  return typeof state === 'string' ? state : serializeStateValue(state);
}

function assertText(value, label) {
  if (typeof value !== 'string' || !value.trim()) invalid(`${label} must be a non-empty string.`);
  return value;
}

function renderOption(name, description) {
  if (description !== null && typeof description !== 'string') invalid('Option descriptions must be strings or null.');
  return description && description !== name ? `${name}: ${description}` : name;
}

function choiceOptions(criteria) {
  let entries;
  if (Array.isArray(criteria)) {
    entries = arrayValues(criteria, 'Choice criteria').map((name) => [name, null]);
  } else {
    entries = recordEntries(criteria, 'Choice criteria');
  }
  if (entries.length < 2) invalid('Choice criteria must contain at least 2 options.');
  const names = new Set();
  const options = entries.map(([name, description]) => {
    assertText(name, 'Choice option names');
    if (names.has(name)) invalid('Choice option names must be unique.');
    names.add(name);
    return {name, rendered: renderOption(name, description)};
  });
  if (new Set(options.map(({rendered}) => rendered)).size !== options.length)
    invalid('Choice criteria render to duplicate model options.');
  return options;
}

function scoreOptions(criteria) {
  const levels = arrayValues(criteria, 'Score criteria');
  if (levels.length < 2) invalid('Score criteria must contain at least 2 levels.');
  return levels.map((description, index) => {
    assertText(description, 'Score levels');
    const name = String(index);
    return {name, label: description, rendered: renderOption(name, description)};
  });
}

function noulOptions(criteria) {
  let descriptions = {true: 'Yes', false: 'No'};
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

export function prepareOpenDeciderNanoSystemOne(state, questions) {
  const entries = recordEntries(questions, 'Questions');
  if (entries.length < 1) invalid('Provide at least 1 named question.');

  const metadata = [];
  const rows = entries.map(([name, question]) => {
    if (!question || typeof question !== 'object' || Array.isArray(question))
      invalid(`Question ${name} must be an object.`);
    const fields = Object.fromEntries(recordEntries(question, `Question ${name}`));
    if (Object.keys(fields).some((key) => !['type', 'instructions', 'criteria'].includes(key)))
      invalid(`Question ${name} has an unsupported field.`);
    const instructions = assertText(fields.instructions, `Question ${name} instructions`);
    let type, options;
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
    metadata.push({name, type, options});
    return {instructions, options: options.map(({rendered}) => rendered)};
  });

  return {request: {state: serializeState(state), questions: rows}, metadata};
}

function probabilityRecord(options, answer) {
  return Object.fromEntries(options.map(({name}, index) => [name, answer.probabilities[index].probability]));
}

function finishAnswer(type, options, answer) {
  const probabilities = probabilityRecord(options, answer);
  const confidence = Math.max(...Object.values(probabilities));
  if (type === 'choice') return {type, choice: options[answer.selectedIndex].name, probabilities, confidence};
  if (type === 'score')
    return {
      type,
      score: answer.selectedIndex,
      expected: options.reduce((sum, option, index) => sum + index * probabilities[option.name], 0),
      legend: Object.fromEntries(options.map(({name, label}, index) => [String(index), label ?? name])),
      probabilities,
      confidence
    };
  const yes = probabilities.yes;
  return {type, noul: yes, probabilities: {true: yes, false: 1 - yes}, confidence: Math.max(yes, 1 - yes)};
}

export function finishOpenDeciderNanoSystemOne(prepared, result, packed) {
  const answers = Object.fromEntries(
    prepared.metadata.map(({name, type, options}, index) => {
      const answer = finishAnswer(type, options, result.answers[index]);
      return [name, packed.questions[index].truncated ? {...answer, truncated: true} : answer];
    })
  );
  return {
    model: OPENDECIDER_NANO_MODEL,
    tokenCount: result.tokenCount,
    ...(result.truncated ? {truncated: true} : {}),
    answers
  };
}
