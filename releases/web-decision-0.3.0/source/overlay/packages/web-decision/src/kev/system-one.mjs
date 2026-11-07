import {KEV_MAX_OPTIONS, WebDecisionError} from '../kev.js';

const MODEL_ID = 'kev-0.6b-q4f16';
const containerValue = Symbol('container value');
const roundingBuffer = new ArrayBuffer(8);
const roundingView = new DataView(roundingBuffer);
const invalid = (message) => {
  throw new WebDecisionError('INVALID_INPUT', message);
};

function addToBudget(budget, characters) {
  if (!budget) return;
  budget.used += characters;
  if (budget.used > budget.max) invalid('Kev input exceeds its supported text size.');
}

function checkBigIntBudget(value, budget) {
  if (!budget || value === 0n) return;
  const negative = value < 0n;
  const maxDigits = budget.max - budget.used - (negative ? 1 : 0);
  if (maxDigits < 1) invalid('Kev input exceeds its supported text size.');

  // Since 2^10 > 10^3, 2^ceil(10n/3) is strictly greater than 10^n.
  // This exact lower bound proves overflow without formatting the BigInt.
  const minimumOverflowBits = (10n * BigInt(maxDigits) + 2n) / 3n;
  const magnitude = negative ? -value : value;
  if (magnitude >> minimumOverflowBits !== 0n) invalid('Kev input exceeds its supported text size.');
}

function isMap(value) {
  try {
    Map.prototype.entries.call(value);
    return true;
  } catch {
    return false;
  }
}

function validateRecord(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  if (isMap(value)) return;
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && Object.getPrototypeOf(prototype) !== null) invalid(`${label} must be a plain object.`);
}

function recordEntries(value, label, maxEntries = Infinity, budget) {
  validateRecord(value, label);
  if (isMap(value)) {
    const entries = [];
    for (const [key, item] of Map.prototype.entries.call(value)) {
      if (entries.length >= maxEntries) invalid(`${label} exceeds its supported number of entries.`);
      if (typeof key !== 'string') invalid(`${label} must contain string keys.`);
      addToBudget(budget, key.length);
      entries.push([key, item]);
    }
    return entries;
  }

  const keys = Reflect.ownKeys(value);
  if (keys.length > maxEntries) invalid(`${label} exceeds its supported number of entries.`);
  const entries = [];
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    addToBudget(budget, key.length);
    entries.push([key, descriptor.value]);
  }
  return entries;
}

function arrayValues(value, label, maxValues = Infinity) {
  if (!Array.isArray(value)) invalid(`${label} must be an array.`);
  if (value.length > maxValues) invalid(`${label} exceeds its supported number of entries.`);
  const keys = Reflect.ownKeys(value);
  if (keys.length > maxValues + 1) invalid(`${label} cannot have extra properties.`);
  const values = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value')) invalid(`${label} must be a dense data array.`);
    values.push(descriptor.value);
  }
  for (const key of keys)
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= value.length))
      invalid(`${label} cannot have extra properties.`);
  return values;
}

function primitiveText(value, budget, arrayItem) {
  if (value === null) return '';
  if (typeof value === 'string') {
    const trimmed = arrayItem ? pythonWhitespacePrefixLength(value) : 0;
    addToBudget(budget, value.length - trimmed);
    return value;
  }
  if (typeof value === 'boolean') {
    const text = value ? 'True' : 'False';
    addToBudget(budget, text.length);
    return text;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) invalid('JSON numbers must be finite.');
    if (Number.isInteger(value) && !Number.isSafeInteger(value))
      invalid('Integer JSON numbers must be safe integers; use bigint for exact integers.');
    const text = String(value);
    addToBudget(budget, text.length);
    return text;
  }
  if (typeof value === 'bigint') {
    checkBigIntBudget(value, budget);
    const text = value.toString();
    addToBudget(budget, text.length);
    return text;
  }
  if (!value || typeof value !== 'object') invalid('Values must be JSON values.');
  return containerValue;
}

function enterContainer(frame, value, budget, ancestors) {
  if (ancestors.has(value)) invalid('JSON values cannot contain cycles.');
  ancestors.add(value);
  frame.container = true;
  frame.entered = true;
  frame.lines = [];
  frame.index = 0;
  if (Array.isArray(value)) {
    frame.kind = 'array';
    frame.children = arrayValues(value, 'JSON arrays', budget?.max ?? Infinity);
  } else {
    frame.kind = 'object';
    frame.children = recordEntries(value, 'JSON objects', budget?.max ?? Infinity);
  }
}

function appendRenderedChild(parent, text, budget) {
  const index = parent.index - 1;
  const padLength = parent.indent * 2;
  if (parent.kind === 'array') {
    const leading = pythonWhitespacePrefixLength(text);
    const trimmedPad = parent.arrayItem && index === 0 ? padLength : 0;
    addToBudget(budget, padLength + 2 - trimmedPad);
    const pad = ' '.repeat(padLength);
    const rendered = text.slice(leading);
    parent.lines.push(`${pad}- ${rendered}`);
    return;
  }

  const [key, item] = parent.children[index];
  const separator = item && typeof item === 'object' ? ':\n' : ': ';
  const trimmedPrefix = parent.arrayItem && index === 0 ? padLength + pythonWhitespacePrefixLength(key) : 0;
  addToBudget(budget, padLength + key.length + separator.length - trimmedPrefix);
  const pad = ' '.repeat(padLength);
  parent.lines.push(`${pad}${key}${separator}${text}`);
}

function pythonWhitespacePrefixLength(text) {
  let length = 0;
  for (const character of text) {
    const codePoint = character.codePointAt(0);
    const whitespace =
      (codePoint >= 0x09 && codePoint <= 0x0d) ||
      (codePoint >= 0x1c && codePoint <= 0x20) ||
      codePoint === 0x85 ||
      codePoint === 0xa0 ||
      codePoint === 0x1680 ||
      (codePoint >= 0x2000 && codePoint <= 0x200a) ||
      codePoint === 0x2028 ||
      codePoint === 0x2029 ||
      codePoint === 0x202f ||
      codePoint === 0x205f ||
      codePoint === 0x3000;
    if (!whitespace) break;
    length += character.length;
  }
  return length;
}

function pushChildFrame(frames, parent) {
  const childIndex = parent.index++;
  const child = parent.kind === 'array' ? parent.children[childIndex] : parent.children[childIndex][1];
  const arrayItem = parent.kind === 'array';
  frames.push({value: child, indent: parent.indent + 1, entered: false, arrayItem});
}

function render(value, budget) {
  const ancestors = new Set();
  const frames = [{value, indent: 0, entered: false}];
  let result;

  function finish(text) {
    const finished = frames.pop();
    if (finished.container) ancestors.delete(finished.value);
    if (frames.length === 0) {
      result = text;
      return;
    }
    appendRenderedChild(frames.at(-1), text, budget);
  }

  while (frames.length > 0) {
    const frame = frames.at(-1);
    if (!frame.entered) {
      const text = primitiveText(frame.value, budget, frame.arrayItem);
      if (text !== containerValue) finish(text);
      else enterContainer(frame, frame.value, budget, ancestors);
      continue;
    }

    if (frame.index < frame.children.length) {
      pushChildFrame(frames, frame);
      continue;
    }

    addToBudget(budget, Math.max(0, frame.lines.length - 1));
    finish(frame.lines.join('\n'));
  }

  return result;
}

function optionText(name, description, budget, nameAlreadyCounted = false) {
  if (!nameAlreadyCounted) addToBudget(budget, name.length);
  if (description === null || description === '') return name;
  const rendered = render(description, budget);
  addToBudget(budget, 2);
  return `${name}: ${rendered}`;
}

function questionFields(question, name) {
  validateRecord(question, `Question ${name}`);
  const fields = {};
  for (const key of Reflect.ownKeys(question)) {
    if (typeof key !== 'string' || !['type', 'instructions', 'criteria'].includes(key)) continue;
    const descriptor = Object.getOwnPropertyDescriptor(question, key);
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`Question ${name} fields must be enumerable data properties.`);
    fields[key] = descriptor.value;
  }
  if (fields.instructions === undefined && Object.hasOwn(fields, 'instructions'))
    invalid(`Question ${name} instructions must be a JSON value.`);
  return fields;
}

function choiceQuestion(name, fields, index, budget) {
  const entries = recordEntries(fields.criteria, `Question ${name} criteria`, KEV_MAX_OPTIONS, budget);
  if (entries.length < 1 || entries.length > KEV_MAX_OPTIONS)
    invalid(`Question ${name} needs 1 to ${KEV_MAX_OPTIONS} options.`);
  const options = entries.map(([key, description]) => {
    return {key, label: optionText(key, description, budget, true)};
  });
  const labels = options.map((option) => option.label);
  return {
    metadata: {name, type: 'choice', options},
    request: {instructions: render(fields.instructions ?? null, budget), options: labels},
    allowDuplicateOptions: true,
    index
  };
}

function scoreQuestion(name, fields, index, budget) {
  const levels = arrayValues(fields.criteria, `Question ${name} criteria`, KEV_MAX_OPTIONS);
  if (levels.length < 1 || levels.length > KEV_MAX_OPTIONS)
    invalid(`Question ${name} needs 1 to ${KEV_MAX_OPTIONS} score levels.`);
  const labels = levels.map((level) => render(level, budget));
  return {
    metadata: {name, type: 'score', options: labels.map((label, optionIndex) => ({key: String(optionIndex), label}))},
    request: {instructions: render(fields.instructions ?? null, budget), options: labels},
    allowDuplicateOptions: true,
    index
  };
}

function noulCriteriaValue(criteria, key, name) {
  if (isMap(criteria)) {
    if (!Map.prototype.has.call(criteria, key)) return null;
    return Map.prototype.get.call(criteria, key);
  }

  const descriptor = Object.getOwnPropertyDescriptor(criteria, key);
  if (!descriptor) return null;
  if (!descriptor.enumerable || !Object.hasOwn(descriptor, 'value'))
    invalid(`Question ${name} criteria must contain enumerable string data properties.`);
  return descriptor.value;
}

function noulQuestion(name, fields, budget) {
  const descriptions = {true: null, false: null};
  if (Object.hasOwn(fields, 'criteria') && fields.criteria !== null) {
    const criteria = fields.criteria;
    validateRecord(criteria, `Question ${name} criteria`);
    descriptions.true = noulCriteriaValue(criteria, 'true', name);
    descriptions.false = noulCriteriaValue(criteria, 'false', name);
  }
  const options = [optionText('no', descriptions.false, budget), optionText('yes', descriptions.true, budget)];
  return {
    metadata: {
      name,
      type: 'noul',
      options: [
        {key: 'false', label: options[0]},
        {key: 'true', label: options[1]}
      ]
    },
    request: {instructions: render(fields.instructions ?? null, budget), options}
  };
}

export function prepareKevSystemOne(state, questions, maxInputCharacters = Infinity) {
  const budget = Number.isSafeInteger(maxInputCharacters) ? {max: maxInputCharacters, used: 0} : null;
  const entries = recordEntries(questions, 'Questions');
  if (entries.length < 1) invalid('Provide at least one named question.');
  const metadata = [];
  const requestQuestions = [];
  const allowDuplicateOptionsFor = new Set();

  for (const [name, value] of entries) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`Question ${name} must be an object.`);
    const fields = questionFields(value, name);
    let prepared;
    if (fields.type === 'choice') prepared = choiceQuestion(name, fields, requestQuestions.length, budget);
    else if (fields.type === 'score') prepared = scoreQuestion(name, fields, requestQuestions.length, budget);
    else if (fields.type === 'noul') prepared = noulQuestion(name, fields, budget);
    else invalid(`Question ${name} type must be choice, score, or noul.`);

    metadata.push(prepared.metadata);
    requestQuestions.push(prepared.request);
    if (prepared.allowDuplicateOptions) allowDuplicateOptionsFor.add(prepared.index);
  }

  return {
    request: {state: render(state, budget), questions: requestQuestions},
    metadata,
    packing: {allowEmptyInstructions: true, allowEmptyOptions: true, allowDuplicateOptionsFor}
  };
}

function normalizedProbabilities(probabilities) {
  const total = probabilities.reduce((sum, probability) => sum + probability, 0);
  if (!Number.isFinite(total) || probabilities.some((probability) => !Number.isFinite(probability) || probability < 0))
    throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid probabilities.');
  if (total === 0) return probabilities.map(() => 1 / probabilities.length);
  return probabilities.map((probability) => probability / total);
}

function choiceConfidence(probabilities) {
  const count = probabilities.length;
  if (count === 1) return 1;
  const normalized = normalizedProbabilities(probabilities);
  return (Math.max(...normalized) - 1 / count) / (1 - 1 / count);
}

function scoreConfidence(probabilities) {
  const levels = probabilities.length;
  if (levels === 1) return 1;
  const normalized = normalizedProbabilities(probabilities);
  let mode = 0;
  for (let index = 1; index < levels; index++) if (normalized[index] > normalized[mode]) mode = index;
  const expectedDistance = normalized.reduce(
    (sum, probability, index) => sum + probability * Math.abs(index - mode),
    0
  );
  let uniformDistance = 0;
  for (let index = 0; index < levels; index++) uniformDistance += Math.abs(index - (levels - 1) / 2);
  const meanAbsoluteDeviation = uniformDistance / levels;
  return Math.max(0, 1 - expectedDistance / meanAbsoluteDeviation);
}

function roundFourDecimals(value) {
  if (value === 0) return value;
  // Match Python round(value, 4) for Kev's JSON response fields.
  roundingView.setFloat64(0, value, false);
  const bits = roundingView.getBigUint64(0, false);
  const negative = bits >> 63n !== 0n;
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & 0x000fffffffffffffn;
  const significand = exponentBits === 0 ? fraction : fraction | 0x0010000000000000n;
  const exponent = exponentBits === 0 ? -1074 : exponentBits - 1075;
  const numerator = significand * 625n;
  const shift = exponent + 4;
  // Typed outputs are probabilities/confidence or a score expectation in [0, 254],
  // so their binary exponent keeps shift negative.
  const denominator = 1n << BigInt(-shift);
  let units = numerator / denominator;
  const remainder = numerator % denominator;
  const doubledRemainder = remainder * 2n;
  if (doubledRemainder > denominator || (doubledRemainder === denominator && units % 2n === 1n)) units++;

  return (negative ? -Number(units) : Number(units)) / 10000;
}

function finishAnswer(metadata, answer) {
  const probabilities = answer.probabilities.map(({probability}) => probability);
  if (metadata.type === 'choice') {
    const probabilityEntries = metadata.options.map(({key}, index) => [key, roundFourDecimals(probabilities[index])]);
    const roundedProbabilities = Object.fromEntries(probabilityEntries);
    return {
      type: 'choice',
      choice: metadata.options[answer.selectedIndex].key,
      confidence: roundFourDecimals(choiceConfidence(probabilities)),
      probabilities: roundedProbabilities,
      probabilityEntries
    };
  }
  if (metadata.type === 'score') {
    const legend = Object.fromEntries(metadata.options.map(({label}, index) => [String(index), label]));
    const probabilityEntries = probabilities.map((probability, index) => [
      String(index),
      roundFourDecimals(probability)
    ]);
    const distribution = Object.fromEntries(probabilityEntries);
    const score = probabilities.reduce((sum, probability, index) => sum + index * probability, 0);
    return {
      type: 'score',
      score: roundFourDecimals(score),
      legend,
      probabilities: distribution,
      probabilityEntries,
      confidence: roundFourDecimals(scoreConfidence(probabilities))
    };
  }
  return {type: 'noul', noul: roundFourDecimals(probabilities[1])};
}

export function finishKevSystemOne(prepared, result, model = MODEL_ID) {
  const answerEntries = prepared.metadata.map((metadata, index) => [
    metadata.name,
    finishAnswer(metadata, result.answers[index])
  ]);
  const answers = Object.fromEntries(answerEntries);
  return {model, tokenCount: result.tokenCount, answers, answerEntries};
}
