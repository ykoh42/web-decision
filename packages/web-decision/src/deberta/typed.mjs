import {WebDecisionError} from '../kev.js';

const MODEL_ID = 'open-jev-deberta-v3-large-q4f16';
const invalid = (message) => {
  throw new WebDecisionError('INVALID_INPUT', message);
};

function entries(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) invalid(`${label} must be a plain object.`);
  const result = [];
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    result.push([key, descriptor.value]);
  }
  return result;
}

function values(value, label) {
  if (!Array.isArray(value)) invalid(`${label} must be an array.`);
  const result = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value')) invalid(`${label} must be a dense data array.`);
    result.push(descriptor.value);
  }
  for (const key of Reflect.ownKeys(value))
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= value.length))
      invalid(`${label} cannot have extra properties.`);
  return result;
}

function questionFields(question, name) {
  const fields = Object.fromEntries(entries(question, `Question ${name}`));
  if (Object.keys(fields).some((key) => !['type', 'instructions', 'options'].includes(key)))
    invalid(`Question ${name} has an unsupported field.`);
  if (typeof fields.instructions !== 'string' || !fields.instructions.trim())
    invalid(`Question ${name} needs non-empty instructions.`);
  return fields;
}

function typedOptions(value, name, type) {
  const options = values(value, `Question ${name} options`);
  const minimum = 2;
  const maximum = type === 'score' ? 10 : 255;
  if (options.length < minimum || options.length > maximum)
    invalid(`Question ${name} needs ${minimum} to ${maximum} ${type === 'score' ? 'levels' : 'options'}.`);
  const seen = new Set();
  for (const option of options) {
    if (typeof option !== 'string' || !option.trim() || seen.has(option))
      invalid(`Question ${name} has an empty or duplicate option.`);
    seen.add(option);
  }
  return options;
}

/** Prepare the pinned Open-Jev typed protocol without changing its option text or order. */
export function prepareDebertaSystemOne(state, questions) {
  if (typeof state !== 'string') invalid('DeBERTa state must be a string.');
  const questionEntries = entries(questions, 'Questions');
  if (questionEntries.length < 1) invalid('Provide at least one named question.');

  const metadata = [];
  const requestQuestions = [];
  for (const [name, question] of questionEntries) {
    if (!name.trim()) invalid('Question names cannot be empty.');
    const fields = questionFields(question, name);
    if (fields.type === 'choice' || fields.type === 'score') {
      const options = typedOptions(fields.options, name, fields.type);
      metadata.push({name, type: fields.type, options});
      requestQuestions.push({instructions: fields.instructions, options});
    } else if (fields.type === 'noul') {
      if (Object.hasOwn(fields, 'options')) invalid(`Question ${name} Noul uses fixed no/yes options.`);
      metadata.push({name, type: 'noul', options: ['no', 'yes']});
      requestQuestions.push({instructions: fields.instructions, options: ['no', 'yes']});
    } else {
      invalid(`Question ${name} type must be choice, score, or noul.`);
    }
  }
  return {request: {state, questions: requestQuestions}, metadata};
}

function finishAnswer(question, answer) {
  const probabilities = answer.probabilities.map(({probability}) => probability);
  const confidence = Math.max(...probabilities);
  if (question.type === 'choice') {
    return {
      type: 'choice',
      choice: question.options[answer.selectedIndex],
      confidence,
      probabilities: Object.fromEntries(question.options.map((option, index) => [option, probabilities[index]]))
    };
  }
  if (question.type === 'score') {
    const distribution = Object.fromEntries(probabilities.map((probability, index) => [String(index), probability]));
    const legend = Object.fromEntries(question.options.map((option, index) => [String(index), option]));
    return {
      type: 'score',
      score: probabilities.reduce((sum, probability, index) => sum + index * probability, 0),
      legend,
      probabilities: distribution,
      confidence
    };
  }
  return {
    type: 'noul',
    noul: probabilities[1],
    probabilities: {false: probabilities[0], true: probabilities[1]},
    confidence
  };
}

export function finishDebertaSystemOne(prepared, result) {
  const answers = Object.fromEntries(
    prepared.metadata.map((question, index) => [question.name, finishAnswer(question, result.answers[index])])
  );
  return {model: MODEL_ID, tokenCount: result.tokenCount, answers};
}
