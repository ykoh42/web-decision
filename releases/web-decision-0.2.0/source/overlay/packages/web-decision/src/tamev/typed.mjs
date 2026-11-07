import {WebDecisionError} from '../kev.js';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const QUESTION_FIELDS = new Set(['type', 'instructions', 'criteria']);

function stringValue(value) {
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  return String(value);
}

export function renderTamevValue(value, indent = 0, stack = new Set()) {
  const pad = '  '.repeat(indent);
  if (value == null) return '';
  if (typeof value === 'string' || typeof value === 'boolean') return stringValue(value);
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (typeof value !== 'object') throw invalid('TAMEV values must be JSON text, numbers, booleans, lists or objects.');
  if (stack.has(value)) throw invalid('TAMEV values cannot contain cycles.');
  stack.add(value);
  try {
    if (Array.isArray(value)) {
      return value
        .map((item, index) => {
          if (!Object.hasOwn(value, index)) throw invalid('TAMEV lists cannot contain holes.');
          return `${pad}- ${renderTamevValue(item, indent + 1, stack).trimStart()}`;
        })
        .join('\n');
    }
    return objectEntries(value, 'TAMEV objects')
      .map(([key, item]) => {
        if (item && typeof item === 'object') return `${pad}${key}:\n${renderTamevValue(item, indent + 1, stack)}`;
        return `${pad}${key}: ${renderTamevValue(item, indent, stack)}`;
      })
      .join('\n');
  } finally {
    stack.delete(value);
  }
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

function questionEntries(questions) {
  const entries = objectEntries(questions, 'TAMEV questions');
  if (entries.length < 1) throw invalid('TAMEV needs at least one named question.');
  return entries.map(([id, value]) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`Question ${id} must be an object.`);
    const fields = Object.create(null);
    for (const [key, item] of objectEntries(value, `Question ${id}`)) {
      if (!QUESTION_FIELDS.has(key)) throw invalid(`Question ${id} has an unsupported field.`);
      fields[key] = item;
    }
    return [id, fields];
  });
}

function criteriaObject(criteria, id) {
  const entries = objectEntries(criteria, `Question ${id} criteria`);
  if (entries.length < 2) throw invalid(`Question ${id} needs at least 2 criteria.`);
  return entries;
}

function option(label, description) {
  const rendered = description == null || description === '' ? label : `${label}: ${renderTamevValue(description)}`;
  return {label, rendered};
}

function truthy(value) {
  return value != null && value !== false && value !== 0 && value !== '';
}

function prepareQuestion(id, question) {
  if (!['choice', 'noul', 'score'].includes(question.type))
    throw invalid(`Question ${id} type must be choice, noul or score.`);
  const instructions = question.instructions == null ? '' : renderTamevValue(question.instructions);
  if (question.type === 'choice') {
    const criteria = criteriaObject(question.criteria, id);
    const options = criteria.map(([label, description]) => option(label, description));
    if (options.some(({label}) => !label.trim())) throw invalid(`Question ${id} has an empty criterion name.`);
    return {
      id,
      type: 'choice',
      instructions,
      labels: options.map(({label}) => label),
      options: options.map(({rendered}) => rendered)
    };
  }
  if (question.type === 'noul') {
    const criteria = question.criteria == null ? [] : objectEntries(question.criteria, `Question ${id} criteria`);
    const values = new Map(criteria);
    const falseValue = values.get('false') || values.get('no') || values.get(false);
    const trueValue = values.get('true') || values.get('yes') || values.get(true);
    const no = option('No / False', truthy(falseValue) ? falseValue : null);
    const yes = option('Yes / True', truthy(trueValue) ? trueValue : null);
    return {id, type: 'noul', instructions, labels: ['false', 'true'], options: [no.rendered, yes.rendered]};
  }
  if (!Array.isArray(question.criteria) || question.criteria.length < 2)
    throw invalid(`Question ${id} score criteria need at least 2 ordered levels.`);
  const criteria = question.criteria.map((value, index) => {
    if (!Object.hasOwn(question.criteria, index)) throw invalid(`Question ${id} score criteria cannot contain holes.`);
    return value;
  });
  return {
    id,
    type: 'score',
    instructions,
    labels: criteria.map((_, index) => String(index)),
    legend: criteria,
    options: criteria.map((value) => renderTamevValue(value))
  };
}

export function prepareTamevSystemOne(state, questions, tokenizer) {
  const stateText = typeof state === 'string' ? state : renderTamevValue(state);
  const preparedQuestions = questionEntries(questions).map(([id, question]) => prepareQuestion(id, question));
  return {
    questions: preparedQuestions.map((question) => {
      const contextText = question.instructions
        ? `${stateText} [SEP] ${question.instructions}`.trim()
        : stateText.trim();
      const context = tokenizer.encodeContext(contextText);
      const options = question.options.map((text) => tokenizer.encodeOption(text));
      return {
        ...question,
        contextIds: context.ids,
        optionRows: options.map(({ids}) => ids),
        tokenCount: context.ids.length + options.reduce((count, optionTokens) => count + optionTokens.ids.length, 0),
        truncated: context.truncated || options.some((optionTokens) => optionTokens.truncated)
      };
    })
  };
}

function rounded(value) {
  return Number(value.toFixed(4));
}

function firstMaximumIndex(values) {
  let index = 0;
  for (let candidate = 1; candidate < values.length; candidate++) {
    if (values[candidate] > values[index]) index = candidate;
  }
  return index;
}

function choiceConfidence(probabilities, maximumIndex) {
  const count = probabilities.length;
  if (count <= 1) return 1;
  return rounded((probabilities[maximumIndex] - 1 / count) / (1 - 1 / count));
}

function scoreConfidence(probabilities) {
  const mode = firstMaximumIndex(probabilities);
  const expectedDistance = probabilities.reduce(
    (total, probability, index) => total + probability * Math.abs(index - mode),
    0
  );
  return rounded(1 - expectedDistance / (probabilities.length - 1));
}

function finishAnswer(question, probabilities) {
  if (question.type === 'noul') return {type: 'noul', noul: rounded(probabilities[1])};
  const values = Object.fromEntries(question.labels.map((label, index) => [label, rounded(probabilities[index])]));
  if (question.type === 'choice') {
    const choiceIndex = firstMaximumIndex(probabilities);
    return {
      type: 'choice',
      choice: question.labels[choiceIndex],
      confidence: choiceConfidence(probabilities, choiceIndex),
      probabilities: values
    };
  }
  const score = probabilities.reduce((total, probability, index) => total + index * probability, 0);
  return {
    type: 'score',
    score: rounded(score),
    confidence: scoreConfidence(probabilities),
    legend: Object.fromEntries(question.legend.map((label, index) => [String(index), label])),
    probabilities: values
  };
}

export function finishTamevSystemOne(prepared, rows, model) {
  if (rows.length !== prepared.questions.length) throw new TypeError('TAMEV Nano result rows are incomplete.');
  const answers = Object.fromEntries(
    prepared.questions.map((question, index) => [question.id, finishAnswer(question, rows[index].probabilities)])
  );
  const tokenCount = prepared.questions.reduce((count, question) => count + question.tokenCount, 0);
  return {
    model,
    tokenCount,
    ...(prepared.questions.some((question) => question.truncated) ? {truncated: true} : {}),
    answers
  };
}
