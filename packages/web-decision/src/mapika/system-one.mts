import {WebDecisionError} from '../kev.js';
import {packMapikaRequest} from '../mapika.js';

type ObjectEntry = [string, unknown];
type QuestionFields = Record<string, unknown>;
type QuestionMetadata =
  | {name: string; type: 'choice'; options: string[]}
  | {name: string; type: 'noul'}
  | {name: string; type: 'score'; levels: string[]; isolated: boolean};
type MapikaLabels = Parameters<typeof packMapikaRequest>[1];
type MapikaLimits = Parameters<typeof packMapikaRequest>[3];
type PackedRequest = ReturnType<typeof packMapikaRequest>;
type DecisionResult = ReturnType<typeof import('../mapika.js').unpackMapikaScores>;
interface PreparedQuestion {
  metadata: QuestionMetadata;
  requests: {instructions: string; options: string[]}[];
}
interface PreparedRow {
  metadata: QuestionMetadata;
  levelIndex: number | undefined;
  packed: PackedRequest;
}
interface PreparedRequest {
  rows: PreparedRow[];
  questions: QuestionMetadata[];
  tokenCount: number;
}

const MAX_OPTIONS = 255;
const MAX_SCORE_LEVELS = 10;
const MAX_ROWS = 512;
const NOUL_WITHOUT_INSTRUCTIONS = 'Which answer fits the context?';
const orderedEntries = Symbol('ordered entries');

function invalid(message: string): never {
  throw new WebDecisionError('INVALID_INPUT', message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function recordEntries(value: unknown, label: string): ObjectEntry[] {
  if (!isRecord(value)) invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) invalid(`${label} must be a plain object.`);
  const entries: ObjectEntry[] = [];
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      invalid(`${label} must contain enumerable string data properties.`);
    entries.push([key, descriptor.value]);
  }
  return entries;
}

function objectEntries(value: unknown, label: string): ObjectEntry[] {
  const descriptor = value && Object.getOwnPropertyDescriptor(value as object, orderedEntries);
  return descriptor ? (descriptor as PropertyDescriptor).value : recordEntries(value, label);
}

function arrayValues(value: unknown, label: string): unknown[] {
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

function jsonNumber(value: number) {
  if (!Number.isFinite(value)) invalid('JSON numbers must be finite.');
  if (Object.is(value, -0)) return '-0.0';
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e16)) {
    return value.toExponential().replace(/e([+-])(\d)$/, (_, sign, digit) => `e${sign}0${digit}`);
  }
  return String(value);
}

function pythonQuote(value: string) {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const escaped = value
    .replaceAll('\\', '\\\\')
    .replaceAll('\n', '\\n')
    .replaceAll('\r', '\\r')
    .replaceAll('\t', '\\t')
    .replaceAll(quote, `\\${quote}`);
  return `${quote}${escaped}${quote}`;
}

function pythonRepr(value: unknown, ancestors = new Set<object>(), depth = 0): string {
  if (typeof value === 'string') return pythonQuote(value);
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'number') return jsonNumber(value);
  if (!value || typeof value !== 'object') invalid('Choice list options must be JSON values.');
  if (ancestors.has(value)) invalid('JSON values cannot contain cycles.');
  if (depth > 256) invalid('JSON values are nested too deeply.');

  ancestors.add(value);
  let text;
  if (Array.isArray(value)) {
    text = `[${arrayValues(value, 'Choice list options')
      .map((item) => pythonRepr(item, ancestors, depth + 1))
      .join(', ')}]`;
  } else {
    text = `{${recordEntries(value, 'Choice list options')
      .map(([key, item]) => `${pythonQuote(key)}: ${pythonRepr(item, ancestors, depth + 1)}`)
      .join(', ')}}`;
  }
  ancestors.delete(value);
  return text;
}

function orderedRecord(entries: ObjectEntry[]) {
  return {[orderedEntries]: entries};
}

function indexedRecord(entries: ObjectEntry[], index: number) {
  const indexed: ObjectEntry[] = [['_index', index]];
  for (const [key, value] of entries) {
    const existing = indexed.findIndex(([current]) => current === key);
    if (existing < 0) indexed.push([key, value]);
    else indexed[existing] = [key, value];
  }
  return orderedRecord(indexed);
}

function jsonText(value: unknown, ancestors = new Set<object>(), depth = 0): string {
  if (value === null) return 'null';
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    return jsonNumber(value);
  }
  if (!value || typeof value !== 'object') invalid('Values must be JSON values.');
  if (ancestors.has(value)) invalid('JSON values cannot contain cycles.');
  if (depth > 256) invalid('JSON values are nested too deeply.');

  ancestors.add(value);
  let text;
  if (Array.isArray(value)) {
    text = `[${arrayValues(value, 'JSON arrays')
      .map((item) => jsonText(item, ancestors, depth + 1))
      .join(', ')}]`;
  } else {
    const entries = objectEntries(value, 'JSON objects');
    text = `{${entries
      .map(([key, item]) => `${JSON.stringify(key)}: ${jsonText(item, ancestors, depth + 1)}`)
      .join(', ')}}`;
  }
  ancestors.delete(value);
  return text;
}

function annotateIndices(value: unknown, ancestors = new Set<object>(), depth = 0): unknown {
  if (!value || typeof value !== 'object') return value;
  if (ancestors.has(value)) invalid('JSON values cannot contain cycles.');
  if (depth > 256) invalid('JSON values are nested too deeply.');

  ancestors.add(value);
  let result;
  if (Array.isArray(value)) {
    const values = arrayValues(value, 'JSON arrays');
    result = values.map((item, index) => {
      const child = annotateIndices(item, ancestors, depth + 1);
      if (values.length < 8) return child;
      if (isRecord(item)) return indexedRecord(objectEntries(child, 'JSON object'), index);
      if (Array.isArray(item))
        return orderedRecord([
          ['_index', index],
          ['value', child]
        ]);
      return orderedRecord([
        ['_index', index],
        ['value', child]
      ]);
    });
  } else {
    result = orderedRecord(
      objectEntries(value, 'JSON object').map(([key, child]) => [key, annotateIndices(child, ancestors, depth + 1)])
    );
  }
  ancestors.delete(value);
  return result;
}

function renderState(state: unknown) {
  if (typeof state === 'string') return state;
  return jsonText(annotateIndices(state));
}

function renderValue(value: unknown) {
  return typeof value === 'string' ? value : jsonText(value);
}

function sourceString(value: unknown) {
  return typeof value === 'string' ? value : pythonRepr(value);
}

function selectField(fields: QuestionFields, primary: string, alias: string | null, label: string) {
  if (Object.hasOwn(fields, primary)) return fields[primary];
  if (alias && Object.hasOwn(fields, alias)) return fields[alias];
  invalid(`${label} is required.`);
}

function questionFields(value: unknown, name: string): QuestionFields {
  const entries = recordEntries(value, `Question ${name}`);
  const allowed = new Set(['type', 'instructions', 'question', 'criteria', 'options', 'isolated']);
  if (entries.some(([key]) => !allowed.has(key))) invalid(`Question ${name} has an unsupported field.`);
  return Object.fromEntries(entries);
}

function choiceOptions(criteria: unknown, name: string) {
  let entries: ObjectEntry[];
  if (Array.isArray(criteria)) {
    entries = arrayValues(criteria, `Question ${name} criteria`).map((value) => [sourceString(value), null]);
  } else {
    entries = recordEntries(criteria, `Question ${name} criteria`);
  }
  // Upstream first converts list criteria into a dict, which keeps the first
  // occurrence of each option name while preserving its original order.
  entries = entries.filter(([key], index) => entries.findIndex(([candidate]) => candidate === key) === index);
  if (entries.length < 2 || entries.length > MAX_OPTIONS)
    invalid(`Question ${name} needs 2 to ${MAX_OPTIONS} Choice options.`);
  const options = entries.map(([key, description]) => {
    if (!key.trim()) invalid(`Question ${name} has an empty option name.`);
    return {key, label: description === null || description === '' ? key : `${key}: ${renderValue(description)}`};
  });
  if (new Set(options.map(({key}) => key)).size !== options.length)
    invalid(`Question ${name} has duplicate option names.`);
  if (new Set(options.map(({label}) => label)).size !== options.length)
    invalid(`Question ${name} has duplicate rendered options.`);
  return options;
}

function scoreLevels(criteria: unknown, name: string) {
  let values;
  if (Array.isArray(criteria)) {
    values = arrayValues(criteria, `Question ${name} criteria`);
  } else {
    const entries = recordEntries(criteria, `Question ${name} criteria`).map(([key, value]) => {
      const order = Number(key);
      if (!key.trim() || !Number.isFinite(order)) invalid(`Question ${name} Score keys must be numbers.`);
      return {order, value};
    });
    values = entries.sort((left, right) => left.order - right.order).map(({value}) => value);
  }
  if (values.length < 2 || values.length > MAX_SCORE_LEVELS)
    invalid(`Question ${name} needs 2 to ${MAX_SCORE_LEVELS} Score levels.`);
  return values.map((value) => renderValue(value));
}

function hasNoulDescription(criteria: unknown, name: string) {
  if (criteria == null) return false;
  const entries = recordEntries(criteria, `Question ${name} criteria`);
  return entries.some(([key, value]) => (key === 'true' || key === 'false') && value !== null && value !== '');
}

function questionInstructions(fields: QuestionFields, name: string, type: unknown, criteria: unknown) {
  const value = Object.hasOwn(fields, 'instructions')
    ? fields.instructions
    : Object.hasOwn(fields, 'question')
      ? fields.question
      : '';
  if ((type === 'noul' || type === 'bool') && (value === null || value === '')) {
    if (!hasNoulDescription(criteria, name))
      invalid(`Question ${name} needs instructions or a true/false description.`);
    return NOUL_WITHOUT_INSTRUCTIONS;
  }
  const instructions = renderValue(value);
  if (!instructions) invalid(`Question ${name} needs instructions.`);
  return instructions;
}

function noulOptions(criteria: unknown, name: string) {
  const descriptions = criteria == null ? [] : recordEntries(criteria, `Question ${name} criteria`);
  if (descriptions.some(([key]) => key !== 'true' && key !== 'false'))
    invalid(`Question ${name} Noul criteria can only describe true and false.`);
  const values = Object.fromEntries(descriptions);
  return [
    values.false == null || values.false === '' ? 'no' : `no: ${renderValue(values.false)}`,
    values.true == null || values.true === '' ? 'yes' : `yes: ${renderValue(values.true)}`
  ];
}

function questionCriteria(fields: QuestionFields, type: unknown, name: string) {
  if (type === 'noul' || type === 'bool') {
    if (Object.hasOwn(fields, 'criteria')) return fields.criteria;
    if (Object.hasOwn(fields, 'options')) return fields.options;
    return null;
  }
  return selectField(fields, 'criteria', 'options', `Question ${name} criteria`);
}

function prepareScoreQuestion(
  name: string,
  fields: QuestionFields,
  instructions: string,
  criteria: unknown
): PreparedQuestion {
  const levels = scoreLevels(criteria, name);
  if (Object.hasOwn(fields, 'isolated') && typeof fields.isolated !== 'boolean')
    invalid(`Question ${name} isolated must be a boolean.`);
  const isolated = fields.isolated !== false;
  const requests = isolated
    ? levels.map((level) => ({
        instructions: `${instructions}\nProposed answer: ${level.replace(/^\s*-?\d+\s*:\s*/, '')}\nDoes the proposed answer fit?`,
        options: ['no', 'yes']
      }))
    : [{instructions, options: levels.map((level, index) => `${index}: ${level}`)}];
  return {metadata: {name, type: 'score', levels, isolated}, requests};
}

function prepareTypedQuestion(
  name: string,
  fields: QuestionFields,
  type: unknown,
  instructions: string,
  criteria: unknown
): PreparedQuestion {
  if (type === 'choice') {
    const options = choiceOptions(criteria, name);
    return {
      metadata: {name, type, options: options.map(({key}) => key)},
      requests: [{instructions, options: options.map(({label}) => label)}]
    };
  }
  if (type === 'noul' || type === 'bool') {
    return {metadata: {name, type: 'noul'}, requests: [{instructions, options: noulOptions(criteria, name)}]};
  }
  if (type === 'score') return prepareScoreQuestion(name, fields, instructions, criteria);
  invalid(`Question ${name} type must be choice, score, noul, or bool.`);
}

function prepareQuestion(
  name: string,
  value: unknown,
  state: string,
  tokenizer: unknown,
  labels: MapikaLabels,
  limits: MapikaLimits,
  remainingRows: number
) {
  const fields = questionFields(value, name);
  const type = Object.hasOwn(fields, 'type') ? fields.type : 'choice';
  const criteria = questionCriteria(fields, type, name);
  const instructions = questionInstructions(fields, name, type, criteria);
  const prepared = prepareTypedQuestion(name, fields, type, instructions, criteria);
  if (prepared.requests.length > remainingRows) invalid(`A request can contain at most ${MAX_ROWS} independent rows.`);

  return {
    metadata: prepared.metadata,
    rows: prepared.requests.map((question, index) => ({
      metadata: prepared.metadata,
      levelIndex: prepared.metadata.type === 'score' && prepared.metadata.isolated ? index : undefined,
      packed: packMapikaRequest(tokenizer, labels, {state, questions: [question]}, limits)
    }))
  };
}

function uniqueTokenCount(rows: PreparedRow[]) {
  const ids = rows.map(({packed}) => packed.ids);
  if (ids.length === 1) return ids[0]!.length;
  const short = Math.min(...ids.map((row) => row.length));
  let common = 0;
  while (common < short && ids.every((row) => row[common] === ids[0]![common])) common++;
  return common + ids.reduce((total, row) => total + row.length - common, 0);
}

export function prepareMapikaSystemOne(
  state: unknown,
  questions: unknown,
  tokenizer: unknown,
  labels: MapikaLabels,
  limits: MapikaLimits
) {
  const stateText = renderState(state);
  const entries = recordEntries(questions, 'Questions');
  if (!entries.length) invalid('Provide at least one named question.');
  const prepared: QuestionMetadata[] = [];
  const rows: PreparedRow[] = [];
  for (const [name, question] of entries) {
    const next = prepareQuestion(name, question, stateText, tokenizer, labels, limits, MAX_ROWS - rows.length);
    prepared.push(next.metadata);
    rows.push(...next.rows);
  }
  return {
    rows,
    questions: prepared,
    tokenCount: uniqueTokenCount(rows)
  };
}

function certainty(probabilities: number[]) {
  const entropy = -probabilities.reduce(
    (sum, probability) => (probability > 0 ? sum + probability * Math.log(probability) : sum),
    0
  );
  return Math.max(0, 1 - entropy / Math.log(probabilities.length));
}

function rounded(value: number, digits: number) {
  return Number(value.toFixed(digits));
}

function normalized(probabilities: number[]) {
  const total = probabilities.reduce((sum, probability) => sum + probability, 0);
  return total === 0 ? probabilities.map(() => 1 / probabilities.length) : probabilities.map((value) => value / total);
}

function choiceConfidence(probabilities: number[]) {
  const values = normalized(probabilities);
  const count = values.length;
  if (count <= 1) return 1;
  return Math.max(0, Math.min(1, (count * Math.max(...values) - 1) / (count - 1)));
}

function scoreConfidence(probabilities: number[]) {
  const values = normalized(probabilities);
  if (values.length <= 1) return 1;
  const mode = values.indexOf(Math.max(...values));
  const spread = values.reduce((sum, probability, index) => sum + probability * Math.abs(index - mode), 0);
  const middle = (values.length - 1) / 2;
  const uniform = values.reduce((sum, _, index) => sum + Math.abs(index - middle), 0) / values.length;
  return Math.max(0, Math.min(1, 1 - spread / uniform));
}

function answerProbabilities(result: DecisionResult | undefined, index: number): number[] {
  const answer = result?.answers?.[0];
  if (!answer || !Array.isArray(answer.probabilities) || answer.probabilities.length !== index)
    throw new WebDecisionError('INFERENCE_ERROR', 'Mapika returned an unexpected typed answer.');
  return answer.probabilities.map(({probability}) => probability);
}

function finishQuestion(question: QuestionMetadata, results: DecisionResult[]) {
  if (question.type === 'choice') {
    const probabilities = answerProbabilities(results[0], question.options.length);
    const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
    return {
      type: 'choice',
      choice: question.options[selectedIndex],
      confidence: rounded(choiceConfidence(probabilities), 4),
      x_p_max: rounded(probabilities[selectedIndex]!, 4),
      certainty: rounded(certainty(probabilities), 4),
      probabilities: Object.fromEntries(question.options.map((key, index) => [key, rounded(probabilities[index]!, 4)]))
    };
  }
  if (question.type === 'noul') return {type: 'noul', noul: rounded(answerProbabilities(results[0], 2)[1]!, 4)};

  if (!question.isolated) {
    const probabilities = answerProbabilities(results[0], question.levels.length);
    const peak = Math.max(...probabilities);
    return {
      type: 'score',
      score: rounded(
        probabilities.reduce((sum, probability, index) => sum + index * probability, 0),
        2
      ),
      confidence: rounded(scoreConfidence(probabilities), 4),
      x_p_max: rounded(peak, 4),
      certainty: rounded(certainty(probabilities), 4),
      legend: Object.fromEntries(question.levels.map((level, index) => [String(index), level])),
      probabilities: Object.fromEntries(
        probabilities.map((probability, index) => [String(index), rounded(probability, 4)])
      )
    };
  }

  const levelFit = results.map((result) => answerProbabilities(result, 2)[1]!);
  const fitMass = levelFit.reduce((sum, probability) => sum + probability, 0);
  const probabilities = fitMass > 0 ? levelFit.map((probability) => probability / fitMass) : levelFit.map(() => 0);
  const confidence = scoreConfidence(probabilities);
  const peak = Math.max(...probabilities);
  return {
    type: 'score',
    score: rounded(
      probabilities.reduce((sum, probability, index) => sum + index * probability, 0),
      2
    ),
    confidence: rounded(confidence, 4),
    x_p_max: rounded(peak, 4),
    certainty: rounded(certainty(probabilities), 4),
    legend: Object.fromEntries(question.levels.map((level, index) => [String(index), level])),
    probabilities: Object.fromEntries(
      probabilities.map((probability, index) => [String(index), rounded(probability, 4)])
    ),
    levelFit: Object.fromEntries(levelFit.map((probability, index) => [String(index), rounded(probability, 4)])),
    fitMass: rounded(fitMass, 4)
  };
}

export function finishMapikaSystemOne(prepared: PreparedRequest, results: DecisionResult[], model: string) {
  if (results.length !== prepared.rows.length)
    throw new WebDecisionError('INFERENCE_ERROR', 'Mapika returned an unexpected number of typed rows.');
  let offset = 0;
  const answers = Object.fromEntries(
    prepared.questions.map((question) => {
      const count = question.type === 'score' && question.isolated ? question.levels.length : 1;
      const answer = finishQuestion(question, results.slice(offset, offset + count));
      offset += count;
      return [question.name, answer];
    })
  );
  return {model, tokenCount: prepared.tokenCount, answers};
}
