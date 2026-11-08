import {WebDecisionError} from '../kev.js';

export const STRANDS_GEMMA4_MAX_LENGTH = 4096;

const whitespace =
  '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const trimStart = new RegExp(`^${whitespace}+`, 'u');
const trimEnd = new RegExp(`${whitespace}+$`, 'u');
const collapseWhitespace = new RegExp(`${whitespace}+`, 'gu');
const invalid = (message, cause) => new WebDecisionError('INVALID_INPUT', message, cause ? {cause} : undefined);
const invalidTokenizer = (message, cause) =>
  new WebDecisionError('INVALID_ASSET', message, cause ? {cause} : undefined);
const invalidLogits = () => new WebDecisionError('INFERENCE_ERROR', 'Strands Gemma 4 returned invalid pointer logits.');

function pythonStrip(value) {
  return value.replace(trimStart, '').replace(trimEnd, '');
}

function assertUnicode(value, label) {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw invalid(`${label} contains an unpaired surrogate.`);
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw invalid(`${label} contains an unpaired surrogate.`);
    }
  }
}

function objectEntries(value, label, allowMap = false) {
  if (allowMap && value instanceof Map) {
    if (Reflect.ownKeys(value).length) throw invalid(`${label} maps cannot have extra properties.`);
    let entries;
    try {
      entries = Array.from(Map.prototype.entries.call(value));
    } catch (cause) {
      throw invalid(`${label} must be a valid string-keyed map.`, cause);
    }
    if (entries.some(([key]) => typeof key !== 'string')) throw invalid(`${label} map keys must be strings.`);
    return entries;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && Object.getPrototypeOf(prototype) !== null)
    throw invalid(`${label} must be a plain object.`);
  const entries = [];
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} must contain enumerable string data properties.`);
    entries.push([key, descriptor.value]);
  }
  return entries;
}

function arrayValues(value, label) {
  if (!Array.isArray(value)) throw invalid(`${label} must be an array.`);
  const values = [];
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`${label} must be a dense data array.`);
    values.push(descriptor.value);
  }
  for (const key of Reflect.ownKeys(value))
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9]\d*)$/u.test(key) || Number(key) >= value.length))
      throw invalid(`${label} cannot have extra properties.`);
  return values;
}

// The JSON wire decides whether Python parses a number as an int or a float.
function pythonJsonNumber(value) {
  if (!Number.isFinite(value)) throw invalid('Content numbers must be finite.');
  const wire = JSON.stringify(value);
  if (!/[.e]/u.test(wire)) return wire;
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e16)) {
    const [mantissa, exponent] = value.toExponential().split('e');
    const power = Number(exponent);
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`;
  }
  return wire;
}

function renderJson(value, depth = 0, ancestors = new Set()) {
  if (value === null || typeof value === 'boolean') return String(value);
  if (typeof value === 'string') {
    assertUnicode(value, 'Content');
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    return pythonJsonNumber(value);
  }
  if (!value || typeof value !== 'object') throw invalid('Content must be JSON data.');
  if (ancestors.has(value)) throw invalid('Content cannot contain a cycle.');

  ancestors.add(value);
  const indent = ' '.repeat((depth + 1) * 2);
  const closing = ' '.repeat(depth * 2);
  let rendered;
  if (Array.isArray(value)) {
    const values = arrayValues(value, 'Content arrays');
    rendered = values.length
      ? `[\n${values.map((item) => `${indent}${renderJson(item, depth + 1, ancestors)}`).join(',\n')}\n${closing}]`
      : '[]';
  } else {
    const entries = objectEntries(value, 'Content objects');
    rendered = entries.length
      ? `{\n${entries
          .map(([key, item]) => {
            assertUnicode(key, 'Content keys');
            return `${indent}${JSON.stringify(key)}: ${renderJson(item, depth + 1, ancestors)}`;
          })
          .join(',\n')}\n${closing}}`
      : '{}';
  }
  ancestors.delete(value);
  return rendered;
}

function renderContent(value, label, allowNull = false) {
  if (value === null && allowNull) return '';
  if (typeof value === 'string') {
    assertUnicode(value, label);
    return pythonStrip(value);
  }
  if (Array.isArray(value) || (value && typeof value === 'object')) return renderJson(value);
  throw invalid(`${label} must be text or JSON object data.`);
}

function snapshotChoice(fields, name, instructions) {
  const entries = Array.isArray(fields.criteria)
    ? arrayValues(fields.criteria, `Question ${name} criteria`).map((label) => [label, null])
    : objectEntries(fields.criteria, `Question ${name} criteria`, true);
  if (entries.length < 2) throw invalid(`Question ${name} choice requires at least 2 options.`);
  const labels = new Set();
  const options = entries.map(([label, description]) => {
    if (typeof label !== 'string' || labels.has(label))
      throw invalid(`Question ${name} choice labels must be distinct strings.`);
    assertUnicode(label, 'Option labels');
    labels.add(label);
    return {label, description: renderContent(description, 'Option descriptions', true)};
  });
  return {type: 'choice', instructions, options};
}

function snapshotScore(fields, name, instructions) {
  const levels = arrayValues(fields.criteria, `Question ${name} criteria`);
  if (levels.length < 2 || levels.length > 10) throw invalid(`Question ${name} score requires 2 to 10 levels.`);
  const options = levels.map((description, index) => {
    if (typeof description !== 'string') throw invalid(`Question ${name} score level ${index} must be text.`);
    assertUnicode(description, 'Score levels');
    return {label: String(index), description: pythonStrip(description)};
  });
  return {type: 'score', instructions, options};
}

function snapshotNoul(fields, name, instructions) {
  const criteria =
    fields.criteria === undefined || fields.criteria === null
      ? []
      : objectEntries(fields.criteria, `Question ${name} criteria`, true);
  if (criteria.some(([key]) => key !== 'true' && key !== 'false'))
    throw invalid(`Question ${name} noul criteria can only describe false and true.`);
  const descriptions = Object.fromEntries(criteria);
  const falseDescription = Object.hasOwn(descriptions, 'false')
    ? descriptions.false
    : 'the statement does not hold for this state';
  const trueDescription = Object.hasOwn(descriptions, 'true')
    ? descriptions.true
    : 'the statement holds for this state';
  return {
    type: 'noul',
    instructions,
    options: [
      {label: 'false', description: renderContent(falseDescription, 'Noul descriptions', true)},
      {label: 'true', description: renderContent(trueDescription, 'Noul descriptions', true)}
    ]
  };
}

const questionSnapshots = {choice: snapshotChoice, score: snapshotScore, noul: snapshotNoul};

function snapshotQuestion(question, name) {
  const fields = Object.fromEntries(objectEntries(question, `Question ${name}`));
  const unsupported = Object.keys(fields).some((key) => !['type', 'instructions', 'criteria'].includes(key));
  if (unsupported) throw invalid(`Question ${name} has an unsupported field.`);
  if (!Object.hasOwn(questionSnapshots, fields.type))
    throw invalid(`Question ${name} type must be choice, score, or noul.`);
  const snapshot = questionSnapshots[fields.type];
  const instructions = renderContent(fields.instructions, `Question ${name} instructions`);
  return snapshot(fields, name, instructions);
}

function renderOptionBlock(options) {
  const lines = [];
  const spans = [];
  let cursor = 0;
  for (let index = 0; index < options.length; index++) {
    const {label, description} = options[index];
    const compact = description.replace(collapseWhitespace, ' ').replace(/^ +| +$/gu, '');
    const line = `${index + 1}. ${label}${compact ? ` — ${compact}` : ''}`;
    lines.push(line);
    spans.push([cursor, cursor + line.length]);
    cursor += line.length + 1;
  }
  return {text: lines.join('\n'), spans};
}

function renderQuestion(question) {
  const headers = {
    choice: 'Select exactly one option.',
    score: 'Rate the state against the ordered levels below (lowest first).',
    noul: 'Decide whether the statement is true of the state.'
  };
  const block = renderOptionBlock(question.options);
  const prefix = `<question type="${question.type}">\n${headers[question.type]}\n${question.instructions}\n<options>\n`;
  const text = `${prefix}${block.text}\n</options>\n</question>\n<answer>`;
  return {
    type: question.type,
    options: question.options,
    text,
    spans: block.spans.map(([start, end]) => [prefix.length + start, prefix.length + end])
  };
}

function snapshotRequest(state, questions) {
  const stateText = renderContent(state, 'State');
  const entries = objectEntries(questions, 'Questions', true);
  if (!entries.length) throw invalid('Provide at least one named question.');
  const snapshots = entries.map(([name, question]) => ({name, ...snapshotQuestion(question, name)}));
  return {stateText: `<state>\n${stateText}\n</state>\n`, questions: snapshots};
}

function tokenIds(encoded) {
  const ids = encoded?.ids;
  if (!Array.isArray(ids)) throw invalidTokenizer('Strands Gemma 4 tokenizer returned invalid token IDs.');
  const copy = [];
  for (let index = 0; index < ids.length; index++) {
    if (!Object.hasOwn(ids, index) || !Number.isSafeInteger(ids[index]) || ids[index] < 0 || ids[index] >= 262144)
      throw invalidTokenizer('Strands Gemma 4 tokenizer returned invalid token IDs.');
    copy.push(ids[index]);
  }
  return copy;
}

function tokenOffsets(tokenizer, text, ids) {
  if (typeof tokenizer?.decode !== 'function')
    throw invalidTokenizer('Strands Gemma 4 tokenizer cannot recover token offsets.');
  const spans = [];
  const pieces = [];
  let cursor = 0;
  for (const id of ids) {
    let piece;
    try {
      piece = tokenizer.decode([id], {skip_special_tokens: false, clean_up_tokenization_spaces: false});
    } catch (cause) {
      throw invalidTokenizer('Strands Gemma 4 tokenizer could not decode token offsets.', cause);
    }
    if (typeof piece !== 'string' || !piece.length)
      throw invalidTokenizer('Strands Gemma 4 tokenizer returned an empty token piece.');
    pieces.push(piece);
    spans.push([cursor, cursor + piece.length]);
    cursor += piece.length;
  }
  if (pieces.join('') !== text)
    throw invalidTokenizer('Strands Gemma 4 tokenizer cannot align decoded tokens with the exact prompt text.');
  return spans;
}

function encode(tokenizer, text, addSpecialTokens) {
  if (typeof tokenizer?.encode !== 'function') throw invalidTokenizer('Strands Gemma 4 tokenizer is unavailable.');
  try {
    return tokenIds(tokenizer.encode(text, {add_special_tokens: addSpecialTokens}));
  } catch (cause) {
    if (cause instanceof WebDecisionError) throw cause;
    throw invalidTokenizer('Strands Gemma 4 tokenizer failed.', cause);
  }
}

function optionIndices(question, offsets, prefixLength) {
  return question.spans.map(([start, end]) => {
    let last = -1;
    for (let index = 0; index < offsets.length; index++) {
      const [left, right] = offsets[index];
      if (right > left && left >= start && right <= end) last = index;
    }
    if (last < 0) throw invalidTokenizer('A Strands Gemma 4 option has no token in its rendered span.');
    return prefixLength + last;
  });
}

/** Encode the upstream state/question split, preserving every token and refusing truncation. */
export function packStrandsGemma4Request(tokenizer, state, questions, maxLength = STRANDS_GEMMA4_MAX_LENGTH) {
  if (!Number.isSafeInteger(maxLength) || maxLength < 1) throw invalid('Invalid Strands Gemma 4 context length.');
  const request = snapshotRequest(state, questions);
  const stateIds = encode(tokenizer, request.stateText, true);
  const rows = request.questions.map((source) => {
    const question = renderQuestion(source);
    const ids = encode(tokenizer, question.text, false);
    const offsets = tokenOffsets(tokenizer, question.text, ids);
    if (stateIds.length + ids.length > maxLength)
      throw invalid(`Question ${source.name} exceeds the ${maxLength}-token context window.`);
    const inputIds = stateIds.concat(ids);
    return Object.freeze({
      name: source.name,
      type: source.type,
      options: Object.freeze(source.options.map(({label}) => label)),
      descriptions: Object.freeze(source.options.map(({description}) => description)),
      inputIds: Object.freeze(inputIds),
      optionIndices: Object.freeze(optionIndices(question, offsets, stateIds.length)),
      answerIndex: inputIds.length - 1,
      stateText: request.stateText,
      questionText: question.text
    });
  });
  return Object.freeze({
    rows: Object.freeze(rows),
    tokenCount: rows.reduce((total, row) => total + row.inputIds.length, 0)
  });
}

function roundPython(value, places = 4) {
  if (!Number.isFinite(value)) return value;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, Math.abs(value));
  const bits = view.getBigUint64(0);
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  const mantissa = exponentBits === 0 ? fraction : fraction | (1n << 52n);
  const exponent = exponentBits === 0 ? -1074 : exponentBits - 1023 - 52;
  const scaled = mantissa * 10n ** BigInt(places);
  let rounded;
  if (exponent >= 0) rounded = scaled << BigInt(exponent);
  else {
    const divisor = 1n << BigInt(-exponent);
    const quotient = scaled / divisor;
    const remainder = scaled % divisor;
    const twice = remainder * 2n;
    rounded = twice > divisor || (twice === divisor && quotient % 2n === 1n) ? quotient + 1n : quotient;
  }
  const result = Number(rounded) / 10 ** places;
  return value < 0 ? -result : result;
}

function softmax(logits, temperature) {
  const scaled = logits.map((value) => value / temperature);
  if (scaled.some((value) => !Number.isFinite(value))) throw invalidLogits();
  const maximum = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - maximum));
  const total = exps.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) throw invalidLogits();
  return exps.map((value) => value / total);
}

function confidence(probabilities) {
  if (probabilities.length <= 1) return 1;
  const maximum = Math.max(...probabilities);
  return Math.max(0, Math.min(1, (probabilities.length * maximum - 1) / (probabilities.length - 1)));
}

function scoreConfidence(probabilities, smoothing) {
  const count = probabilities.length;
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  if (count <= 1) return 1;
  if (!Number.isFinite(smoothing) || smoothing < 0) throw invalid('Invalid Strands Gemma 4 ordinal smoothing.');
  if (total <= 0) return 0;
  const normalized = probabilities.map((value) => value / total);
  const mean = normalized.reduce((sum, value, index) => sum + index * value, 0);
  const variance = normalized.reduce((sum, value, index) => sum + value * (index - mean) ** 2, 0);
  const sigma = Math.sqrt(variance);
  const maximum = (count - 1) / 2;
  const floor = Math.sqrt(smoothing);
  if (maximum <= floor) return sigma <= floor ? 1 : 0;
  return Math.max(0, Math.min(1, (maximum - sigma) / (maximum - floor)));
}

function validateLogits(logits, count) {
  if (
    (!Array.isArray(logits) && !(ArrayBuffer.isView(logits) && !(logits instanceof DataView))) ||
    logits.length !== count
  )
    throw invalidLogits();
  const values = [];
  for (let index = 0; index < count; index++) {
    if (Array.isArray(logits) && !Object.hasOwn(logits, index)) throw invalidLogits();
    const value = logits[index];
    if (!Number.isFinite(value)) throw invalidLogits();
    values.push(value);
  }
  return values;
}

/** Apply the checkpoint's per-question temperature and native Strands response rules. */
export function finishStrandsGemma4Answer(row, logits, config) {
  const count = row?.options?.length;
  if (
    !Number.isSafeInteger(count) ||
    count < 2 ||
    (row.type === 'score' && count > 10) ||
    (row.type === 'noul' && count !== 2) ||
    !['choice', 'score', 'noul'].includes(row.type) ||
    (row.type === 'score' && (!Array.isArray(row.descriptions) || row.descriptions.length !== count))
  )
    throw invalidLogits();
  const temperature = config?.temperatureByKind?.[row.type];
  if (!Number.isFinite(temperature) || temperature <= 0) throw invalid('Invalid Strands Gemma 4 temperature.');
  const probabilities = softmax(validateLogits(logits, count), temperature);
  if (row.type === 'noul') return {type: 'noul', noul: roundPython(probabilities[1])};
  const probabilityRecord = Object.fromEntries(
    row.options.map((label, index) => [label, roundPython(probabilities[index])])
  );
  if (row.type === 'choice') {
    const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
    return {
      type: 'choice',
      choice: row.options[selectedIndex],
      probabilities: probabilityRecord,
      confidence: roundPython(confidence(probabilities))
    };
  }
  const expected = probabilities.reduce((sum, probability, index) => sum + index * probability, 0);
  const legend = Object.fromEntries(row.options.map((label, index) => [String(index), row.descriptions[index]]));
  return {
    type: 'score',
    score: roundPython(expected),
    legend,
    probabilities: probabilityRecord,
    confidence: roundPython(scoreConfidence(probabilities, config.ordinalSmoothing))
  };
}

/** Restore question names and model identity after one graph result per typed prompt. */
export function unpackStrandsGemma4Scores(packed, scoreRows, config, model) {
  if (
    !Array.isArray(packed?.rows) ||
    !Array.isArray(scoreRows) ||
    scoreRows.length !== packed.rows.length ||
    !Number.isSafeInteger(packed.tokenCount) ||
    packed.tokenCount < 0 ||
    typeof model !== 'string' ||
    !model
  )
    throw invalidLogits();
  const answers = [];
  const names = new Set();
  for (let index = 0; index < packed.rows.length; index++) {
    if (!Object.hasOwn(packed.rows, index) || !Object.hasOwn(scoreRows, index)) throw invalidLogits();
    const row = packed.rows[index];
    if (typeof row?.name !== 'string' || names.has(row.name)) throw invalidLogits();
    names.add(row.name);
    answers.push([row.name, finishStrandsGemma4Answer(row, scoreRows[index], config)]);
  }
  return {model, tokenCount: packed.tokenCount, answers: Object.fromEntries(answers)};
}
