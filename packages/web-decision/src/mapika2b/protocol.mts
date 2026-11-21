import {readKananaFloat} from '../kanana/json-float.ts';
import {WebDecisionError} from '../kev.js';
import {buildMapikaLabels} from '../mapika.js';
import {
  MAPIKA2B_BASE_TEMPERATURE,
  MAPIKA2B_LIMITS,
  MAPIKA2B_MAX_OPTIONS,
  MAPIKA2B_MAX_SCORE_LEVELS,
  MAPIKA2B_TEMPERATURES,
  MAPIKA2B_VISION,
  MAPIKA2B_VOCAB_SIZE,
  type Mapika2bImageAsset,
  type Mapika2bLabels,
  type Mapika2bLimits,
  type Mapika2bPlan,
  type Mapika2bQuestion,
  type Mapika2bQuestionKind,
  type Mapika2bRow,
  type Mapika2bSlot,
  type Mapika2bTemperatureOptions,
  type Mapika2bTemperatures,
  type Mapika2bTokenizer
} from './contract.mts';

type Entry = readonly [string, unknown];
type Fields = ReadonlyMap<string, unknown>;
type ScoringSlot = Readonly<{instructions: string; options: readonly string[]; kind: Mapika2bQuestionKind}>;
type RenderedQuestion =
  | Readonly<{kind: 'choice'; instructions: string; options: readonly string[]; names: readonly string[]}>
  | Readonly<{kind: 'noul'; instructions: string; options: readonly string[]}>
  | Readonly<{
      kind: 'score';
      instructions: string;
      options: readonly string[];
      legend: readonly string[];
      isolated: boolean;
    }>;

const NARROW_LABELS = 'ABCDEFGHIJ';
const NOUL_FALLBACK = 'Which answer fits the context?';
const BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;

function invalid(message: string): never {
  throw new WebDecisionError('INVALID_INPUT', message);
}

function hasPlainPrototype(value: object): boolean {
  const prototype = Object.getPrototypeOf(value);
  return prototype === null || prototype === Object.prototype;
}

function dataEntries(value: unknown, field: string): readonly Entry[] {
  if (value instanceof Map) {
    const entries = Array.from(Map.prototype.entries.call(value) as Iterable<readonly [unknown, unknown]>);
    if (entries.some(([key]) => typeof key !== 'string')) invalid(field + ' must have string keys.');
    return entries as Entry[];
  }
  if (!value || typeof value !== 'object' || !hasPlainPrototype(value)) invalid(field + ' must be an object.');
  return Object.keys(value).map((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) invalid(field + ' must contain data properties.');
    return [key, descriptor.value];
  });
}

function positiveTemperature(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)
    invalid('Mapika temperatures must be finite positive numbers.');
  return value;
}

/** Match the pinned Decider's scalar and per-type temperature precedence. */
export function resolveMapika2bTemperatures({
  temperature,
  temperatureByType
}: Mapika2bTemperatureOptions = {}): Mapika2bTemperatures {
  if (temperature === undefined && temperatureByType === undefined) return MAPIKA2B_TEMPERATURES;
  const fallback = temperature === undefined ? MAPIKA2B_BASE_TEMPERATURE : positiveTemperature(temperature);
  const resolved = {choice: fallback, noul: fallback, score: fallback};
  if (temperatureByType === undefined) return Object.freeze(resolved);
  for (const [type, value] of dataEntries(temperatureByType, 'Mapika temperatureByType')) {
    if (type !== 'choice' && type !== 'noul' && type !== 'score') invalid('Unknown Mapika question type: ' + type);
    resolved[type] = positiveTemperature(value);
  }
  return Object.freeze(resolved);
}

function arrayItems(value: readonly unknown[]): unknown[] {
  return Array.from({length: value.length}, (_, index) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, index);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) invalid('Mapika JSON arrays must contain data values.');
    return descriptor.value;
  });
}

function isAbsent(value: unknown): boolean {
  return value == null || value === '';
}

function isPythonTruthy(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (value instanceof Map) return value.size > 0;
  if (typeof value === 'object' && value !== null) return Object.keys(value).length > 0;
  return Boolean(value);
}

function formatNumber(value: number, forceFloat: boolean): string {
  if (Object.is(value, -0)) return '-0.0';
  if (!Number.isFinite(value)) return String(value);
  const wire = String(value);
  if (!forceFloat && /^-?\d+$/u.test(wire)) return wire;
  const magnitude = Math.abs(value);
  if (magnitude > 0 && (magnitude < 1e-4 || magnitude >= 1e16)) {
    return value.toExponential().replace(/e([+-])(\d)$/u, (_, sign, digit) => 'e' + sign + '0' + digit);
  }
  return Number.isInteger(value) ? wire + '.0' : wire;
}

function pythonQuote(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const escape = (character: string): string => {
    if (character === '\\' || character === quote) return '\\' + character;
    if (character === '\n') return '\\n';
    if (character === '\r') return '\\r';
    if (character === '\t') return '\\t';
    if (character === ' ' || !/[\p{C}\p{Z}]/u.test(character)) return character;
    const code = character.codePointAt(0)!;
    if (code <= 0xff) return '\\x' + code.toString(16).padStart(2, '0');
    if (code <= 0xffff) return '\\u' + code.toString(16).padStart(4, '0');
    return '\\U' + code.toString(16).padStart(8, '0');
  };
  return quote + Array.from(value, escape).join('') + quote;
}

function renderAtom(value: unknown, repr: boolean): string | undefined {
  if (value === null) return repr ? 'None' : 'null';
  switch (typeof value) {
    case 'boolean':
      return repr ? (value ? 'True' : 'False') : String(value);
    case 'string':
      return repr ? pythonQuote(value) : JSON.stringify(value);
    case 'bigint':
      return value.toString();
    case 'number': {
      const text = formatNumber(value, false);
      return repr && !Number.isFinite(value) ? text.toLowerCase().replace('infinity', 'inf') : text;
    }
  }
  const float = readKananaFloat(value);
  return float === undefined ? undefined : formatNumber(float, true);
}

function addIndex(value: unknown, index: number): Fields {
  const fields =
    Array.isArray(value) || !value || typeof value !== 'object' || readKananaFloat(value) !== undefined
      ? new Map([['value', value]])
      : new Map(dataEntries(value, 'State'));
  return new Map([['_index', index], ...fields]);
}

function renderArray(value: readonly unknown[], repr: boolean, annotate: boolean, ancestors: Set<object>): string {
  const items = arrayItems(value);
  const rendered = items.map((item, index) =>
    renderJson(annotate && items.length >= 8 ? addIndex(item, index) : item, repr, annotate, ancestors)
  );
  return '[' + rendered.join(', ') + ']';
}

function renderJson(value: unknown, repr = false, annotate = false, ancestors = new Set<object>()): string {
  const atom = renderAtom(value, repr);
  if (atom !== undefined) return atom;
  if (!value || typeof value !== 'object') invalid('Mapika inputs must contain JSON data.');
  if (ancestors.has(value)) invalid('Mapika inputs cannot contain cycles.');
  ancestors.add(value);
  const rendered = Array.isArray(value)
    ? renderArray(value, repr, annotate, ancestors)
    : '{' +
      dataEntries(value, 'JSON data')
        .map(
          ([key, item]) =>
            (repr ? pythonQuote(key) : JSON.stringify(key)) + ': ' + renderJson(item, repr, annotate, ancestors)
        )
        .join(', ') +
      '}';
  ancestors.delete(value);
  return rendered;
}

function renderText(value: unknown): string {
  return typeof value === 'string' ? value : renderJson(value);
}

function preferred(fields: Fields, first: string, second: string, fallback: unknown): unknown {
  if (fields.has(first)) return fields.get(first);
  return fields.has(second) ? fields.get(second) : fallback;
}

function choiceEntries(criteria: unknown): readonly Entry[] {
  if (!Array.isArray(criteria)) return dataEntries(criteria, 'Choice criteria');
  const entries = arrayItems(criteria).map(
    (value) => [typeof value === 'string' ? value : renderJson(value, true), null] as const
  );
  return Array.from(new Map(entries));
}

function formatOption(name: string, description: unknown): string {
  return isAbsent(description) ? name : name + ': ' + renderText(description);
}

function renderChoice(instructions: string, criteria: unknown): RenderedQuestion {
  const entries = choiceEntries(criteria);
  if (entries.length < 2 || entries.length > MAPIKA2B_MAX_OPTIONS)
    invalid('Choice criteria must have 2 to 255 options.');
  return {
    kind: 'choice',
    instructions,
    names: entries.map(([name]) => name),
    options: entries.map(([name, description]) => formatOption(name, description))
  };
}

function scoreOrder(key: string): number {
  const text = key.trim();
  if (!/^[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?$/u.test(text))
    invalid('Score criteria map keys must be numbers.');
  return Number(text.replaceAll('_', ''));
}

function renderScore(instructions: string, criteria: unknown, isolated: boolean): RenderedQuestion {
  const levels = Array.isArray(criteria)
    ? arrayItems(criteria)
    : dataEntries(criteria, 'Score criteria')
        .map(([key, value]) => ({order: scoreOrder(key), value}))
        .sort((left, right) => left.order - right.order)
        .map(({value}) => value);
  if (levels.length < 2 || levels.length > MAPIKA2B_MAX_SCORE_LEVELS)
    invalid('Score criteria must have 2 to 10 levels.');
  const legend = levels.map(renderText);
  return {kind: 'score', instructions, options: legend.map((value, index) => index + ': ' + value), legend, isolated};
}

function renderNoul(instructions: unknown, criteria: unknown): RenderedQuestion {
  const options = new Map(criteria == null ? [] : dataEntries(criteria, 'Noul criteria'));
  const no = options.get('false');
  const yes = options.get('true');
  if (isAbsent(instructions) && isAbsent(no) && isAbsent(yes))
    invalid('A Noul question without instructions must describe true or false.');
  return {
    kind: 'noul',
    instructions: isAbsent(instructions) ? NOUL_FALLBACK : renderText(instructions),
    options: [formatOption('no', no), formatOption('yes', yes)]
  };
}

function renderQuestion(spec: unknown): RenderedQuestion {
  const fields = new Map(dataEntries(spec, 'Question'));
  const kind = fields.has('type') ? fields.get('type') : 'choice';
  const criteria = preferred(fields, 'criteria', 'options', null);
  const input = preferred(fields, 'instructions', 'question', '');
  if (kind === 'noul' || kind === 'bool') return renderNoul(input, criteria);
  const instructions = renderText(input);
  if (!instructions) invalid('Provide instructions for each question.');
  if (kind === 'choice') return renderChoice(instructions, criteria);
  if (kind === 'score')
    return renderScore(instructions, criteria, !fields.has('isolated') || isPythonTruthy(fields.get('isolated')));
  return invalid('Mapika question type must be choice, score, noul, or bool.');
}

function decodeAsset(reference: string, value: unknown): Mapika2bImageAsset {
  const fields = new Map(dataEntries(value, 'Image asset'));
  const type = fields.has('type') ? fields.get('type') : 'image';
  const format = String(fields.get('format') ?? '').toLowerCase();
  const base64 = fields.get('data_base64');
  if (!reference || type !== 'image' || (format !== 'png' && format !== 'jpg' && format !== 'jpeg'))
    invalid('Each asset requires an image reference with png, jpg, or jpeg format.');
  if (typeof base64 !== 'string' || !base64 || !BASE64.test(base64))
    invalid('Image asset ' + reference + ' has invalid base64 data.');
  return {reference, format, data: Uint8Array.from(atob(base64), (character) => character.charCodeAt(0))};
}

function imageReferences(state: string, assets: readonly Mapika2bImageAsset[]) {
  const sorted = [...assets].sort((left, right) => right.reference.length - left.reference.length);
  const hits: {index: number; asset: Mapika2bImageAsset}[] = [];
  for (let index = 0; index < state.length;) {
    const asset = sorted.find(({reference}) => state.startsWith(reference, index));
    if (!asset) index++;
    else {
      hits.push({index, asset});
      index += asset.reference.length;
    }
  }
  return hits;
}

function placeImages(state: string, questions: unknown, raw: unknown, maxImages: number) {
  const values = raw == null ? [] : dataEntries(raw, 'Assets');
  if (!values.length) return {state, images: []};
  if (values.length > maxImages) invalid('Mapika request exceeds its image asset limit.');
  const assets = values.map(([name, value]) => decodeAsset(name, value));
  const hits = imageReferences(state, assets);
  const used = new Set(hits.map(({asset}) => asset.reference));
  for (const {reference} of assets) {
    if (used.has(reference)) continue;
    if (renderJson(questions).includes(reference)) invalid('Images can be referenced from the state only.');
    invalid('Image asset ' + reference + ' is not referenced in the state.');
  }
  if (hits.length > maxImages) invalid('Mapika request exceeds its image reference limit.');
  let start = 0;
  const text = hits.map(({index, asset}, position) => {
    const before = state.slice(start, index);
    start = index + asset.reference.length;
    return before + (hits.length === 1 ? 'the image' : 'image ' + (position + 1));
  });
  return {state: text.join('') + state.slice(start), images: hits.map(({asset}) => asset)};
}

function encode(tokenizer: Mapika2bTokenizer, text: string): number[] {
  const output = tokenizer.encode(text, {add_special_tokens: false});
  if (!output || typeof output.ids?.[Symbol.iterator] !== 'function')
    throw new WebDecisionError('INVALID_ASSET', 'Mapika tokenizer output has no token IDs.');
  const ids = Array.from(output.ids);
  if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= MAPIKA2B_VOCAB_SIZE))
    throw new WebDecisionError('INVALID_ASSET', 'Mapika tokenizer emitted an invalid token ID.');
  return ids;
}

export function createMapika2bLabels(tokenizer: Mapika2bTokenizer): Mapika2bLabels {
  const {names, ids, open} = buildMapikaLabels(tokenizer);
  if (!names.join('').startsWith(NARROW_LABELS) || [...ids, ...open].some((id) => id >= MAPIKA2B_VOCAB_SIZE))
    throw new WebDecisionError('INVALID_ASSET', 'Mapika tokenizer does not support the pinned label vocabulary.');
  return {names, ids, open};
}

function questionTokenIds(tokenizer: Mapika2bTokenizer, labels: Mapika2bLabels, slot: ScoringSlot, number: string) {
  const head = '\n\nQuestion' + number + ': ' + slot.instructions + '\nOptions:';
  const tail = '\nAnswer' + number + ': (';
  if (slot.options.length <= NARROW_LABELS.length) {
    const options = slot.options.map((option, index) => '\n(' + NARROW_LABELS[index] + ') ' + option).join('');
    return encode(tokenizer, head + options + tail);
  }
  const ids = encode(tokenizer, head);
  for (const [index, option] of slot.options.entries()) {
    ids.push(...labels.open, labels.ids[index]!);
    for (const id of encode(tokenizer, ') ' + option)) ids.push(id);
  }
  return ids.concat(encode(tokenizer, tail));
}

function slotsForQuestion(question: RenderedQuestion, isolated: boolean): ScoringSlot[] {
  if (question.kind !== 'score' || !isolated) return [question];
  return question.legend.map((level) => ({
    kind: 'score',
    instructions:
      question.instructions +
      '\nProposed answer: ' +
      level.replace(/^\s*-?\d+\s*:\s*/u, '') +
      '\nDoes the proposed answer fit?',
    options: ['no', 'yes']
  }));
}

function answerQuestion(
  name: string,
  question: RenderedQuestion,
  firstSlot: number,
  isolated: boolean
): Mapika2bQuestion {
  if (question.kind === 'noul') return {name, kind: 'noul', firstSlot};
  if (question.kind === 'choice') return {name, kind: 'choice', names: question.names, firstSlot};
  return {name, kind: 'score', legend: question.legend, isolated, firstSlot};
}

function planQuestions(questionEntries: readonly Entry[], independent: boolean) {
  const questions: Mapika2bQuestion[] = [];
  const slots: ScoringSlot[] = [];
  for (const [name, spec] of questionEntries) {
    const question = renderQuestion(spec);
    const isolated = question.kind === 'score' && question.isolated && independent;
    questions.push(answerQuestion(name, question, slots.length, isolated));
    slots.push(...slotsForQuestion(question, isolated));
  }
  return {questions, slots};
}

function buildRow(
  tokenizer: Mapika2bTokenizer,
  labels: Mapika2bLabels,
  stateIds: readonly number[],
  scoring: readonly ScoringSlot[],
  maxTokens: number,
  temperatures: Mapika2bTemperatures
): Mapika2bRow {
  const inputIds = [...stateIds];
  const slots: Mapika2bSlot[] = [];
  for (const [index, slot] of scoring.entries()) {
    const piece = questionTokenIds(tokenizer, labels, slot, scoring.length > 1 ? ' ' + (index + 1) : '');
    if (inputIds.length + piece.length > maxTokens) invalid('Mapika input exceeds its per-row token limit.');
    for (const id of piece) inputIds.push(id);
    slots.push({
      tokenIndex: inputIds.length - 1,
      optionCount: slot.options.length,
      temperature: temperatures[slot.kind]
    });
  }
  return {inputIds, slots};
}

function sharedTokens(rows: readonly Mapika2bRow[], stateTokens: number): number {
  const first = rows[0];
  if (!first) return 0;
  let common = stateTokens;
  while (common < first.inputIds.length && rows.every(({inputIds}) => inputIds[common] === first.inputIds[common]))
    common++;
  return common + rows.reduce((sum, {inputIds}) => sum + inputIds.length - common, 0);
}

function checkRowLengths(rows: readonly Mapika2bRow[], limits: Mapika2bLimits) {
  let tokens = 0;
  for (const {inputIds} of rows) {
    if (inputIds.length > limits.maxRowTokens) invalid('Mapika input exceeds its per-row token limit.');
    tokens += inputIds.length;
  }
  if (tokens > limits.maxRequestTokens) invalid('Mapika input exceeds its total-request token limit.');
  return tokens;
}

export function checkMapika2bLimits(limits: Mapika2bLimits): void {
  const values = [
    limits.maxStateTokens,
    limits.maxRowTokens,
    limits.maxRequestTokens,
    limits.maxRows,
    limits.maxImages
  ];
  if (values.some((value) => !Number.isSafeInteger(value) || value < 0))
    invalid('Mapika serving limits must be nonnegative integers.');
}

export function resolveMapika2bLimits(value?: Partial<Mapika2bLimits>): Mapika2bLimits {
  if (value === undefined) return MAPIKA2B_LIMITS;
  const limits: Record<keyof Mapika2bLimits, number> = {...MAPIKA2B_LIMITS};
  for (const [key, entry] of dataEntries(value, 'Mapika limits')) {
    if (!Object.hasOwn(limits, key)) invalid('Unknown Mapika serving limit: ' + key);
    if (typeof entry !== 'number' || !Number.isSafeInteger(entry) || entry < 0)
      invalid('Mapika serving limits must be nonnegative integers.');
    limits[key as keyof Mapika2bLimits] = entry;
  }
  checkMapika2bLimits(limits);
  return Object.freeze(limits);
}

export function prepareMapika2bRequest(
  request: unknown,
  tokenizer: Mapika2bTokenizer,
  labels: Mapika2bLabels = createMapika2bLabels(tokenizer),
  limits: Mapika2bLimits = MAPIKA2B_LIMITS,
  temperatures: Mapika2bTemperatures = MAPIKA2B_TEMPERATURES
): Mapika2bPlan {
  checkMapika2bLimits(limits);
  const fields = new Map(dataEntries(request, 'Request'));
  if (!fields.has('state')) invalid('Mapika requires a state.');
  const questionSource = fields.get('questions');
  const questionEntries = dataEntries(questionSource, 'Questions');
  const independent = fields.has('independent') ? fields.get('independent') : true;
  if (typeof independent !== 'boolean') invalid('Mapika independent must be a boolean.');
  const rawState = fields.get('state');
  const stateText = typeof rawState === 'string' ? rawState : renderJson(rawState, false, true);
  const {state, images} = placeImages(stateText, questionSource, fields.get('assets'), limits.maxImages);
  const {questions, slots} = planQuestions(questionEntries, independent);
  const scoring = independent ? slots.map((slot) => [slot]) : [slots];
  if (scoring.length > limits.maxRows) invalid('Mapika request expands beyond its scoring row limit.');
  const stateIds = encode(tokenizer, 'Context:\n' + state).slice(0, limits.maxStateTokens);
  const rows = scoring.map((row) => buildRow(tokenizer, labels, stateIds, row, limits.maxRowTokens, temperatures));
  checkRowLengths(rows, limits);
  return {
    rows,
    questions,
    images,
    limits,
    stateTokenCount: stateIds.length,
    imagePrefixLength: 0,
    inputTokenCount: sharedTokens(rows, stateIds.length)
  };
}

export type Mapika2bImageGrid = readonly [frames: number, height: number, width: number];

function imageBlock([frames, height, width]: Mapika2bImageGrid): number[] {
  const {imageToken, mergeSize, startToken, endToken} = MAPIKA2B_VISION;
  if (
    frames !== 1 ||
    !Number.isSafeInteger(height) ||
    !Number.isSafeInteger(width) ||
    height < mergeSize ||
    width < mergeSize ||
    height % mergeSize !== 0 ||
    width % mergeSize !== 0
  )
    invalid('Mapika received an invalid Qwen image processor grid.');
  const count = (height / mergeSize) * (width / mergeSize);
  if (!Number.isSafeInteger(count) || count > MAPIKA2B_VISION.maxPixels / (MAPIKA2B_VISION.patchSize * mergeSize) ** 2)
    invalid('Mapika image processor grid exceeds the checkpoint pixel limit.');
  return [startToken, ...Array<number>(count).fill(imageToken), endToken];
}

export function prefixMapika2bImages(plan: Mapika2bPlan, grids: readonly Mapika2bImageGrid[]): Mapika2bPlan {
  if (plan.imagePrefixLength || grids.length !== plan.images.length)
    invalid('Mapika requires one image grid per state reference and one vision prefix per row.');
  if (!grids.length) return plan;
  const prefix = grids.flatMap(imageBlock);
  const rows = plan.rows.map(({inputIds, slots}) => ({
    inputIds: prefix.concat(inputIds),
    slots: slots.map((slot) => ({...slot, tokenIndex: slot.tokenIndex + prefix.length}))
  }));
  const inputTokenCount = checkRowLengths(rows, plan.limits);
  return {...plan, rows, inputTokenCount, imagePrefixLength: prefix.length};
}
