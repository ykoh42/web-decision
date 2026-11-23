import {WebDecisionError} from '../kev.js';
import {readKananaFloat} from '../kanana/json-float.ts';

/** Strands v2 prompt rendering. The tokenizer must encode special-token spellings as plain text. */
export type Gemma4V2Content = string | Record<string, unknown> | ReadonlyMap<string, unknown> | unknown[];
export type Gemma4V2Dictionary<T> = Record<string, T> | ReadonlyMap<string, T>;
type Criteria = Gemma4V2Dictionary<Gemma4V2Content | null> | string[] | null;
export type Gemma4V2Question = {type?: 'choice' | 'score' | 'noul'; instructions: Gemma4V2Content; criteria?: Criteria};
type Request = {state?: Gemma4V2Content; questions: Gemma4V2Dictionary<Gemma4V2Question>};
type Tokenizer = {encode(text: string, options: {add_special_tokens: false}): {ids: number[]}};

const invalid = (message: string) => new WebDecisionError('INVALID_INPUT', message);

const WS = '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const trim = (s: string) => s.replace(new RegExp(`^${WS}+|${WS}+$`, 'gu'), '');
const compact = (s: string) => trim(s).replace(new RegExp(`${WS}+`, 'gu'), ' ');
const HEADERS = {
  choice: 'Select exactly one option.',
  score: 'Rate the state against the ordered levels below (lowest first).',
  noul: 'Decide whether the statement is true of the state.'
} as const;
const NOUL_DEFAULT = {false: 'the statement does not hold for this state', true: 'the statement holds for this state'};

function validText(value: string): void {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw invalid('Content contains an unpaired surrogate.');
    } else if (code >= 0xdc00 && code <= 0xdfff) throw invalid('Content contains an unpaired surrogate.');
  }
}

function record(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === null || Object.getPrototypeOf(prototype) === null;
}

function dictionaryEntries(value: unknown): Array<[string, unknown]> {
  if (value instanceof Map) {
    if (Reflect.ownKeys(value).length) throw invalid('Maps cannot have extra properties.');
    const entries = Array.from(Map.prototype.entries.call(value)) as Array<[unknown, unknown]>;
    if (entries.some(([key]) => typeof key !== 'string')) throw invalid('Map keys must be strings.');
    return entries as Array<[string, unknown]>;
  }
  if (!record(value)) throw invalid('Expected a string-keyed object or Map.');
  return Object.entries(value);
}

function dictionary(value: unknown): value is Record<string, unknown> | ReadonlyMap<string, unknown> {
  return value instanceof Map || record(value);
}

function pythonNumber(value: number): string {
  // The pinned Python JSON parser and renderer accept these non-standard JSON values.
  if (Number.isNaN(value)) return 'NaN';
  if (value === Infinity) return 'Infinity';
  if (value === -Infinity) return '-Infinity';
  const wire = JSON.stringify(value);
  if (!/[.e]/u.test(wire)) return wire;
  const n = Math.abs(value);
  if (n !== 0 && (n < 1e-4 || n >= 1e16)) {
    const [mantissa, exponent] = value.toExponential().split('e');
    const power = Number(exponent);
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`;
  }
  return wire;
}

function pythonFloat(value: number): string {
  if (Object.is(value, -0)) return '-0.0';
  if (value === 0) return '0.0';
  const magnitude = Math.abs(value);
  if (magnitude < 1e-4 || magnitude >= 1e16) {
    const [mantissa, exponent] = value.toExponential().split('e');
    const power = Number(exponent);
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`;
  }
  const result = JSON.stringify(value);
  return result.includes('.') ? result : `${result}.0`;
}

function renderScalar(value: unknown): string | undefined {
  if (typeof value === 'string') {
    validText(value);
    return JSON.stringify(value);
  }
  if (value === null || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number') return pythonNumber(value);
  if (typeof value === 'bigint') return value.toString();
  const markedFloat = readKananaFloat(value);
  if (markedFloat !== undefined) return pythonFloat(markedFloat);
  return undefined;
}

function renderJson(value: unknown, depth = 0, ancestors = new Set<object>()): string {
  const scalar = renderScalar(value);
  if (scalar !== undefined) return scalar;
  if (!Array.isArray(value) && !dictionary(value)) throw invalid('Content must be JSON data.');
  if (ancestors.has(value)) throw invalid('Content cannot contain a cycle.');
  ancestors.add(value);
  const entries = Array.isArray(value) ? Array.from(value, (item) => [null, item] as const) : dictionaryEntries(value);
  for (const [key] of entries) if (key !== null) validText(key);
  if (!entries.length) {
    ancestors.delete(value);
    return Array.isArray(value) ? '[]' : '{}';
  }
  const indent = '  '.repeat(depth + 1);
  const close = '  '.repeat(depth);
  const lines = entries.map(
    ([key, item]) =>
      `${indent}${key === null ? '' : `${JSON.stringify(key)}: `}${renderJson(item, depth + 1, ancestors)}`
  );
  ancestors.delete(value);
  return Array.isArray(value) ? `[\n${lines.join(',\n')}\n${close}]` : `{\n${lines.join(',\n')}\n${close}}`;
}

function content(value: Gemma4V2Content | null): string {
  if (value === null) return '';
  if (typeof value === 'string') {
    validText(value);
    return trim(value);
  }
  if (!Array.isArray(value) && !dictionary(value)) throw invalid('Content must be text or JSON data.');
  return renderJson(value);
}

function code(index: number): string {
  // Gemma tokenizes GZ as two pieces, so upstream leaves it out.
  if (index >= 207) index++;
  return index < 26
    ? String.fromCharCode(65 + index)
    : String.fromCharCode(65 + Math.floor((index - 26) / 26), 65 + ((index - 26) % 26));
}

function normalizeNoul(criteria: Criteria, entries: Array<[string, unknown]>) {
  if (Array.isArray(criteria) || entries.some(([key]) => key !== 'true' && key !== 'false'))
    throw invalid('Invalid noul criteria.');
  const values = new Map(entries);
  return {
    kind: 'noul' as const,
    options: ['false', 'true'],
    descriptions: ['false', 'true'].map((key) =>
      values.has(key) ? (values.get(key) as Gemma4V2Content | null) : NOUL_DEFAULT[key as keyof typeof NOUL_DEFAULT]
    )
  };
}

function normalize(question: Gemma4V2Question) {
  const criteria = question.criteria ?? null;
  if (criteria !== null && !Array.isArray(criteria) && !dictionary(criteria))
    throw invalid('Invalid question criteria.');
  const entries = criteria !== null && !Array.isArray(criteria) ? dictionaryEntries(criteria) : [];
  const isNoul =
    criteria === null || (!Array.isArray(criteria) && entries.every(([key]) => key === 'true' || key === 'false'));
  const kind = question.type ?? (isNoul ? 'noul' : Array.isArray(criteria) ? 'score' : 'choice');
  if (!['noul', 'choice', 'score'].includes(kind)) throw invalid('Unknown question type.');
  if (kind === 'noul') return normalizeNoul(criteria, entries);
  if (kind === 'score') {
    if (
      !Array.isArray(criteria) ||
      criteria.length < 2 ||
      criteria.length > 10 ||
      criteria.some((x) => typeof x !== 'string')
    )
      throw invalid('Invalid score criteria.');
    return {kind, options: criteria.map((_, index) => String(index)), descriptions: criteria};
  }
  if (criteria === null || Array.isArray(criteria)) throw invalid('Invalid choice criteria.');
  if (entries.length < 2 || entries.length > 255) throw invalid('A choice requires 2 to 255 options.');
  return {
    kind,
    options: entries.map(([key]) => key),
    descriptions: entries.map(([, value]) => value as Gemma4V2Content | null)
  };
}

export function renderStrandsV2(request: Request, tokenizer: Tokenizer, imageTokens: number[] = []) {
  if (!record(request)) throw invalid('A request object is required.');
  const questionEntries = dictionaryEntries(request.questions);
  if (!questionEntries.length) throw invalid('At least one named question is required.');
  if (request.state === null) throw invalid('State must be text or JSON data.');
  const encode = (text: string) => tokenizer.encode(text, {add_special_tokens: false}).ids;
  const text = content(request.state ?? '');
  const stateText = `${text ? `${text}\n` : ''}</state>\n`;
  const stateIds = [2, 105];
  let pendingText = 'user\n<state>\n';
  const images: Array<{start: number; end: number}> = [];
  for (const count of imageTokens) {
    if (!Number.isInteger(count) || count < 1 || count > 280) throw invalid('Invalid image token count.');
    for (const id of encode(pendingText)) stateIds.push(id);
    stateIds.push(255999);
    const start = stateIds.length;
    for (let i = 0; i < count; i++) stateIds.push(258880);
    images.push({start, end: stateIds.length});
    stateIds.push(258882);
    pendingText = '\n';
  }
  for (const id of encode(pendingText + stateText)) stateIds.push(id);
  const questions = questionEntries.map(([name, question]) => {
    validText(name);
    if (!record(question) || !Object.hasOwn(question, 'instructions') || question.instructions === null)
      throw invalid('Invalid question.');
    const typedQuestion = question as Gemma4V2Question;
    const {kind, options, descriptions} = normalize(typedQuestion);
    for (const option of options) validText(option);
    const codes = options.map((_, i) => code(i));
    const instruction = content(typedQuestion.instructions);
    const renderedDescriptions = descriptions.map((description) => content(description ?? null));
    const lines = options
      .map(
        (option, i) =>
          `${codes[i]}.${compact(option) ? ` ${compact(option)}` : ''}${renderedDescriptions[i] ? ` — ${compact(renderedDescriptions[i]!)}` : ''}\n`
      )
      .join('');
    const questionText = `<question type="${kind}">\n${HEADERS[kind]}\n${instruction ? `${instruction}\n` : ''}<options>\n${lines}</options>\nAnswer with the code of one option.\n</question>`;
    const questionIds = [...encode(questionText), 106, ...encode('\n'), 105, ...encode('model\n')];
    return {
      name,
      kind,
      options,
      descriptions: renderedDescriptions,
      codes,
      questionText: `${questionText}<turn|>\n<|turn>model\n`,
      questionIds
    };
  });
  return {
    stateIds,
    images,
    questions,
    inputTokens: stateIds.length + questions.reduce((n, q) => n + q.questionIds.length, 0)
  };
}
