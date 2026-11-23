/** Candidate Strands v2 prompt rendering. The caller supplies a plain-text tokenizer. */
type Content = string | Record<string, unknown> | unknown[];
type Criteria = Record<string, Content | null> | string[] | null;
type Question = {type?: 'choice' | 'score' | 'noul'; instructions: Content; criteria?: Criteria};
type Request = {state?: Content; questions: Record<string, Question>};
type Tokenizer = {encode(text: string, options: {add_special_tokens: false}): {ids: number[]}};

const WS = '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const trim = (s: string) => s.replace(new RegExp(`^${WS}+|${WS}+$`, 'gu'), '');
const compact = (s: string) => trim(s).replace(new RegExp(`${WS}+`, 'gu'), ' ');
const HEADERS = {
  choice: 'Select exactly one option.',
  score: 'Rate the state against the ordered levels below (lowest first).',
  noul: 'Decide whether the statement is true of the state.'
} as const;
const NOUL_DEFAULT = {false: 'the statement does not hold for this state', true: 'the statement holds for this state'};

function pythonNumber(value: number): string {
  if (!Number.isFinite(value)) throw new Error('Invalid JSON number');
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

function renderJson(value: unknown, depth = 0): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') return pythonNumber(value);
  if (!value || typeof value !== 'object') throw new Error('Invalid JSON content');
  const entries = Array.isArray(value) ? value.map((item) => [null, item] as const) : Object.entries(value);
  if (!entries.length) return Array.isArray(value) ? '[]' : '{}';
  const indent = '  '.repeat(depth + 1);
  const close = '  '.repeat(depth);
  const lines = entries.map(
    ([key, item]) => `${indent}${key === null ? '' : `${JSON.stringify(key)}: `}${renderJson(item, depth + 1)}`
  );
  return Array.isArray(value) ? `[\n${lines.join(',\n')}\n${close}]` : `{\n${lines.join(',\n')}\n${close}}`;
}

function content(value: Content | null): string {
  if (value === null) return '';
  if (typeof value === 'string') return trim(value);
  return renderJson(value);
}

function code(index: number): string {
  // Gemma tokenizes GZ as two pieces, so upstream leaves it out.
  if (index >= 207) index++;
  return index < 26
    ? String.fromCharCode(65 + index)
    : String.fromCharCode(65 + Math.floor((index - 26) / 26), 65 + ((index - 26) % 26));
}

function normalize(question: Question) {
  const criteria = question.criteria ?? null;
  const kind =
    question.type ??
    (criteria === null ||
    (!Array.isArray(criteria) && Object.keys(criteria).every((k) => k === 'true' || k === 'false'))
      ? 'noul'
      : Array.isArray(criteria)
        ? 'score'
        : 'choice');
  if (kind === 'noul') {
    if (
      criteria !== null &&
      (Array.isArray(criteria) || Object.keys(criteria).some((k) => k !== 'true' && k !== 'false'))
    )
      throw new Error('Invalid noul criteria');
    return {
      kind,
      options: ['false', 'true'],
      descriptions: ['false', 'true'].map((key) =>
        criteria !== null && Object.hasOwn(criteria, key)
          ? criteria[key]
          : NOUL_DEFAULT[key as keyof typeof NOUL_DEFAULT]
      )
    };
  }
  if (kind === 'score') {
    if (
      !Array.isArray(criteria) ||
      criteria.length < 2 ||
      criteria.length > 10 ||
      criteria.some((x) => typeof x !== 'string')
    )
      throw new Error('Invalid score criteria');
    return {kind, options: criteria.map((_, index) => String(index)), descriptions: criteria};
  }
  if (!criteria || Array.isArray(criteria)) throw new Error('Invalid choice criteria');
  const entries = Object.entries(criteria);
  if (entries.length < 2 || entries.length > 255) throw new Error('Invalid choice count');
  return {kind, options: entries.map(([key]) => key), descriptions: entries.map(([, value]) => value)};
}

export function renderStrandsV2(request: Request, tokenizer: Tokenizer, imageTokens: number[] = []) {
  const encode = (text: string) => tokenizer.encode(text, {add_special_tokens: false}).ids;
  const text = content(request.state ?? '');
  const stateText = `${text ? `${text}\n` : ''}</state>\n`;
  const stateIds = [2, 105];
  let pendingText = 'user\n<state>\n';
  const images: Array<{start: number; end: number}> = [];
  for (const count of imageTokens) {
    if (!Number.isInteger(count) || count < 1) throw new Error('Invalid image token count');
    stateIds.push(...encode(pendingText), 255999);
    const start = stateIds.length;
    for (let i = 0; i < count; i++) stateIds.push(258880);
    images.push({start, end: stateIds.length});
    stateIds.push(258882);
    pendingText = '\n';
  }
  stateIds.push(...encode(pendingText + stateText));
  const questions = Object.entries(request.questions).map(([name, question]) => {
    const {kind, options, descriptions} = normalize(question);
    const codes = options.map((_, i) => code(i));
    const instruction = content(question.instructions);
    const lines = options
      .map(
        (option, i) =>
          `${codes[i]}.${compact(option) ? ` ${compact(option)}` : ''}${content(descriptions[i] ?? null) ? ` — ${compact(content(descriptions[i] ?? null))}` : ''}\n`
      )
      .join('');
    const questionText = `<question type="${kind}">\n${HEADERS[kind]}\n${instruction ? `${instruction}\n` : ''}<options>\n${lines}</options>\nAnswer with the code of one option.\n</question>`;
    const questionIds = [...encode(questionText), 106, ...encode('\n'), 105, ...encode('model\n')];
    return {name, kind, options, codes, questionText: `${questionText}<turn|>\n<|turn>model\n`, questionIds};
  });
  if (!questions.length) throw new Error('At least one question is required');
  return {
    stateIds,
    images,
    questions,
    inputTokens: stateIds.length + questions.reduce((n, q) => n + q.questionIds.length, 0)
  };
}
