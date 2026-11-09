// Text-only Nari v3.2 prototype, based on the Apache-2.0 upstream
// modeling_sysone.py at 029c4628a38859dce5b0db1742e9c97545aabe12.
// It rejects unsafe control tokens and state truncation, and does not run a model.
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const SYSTEM =
  'You are a careful decision model. Read the state and answer each multiple-choice question ' +
  'with the letter of exactly one option.';
const OPEN = `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n`;
const CLOSE = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';
const REOPEN = '<|im_end|>\n<|im_start|>user\n';
const HEADER = '### 상태 (state)\n';
const INSTRUCTION = 'Answer with one letter.';
const VOCAB_SIZE = 248320;

interface NariTokenizerEncoding {
  readonly ids?: unknown;
}

interface NariTokenizer {
  encode(value: string, options: {readonly add_special_tokens: false}): NariTokenizerEncoding | null | undefined;
}

interface NariQuestionCandidate {
  readonly instructions?: unknown;
  readonly options?: unknown;
}

interface NariQuestion {
  readonly instructions: string;
  readonly options: readonly string[];
}

interface NariRequestCandidate {
  readonly state?: unknown;
  readonly questions?: unknown;
}

interface NariEncodedQuestions {
  readonly qids: number[];
  readonly optRel: number[];
  readonly optQ: number[];
  readonly optType: number[];
  readonly slots: number[];
  readonly nLetters: number[];
}

interface NariTemperatures {
  readonly [key: string]: number | undefined;
}

export interface NariPackedRequest {
  readonly inputIds: number[];
  readonly optPos: number[];
  readonly optQ: number[];
  readonly optType: number[];
  readonly slotPos: number[];
  readonly nLetters: number[];
  readonly options: string[][];
}

export interface NariChoiceProbability {
  readonly option: string;
  readonly probability: number;
  readonly logit: number;
}

export interface NariChoiceResult {
  readonly selected: string;
  readonly selectedIndex: number;
  readonly probabilities: readonly NariChoiceProbability[];
}

function encodeIds(tokenizer: NariTokenizer, value: string): number[] {
  const tokens = tokenizer.encode(value, {add_special_tokens: false})?.ids;
  if (!tokens || typeof (tokens as {readonly [Symbol.iterator]?: unknown})[Symbol.iterator] !== 'function')
    throw new TypeError('Invalid tokenizer output');
  const result = Array.from(tokens as Iterable<unknown>);
  if (result.some((id) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= VOCAB_SIZE))
    throw new RangeError('Tokenizer returned an out-of-bounds token ID');
  return result as number[];
}

function assertUserText(
  tokenizer: NariTokenizer,
  value: unknown,
  controlIds: ReadonlySet<number>
): asserts value is string {
  if (typeof value !== 'string') throw new TypeError('Text fields must be strings');
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new RangeError('Unpaired UTF-16 surrogate');
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new RangeError('Unpaired UTF-16 surrogate');
    }
  }
  if (encodeIds(tokenizer, value).some((id) => controlIds.has(id)))
    throw new RangeError('User input contains a reserved control token');
}

function validateQuestion(
  tokenizer: NariTokenizer,
  question: unknown,
  controlIds: ReadonlySet<number>
): asserts question is NariQuestion {
  const candidate = question as NariQuestionCandidate;
  if (!question || typeof candidate.instructions !== 'string' || !candidate.instructions.trim())
    throw new RangeError('Question needs nonempty instructions');
  if (!Array.isArray(candidate.options) || candidate.options.length < 2 || candidate.options.length > 255)
    throw new RangeError('Question needs 2 to 255 options');
  if ((candidate.options as unknown[]).some((option: unknown) => typeof option !== 'string' || !option.trim()))
    throw new RangeError('Options must be nonempty strings');
  if (new Set(candidate.options as unknown[]).size !== candidate.options.length)
    throw new RangeError('Options must be unique');
  for (const value of [candidate.instructions, ...(candidate.options as string[])])
    assertUserText(tokenizer, value, controlIds);
}

function encodeQuestions(ids: (text: string) => number[], questions: readonly NariQuestion[]): NariEncodedQuestions {
  const qids = [],
    optRel = [],
    optQ = [],
    optType = [],
    slots = [],
    nLetters = [];
  let inAnswer = false;
  for (const [index, question] of questions.entries()) {
    if (inAnswer) {
      qids.push(...ids(REOPEN));
      inAnswer = false;
    }
    qids.push(...ids(`\n### 질문 ${index + 1}: q${index} [하나 고르기]\n${question.instructions}\n`));
    const hasLetters = question.options.length <= 26;
    for (const [optionIndex, option] of question.options.entries()) {
      const shown = hasLetters ? `${LETTERS[optionIndex]!}. ${option}\n` : `- ${option}\n`;
      qids.push(...ids(shown));
      optRel.push(qids.length - 1);
      optQ.push(index);
      optType.push(1);
    }
    if (hasLetters) {
      qids.push(...ids(INSTRUCTION + CLOSE));
      slots.push(qids.length - 1);
      inAnswer = true;
    } else {
      slots.push(-1);
    }
    nLetters.push(hasLetters ? question.options.length : 0);
  }
  return {qids, optRel, optQ, optType, slots, nLetters};
}

/** Reproduce v4chat on accepted text choices, rejecting rather than truncating state. */
export function packNariRequest(tokenizer: NariTokenizer, request: unknown, addedTokenIds: unknown): NariPackedRequest {
  const requestCandidate = request as NariRequestCandidate;
  if (
    !request ||
    typeof request !== 'object' ||
    !Array.isArray(requestCandidate.questions) ||
    !requestCandidate.questions.length
  )
    throw new RangeError('At least one question is required');
  if (
    !Array.isArray(addedTokenIds) ||
    !addedTokenIds.length ||
    addedTokenIds.some(
      (id: unknown) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= VOCAB_SIZE
    ) ||
    new Set(addedTokenIds as unknown[]).size !== addedTokenIds.length
  )
    throw new RangeError('Invalid tokenizer control-token binding');
  const controlIds = new Set(addedTokenIds as number[]);
  assertUserText(tokenizer, requestCandidate.state, controlIds);
  for (const question of requestCandidate.questions as unknown[]) validateQuestion(tokenizer, question, controlIds);
  const ids = (text: string) => encodeIds(tokenizer, text);
  const {qids, optRel, optQ, optType, slots, nLetters} = encodeQuestions(
    ids,
    requestCandidate.questions as NariQuestion[]
  );
  const open = ids(OPEN),
    header = ids(HEADER),
    state = ids(requestCandidate.state as string);
  const budget = 2048 - open.length - header.length - qids.length;
  if (budget < 32 || state.length > budget) throw new RangeError('Input exceeds token limit; no truncation applied');
  const base = open.length + header.length + state.length;
  const inputIds = [...open, ...header, ...state, ...qids];
  return {
    inputIds,
    optPos: optRel.map((pos) => base + pos),
    optQ,
    optType,
    slotPos: slots.map((pos) => (pos < 0 ? -1 : base + pos)),
    nLetters,
    options: (requestCandidate.questions as NariQuestion[]).map((question) => [...question.options])
  };
}

function logSoftmax(values: readonly number[]): number[] {
  const max = Math.max(...values);
  const normalizer = Math.log(values.reduce((sum, value) => sum + Math.exp(value - max), 0));
  return values.map((value) => value - max - normalizer);
}

/** Reproduce the combined head/letter score and type-specific temperature. */
export function decodeNariChoice(
  options: readonly string[],
  headScores: ArrayLike<number> | Iterable<number> | null | undefined,
  letterScores: ArrayLike<number> | Iterable<number> | null | undefined,
  temperatures: NariTemperatures | null | undefined,
  lmWeight = 0.5
): NariChoiceResult {
  const head = Array.from(headScores ?? []);
  const letter = letterScores == null ? null : Array.from(letterScores);
  if (
    options.length < 2 ||
    options.length > 255 ||
    head.length !== options.length ||
    head.some((x) => !Number.isFinite(x))
  )
    throw new RangeError('Invalid head scores');
  if (!Number.isFinite(lmWeight) || lmWeight < 0 || lmWeight > 1) throw new RangeError('Invalid LM weight');
  const hasLetters = options.length <= 26;
  if (
    hasLetters !== (letter !== null) ||
    (letter && (letter.length !== head.length || letter.some((x) => !Number.isFinite(x))))
  )
    throw new RangeError('Invalid letter scores');
  let logits = head;
  if (letter) {
    const a = logSoftmax(head),
      b = logSoftmax(letter);
    logits = a.map((value, i) => (1 - lmWeight) * value + lmWeight * b[i]!);
  }
  const bucket = hasLetters ? (options.length <= 2 ? '2' : options.length <= 5 ? '3-5' : '6+') : '27+';
  const temperature = temperatures?.[`choice:${bucket}`] ?? temperatures?.['*'] ?? 1;
  if (!Number.isFinite(temperature) || temperature <= 0) throw new RangeError('Invalid temperature');
  const max = Math.max(...logits);
  const exp = logits.map((value) => Math.exp((value - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  const probabilities = exp.map((value) => value / sum);
  const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
  return {
    selected: options[selectedIndex]!,
    selectedIndex,
    probabilities: options.map((option, i) => ({option, probability: probabilities[i]!, logit: logits[i]!}))
  };
}
