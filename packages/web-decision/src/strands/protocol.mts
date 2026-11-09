// Experimental text-choice adapter for the Strands v1 protocol. It does not
// load a model or claim support in web-decision.
import {STRANDS_CHECKPOINT_MAX_LENGTH} from './contract.mjs';

const CHOICE_TEMPERATURE = 0.8629556148274228;
const PY_WHITESPACE =
  '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const stripStart = new RegExp(`^${PY_WHITESPACE}+`, 'u');
const stripEnd = new RegExp(`${PY_WHITESPACE}+$`, 'u');
const encoder = new TextEncoder();

interface ChoiceQuestionSource {
  readonly instructions?: unknown;
  readonly options?: unknown;
}

interface ChoiceQuestion {
  readonly instructions: string;
  readonly options: string[];
}

interface RenderedChoice {
  readonly stateText: string;
  readonly questionText: string;
  readonly spans: [number, number][];
}

interface EncodedTokens {
  readonly ids: number[];
  readonly tokens: string[];
}

interface TokenizerRuntime {
  readonly decoder?:
    | {
        readonly byte_decoder?: Readonly<Record<string, string | number>> | null | undefined;
      }
    | null
    | undefined;
  get_added_tokens_decoder?: (() => {has(value: number): boolean} | null | undefined) | undefined;
  encode(text: string, options: {readonly add_special_tokens: boolean}): unknown;
}

interface EncodedChoice {
  readonly inputIds: number[];
  readonly optionIndices: number[];
  readonly answerIndex: number;
  readonly stateText: string;
  readonly questionText: string;
}

interface ChoiceProbability {
  readonly option: string;
  readonly logit: number;
  readonly probability: number;
}

interface ChoiceScore {
  readonly selected: string;
  readonly selectedIndex: number;
  readonly probabilities: ChoiceProbability[];
}

function requireText(value: unknown, name: string): string {
  if (typeof value !== 'string') throw new TypeError(`${name} must be a string`);
  // Python cannot encode an isolated surrogate as UTF-8. Reject it before the
  // JavaScript tokenizer or TextEncoder can silently replace it.
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new TypeError(`${name} contains an unpaired surrogate`);
    } else if (code >= 0xdc00 && code <= 0xdfff) throw new TypeError(`${name} contains an unpaired surrogate`);
  }
  return value;
}

function pythonStrip(value: string): string {
  return value.replace(stripStart, '').replace(stripEnd, '');
}

/** Capture indexed question data before tokenization or queued GPU work. */
export function snapshotChoiceQuestion(question: unknown): ChoiceQuestion {
  if (!question || typeof question !== 'object') throw new TypeError('question is required');
  const instructions = requireText((question as ChoiceQuestionSource).instructions, 'instructions');
  if (!pythonStrip(instructions)) throw new TypeError('instructions must not be empty');
  const source = (question as ChoiceQuestionSource).options;
  if (!Array.isArray(source)) throw new RangeError('choice must contain 2 to 255 options');
  const seen = new Set<string>(),
    options: string[] = [],
    count = source.length;
  if (count < 2 || count > 255) throw new RangeError('choice must contain 2 to 255 options');
  for (let i = 0; i < count; i++) {
    if (!Object.hasOwn(source, i)) throw new TypeError('option entry is missing');
    const option = (source as unknown[])[i];
    requireText(option, 'option');
    if (!pythonStrip(option as string)) throw new TypeError('options must not be empty');
    if (seen.has(option as string)) throw new TypeError('options must be distinct');
    seen.add(option as string);
    options.push(option as string);
  }
  return {instructions, options};
}

/** Render the upstream choice format and preserve option spans as JS indices. */
export function renderChoice(state: unknown, question: unknown): RenderedChoice {
  requireText(state, 'state');
  const {instructions, options} = snapshotChoiceQuestion(question);
  let questionText = `<question type="choice">\nSelect exactly one option.\n${pythonStrip(instructions)}\n<options>\n`;
  const spans: [number, number][] = [];
  for (let i = 0; i < options.length; i++) {
    if (i) questionText += '\n';
    const start = questionText.length;
    questionText += `${i + 1}. ${options[i]!}`;
    spans.push([start, questionText.length]);
  }
  questionText += '\n</options>\n</question>\n<answer>';
  return {stateText: `<state>\n${pythonStrip(state as string)}\n</state>\n`, questionText, spans};
}

/** Recover byte spans from this checkpoint's NFC + ByteLevel tokenizer output. */
function recoverTokenByteSpans(tokenizer: unknown, text: string, encoded: EncodedTokens): [number, number][] {
  const decoder = (tokenizer as TokenizerRuntime | null | undefined)?.decoder?.byte_decoder;
  const added = (tokenizer as TokenizerRuntime | null | undefined)?.get_added_tokens_decoder?.();
  if (!decoder || !added || encoded.ids.length !== encoded.tokens.length)
    throw new Error('Expected the checkpoint ByteLevel tokenizer and matching IDs/tokens');
  const expected = encoder.encode(text.normalize('NFC'));
  const spans: [number, number][] = [];
  let cursor = 0;
  for (let i = 0; i < encoded.ids.length; i++) {
    const token = encoded.tokens[i]!;
    const isAdded = added.has(encoded.ids[i]!);
    const bytes = isAdded
      ? Array.from(encoder.encode(token))
      : Array.from(token, (ch) => {
          const byte = decoder[ch];
          if (byte === undefined) throw new Error('Tokenizer returned a non-ByteLevel token');
          // The JS tokenizer's decoder stores byte values as numeric strings.
          return Number(byte);
        });
    const start = cursor;
    for (const byte of bytes) {
      if (cursor >= expected.length || expected[cursor]! !== byte)
        throw new Error('Tokenizer bytes do not match the NFC input');
      cursor++;
    }
    spans.push([start, cursor]);
  }
  if (cursor !== expected.length) throw new Error('Tokenizer did not cover the entire input');
  return spans;
}

function snapshotTokenIds(encoded: unknown): number[] {
  const source = (encoded as {readonly ids?: unknown} | null | undefined)?.ids,
    array = Array.isArray(source);
  if (!array && !(ArrayBuffer.isView(source) && !(source instanceof DataView)))
    throw new TypeError('Tokenizer did not return indexed IDs');
  const ids: number[] = [],
    count = (source as {readonly length: number}).length;
  for (let i = 0; i < count; i++) {
    if (array && !Object.hasOwn(source, i)) throw new TypeError('Tokenizer returned invalid indexed IDs');
    const id = (source as ArrayLike<unknown>)[i];
    if (!Number.isSafeInteger(id as number)) throw new TypeError('Tokenizer returned invalid indexed IDs');
    ids.push(id as number);
  }
  return ids;
}

function snapshotTokens(encoded: unknown, count: number): string[] {
  const source = (encoded as {readonly tokens?: unknown} | null | undefined)?.tokens;
  if (!Array.isArray(source) || source.length !== count) throw new TypeError('Tokenizer IDs and tokens differ');
  const tokens: string[] = [];
  for (let i = 0; i < count; i++) {
    if (!Object.hasOwn(source, i)) throw new TypeError('Tokenizer returned invalid tokens');
    const token = (source as unknown[])[i];
    if (typeof token !== 'string') throw new TypeError('Tokenizer returned invalid tokens');
    tokens.push(token as string);
  }
  return tokens;
}

/** Build one independent prompt per question, refusing unsupported truncation. */
export function encodeChoice(
  tokenizer: unknown,
  state: unknown,
  question: unknown,
  maxLength = STRANDS_CHECKPOINT_MAX_LENGTH
): EncodedChoice {
  if (!Number.isSafeInteger(maxLength) || maxLength <= 0) throw new RangeError('invalid maximum length');
  const {stateText, questionText, spans} = renderChoice(state, question);
  const prefixIds = snapshotTokenIds((tokenizer as TokenizerRuntime).encode(stateText, {add_special_tokens: true}));
  const encoded = (tokenizer as TokenizerRuntime).encode(questionText, {add_special_tokens: false});
  const suffixIds = snapshotTokenIds(encoded),
    suffix = {ids: suffixIds, tokens: snapshotTokens(encoded, suffixIds.length)};
  const tokens = recoverTokenByteSpans(tokenizer, questionText, suffix);
  const optionIndices = spans.map(([start, end]) => {
    const left = questionText.slice(0, start).normalize('NFC');
    const middle = questionText.slice(start, end).normalize('NFC');
    const right = questionText.slice(end).normalize('NFC');
    if (left + middle + right !== questionText.normalize('NFC'))
      throw new Error('Normalization crossed an option boundary');
    const firstByte = encoder.encode(left).length;
    const lastByte = firstByte + encoder.encode(middle).length;
    let last = -1;
    for (let i = 0; i < tokens.length; i++) {
      const [lo, hi] = tokens[i]!;
      if (hi > lo && lo >= firstByte && hi <= lastByte) last = i;
    }
    if (last < 0) throw new RangeError('Option has no token in its span');
    return prefixIds.length + last;
  });
  const inputIds = prefixIds.concat(suffixIds);
  if (inputIds.length > maxLength) throw new RangeError(`Prompt exceeds ${maxLength} tokens`);
  return {inputIds, optionIndices, answerIndex: inputIds.length - 1, stateText, questionText};
}

/** Convert raw pointer logits into the upstream choice-calibrated distribution. */
export function scoreChoice(options: unknown, logits: unknown, temperature = CHOICE_TEMPERATURE): ChoiceScore {
  if (!Array.isArray(options) || !Array.isArray(logits))
    throw new TypeError('options and logits must have the same length of at least 2');
  const count = logits.length;
  if (options.length !== count || count < 2)
    throw new TypeError('options and logits must have the same length of at least 2');
  if (!Number.isFinite(temperature) || temperature <= 0) throw new RangeError('temperature and logits must be finite');
  const optionValues: string[] = [],
    values: number[] = [];
  for (let i = 0; i < count; i++) {
    if (!Object.hasOwn(options, i) || !Object.hasOwn(logits, i))
      throw new RangeError('options and logits must have valid indexed values');
    const option = (options as unknown[])[i],
      value = (logits as unknown[])[i];
    if (typeof option !== 'string' || !Number.isFinite(value as number))
      throw new RangeError('options and logits must have valid indexed values');
    optionValues.push(option as string);
    values.push(value as number);
  }
  const scaled = values.map((x) => x / temperature);
  if (scaled.some((x) => !Number.isFinite(x))) throw new RangeError('scaled logits must be finite');
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probabilities = exps.map((x) => x / sum);
  const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
  return {
    selected: optionValues[selectedIndex]!,
    selectedIndex,
    probabilities: optionValues.map((option, i) => ({option, logit: values[i]!, probability: probabilities[i]!}))
  };
}
