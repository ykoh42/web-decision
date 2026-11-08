import {encodeKananaQuery, type KananaTokenizer} from './tokenizer.js';
import {pythonStrip} from './chat-template.js';
import {renderKananaQueryInput} from './protocol.js';

type QueryMessage = {role?: unknown; content: unknown; [key: string]: unknown};
type QueryPart = {type?: unknown; text?: unknown; [key: string]: unknown};
type TextTokenPart = {index: number; ids: number[]};
type QueryTruncationOptions = {
  editableUserTextIndexes?: unknown;
  maxLength?: number;
  truncateInputs?: boolean;
};
type CandidateTruncationOptions = {
  maxLength?: number;
  truncateInputs?: boolean;
  truncateQuestionCandidates?: boolean;
};
type DocumentTruncationOptions = {maxLength?: number; instruction?: string; truncateInputs?: boolean};
type PreparedQuestionEntry = {id: unknown; question?: unknown; [key: string]: unknown};
type PreparedCandidateRequest = Record<string, unknown> & {questionEntries: PreparedQuestionEntry[]};
type EncodedQuery = ReturnType<typeof encodeKananaQuery>;

export const KANANA_MAX_LENGTH = 20_480;
export const KANANA_DOCUMENT_INSTRUCTION = "Represent the user's input.";

class KananaInputLengthError extends RangeError {
  declare code: string;
  declare maxLength: number;
  declare tokenCount: number;

  constructor(maxLength: number, tokenCount: number, message?: string) {
    super(message ?? `Kanana query has ${tokenCount} tokens, exceeding the configured limit of ${maxLength}.`);
    this.name = 'KananaInputLengthError';
    this.code = 'input_too_long';
    this.maxLength = maxLength;
    this.tokenCount = tokenCount;
  }
}

function validateMaxLength(maxLength: number): void {
  if (!Number.isSafeInteger(maxLength) || maxLength < 1)
    throw new RangeError('Kanana maxLength must be a positive safe integer.');
}

function cloneQueryMessages(messages: unknown): QueryMessage[] {
  if (!Array.isArray(messages)) throw new TypeError('Kanana query messages must be an array.');
  return (messages as unknown[]).map((message: unknown) => {
    if (!message || typeof message !== 'object' || Array.isArray(message))
      throw new TypeError('Kanana query messages must be objects.');
    const messageRecord = message as Record<string, unknown>;
    const content = messageRecord.content;
    return {
      ...messageRecord,
      content: Array.isArray(content) ? (content as unknown[]).map((part: unknown) => ({...(part as object)})) : content
    };
  });
}

function validateEditableIndexes(messages: unknown, editableUserTextIndexes: unknown): number[] {
  if (!Array.isArray(editableUserTextIndexes))
    throw new TypeError('editableUserTextIndexes must be returned by renderKananaQueryInput().');
  const queryMessages = messages as {length: number; [index: number]: unknown};
  const systemMessage = queryMessages[0] as {role?: unknown} | null | undefined;
  const userMessage = queryMessages[1] as {role?: unknown; content?: unknown} | null | undefined;
  if (queryMessages.length !== 2 || systemMessage?.role !== 'system' || userMessage?.role !== 'user')
    throw new TypeError('Kanana truncation expects one system and one user query message.');
  const content = userMessage.content;
  if (!Array.isArray(content) || content.length < 2) throw new TypeError('Kanana user content must be an array.');
  const parts = content as unknown[];
  const prefix = parts[0] as {type?: unknown; text?: unknown} | null | undefined;
  if (prefix?.type !== 'text' || prefix.text !== 'State:\n')
    throw new TypeError('Kanana query state prefix is protected.');
  const suffix = parts.at(-1) as {type?: unknown; text: string} | null | undefined;
  if (suffix?.type !== 'text' || !suffix.text.startsWith('\n\nTask:\n'))
    throw new TypeError('Kanana task and options suffix is protected.');

  const seen = new Set<number>();
  for (const rawIndex of editableUserTextIndexes as unknown[]) {
    if (
      typeof rawIndex !== 'number' ||
      !Number.isSafeInteger(rawIndex) ||
      rawIndex <= 0 ||
      rawIndex >= parts.length - 1 ||
      seen.has(rawIndex) ||
      (parts[rawIndex] as {type?: unknown} | null | undefined)?.type !== 'text' ||
      typeof (parts[rawIndex] as {text?: unknown} | null | undefined)?.text !== 'string'
    )
      throw new TypeError(
        'Editable query boundaries must identify unique state text parts between the protected prefix and suffix.'
      );
    seen.add(rawIndex);
  }
  return [...seen];
}

function encodeText(tokenizer: KananaTokenizer, value: string): number[] {
  const encoded = tokenizer.encode(value, {add_special_tokens: false});
  const source = (encoded as {ids?: unknown} | null | undefined)?.ids;
  const isArray = Array.isArray(source);
  if (!isArray && !(ArrayBuffer.isView(source) && !(source instanceof DataView)))
    throw new TypeError('Kanana tokenizer must return indexed token IDs.');
  const idsSource = source as ArrayLike<unknown>;
  const result: number[] = [];
  for (let index = 0; index < idsSource.length; index++) {
    if (isArray && !Object.hasOwn(source as object, index))
      throw new TypeError('Kanana tokenizer returned a sparse token ID array.');
    const id = idsSource[index];
    if (typeof id !== 'number' || !Number.isSafeInteger(id) || id < 0)
      throw new TypeError('Kanana tokenizer returned an invalid token ID.');
    result.push(id);
  }
  return result;
}

function renderRetainedText(
  tokenizer: KananaTokenizer,
  textParts: TextTokenPart[],
  keepCount: number
): [number, string][] {
  const allIds = textParts.flatMap(({ids}) => ids);
  const headCount = Math.ceil(keepCount / 2);
  const tailCount = Math.floor(keepCount / 2);
  const tailStart = allIds.length - tailCount;
  let cursor = 0;

  return textParts.map(({index, ids}) => {
    const start = cursor;
    const end = start + ids.length;
    cursor = end;

    const kept: number[] = [];
    const headEnd = Math.min(end, headCount);
    if (start < headEnd) kept.push(...allIds.slice(start, headEnd));
    const keptTailStart = Math.max(start, tailStart);
    if (keptTailStart < end) kept.push(...allIds.slice(keptTailStart, end));

    return [index, tokenizer.decode!(kept, {skip_special_tokens: false})];
  });
}

function candidateMessages(
  originalMessages: QueryMessage[],
  textParts: TextTokenPart[],
  tokenizer: KananaTokenizer,
  keepCount: number
): QueryMessage[] {
  const messages = cloneQueryMessages(originalMessages);
  const userContent = messages[1]!.content as QueryPart[];
  for (const [index, text] of renderRetainedText(tokenizer, textParts, keepCount)) userContent[index]!.text = text;
  return messages;
}

function countPrompt(tokenizer: KananaTokenizer, messages: unknown, imageTokenThw: unknown): EncodedQuery {
  return encodeKananaQuery(tokenizer, messages, imageTokenThw);
}

function failInputLength(maxLength: number, tokenCount: number, message?: string): never {
  throw new KananaInputLengthError(maxLength, tokenCount, message);
}

function queryResult(
  tokenizer: KananaTokenizer,
  messages: QueryMessage[],
  imageTokenThw: unknown,
  truncated: boolean,
  retainedTokens: number | null = null
) {
  return {
    ...countPrompt(tokenizer, messages, imageTokenThw),
    messages,
    truncated,
    retainedTokens
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) && !(value instanceof Map);
}

function preparedEntries(prepared: unknown): {request: PreparedCandidateRequest; entries: PreparedQuestionEntry[]} {
  const entries = (prepared as {questionEntries?: unknown} | null | undefined)?.questionEntries;
  if (!Array.isArray(entries)) throw new TypeError('Expected a prepared Kanana request.');
  return {request: prepared as PreparedCandidateRequest, entries: entries as PreparedQuestionEntry[]};
}

function candidatePrepared(prepared: unknown, questionId: string, candidate: string): unknown {
  const {request, entries} = preparedEntries(prepared);
  const index = entries.findIndex((entry) => entry.id === questionId);
  if (index < 0) throw new RangeError(`Unknown Kanana question ID: ${String(questionId)}`);

  const entry = entries[index]!;
  const question = entry.question as {instructions?: unknown; [key: string]: unknown} | null | undefined;
  const instructions = question?.instructions;
  if (!isRecord(instructions) || typeof instructions.candidate !== 'string')
    throw new TypeError('Kanana Question candidate truncation requires instructions.candidate to be a string.');

  const questionEntries = [...entries];
  questionEntries[index] = {
    ...entry,
    question: {...(entry.question as object), instructions: {...instructions, candidate}}
  };
  return {...request, questionEntries};
}

function structuredCandidate(prepared: unknown, questionId: string): string | null {
  const {entries} = preparedEntries(prepared);
  const entry = entries.find((item) => item.id === questionId);
  if (!entry) throw new RangeError(`Unknown Kanana question ID: ${String(questionId)}`);
  const question = entry.question as {instructions?: unknown} | null | undefined;
  const instructions = question?.instructions;
  if (!isRecord(instructions) || typeof instructions.candidate !== 'string') return null;
  return instructions.candidate;
}

function renderCandidateInput(
  prepared: unknown,
  questionId: string,
  candidate: string
): ReturnType<typeof renderKananaQueryInput> {
  return renderKananaQueryInput(candidatePrepared(prepared, questionId, candidate), questionId);
}

function decodeHeadTail(tokenizer: KananaTokenizer, ids: number[], keepCount: number): string {
  const headCount = Math.ceil(keepCount / 2);
  const tailCount = Math.floor(keepCount / 2);
  const head = tokenizer.decode!(ids.slice(0, headCount), {skip_special_tokens: false});
  const tail = tailCount ? tokenizer.decode!(ids.slice(-tailCount), {skip_special_tokens: false}) : '';
  return head + tail;
}

/**
 * Match the pinned Kanana query-state truncation behavior while preserving every
 * non-state part of the rendered prompt. The limit defaults to the model config
 * value and is never clamped to an adapter-specific lower cap.
 */
export function truncateKananaQuery(
  tokenizer: KananaTokenizer,
  messages: unknown,
  imageTokenThw: unknown = [],
  {editableUserTextIndexes, maxLength = KANANA_MAX_LENGTH, truncateInputs = true}: QueryTruncationOptions = {}
) {
  validateMaxLength(maxLength);
  if (typeof tokenizer?.encode !== 'function') throw new TypeError('A Kanana tokenizer with encode() is required.');
  const editableIndexes = validateEditableIndexes(messages, editableUserTextIndexes);
  const originalMessages = cloneQueryMessages(messages);
  const original = countPrompt(tokenizer, originalMessages, imageTokenThw);
  if (original.tokenCount <= maxLength)
    return {...original, messages: originalMessages, truncated: false, retainedStateTokens: null};
  if (!truncateInputs) failInputLength(maxLength, original.tokenCount);

  const userContent = originalMessages[1]!.content as QueryPart[];
  const textParts = editableIndexes.map((index) => ({
    index,
    ids: encodeText(tokenizer, userContent[index]!.text as string)
  }));
  const totalStateTokens = textParts.reduce((total, part) => total + part.ids.length, 0);
  if (!totalStateTokens) failInputLength(maxLength, original.tokenCount);
  if (typeof tokenizer.decode !== 'function')
    throw new TypeError('Kanana tokenizer with decode() is required to truncate query state.');

  const emptyStateMessages = candidateMessages(originalMessages, textParts, tokenizer, 0);
  const protectedPrompt = countPrompt(tokenizer, emptyStateMessages, imageTokenThw);
  if (protectedPrompt.tokenCount > maxLength) failInputLength(maxLength, protectedPrompt.tokenCount);

  let low = 0;
  let high = totalStateTokens - 1;
  let best = {messages: emptyStateMessages, encoded: protectedPrompt, retainedStateTokens: 0};

  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = candidateMessages(originalMessages, textParts, tokenizer, middle);
    const encoded = countPrompt(tokenizer, candidate, imageTokenThw);
    if (encoded.tokenCount <= maxLength) {
      best = {messages: candidate, encoded, retainedStateTokens: middle};
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }

  if (best.encoded.tokenCount > maxLength) failInputLength(maxLength, best.encoded.tokenCount);
  return {
    ...best.encoded,
    messages: best.messages,
    truncated: true,
    retainedStateTokens: best.retainedStateTokens
  };
}

/** Apply the pinned fallback that shortens only a structured Question candidate description. */
export function truncateKananaQuestionCandidate(
  tokenizer: KananaTokenizer,
  prepared: unknown,
  questionId: string,
  imageTokenThw: unknown = [],
  {
    maxLength = KANANA_MAX_LENGTH,
    truncateInputs = true,
    truncateQuestionCandidates = true
  }: CandidateTruncationOptions = {}
) {
  validateMaxLength(maxLength);
  if (typeof tokenizer?.encode !== 'function') throw new TypeError('A Kanana tokenizer with encode() is required.');
  const candidate = structuredCandidate(prepared, questionId);
  const input = renderKananaQueryInput(prepared, questionId);
  const original = countPrompt(tokenizer, input.messages, imageTokenThw);
  if (original.tokenCount <= maxLength)
    return {...original, messages: input.messages, truncated: false, retainedTokens: null};
  if (!truncateInputs || !truncateQuestionCandidates) failInputLength(maxLength, original.tokenCount);
  if (candidate === null)
    failInputLength(
      maxLength,
      original.tokenCount,
      'Question candidate truncation requires a structured Question with a candidate string.'
    );

  const ids = encodeText(tokenizer, candidate);
  if (!ids.length)
    failInputLength(
      maxLength,
      original.tokenCount,
      'Question candidate is empty; protected input cannot be shortened.'
    );
  if (typeof tokenizer.decode !== 'function')
    throw new TypeError('Kanana tokenizer with decode() is required to truncate candidate descriptions.');

  const render = (keepCount: number) =>
    renderCandidateInput(prepared, questionId, decodeHeadTail(tokenizer, ids, keepCount));
  const emptyCandidate = render(0);
  const protectedPrompt = countPrompt(tokenizer, emptyCandidate.messages, imageTokenThw);
  if (protectedPrompt.tokenCount > maxLength) failInputLength(maxLength, protectedPrompt.tokenCount);

  let low = 0;
  let high = ids.length;
  while (low < high) {
    const keepCount = Math.ceil((low + high) / 2);
    const attempt = render(keepCount);
    if (countPrompt(tokenizer, attempt.messages, imageTokenThw).tokenCount <= maxLength) low = keepCount;
    else high = keepCount - 1;
  }
  if (!low)
    failInputLength(maxLength, protectedPrompt.tokenCount, 'No token budget remains for the Question candidate.');

  const result = render(low);
  const encoded = countPrompt(tokenizer, result.messages, imageTokenThw);
  if (encoded.tokenCount > maxLength) failInputLength(maxLength, encoded.tokenCount);
  return {...encoded, messages: result.messages, truncated: true, retainedTokens: low};
}

/** Encode a candidate document, retaining its prefix if the pinned context limit requires truncation. */
export function truncateKananaDocument(
  tokenizer: KananaTokenizer,
  value: string,
  imageTokenThw: unknown = [],
  {
    maxLength = KANANA_MAX_LENGTH,
    instruction = KANANA_DOCUMENT_INSTRUCTION,
    truncateInputs = true
  }: DocumentTruncationOptions = {}
) {
  validateMaxLength(maxLength);
  if (typeof tokenizer?.encode !== 'function') throw new TypeError('A Kanana tokenizer with encode() is required.');
  if (typeof value !== 'string' || !pythonStrip(value)) {
    const error = new TypeError('Kanana candidate text must be a nonempty string.') as TypeError & {code: string};
    error.code = 'empty_candidate';
    throw error;
  }
  if (typeof instruction !== 'string') throw new TypeError('Kanana document instruction must be a string.');

  const messages: QueryMessage[] = [
    {role: 'system', content: [{type: 'text', text: instruction}]},
    {role: 'user', content: [{type: 'text', text: value}]}
  ];
  const original = queryResult(tokenizer, messages, imageTokenThw, false);
  if (original.tokenCount <= maxLength) return original;
  if (!truncateInputs) failInputLength(maxLength, original.tokenCount);

  const ids = encodeText(tokenizer, value);
  const protectedMessages = cloneQueryMessages(messages);
  (protectedMessages[1]!.content as QueryPart[])[0]!.text = '';
  const protectedPrompt = countPrompt(tokenizer, protectedMessages, imageTokenThw);
  if (protectedPrompt.tokenCount > maxLength) failInputLength(maxLength, protectedPrompt.tokenCount);
  if (!ids.length) failInputLength(maxLength, original.tokenCount);
  if (typeof tokenizer.decode !== 'function')
    throw new TypeError('Kanana tokenizer with decode() is required to truncate candidate documents.');

  let low = 0;
  let high = ids.length - 1;
  let best = {messages: protectedMessages, encoded: protectedPrompt, retainedTokens: 0};
  while (low <= high) {
    const keepCount = Math.floor((low + high) / 2);
    const candidateMessages = cloneQueryMessages(messages);
    (candidateMessages[1]!.content as QueryPart[])[0]!.text = tokenizer.decode(ids.slice(0, keepCount), {
      skip_special_tokens: false
    });
    const encoded = countPrompt(tokenizer, candidateMessages, imageTokenThw);
    if (encoded.tokenCount <= maxLength) {
      best = {messages: candidateMessages, encoded, retainedTokens: keepCount};
      low = keepCount + 1;
    } else {
      high = keepCount - 1;
    }
  }

  if (best.encoded.tokenCount > maxLength) failInputLength(maxLength, best.encoded.tokenCount);
  return {...best.encoded, messages: best.messages, truncated: true, retainedTokens: best.retainedTokens};
}
