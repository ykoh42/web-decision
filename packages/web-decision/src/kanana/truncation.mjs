import {encodeKananaQuery} from './tokenizer.mjs';
import {pythonStrip} from './chat-template.mjs';
import {renderKananaQueryInput} from './protocol.mjs';

export const KANANA_MAX_LENGTH = 20_480;
export const KANANA_DOCUMENT_INSTRUCTION = "Represent the user's input.";

class KananaInputLengthError extends RangeError {
  constructor(maxLength, tokenCount, message) {
    super(message ?? `Kanana query has ${tokenCount} tokens, exceeding the configured limit of ${maxLength}.`);
    this.name = 'KananaInputLengthError';
    this.code = 'input_too_long';
    this.maxLength = maxLength;
    this.tokenCount = tokenCount;
  }
}

function validateMaxLength(maxLength) {
  if (!Number.isSafeInteger(maxLength) || maxLength < 1)
    throw new RangeError('Kanana maxLength must be a positive safe integer.');
}

function cloneQueryMessages(messages) {
  if (!Array.isArray(messages)) throw new TypeError('Kanana query messages must be an array.');
  return messages.map((message) => {
    if (!message || typeof message !== 'object' || Array.isArray(message))
      throw new TypeError('Kanana query messages must be objects.');
    return {
      ...message,
      content: Array.isArray(message.content) ? message.content.map((part) => ({...part})) : message.content
    };
  });
}

function validateEditableIndexes(messages, editableUserTextIndexes) {
  if (!Array.isArray(editableUserTextIndexes))
    throw new TypeError('editableUserTextIndexes must be returned by renderKananaQueryInput().');
  if (messages.length !== 2 || messages[0]?.role !== 'system' || messages[1]?.role !== 'user')
    throw new TypeError('Kanana truncation expects one system and one user query message.');
  const content = messages[1].content;
  if (!Array.isArray(content) || content.length < 2) throw new TypeError('Kanana user content must be an array.');
  if (content[0]?.type !== 'text' || content[0].text !== 'State:\n')
    throw new TypeError('Kanana query state prefix is protected.');
  const suffix = content.at(-1);
  if (suffix?.type !== 'text' || !suffix.text.startsWith('\n\nTask:\n'))
    throw new TypeError('Kanana task and options suffix is protected.');

  const seen = new Set();
  for (const index of editableUserTextIndexes) {
    if (
      !Number.isSafeInteger(index) ||
      index <= 0 ||
      index >= content.length - 1 ||
      seen.has(index) ||
      content[index]?.type !== 'text' ||
      typeof content[index].text !== 'string'
    )
      throw new TypeError(
        'Editable query boundaries must identify unique state text parts between the protected prefix and suffix.'
      );
    seen.add(index);
  }
  return [...seen];
}

function encodeText(tokenizer, value) {
  const encoded = tokenizer.encode(value, {add_special_tokens: false});
  const ids = encoded?.ids;
  if (!Array.isArray(ids) && !(ArrayBuffer.isView(ids) && !(ids instanceof DataView)))
    throw new TypeError('Kanana tokenizer must return indexed token IDs.');
  const result = [];
  for (let index = 0; index < ids.length; index++) {
    if (Array.isArray(ids) && !Object.hasOwn(ids, index))
      throw new TypeError('Kanana tokenizer returned a sparse token ID array.');
    const id = ids[index];
    if (!Number.isSafeInteger(id) || id < 0) throw new TypeError('Kanana tokenizer returned an invalid token ID.');
    result.push(id);
  }
  return result;
}

function renderRetainedText(tokenizer, textParts, keepCount) {
  const allIds = textParts.flatMap(({ids}) => ids);
  const headCount = Math.ceil(keepCount / 2);
  const tailCount = Math.floor(keepCount / 2);
  const tailStart = allIds.length - tailCount;
  let cursor = 0;

  return textParts.map(({index, ids}) => {
    const start = cursor;
    const end = start + ids.length;
    cursor = end;

    const kept = [];
    const headEnd = Math.min(end, headCount);
    if (start < headEnd) kept.push(...allIds.slice(start, headEnd));
    const keptTailStart = Math.max(start, tailStart);
    if (keptTailStart < end) kept.push(...allIds.slice(keptTailStart, end));

    return [index, tokenizer.decode(kept, {skip_special_tokens: false})];
  });
}

function candidateMessages(originalMessages, textParts, tokenizer, keepCount) {
  const messages = cloneQueryMessages(originalMessages);
  const userContent = messages[1].content;
  for (const [index, text] of renderRetainedText(tokenizer, textParts, keepCount)) userContent[index].text = text;
  return messages;
}

function countPrompt(tokenizer, messages, imageTokenThw) {
  return encodeKananaQuery(tokenizer, messages, imageTokenThw);
}

function failInputLength(maxLength, tokenCount, message) {
  throw new KananaInputLengthError(maxLength, tokenCount, message);
}

function queryResult(tokenizer, messages, imageTokenThw, truncated, retainedTokens = null) {
  return {
    ...countPrompt(tokenizer, messages, imageTokenThw),
    messages,
    truncated,
    retainedTokens
  };
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value) && !(value instanceof Map);
}

function candidatePrepared(prepared, questionId, candidate) {
  if (!Array.isArray(prepared?.questionEntries)) throw new TypeError('Expected a prepared Kanana request.');
  const index = prepared.questionEntries.findIndex((entry) => entry.id === questionId);
  if (index < 0) throw new RangeError(`Unknown Kanana question ID: ${String(questionId)}`);

  const entry = prepared.questionEntries[index];
  const instructions = entry.question?.instructions;
  if (!isRecord(instructions) || typeof instructions.candidate !== 'string')
    throw new TypeError('Kanana Question candidate truncation requires instructions.candidate to be a string.');

  const questionEntries = [...prepared.questionEntries];
  questionEntries[index] = {
    ...entry,
    question: {...entry.question, instructions: {...instructions, candidate}}
  };
  return {...prepared, questionEntries};
}

function structuredCandidate(prepared, questionId) {
  if (!Array.isArray(prepared?.questionEntries)) throw new TypeError('Expected a prepared Kanana request.');
  const entry = prepared.questionEntries.find((item) => item.id === questionId);
  if (!entry) throw new RangeError(`Unknown Kanana question ID: ${String(questionId)}`);
  const instructions = entry.question?.instructions;
  if (!isRecord(instructions) || typeof instructions.candidate !== 'string') return null;
  return instructions.candidate;
}

function renderCandidateInput(prepared, questionId, candidate) {
  return renderKananaQueryInput(candidatePrepared(prepared, questionId, candidate), questionId);
}

function decodeHeadTail(tokenizer, ids, keepCount) {
  const headCount = Math.ceil(keepCount / 2);
  const tailCount = Math.floor(keepCount / 2);
  const head = tokenizer.decode(ids.slice(0, headCount), {skip_special_tokens: false});
  const tail = tailCount ? tokenizer.decode(ids.slice(-tailCount), {skip_special_tokens: false}) : '';
  return head + tail;
}

/**
 * Match the pinned Kanana query-state truncation behavior while preserving every
 * non-state part of the rendered prompt. The limit defaults to the model config
 * value and is never clamped to an adapter-specific lower cap.
 */
export function truncateKananaQuery(
  tokenizer,
  messages,
  imageTokenThw = [],
  {editableUserTextIndexes, maxLength = KANANA_MAX_LENGTH, truncateInputs = true} = {}
) {
  validateMaxLength(maxLength);
  if (typeof tokenizer?.encode !== 'function') throw new TypeError('A Kanana tokenizer with encode() is required.');
  const editableIndexes = validateEditableIndexes(messages, editableUserTextIndexes);
  const originalMessages = cloneQueryMessages(messages);
  const original = countPrompt(tokenizer, originalMessages, imageTokenThw);
  if (original.tokenCount <= maxLength)
    return {...original, messages: originalMessages, truncated: false, retainedStateTokens: null};
  if (!truncateInputs) failInputLength(maxLength, original.tokenCount);

  const textParts = editableIndexes.map((index) => ({
    index,
    ids: encodeText(tokenizer, originalMessages[1].content[index].text)
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
  tokenizer,
  prepared,
  questionId,
  imageTokenThw = [],
  {maxLength = KANANA_MAX_LENGTH, truncateInputs = true, truncateQuestionCandidates = true} = {}
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

  const render = (keepCount) => renderCandidateInput(prepared, questionId, decodeHeadTail(tokenizer, ids, keepCount));
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
  tokenizer,
  value,
  imageTokenThw = [],
  {maxLength = KANANA_MAX_LENGTH, instruction = KANANA_DOCUMENT_INSTRUCTION, truncateInputs = true} = {}
) {
  validateMaxLength(maxLength);
  if (typeof tokenizer?.encode !== 'function') throw new TypeError('A Kanana tokenizer with encode() is required.');
  if (typeof value !== 'string' || !pythonStrip(value)) {
    const error = new TypeError('Kanana candidate text must be a nonempty string.');
    error.code = 'empty_candidate';
    throw error;
  }
  if (typeof instruction !== 'string') throw new TypeError('Kanana document instruction must be a string.');

  const messages = [
    {role: 'system', content: [{type: 'text', text: instruction}]},
    {role: 'user', content: [{type: 'text', text: value}]}
  ];
  const original = queryResult(tokenizer, messages, imageTokenThw, false);
  if (original.tokenCount <= maxLength) return original;
  if (!truncateInputs) failInputLength(maxLength, original.tokenCount);

  const ids = encodeText(tokenizer, value);
  const protectedMessages = cloneQueryMessages(messages);
  protectedMessages[1].content[0].text = '';
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
    candidateMessages[1].content[0].text = tokenizer.decode(ids.slice(0, keepCount), {skip_special_tokens: false});
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
