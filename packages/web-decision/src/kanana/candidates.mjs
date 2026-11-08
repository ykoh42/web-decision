import {
  KANANA_DOCUMENT_INSTRUCTION,
  KANANA_MAX_LENGTH,
  truncateKananaDocument,
  truncateKananaQuery,
  truncateKananaQuestionCandidate
} from './truncation.mjs';
import {renderKananaQueryInput} from './protocol.mjs';

function fail(code, message) {
  const error = new TypeError(message);
  error.code = code;
  throw error;
}

function validateEntries(prepared) {
  if (!Array.isArray(prepared?.questionEntries)) throw new TypeError('Expected a prepared Kanana request.');
  for (let index = 0; index < prepared.questionEntries.length; index++) {
    if (!Object.hasOwn(prepared.questionEntries, index))
      fail('questions', 'prepared question entries must not contain empty entries.');
  }
  return prepared.questionEntries;
}

function validateCandidateArray(value, label, questionId) {
  if (!Array.isArray(value)) fail('candidate_count', `${label} for question ${questionId} must be an array.`);
  for (let index = 0; index < value.length; index++) {
    if (!Object.hasOwn(value, index)) fail('candidate_count', `${label} must not contain empty entries.`);
  }
  return value;
}

function candidateSequence(inputIds) {
  return inputIds.join(',');
}

function prepareQuestionCandidates(tokenizer, entry, options, truncateInputs) {
  const candidateKeys = validateCandidateArray(entry?.candidateKeys, 'candidateKeys', entry?.id);
  const candidateTexts = validateCandidateArray(entry?.candidateTexts, 'candidateTexts', entry?.id);
  if (candidateKeys.length !== candidateTexts.length)
    fail('candidate_count', `candidateKeys and candidateTexts for question ${entry?.id} must have equal lengths.`);

  const seen = new Set();
  const candidateDocuments = [];
  for (let index = 0; index < candidateTexts.length; index++) {
    const key = candidateKeys[index];
    const text = candidateTexts[index];
    if (typeof key !== 'string' || !key) fail('candidate_name', 'candidate keys must be nonempty strings.');
    if (typeof text !== 'string') fail('empty_candidate', 'candidate text must be a nonempty string.');

    const document = truncateKananaDocument(tokenizer, text, [], options);
    const sequence = candidateSequence(document.inputIds);
    if (!truncateInputs && seen.has(sequence))
      fail('candidate_token_collision', 'different labels have identical candidate token inputs.');
    seen.add(sequence);
    candidateDocuments.push({
      key,
      text,
      messages: document.messages,
      inputIds: document.inputIds,
      tokenCount: document.tokenCount,
      truncated: document.truncated,
      retainedTokens: document.retainedTokens
    });
  }

  return {
    ...entry,
    candidateKeys: [...candidateKeys],
    candidateTexts: [...candidateTexts],
    candidateDocuments
  };
}

/** Prepare candidate documents in model order and detect untruncated token collisions. */
export function prepareKananaCandidates(
  tokenizer,
  prepared,
  {maxLength = KANANA_MAX_LENGTH, truncateInputs = true, instruction = KANANA_DOCUMENT_INSTRUCTION} = {}
) {
  const questionEntries = validateEntries(prepared);
  if (typeof truncateInputs !== 'boolean') throw new TypeError('truncateInputs must be a boolean.');
  const options = {maxLength, truncateInputs, instruction};
  return questionEntries.map((entry) => prepareQuestionCandidates(tokenizer, entry, options, truncateInputs));
}

/** Truncate query state first, then shorten a structured question candidate only if protected input cannot fit. */
export function truncateKananaQueryWithCandidateFallback(
  tokenizer,
  prepared,
  questionId,
  imageTokenThw = [],
  {maxLength = KANANA_MAX_LENGTH, truncateInputs = true, truncateQuestionCandidates = true} = {}
) {
  const input = renderKananaQueryInput(prepared, questionId);
  try {
    return truncateKananaQuery(tokenizer, input.messages, imageTokenThw, {
      editableUserTextIndexes: input.editableUserTextIndexes,
      maxLength,
      truncateInputs
    });
  } catch (error) {
    if (error?.code !== 'input_too_long' || !truncateInputs || !truncateQuestionCandidates) throw error;
    return truncateKananaQuestionCandidate(tokenizer, prepared, questionId, imageTokenThw, {
      maxLength,
      truncateInputs,
      truncateQuestionCandidates
    });
  }
}
