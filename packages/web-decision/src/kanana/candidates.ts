import {
  KANANA_DOCUMENT_INSTRUCTION,
  KANANA_MAX_LENGTH,
  truncateKananaDocument,
  truncateKananaQuery,
  truncateKananaQuestionCandidate
} from './truncation.js';
import {renderKananaQueryInput} from './protocol.js';

type CandidateTokenizer = {
  encode(text: string, options?: {add_special_tokens?: boolean}): {ids: ArrayLike<number>};
  decode?(ids: number[], options?: {skip_special_tokens?: boolean}): string;
};
type CandidateEntry = {
  id: string;
  candidateKeys: unknown[];
  candidateTexts: unknown[];
  [key: string]: unknown;
};
type CandidateDocument = {
  key: string;
  text: string;
  messages: unknown;
  inputIds: number[];
  tokenCount: number;
  truncated: boolean;
  retainedTokens: unknown;
};
type PreparedCandidateEntry = Omit<CandidateEntry, 'candidateKeys' | 'candidateTexts'> & {
  candidateKeys: string[];
  candidateTexts: string[];
  candidateDocuments: CandidateDocument[];
};
type CandidateOptions = {maxLength?: number; truncateInputs?: boolean; instruction?: string};
type QueryFallbackOptions = {maxLength?: number; truncateInputs?: boolean; truncateQuestionCandidates?: boolean};

function fail(code: string, message: string): never {
  const error = new TypeError(message);
  (error as TypeError & {code: string}).code = code;
  throw error;
}

function validateEntries(prepared: unknown): CandidateEntry[] {
  if (!Array.isArray((prepared as {questionEntries?: unknown} | null | undefined)?.questionEntries))
    throw new TypeError('Expected a prepared Kanana request.');
  const request = prepared as {questionEntries: CandidateEntry[]};
  for (let index = 0; index < request.questionEntries.length; index++) {
    if (!Object.hasOwn(request.questionEntries, index))
      fail('questions', 'prepared question entries must not contain empty entries.');
  }
  return request.questionEntries;
}

function validateCandidateArray(value: unknown, label: string, questionId: string): unknown[] {
  if (!Array.isArray(value)) fail('candidate_count', `${label} for question ${questionId} must be an array.`);
  for (let index = 0; index < value.length; index++) {
    if (!Object.hasOwn(value, index)) fail('candidate_count', `${label} must not contain empty entries.`);
  }
  return value;
}

function candidateSequence(inputIds: number[]): string {
  return inputIds.join(',');
}

function prepareQuestionCandidates(
  tokenizer: CandidateTokenizer,
  entry: CandidateEntry,
  options: {maxLength: number; truncateInputs: boolean; instruction: string},
  truncateInputs: boolean
): PreparedCandidateEntry {
  const candidateKeys = validateCandidateArray(entry?.candidateKeys, 'candidateKeys', entry?.id);
  const candidateTexts = validateCandidateArray(entry?.candidateTexts, 'candidateTexts', entry?.id);
  if (candidateKeys.length !== candidateTexts.length)
    fail('candidate_count', `candidateKeys and candidateTexts for question ${entry?.id} must have equal lengths.`);

  const seen = new Set();
  const candidateDocuments: CandidateDocument[] = [];
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
    candidateKeys: [...candidateKeys] as string[],
    candidateTexts: [...candidateTexts] as string[],
    candidateDocuments
  };
}

/** Prepare candidate documents in model order and detect untruncated token collisions. */
export function prepareKananaCandidates(
  tokenizer: CandidateTokenizer,
  prepared: unknown,
  {
    maxLength = KANANA_MAX_LENGTH,
    truncateInputs = true,
    instruction = KANANA_DOCUMENT_INSTRUCTION
  }: CandidateOptions = {}
): PreparedCandidateEntry[] {
  const questionEntries = validateEntries(prepared);
  if (typeof truncateInputs !== 'boolean') throw new TypeError('truncateInputs must be a boolean.');
  const options = {maxLength, truncateInputs, instruction};
  return questionEntries.map((entry) => prepareQuestionCandidates(tokenizer, entry, options, truncateInputs));
}

/** Truncate query state first, then shorten a structured question candidate only if protected input cannot fit. */
export function truncateKananaQueryWithCandidateFallback(
  tokenizer: CandidateTokenizer,
  prepared: unknown,
  questionId: string,
  imageTokenThw: number[][] = [],
  {maxLength = KANANA_MAX_LENGTH, truncateInputs = true, truncateQuestionCandidates = true}: QueryFallbackOptions = {}
): ReturnType<typeof truncateKananaQuery> | ReturnType<typeof truncateKananaQuestionCandidate> {
  const input = renderKananaQueryInput(prepared, questionId);
  try {
    return truncateKananaQuery(tokenizer, input.messages, imageTokenThw, {
      editableUserTextIndexes: input.editableUserTextIndexes,
      maxLength,
      truncateInputs
    } as {editableUserTextIndexes: number[]; maxLength: number; truncateInputs: boolean});
  } catch (error) {
    if (
      (error as {code?: unknown} | null | undefined)?.code !== 'input_too_long' ||
      !truncateInputs ||
      !truncateQuestionCandidates
    )
      throw error;
    return truncateKananaQuestionCandidate(tokenizer, prepared, questionId, imageTokenThw, {
      maxLength,
      truncateInputs,
      truncateQuestionCandidates
    });
  }
}
