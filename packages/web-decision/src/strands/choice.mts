import {WebDecisionError} from '../kev.js';
import {STRANDS_FUSED_GRAPH_MAX_TOKENS} from './contract.mjs';
import {encodeChoice, scoreChoice, snapshotChoiceQuestion} from './protocol.mjs';

export const MAX_TOKENS = STRANDS_FUSED_GRAPH_MAX_TOKENS;
const VOCAB_SIZE = 248320;
const PYTHON_WHITESPACE_ONLY = new RegExp(
  // Python's whitespace includes these control characters.
  // eslint-disable-next-line no-control-regex
  '^[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]*$',
  'u'
);
interface ChoiceRowSource {
  readonly inputIds?: unknown;
  readonly answerIndex?: unknown;
  readonly optionIndices?: unknown;
  readonly options?: unknown;
}

interface EncodedChoice {
  readonly inputIds: readonly number[];
  readonly optionIndices: readonly number[];
  readonly answerIndex: number;
  readonly stateText: string;
  readonly questionText: string;
}

interface ChoiceQuestion {
  readonly instructions: string;
  readonly options: readonly string[];
}

interface ValidatedStrandsRow {
  readonly inputIds: readonly number[];
  readonly answerIndex: number;
  readonly optionIndices: readonly number[];
  readonly options: readonly string[];
}

interface PackedStrandsRow extends ValidatedStrandsRow {
  readonly stateText: string;
  readonly questionText: string;
}

interface PackedStrandsRequest {
  readonly rows: readonly PackedStrandsRow[];
  readonly tokenCount: number;
}

interface ScoresRequest {
  readonly rows: readonly {readonly options: readonly string[]}[];
  readonly tokenCount: number;
}

interface ChoiceProbability {
  readonly option: string;
  readonly logit: number;
  readonly probability: number;
}

interface ChoiceScore {
  readonly selected: string;
  readonly selectedIndex: number;
  readonly probabilities: readonly ChoiceProbability[];
}

interface UnpackedScores {
  readonly tokenCount: number;
  readonly answers: readonly ChoiceScore[];
}

const invalid = (message: string, cause?: unknown): Error =>
  new WebDecisionError('INVALID_INPUT', message, cause ? {cause} : undefined);

function snapshotTokenIds(sourceIds: unknown, error: () => Error): number[] {
  if (!Array.isArray(sourceIds)) throw error();
  const inputIds: number[] = [],
    count = sourceIds.length;
  if (count < 2 || count > MAX_TOKENS) throw error();
  for (let i = 0; i < count; i++) {
    if (!Object.hasOwn(sourceIds, i)) throw error();
    const id = (sourceIds as unknown[])[i];
    if (!Number.isSafeInteger(id as number) || (id as number) < 0 || (id as number) >= VOCAB_SIZE) throw error();
    inputIds.push(id as number);
  }
  return inputIds;
}

/** Return the exact validated values to use for an experimental graph call. */
export function validateStrandsRow(row: unknown): ValidatedStrandsRow {
  const error = () => invalid('Invalid Strands token IDs, answer, or choice positions.');
  const inputIds = snapshotTokenIds((row as ChoiceRowSource | null | undefined)?.inputIds, error),
    count = inputIds.length;
  const answerIndex = (row as ChoiceRowSource | null | undefined)?.answerIndex;
  if (answerIndex !== count - 1) throw error();
  const sourceIndices = (row as ChoiceRowSource | null | undefined)?.optionIndices,
    sourceOptions = (row as ChoiceRowSource | null | undefined)?.options;
  if (!Array.isArray(sourceIndices) || !Array.isArray(sourceOptions)) throw error();
  const optionIndices: number[] = [],
    options: string[] = [],
    seen = new Set<string>(),
    optionCount = sourceOptions.length;
  if (optionCount < 2 || optionCount > 255 || sourceIndices.length !== optionCount) throw error();
  for (let i = 0; i < optionCount; i++) {
    if (!Object.hasOwn(sourceIndices, i) || !Object.hasOwn(sourceOptions, i)) throw error();
    const index = (sourceIndices as unknown[])[i],
      text = (sourceOptions as unknown[])[i];
    if (
      !Number.isSafeInteger(index as number) ||
      (index as number) < 0 ||
      (index as number) >= (answerIndex as number) ||
      (i > 0 && (index as number) <= optionIndices[i - 1]!) ||
      typeof text !== 'string' ||
      PYTHON_WHITESPACE_ONLY.test(text) ||
      seen.has(text)
    )
      throw error();
    optionIndices.push(index as number);
    options.push(text as string);
    seen.add(text);
  }
  return {inputIds, answerIndex, optionIndices, options};
}

/** Snapshot the entire request, then encode one independent causal row per question. */
export function packStrandsRequest(tokenizer: unknown, request: unknown): PackedStrandsRequest {
  if (!request || typeof request !== 'object' || Array.isArray(request))
    throw invalid('Provide a string state and at least one question.');
  const state = (request as Record<string, unknown>).state;
  const source = (request as Record<string, unknown>).questions;
  if (typeof state !== 'string' || !Array.isArray(source))
    throw invalid('Provide a string state and at least one question.');
  const questions: ChoiceQuestion[] = [],
    questionCount = source.length;
  if (!questionCount) throw invalid('Provide a string state and at least one question.');
  for (let i = 0; i < questionCount; i++) {
    if (!Object.hasOwn(source, i)) throw invalid('Each question needs instructions and options.');
    try {
      questions.push(snapshotChoiceQuestion((source as unknown[])[i]) as ChoiceQuestion);
    } catch (cause) {
      throw invalid('Each question needs instructions and options.', cause);
    }
  }
  const rows: PackedStrandsRow[] = [];
  for (let i = 0; i < questionCount; i++) {
    const question = questions[i]!;
    let encoded: EncodedChoice;
    try {
      encoded = encodeChoice(tokenizer, state, question, MAX_TOKENS) as EncodedChoice;
    } catch (cause) {
      throw invalid('Could not encode a Strands choice question.', cause);
    }
    const snapshot = validateStrandsRow({
      inputIds: encoded.inputIds,
      optionIndices: encoded.optionIndices,
      answerIndex: encoded.answerIndex,
      options: question.options
    });
    const row = {
      ...snapshot,
      inputIds: Object.freeze(snapshot.inputIds),
      optionIndices: Object.freeze(snapshot.optionIndices),
      options: Object.freeze(snapshot.options),
      stateText: encoded.stateText,
      questionText: encoded.questionText
    };
    rows.push(Object.freeze(row));
  }
  return Object.freeze({
    rows: Object.freeze(rows),
    tokenCount: rows.reduce((sum, row) => sum + row.inputIds.length, 0)
  });
}

function snapshotScoreRow(scores: unknown, count: number): number[] {
  const array = Array.isArray(scores);
  if (
    (!array && !(ArrayBuffer.isView(scores) && !(scores instanceof DataView))) ||
    (scores as {readonly length: number}).length !== count
  )
    throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid choice logits.');
  const values: number[] = [];
  for (let i = 0; i < count; i++) {
    if (array && !Object.hasOwn(scores, i))
      throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid choice logits.');
    const value = (scores as ArrayLike<unknown>)[i];
    if (!Number.isFinite(value as number))
      throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid choice logits.');
    values.push(value as number);
  }
  return values;
}

/** The fused graph emits raw FP32 logits; the pinned temperature is applied once. */
export function unpackStrandsScores(rows: unknown, packed: ScoresRequest | null | undefined): UnpackedScores {
  if (!Array.isArray(rows) || rows.length !== packed?.rows?.length)
    throw new WebDecisionError('INFERENCE_ERROR', 'The model returned the wrong number of score rows.');
  const snapshots: number[][] = [];
  for (let index = 0; index < rows.length; index++) {
    if (!Object.hasOwn(rows, index))
      throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid choice logits.');
    snapshots.push(snapshotScoreRow((rows as unknown[])[index], packed!.rows[index]!.options.length));
  }
  return {
    tokenCount: packed!.tokenCount,
    answers: packed!.rows.map((row, index) => scoreChoice(row.options, snapshots[index]!) as ChoiceScore)
  };
}
