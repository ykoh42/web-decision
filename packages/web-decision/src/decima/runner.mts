import {WebDecisionError} from '../kev.js';
import type {InferenceSession, TypedTensor} from 'onnxruntime-web/wasm';
import type {DecimaRawScores} from './protocol.mjs';
import type {DecimaTokenizer} from './tokenizer.mjs';

export type DecimaOrtApi = typeof import('onnxruntime-web/wasm');

interface DecimaBatch {
  readonly ids: BigInt64Array;
  readonly mask: BigInt64Array;
  readonly shape: number[];
}

export interface DecimaEncodedInputs {
  readonly states: TypedTensor<'float32'>;
  readonly mask: TypedTensor<'int64'>;
}

export interface DecimaRunner {
  encode(rows: readonly (readonly number[])[]): Promise<DecimaEncodedInputs>;
  scoreEncoded(state: DecimaEncodedInputs, choices: DecimaEncodedInputs, choiceCount: number): Promise<DecimaRawScores>;
  score(
    stateRows: readonly (readonly number[])[],
    choiceRows: readonly (readonly number[])[]
  ): Promise<DecimaRawScores>;
  dispose(): Promise<void>;
}

const fail = (message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function checkRows(rows: readonly (readonly number[])[], padId: number): DecimaBatch {
  if (!Array.isArray(rows) || !rows.length || rows.some((row) => !Array.isArray(row) || !row.length))
    throw fail('Decima received an empty token batch.');
  const width = rows.reduce((maximum, row) => Math.max(maximum, row.length), 0);
  const ids = new BigInt64Array(rows.length * width);
  const mask = new BigInt64Array(rows.length * width);
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
    const row = rows[rowIndex]!;
    const offset = rowIndex * width;
    for (let column = 0; column < width; column++) {
      ids[offset + column] = BigInt(row[column] ?? padId);
      mask[offset + column] = BigInt(column < row.length ? 1 : 0);
    }
  }
  return {ids, mask, shape: [rows.length, width]};
}

function isFloatTensor(tensor: unknown, minimumRank: number): tensor is TypedTensor<'float32'> {
  return (
    (tensor as TypedTensor<'float32'> | null | undefined)?.type === 'float32' &&
    Array.isArray((tensor as TypedTensor<'float32'>).dims) &&
    (tensor as TypedTensor<'float32'>).dims.length >= minimumRank &&
    (tensor as TypedTensor<'float32'>).dims.every(
      (dimension: number) => Number.isSafeInteger(dimension) && dimension > 0
    ) &&
    ArrayBuffer.isView((tensor as TypedTensor<'float32'>).data) &&
    (tensor as TypedTensor<'float32'>).data.length ===
      (tensor as TypedTensor<'float32'>).dims.reduce((size, value) => size * value, 1)
  );
}

export function createDecimaRunner(
  ort: DecimaOrtApi,
  encoder: InferenceSession,
  scorer: InferenceSession,
  tokenizer: DecimaTokenizer
): DecimaRunner {
  if (
    typeof ort?.Tensor !== 'function' ||
    typeof encoder?.run !== 'function' ||
    typeof scorer?.run !== 'function' ||
    typeof encoder?.release !== 'function' ||
    typeof scorer?.release !== 'function' ||
    !Number.isSafeInteger(tokenizer?.special?.padId)
  )
    throw new TypeError('Decima requires ONNX Runtime sessions and its tokenizer.');
  let closed = false;
  let disposing: Promise<void> | undefined;

  async function encode(rows: readonly (readonly number[])[]): Promise<DecimaEncodedInputs> {
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const batch = checkRows(rows, tokenizer.special.padId);
    const outputs = await encoder.run({
      input_ids: new ort.Tensor('int64', batch.ids, batch.shape),
      attention_mask: new ort.Tensor('int64', batch.mask, batch.shape)
    });
    const states = outputs?.token_states;
    if (!isFloatTensor(states, 3) || states.dims[0] !== rows.length || states.dims[1] !== batch.shape[1])
      throw fail('Decima encoder returned an unexpected tensor.');
    return {states, mask: new ort.Tensor('int64', batch.mask, batch.shape)};
  }

  async function scoreEncoded(
    state: DecimaEncodedInputs,
    choices: DecimaEncodedInputs,
    choiceCount: number
  ): Promise<DecimaRawScores> {
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    if (!state || !choices || !Number.isSafeInteger(choiceCount) || choiceCount < 2)
      throw fail('Decima scorer received invalid encoded inputs.');
    const outputs = await scorer.run({
      state_h: state.states,
      state_mask: state.mask,
      choice_h: choices.states,
      choice_mask: choices.mask
    });
    const scores = outputs?.scores;
    const features = outputs?.z;
    if (
      !isFloatTensor(scores, 1) ||
      scores.data.length !== choiceCount ||
      !isFloatTensor(features, 2) ||
      features.dims[0] !== choiceCount
    )
      throw fail('Decima scorer returned unexpected tensors.');
    const scoreValues = Array.from(scores.data);
    const featureValues = Array.from(features.data);
    if (scoreValues.some((value) => !Number.isFinite(value)) || featureValues.some((value) => !Number.isFinite(value)))
      throw fail('Decima scorer returned a non-finite value.');
    return {scores: scoreValues, features: featureValues};
  }

  return {
    encode,
    scoreEncoded,
    async score(
      stateRows: readonly (readonly number[])[],
      choiceRows: readonly (readonly number[])[]
    ): Promise<DecimaRawScores> {
      if (!Array.isArray(stateRows) || stateRows.length !== 1 || !Array.isArray(choiceRows) || choiceRows.length < 2)
        throw fail('Decima scorer received invalid input rows.');
      const [state, choices] = await Promise.all([encode(stateRows), encode(choiceRows)]);
      return scoreEncoded(state, choices, choiceRows.length);
    },
    dispose(): Promise<void> {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.allSettled([scorer.release(), encoder.release()]).then((results) => {
        const failure = results.find((result) => result.status === 'rejected');
        if (failure) throw fail('Could not release Decima ONNX sessions.', failure.reason);
      });
      return disposing;
    }
  };
}
