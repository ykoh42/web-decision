import {WebDecisionError} from '../kev.js';
import {CRIVO_BATCH_SIZE, CRIVO_BATCH_TOKEN_BUDGET, CRIVO_PAD_MULTIPLE} from './contract.mjs';
import type {CrivoAbortSignalLike, CrivoPreparedRow} from './protocol.mjs';

interface CrivoTensor {
  readonly type: string;
  readonly data: ArrayBufferView;
  readonly dims: number[];
  dispose?(): unknown;
}

interface CrivoTensorConstructor {
  new (type: string, data: ArrayBufferView, dims: number[]): CrivoTensor;
}

interface CrivoOrtRuntime {
  readonly Tensor: CrivoTensorConstructor;
}

interface CrivoOrtSession {
  run(feeds: Record<string, CrivoTensor>, outputNames: string[]): unknown;
  release(): unknown;
}

interface CrivoRunner {
  run(rows: readonly CrivoPreparedRow[], signal?: CrivoAbortSignalLike | null): Promise<Float32Array[]>;
  dispose(): Promise<void>;
}

const fail = (message: string, cause?: unknown) =>
  new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function makeFeeds(ort: CrivoOrtRuntime, rows: readonly CrivoPreparedRow[]) {
  const batchSize = rows.length;
  const sequenceLength =
    Math.ceil(Math.max(...rows.map((row) => row.inputIds.length)) / CRIVO_PAD_MULTIPLE) * CRIVO_PAD_MULTIPLE;
  const optionCount = Math.max(...rows.map((row) => row.optionPositions.length));
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);
  const optionPositions = new BigInt64Array(batchSize * optionCount);
  const optionMask = new Uint8Array(batchSize * optionCount);
  const qtype = new BigInt64Array(batchSize);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const row = rows[rowIndex]!;
    const offset = rowIndex * sequenceLength;
    for (let tokenIndex = 0; tokenIndex < row.inputIds.length; tokenIndex++) {
      inputIds[offset + tokenIndex] = BigInt(row.inputIds[tokenIndex]!);
      attentionMask[offset + tokenIndex] = 1n;
    }
    for (let optionIndex = 0; optionIndex < row.optionPositions.length; optionIndex++) {
      optionPositions[rowIndex * optionCount + optionIndex] = BigInt(row.optionPositions[optionIndex]!);
      optionMask[rowIndex * optionCount + optionIndex] = 1;
    }
    qtype[rowIndex] = BigInt(row.qtype);
  }

  return {
    optionCount,
    values: {
      input_ids: new ort.Tensor('int64', inputIds, [batchSize, sequenceLength]),
      attention_mask: new ort.Tensor('int64', attentionMask, [batchSize, sequenceLength]),
      opt_positions: new ort.Tensor('int64', optionPositions, [batchSize, optionCount]),
      opt_mask: new ort.Tensor('bool', optionMask, [batchSize, optionCount]),
      qtype: new ort.Tensor('int64', qtype, [batchSize])
    }
  };
}

function readLogits(output: unknown, batchSize: number, optionCount: number): Float32Array[] {
  const tensor = (output as {readonly logits?: unknown} | null | undefined)?.logits as
    | {readonly type?: unknown; readonly dims?: unknown; readonly data?: unknown}
    | null
    | undefined;
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== batchSize ||
    tensor.dims[1] !== optionCount ||
    !ArrayBuffer.isView(tensor.data) ||
    (tensor.data as ArrayBufferView & {readonly length: number}).length !== batchSize * optionCount
  )
    throw fail('Crivo ONNX logits have an unexpected shape.');
  const values = Array.from(tensor.data as unknown as ArrayLike<number>);
  if (values.some((value) => !Number.isFinite(value))) throw fail('Crivo ONNX logits are not finite.');
  return Array.from({length: batchSize}, (_, index) =>
    Float32Array.from(values.slice(index * optionCount, (index + 1) * optionCount))
  );
}

function makeBatches(rows: readonly CrivoPreparedRow[]): number[][] {
  const order = rows
    .map((_, index) => index)
    .sort((left, right) => rows[left]!.inputIds.length - rows[right]!.inputIds.length);
  const batches: number[][] = [];
  let batch: number[] = [];
  for (const index of order) {
    const nextLength = rows[index]!.inputIds.length;
    if (
      batch.length > 0 &&
      (batch.length >= CRIVO_BATCH_SIZE || nextLength * (batch.length + 1) > CRIVO_BATCH_TOKEN_BUDGET)
    ) {
      batches.push(batch);
      batch = [];
    }
    batch.push(index);
  }
  if (batch.length > 0) batches.push(batch);
  return batches;
}

export function createCrivoRunner(ort: unknown, session: unknown): CrivoRunner {
  const checkedRuntime = ort as {readonly Tensor?: unknown} | null | undefined;
  const checkedSession = session as {readonly run?: unknown; readonly release?: unknown} | null | undefined;
  if (
    typeof checkedRuntime?.Tensor !== 'function' ||
    typeof checkedSession?.run !== 'function' ||
    typeof checkedSession?.release !== 'function'
  )
    throw new TypeError('Crivo runner components are required.');

  const runtime = ort as CrivoOrtRuntime;
  let activeSession: CrivoOrtSession | null = session as CrivoOrtSession;
  let closed = false;
  let disposing: Promise<void> | undefined;
  return {
    async run(rows: readonly CrivoPreparedRow[], signal?: CrivoAbortSignalLike | null) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Crivo model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1) throw fail('Crivo needs at least one question.');
      const results: Float32Array[] = [];
      results.length = rows.length;
      for (const batch of makeBatches(rows)) {
        signal?.throwIfAborted();
        const batchRows = batch.map((index) => rows[index]!);
        const feeds = makeFeeds(runtime, batchRows);
        let output: unknown;
        try {
          output = await activeSession!.run(feeds.values, ['logits']);
          const logits = readLogits(output, batchRows.length, feeds.optionCount);
          for (let index = 0; index < batch.length; index++) {
            const optionCount = batchRows[index]!.optionPositions.length;
            results[batch[index]!] = logits[index]!.slice(0, optionCount);
          }
        } catch (cause) {
          if (
            cause instanceof WebDecisionError ||
            (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
          )
            throw cause;
          throw fail('Crivo ONNX inference failed.', cause);
        } finally {
          for (const tensor of Object.values(feeds.values)) tensor.dispose?.();
          for (const tensor of Object.values((output ?? {}) as Record<string, unknown>))
            (tensor as {dispose?: () => unknown}).dispose?.();
        }
        signal?.throwIfAborted();
      }
      return results;
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve().then(async () => {
        await activeSession!.release();
        activeSession = null;
      });
      return disposing;
    }
  };
}
