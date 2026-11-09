import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_BATCH_SIZE} from './contract.mjs';

const fail = (message: string, cause?: unknown) =>
  new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

interface SafeDecideInputRow {
  readonly inputIds: number[];
}

interface SafeDecideTensor {
  dispose?(): void;
}

interface SafeDecideTensorConstructor {
  new (type: 'int64', data: BigInt64Array, dims: number[]): SafeDecideTensor;
}

interface SafeDecideOrt {
  readonly Tensor?: unknown;
}

interface SafeDecideOrtWithTensor extends SafeDecideOrt {
  readonly Tensor: SafeDecideTensorConstructor;
}

interface SafeDecideOnnxSession {
  run(feeds: Record<string, SafeDecideTensor>, outputNames: string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface SafeDecideTensorCandidate {
  readonly type?: unknown;
  readonly dims?: unknown;
  readonly data?: unknown;
}

interface SafeDecideSignal {
  throwIfAborted(): void;
}

interface SafeDecideRunner {
  run(rows: unknown[], signal?: SafeDecideSignal | null): Promise<number[]>;
  dispose(): Promise<void>;
}

function makeFeeds(
  ort: SafeDecideOrtWithTensor,
  rows: SafeDecideInputRow[],
  padId: number
): Record<string, SafeDecideTensor> {
  const batchSize = rows.length;
  const sequenceLength = Math.max(...rows.map((row) => row.inputIds.length));
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const ids = rows[rowIndex]!.inputIds;
    const offset = rowIndex * sequenceLength;
    for (let tokenIndex = 0; tokenIndex < sequenceLength; tokenIndex++) {
      inputIds[offset + tokenIndex] = BigInt(ids[tokenIndex] ?? padId);
      attentionMask[offset + tokenIndex] = BigInt(tokenIndex < ids.length ? 1 : 0);
    }
  }

  return {
    input_ids: new ort.Tensor('int64', inputIds, [batchSize, sequenceLength]),
    attention_mask: new ort.Tensor('int64', attentionMask, [batchSize, sequenceLength])
  };
}

function readLogits(tensor: unknown, rows: number): number[] {
  if (
    (tensor as SafeDecideTensorCandidate | null | undefined)?.type !== 'float32' ||
    !Array.isArray((tensor as SafeDecideTensorCandidate).dims) ||
    ((tensor as SafeDecideTensorCandidate).dims as unknown[]).length !== 2 ||
    ((tensor as SafeDecideTensorCandidate).dims as unknown[])[0] !== rows ||
    ((tensor as SafeDecideTensorCandidate).dims as unknown[])[1] !== 1 ||
    !ArrayBuffer.isView((tensor as SafeDecideTensorCandidate).data) ||
    ((tensor as SafeDecideTensorCandidate).data as unknown as {length: number}).length !== rows
  )
    throw fail('SafeDecide ONNX logits have an unexpected shape.');
  const logits = Array.from((tensor as SafeDecideTensorCandidate).data as unknown as ArrayLike<unknown>);
  if (logits.some((value) => !Number.isFinite(value))) throw fail('SafeDecide ONNX logits are not finite.');
  return logits as number[];
}

export function createSafeDecideRunner(
  ort: unknown,
  session: SafeDecideOnnxSession | null,
  padId: number,
  batchSize: number = SAFEDECIDE_BATCH_SIZE
): SafeDecideRunner {
  if (
    typeof (ort as SafeDecideOrt | null | undefined)?.Tensor !== 'function' ||
    typeof session?.run !== 'function' ||
    typeof session?.release !== 'function' ||
    !Number.isSafeInteger(padId) ||
    !Number.isSafeInteger(batchSize) ||
    batchSize < 1
  )
    throw new TypeError('SafeDecide runner components are required.');

  let closed = false;
  let disposing: Promise<void> | undefined;
  return {
    async run(rows: unknown[], signal?: SafeDecideSignal | null): Promise<number[]> {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The SafeDecide model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1) throw fail('SafeDecide needs at least one option pair.');
      // Keep one sparse output slot per input until each batch writes its logits.
      // oxlint-disable-next-line unicorn/no-new-array -- Sparse slots preserve the original batch result layout.
      const logits = new Array(rows.length);
      for (let start = 0; start < rows.length; start += batchSize) {
        signal?.throwIfAborted();
        const chunk = rows.slice(start, start + batchSize);
        const feeds = makeFeeds(ort as SafeDecideOrtWithTensor, chunk as SafeDecideInputRow[], padId);
        let output: unknown;
        try {
          output = await (session as SafeDecideOnnxSession).run(feeds, ['logits']);
          const chunkLogits = readLogits((output as {logits?: unknown} | null | undefined)?.logits, chunk.length);
          for (let index = 0; index < chunk.length; index++) logits[start + index] = chunkLogits[index]!;
        } finally {
          for (const tensor of Object.values(feeds)) tensor.dispose?.();
          for (const tensor of Object.values((output ?? {}) as Record<string, SafeDecideTensor>)) tensor.dispose?.();
        }
        signal?.throwIfAborted();
      }
      return logits;
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve().then(async () => {
        await (session as SafeDecideOnnxSession).release();
        session = null;
      });
      return disposing;
    }
  };
}
