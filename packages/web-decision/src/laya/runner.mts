import {WebDecisionError} from '../kev.js';
import type {PreparedLayaRow} from './protocol.mjs';
import type {LayaTokenizer} from './tokenizer.mjs';

interface LayaTensor {
  dispose?(): void;
}

interface LayaTensorConstructor {
  new (type: 'int64', data: BigInt64Array, dims: number[]): LayaTensor;
  new (type: 'bool', data: Uint8Array, dims: number[]): LayaTensor;
}

interface LayaOrtRuntime {
  readonly Tensor?: unknown;
}

interface LayaOrtWithTensor extends LayaOrtRuntime {
  readonly Tensor: LayaTensorConstructor;
}

interface LayaOnnxSession {
  run(feeds: Record<string, LayaTensor>, outputNames: string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface LayaTensorOutput {
  readonly type?: unknown;
  readonly dims?: unknown;
  readonly data?: unknown;
}

interface LayaOutputRow {
  readonly logits: number[];
  readonly actLogits: number[];
}

interface LayaRunnerOutput {
  readonly rows: LayaOutputRow[];
}

const fail = (message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);
const MAX_BATCH_ROWS = 8;
const MAX_BATCH_TOKENS = 1024;

function nextBatch(rows: PreparedLayaRow[], start: number): PreparedLayaRow[] {
  let end = start;
  let maxSequenceLength = 0;
  while (end < rows.length && end - start < MAX_BATCH_ROWS) {
    const nextLength = Math.max(maxSequenceLength, rows[end]!.inputIds.length);
    const nextRows = end - start + 1;
    if (nextRows > 1 && nextLength * nextRows > MAX_BATCH_TOKENS) break;
    maxSequenceLength = nextLength;
    end++;
  }
  return rows.slice(start, end);
}

function floatTensor(tensor: unknown, rows: number, width: number, label: string): number[] {
  if (
    (tensor as LayaTensorOutput | null | undefined)?.type !== 'float32' ||
    !Array.isArray((tensor as LayaTensorOutput).dims) ||
    (tensor as {dims: number[]}).dims.length !== 2 ||
    (tensor as {dims: number[]}).dims[0] !== rows ||
    (tensor as {dims: number[]}).dims[1] !== width ||
    !ArrayBuffer.isView((tensor as LayaTensorOutput).data) ||
    (tensor as {data: {length: number}}).data.length !== rows * width
  )
    throw fail(`Laya ${label} output has an unexpected shape.`);
  const values = Array.from((tensor as {data: ArrayLike<number>}).data);
  if (values.some((value) => !Number.isFinite(value))) throw fail(`Laya ${label} output contains non-finite values.`);
  return values;
}

function makeFeeds(
  ort: LayaOrtWithTensor,
  rows: PreparedLayaRow[],
  tokenizer: LayaTokenizer
): Record<string, LayaTensor> {
  const sequenceLength = Math.max(...rows.map((row) => row.inputIds.length));
  const markerCount = Math.max(...rows.map((row) => row.markerPositions.length));
  const batchSize = rows.length;
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);
  const markerPositions = new BigInt64Array(batchSize * markerCount);
  const markerMask = new Uint8Array(batchSize * markerCount);
  const qtype = new BigInt64Array(batchSize);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const row = rows[rowIndex]!;
    const tokenOffset = rowIndex * sequenceLength;
    const markerOffset = rowIndex * markerCount;
    qtype[rowIndex] = BigInt(row.qtype);
    for (let tokenIndex = 0; tokenIndex < sequenceLength; tokenIndex++) {
      inputIds[tokenOffset + tokenIndex] = BigInt(row.inputIds[tokenIndex] ?? tokenizer.special.padId);
      attentionMask[tokenOffset + tokenIndex] = BigInt(tokenIndex < row.inputIds.length ? 1 : 0);
    }
    for (let markerIndex = 0; markerIndex < row.markerPositions.length; markerIndex++) {
      markerPositions[markerOffset + markerIndex] = BigInt(row.markerPositions[markerIndex]!);
      markerMask[markerOffset + markerIndex] = 1;
    }
  }

  return {
    input_ids: new ort.Tensor('int64', inputIds, [batchSize, sequenceLength]),
    attention_mask: new ort.Tensor('int64', attentionMask, [batchSize, sequenceLength]),
    marker_pos: new ort.Tensor('int64', markerPositions, [batchSize, markerCount]),
    marker_mask: new ort.Tensor('bool', markerMask, [batchSize, markerCount]),
    qtype: new ort.Tensor('int64', qtype, [batchSize])
  };
}

export function createLayaRunner(
  ort: unknown,
  session: LayaOnnxSession | null,
  tokenizer: LayaTokenizer | null
): {
  run(rows: PreparedLayaRow[], signal?: AbortSignal | null): Promise<LayaRunnerOutput>;
  dispose(): Promise<void>;
} {
  if (
    typeof (ort as LayaOrtRuntime | null | undefined)?.Tensor !== 'function' ||
    typeof session?.run !== 'function' ||
    typeof session?.release !== 'function' ||
    !Number.isSafeInteger((tokenizer as LayaTokenizer | null)?.special?.padId)
  )
    throw new TypeError('Laya requires ONNX Runtime WebGPU, its session, and its tokenizer.');

  let closed = false;
  let disposing: Promise<void> | undefined;

  return {
    async run(rows: PreparedLayaRow[], signal?: AbortSignal | null): Promise<LayaRunnerOutput> {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Laya model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1) throw fail('Laya received an invalid question batch.');
      const outputRows: LayaOutputRow[] = [];
      for (let start = 0; start < rows.length;) {
        signal?.throwIfAborted();
        const chunk = nextBatch(rows, start);
        start += chunk.length;
        const feeds = makeFeeds(ort as LayaOrtWithTensor, chunk, tokenizer as LayaTokenizer);
        let output: unknown;
        try {
          output = await (session as LayaOnnxSession).run(feeds, ['logits', 'act_logits']);
          const width = Math.max(...chunk.map((row) => row.options.length));
          const logits = floatTensor(
            (output as {logits?: unknown} | null | undefined)?.logits,
            chunk.length,
            width,
            'choice logits'
          );
          const actionLogits = floatTensor(
            (output as {act_logits?: unknown} | null | undefined)?.act_logits,
            chunk.length,
            2,
            'action logits'
          );
          for (let rowIndex = 0; rowIndex < chunk.length; rowIndex++) {
            const count = chunk[rowIndex]!.options.length;
            outputRows.push({
              logits: logits.slice(rowIndex * width, rowIndex * width + count),
              actLogits: actionLogits.slice(rowIndex * 2, rowIndex * 2 + 2)
            });
          }
        } finally {
          for (const tensor of Object.values(feeds)) tensor.dispose?.();
          for (const tensor of Object.values((output ?? {}) as Record<string, LayaTensor>)) tensor.dispose?.();
        }
        signal?.throwIfAborted();
      }
      return {rows: outputRows};
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve().then(async () => {
        await (session as LayaOnnxSession).release();
        session = null;
        tokenizer = null;
      });
      return disposing;
    }
  };
}
