import {WebDecisionError} from '../kev.js';
import {VERDICT_MAX_LABELS, VERDICT_MAX_TOKENS} from './contract.mjs';

interface VerdictEncodedRow {
  readonly ids: number[];
  readonly candidateIds: unknown[];
}

interface VerdictTensor {
  dispose?(): void;
}

interface VerdictTensorConstructor {
  new (type: 'int64', data: BigInt64Array, dims: number[]): VerdictTensor;
}

interface VerdictOrtRuntime {
  readonly Tensor?: unknown;
}

interface VerdictOrtWithTensor extends VerdictOrtRuntime {
  readonly Tensor: VerdictTensorConstructor;
}

interface VerdictOnnxSession {
  readonly inputNames?: Iterable<string>;
  readonly outputNames?: Iterable<string>;
  run(feeds: Record<string, VerdictTensor>, outputNames: readonly string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface VerdictTensorCandidate {
  readonly type?: unknown;
  readonly dims?: unknown;
  readonly data?: unknown;
}

interface VerdictSignal {
  throwIfAborted(): void;
}

interface VerdictRunner {
  run(rows: unknown[], signal?: VerdictSignal | null): Promise<number[][]>;
  dispose(): Promise<void>;
}

export const VERDICT_INPUT_NAMES = Object.freeze(['input_ids', 'attention_mask']);
export const VERDICT_OUTPUT_NAMES = Object.freeze(['logits']);

const MAX_BATCH_ROWS = 8;
const MAX_BATCH_TOKENS = 4096;
const fail = (
  code: ConstructorParameters<typeof WebDecisionError>[0],
  message: string,
  cause?: unknown
): WebDecisionError => new WebDecisionError(code, message, cause ? {cause} : undefined);

function validateSession(session: unknown): VerdictOnnxSession {
  const inputs = new Set((session as {inputNames?: Iterable<string>} | null | undefined)?.inputNames ?? []);
  const outputs = new Set((session as {outputNames?: Iterable<string>} | null | undefined)?.outputNames ?? []);
  if (
    !VERDICT_INPUT_NAMES.every((name) => inputs.has(name)) ||
    !VERDICT_OUTPUT_NAMES.every((name) => outputs.has(name))
  )
    throw fail('INVALID_ASSET', 'Verdict ONNX graph has an unexpected interface.');
  return session as VerdictOnnxSession;
}

function validateRow(row: unknown): asserts row is VerdictEncodedRow {
  if (
    !Array.isArray((row as {ids?: unknown} | null | undefined)?.ids) ||
    (row as {ids: unknown[]}).ids.length < 2 ||
    (row as {ids: unknown[]}).ids.length > VERDICT_MAX_TOKENS ||
    (row as {ids: unknown[]}).ids.some(
      (id: unknown) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= 50370
    ) ||
    !Array.isArray((row as {candidateIds?: unknown}).candidateIds) ||
    (row as {candidateIds: unknown[]}).candidateIds.length < 3 ||
    (row as {candidateIds: unknown[]}).candidateIds.length > VERDICT_MAX_LABELS
  )
    throw fail('INVALID_INPUT', 'Verdict received an invalid encoded question.');
}

function nextBatch(rows: VerdictEncodedRow[], start: number): VerdictEncodedRow[] {
  const batch: VerdictEncodedRow[] = [];
  let longest = 0;
  while (start + batch.length < rows.length && batch.length < MAX_BATCH_ROWS) {
    const row = rows[start + batch.length]!;
    const nextLongest = Math.max(longest, row.ids.length);
    if (batch.length && (batch.length + 1) * nextLongest > MAX_BATCH_TOKENS) break;
    longest = nextLongest;
    batch.push(row);
  }
  return batch;
}

function makeFeeds(ort: VerdictOrtWithTensor, rows: VerdictEncodedRow[], padId: number): Record<string, VerdictTensor> {
  const batchSize = rows.length;
  const sequenceLength = Math.max(...rows.map((row) => row.ids.length));
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const row = rows[rowIndex]!;
    const offset = rowIndex * sequenceLength;
    for (let tokenIndex = 0; tokenIndex < sequenceLength; tokenIndex++) {
      const token = row.ids[tokenIndex];
      inputIds[offset + tokenIndex] = BigInt(token ?? padId);
      attentionMask[offset + tokenIndex] = token === undefined ? 0n : 1n;
    }
  }

  return {
    input_ids: new ort.Tensor('int64', inputIds, [batchSize, sequenceLength]),
    attention_mask: new ort.Tensor('int64', attentionMask, [batchSize, sequenceLength])
  };
}

function readLogits(output: unknown, rows: VerdictEncodedRow[]): number[][] {
  const tensor = (output as {logits?: VerdictTensorCandidate | null} | null | undefined)?.logits;
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== rows.length ||
    tensor.dims[1] !== VERDICT_MAX_LABELS ||
    !ArrayBuffer.isView(tensor.data) ||
    (tensor.data as unknown as {length: number}).length !== rows.length * VERDICT_MAX_LABELS
  )
    throw fail('INFERENCE_ERROR', 'Verdict returned logits with an unexpected shape.');

  const logits = Array.from(tensor.data as unknown as ArrayLike<unknown>);
  if (logits.some((value) => !Number.isFinite(value)))
    throw fail('INFERENCE_ERROR', 'Verdict returned non-finite logits.');
  return rows.map((row, index) => {
    const offset = index * VERDICT_MAX_LABELS;
    return (logits as number[]).slice(offset, offset + row.candidateIds.length);
  });
}

export function createVerdictRunner(ort: unknown, onnxSession: unknown, padId: number = 50283): VerdictRunner {
  if (
    typeof (ort as VerdictOrtRuntime | null | undefined)?.Tensor !== 'function' ||
    typeof (onnxSession as {run?: unknown} | null | undefined)?.run !== 'function' ||
    typeof (onnxSession as {release?: unknown} | null | undefined)?.release !== 'function' ||
    !Number.isSafeInteger(padId) ||
    padId < 0 ||
    padId >= 50370
  )
    throw new TypeError('Verdict requires ONNX Runtime WebGPU, a session, and a valid pad ID.');
  let session: VerdictOnnxSession | null = validateSession(onnxSession);
  let closed = false;
  let disposing: Promise<void> | undefined;

  return {
    async run(rows: unknown[], signal?: VerdictSignal | null): Promise<number[][]> {
      if (closed) throw fail('SESSION_CLOSED', 'The Verdict model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1)
        throw fail('INVALID_INPUT', 'Verdict received an invalid question batch.');
      rows.forEach(validateRow);
      // Retain sparse slots until each batch writes its corresponding logits.
      // oxlint-disable-next-line unicorn/no-new-array -- The result requires holes before batch writes complete.
      const results: number[][] = new Array(rows.length);
      let start = 0;
      while (start < rows.length) {
        signal?.throwIfAborted();
        const batch = nextBatch(rows as VerdictEncodedRow[], start);
        const feeds = makeFeeds(ort as VerdictOrtWithTensor, batch, padId);
        let output: unknown;
        try {
          output = await (session as VerdictOnnxSession).run(feeds, VERDICT_OUTPUT_NAMES);
          signal?.throwIfAborted();
          const logits = readLogits(output, batch);
          for (let index = 0; index < batch.length; index++) results[start + index] = logits[index]!;
        } catch (cause) {
          if (
            cause instanceof WebDecisionError ||
            (cause as {name?: unknown} | null | undefined)?.name === 'AbortError'
          )
            throw cause;
          throw fail('INFERENCE_ERROR', 'Verdict ONNX inference failed.', cause);
        } finally {
          for (const tensor of Object.values(feeds)) tensor.dispose?.();
          for (const tensor of Object.values((output ?? {}) as Record<string, VerdictTensor>)) tensor.dispose?.();
        }
        start += batch.length;
      }
      return results;
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve().then(async () => {
        const current = session;
        session = null;
        await current?.release();
      });
      return disposing;
    }
  };
}
