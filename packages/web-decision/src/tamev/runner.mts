import {WebDecisionError} from '../kev.js';

interface TamevOrt {
  readonly Tensor: new (type: string, data: BigInt64Array, shape: number[]) => unknown;
}
interface TamevOnnxSession {
  run(inputs: Record<string, unknown>): PromiseLike<{readonly logits?: unknown; readonly probs?: unknown}>;
  release(): unknown | PromiseLike<unknown>;
}
interface TamevTokenizer {
  encodeContext(text: string): unknown;
  readonly special: {readonly padId: number};
}
interface TensorCandidate {
  readonly type?: unknown;
  readonly dims?: unknown;
  readonly data?: unknown;
}
interface FloatTensor {
  readonly data: ArrayLike<number>;
}

const inferenceFailure = (message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function makeBatch(rows: readonly (readonly number[])[], padId: number) {
  let width = 0;
  for (const row of rows) {
    if (row.length > width) width = row.length;
  }
  const ids = new BigInt64Array(rows.length * width);
  const mask = new BigInt64Array(rows.length * width);
  for (let row = 0; row < rows.length; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      ids[index] = BigInt(rows[row]![column] ?? padId);
      mask[index] = BigInt(column < rows[row]!.length ? 1 : 0);
    }
  }
  return {ids, mask, width};
}

function isExpectedTensor(tensor: unknown, count: number): tensor is FloatTensor {
  return (
    (tensor as TensorCandidate | null | undefined)?.type === 'float32' &&
    Array.isArray((tensor as TensorCandidate).dims) &&
    ((tensor as TensorCandidate).dims as unknown[]).length === 2 &&
    ((tensor as TensorCandidate).dims as unknown[])[0] === 1 &&
    ((tensor as TensorCandidate).dims as unknown[])[1] === count &&
    ArrayBuffer.isView((tensor as TensorCandidate).data) &&
    ((tensor as TensorCandidate).data as {readonly length: number}).length === count &&
    Array.from((tensor as TensorCandidate).data as ArrayLike<number>).every(Number.isFinite)
  );
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

export function createTamevRunner(ort: unknown, session: unknown, tokenizer: unknown) {
  if (
    typeof (ort as TamevOrt | null | undefined)?.Tensor !== 'function' ||
    typeof (session as TamevOnnxSession | null | undefined)?.run !== 'function' ||
    typeof (session as TamevOnnxSession | null | undefined)?.release !== 'function' ||
    typeof (tokenizer as TamevTokenizer | null | undefined)?.encodeContext !== 'function' ||
    !Number.isSafeInteger((tokenizer as TamevTokenizer | null | undefined)?.special?.padId)
  )
    throw new TypeError('TAMEV Micro requires an ONNX session and tokenizer.');

  let closed = false;
  let disposing: Promise<unknown> | undefined;

  return {
    async run(contextIds: unknown, optionRows: unknown) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      if (
        !Array.isArray(contextIds) ||
        contextIds.length < 2 ||
        !Array.isArray(optionRows) ||
        optionRows.length < 2 ||
        optionRows.some((row) => !Array.isArray(row) || row.length < 2)
      )
        throw inferenceFailure('TAMEV Micro received invalid token rows.');

      const context = makeBatch([contextIds], (tokenizer as TamevTokenizer).special.padId);
      const options = makeBatch(optionRows, (tokenizer as TamevTokenizer).special.padId);
      let outputs;
      try {
        outputs = await (session as TamevOnnxSession).run({
          ctx_input_ids: new (ort as TamevOrt).Tensor('int64', context.ids, [1, context.width]),
          ctx_attention_mask: new (ort as TamevOrt).Tensor('int64', context.mask, [1, context.width]),
          opt_input_ids: new (ort as TamevOrt).Tensor('int64', options.ids, [1, optionRows.length, options.width]),
          opt_attention_mask: new (ort as TamevOrt).Tensor('int64', options.mask, [1, optionRows.length, options.width])
        });
      } catch (cause) {
        throw inferenceFailure('TAMEV Micro inference failed.', cause);
      }

      const logits = outputs?.logits;
      const probabilities = outputs?.probs;
      if (!isExpectedTensor(logits, optionRows.length) || !isExpectedTensor(probabilities, optionRows.length))
        throw inferenceFailure('TAMEV Micro returned unexpected tensors.');
      const values = Array.from(probabilities.data);
      const total = sum(values);
      if (values.some((value) => value < 0) || Math.abs(total - 1) > 1e-3)
        throw inferenceFailure('TAMEV Micro returned invalid probabilities.');
      return {logits: Array.from(logits.data), probabilities: values};
    },
    async releaseCachedMemory() {},
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve()
        .then(() => (session as TamevOnnxSession).release())
        .catch((cause) => {
          throw inferenceFailure('Could not release the TAMEV Micro ONNX session.', cause);
        });
      return disposing;
    }
  };
}
