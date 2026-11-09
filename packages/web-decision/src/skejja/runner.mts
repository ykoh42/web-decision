import {WebDecisionError} from '../kev.js';
import type {InferenceSession} from 'onnxruntime-web';

type SkejjaRuntime = {readonly Tensor: typeof import('onnxruntime-web').Tensor};
type SkejjaSession = Pick<InferenceSession, 'run' | 'release'>;

interface LogitTensor {
  readonly type: 'float32';
  readonly dims: unknown[];
  readonly data: ArrayBufferView & ArrayLike<number>;
}

export interface SkejjaRunner {
  run(inputIds: Iterable<number | bigint>, labelCount: number): Promise<number[]>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

const inferenceFailure = (cause: unknown): WebDecisionError =>
  new WebDecisionError('INFERENCE_ERROR', 'Skejja inference failed.', cause ? {cause} : undefined);

function isLogitTensor(tensor: unknown, count: number): tensor is LogitTensor {
  return (
    (tensor as {readonly type?: unknown} | null | undefined)?.type === 'float32' &&
    Array.isArray((tensor as {readonly dims?: unknown}).dims) &&
    (tensor as {readonly dims: unknown[]}).dims.length === 2 &&
    (tensor as {readonly dims: unknown[]}).dims[0] === 1 &&
    (tensor as {readonly dims: unknown[]}).dims[1] === count &&
    ArrayBuffer.isView((tensor as {readonly data: unknown}).data) &&
    (tensor as {readonly data: ArrayBufferView & ArrayLike<number>}).data.length === count &&
    Array.from((tensor as {readonly data: ArrayBufferView & ArrayLike<number>}).data).every(Number.isFinite)
  );
}

export function createSkejjaRunner(
  ort: SkejjaRuntime | null | undefined,
  session: SkejjaSession | null | undefined
): SkejjaRunner {
  if (typeof ort?.Tensor !== 'function' || typeof session?.run !== 'function' || typeof session?.release !== 'function')
    throw new TypeError('Skejja requires an ONNX Runtime Web session.');

  let closed = false;
  let disposing: Promise<void> | undefined;

  return {
    async run(inputIds, labelCount): Promise<number[]> {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const ids = BigInt64Array.from(inputIds, BigInt);
      const mask = new BigInt64Array(ids.length).fill(1n);
      const slots = new Float32Array(labelCount);
      let outputs: Awaited<ReturnType<SkejjaSession['run']>>;
      try {
        outputs = await session.run({
          input_ids: new ort.Tensor('int64', ids, [1, ids.length]),
          attention_mask: new ort.Tensor('int64', mask, [1, ids.length]),
          label_slots: new ort.Tensor('float32', slots, [labelCount])
        });
      } catch (cause) {
        throw inferenceFailure(cause);
      }
      if (!isLogitTensor(outputs?.logits, labelCount))
        throw inferenceFailure(new TypeError('Unexpected logits tensor.'));
      return Array.from((outputs.logits as unknown as LogitTensor).data);
    },
    async releaseCachedMemory() {},
    dispose(): Promise<void> {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve()
        .then(() => session.release())
        .catch((cause) => {
          throw new WebDecisionError('INFERENCE_ERROR', 'Could not release the Skejja ONNX session.', {cause});
        });
      return disposing;
    }
  };
}
