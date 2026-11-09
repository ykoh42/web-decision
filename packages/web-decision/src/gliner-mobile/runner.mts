import {WebDecisionError} from '../kev.js';

interface GlinerTensor {
  dispose?(): void;
}

interface GlinerTensorConstructor {
  new (type: 'int64', data: BigInt64Array, dims: number[]): GlinerTensor;
}

interface GlinerOrtRuntime {
  readonly Tensor?: unknown;
}

interface GlinerOrtWithTensor extends GlinerOrtRuntime {
  readonly Tensor: GlinerTensorConstructor;
}

interface GlinerOnnxSession {
  run(feeds: Record<string, GlinerTensor>, outputNames: string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface GlinerTensorOutput {
  readonly type?: unknown;
  readonly dims?: unknown;
  readonly data?: unknown;
}

interface GlinerSignal {
  throwIfAborted(): void;
}

const fail = (message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function readLogits(output: unknown, markerCount: number): number[] {
  if (
    (output as GlinerTensorOutput | null | undefined)?.type !== 'float32' ||
    !Array.isArray((output as GlinerTensorOutput).dims) ||
    (output as {dims: number[]}).dims.length !== 2 ||
    (output as {dims: number[]}).dims[0] !== 1 ||
    (output as {dims: number[]}).dims[1] !== markerCount ||
    !ArrayBuffer.isView((output as GlinerTensorOutput).data) ||
    (output as {data: {length: number}}).data.length !== markerCount
  )
    throw fail('GLiNER logits have an unexpected shape.');
  const logits = Array.from((output as {data: ArrayLike<unknown>}).data);
  if (logits.some((value) => !Number.isFinite(value))) throw fail('GLiNER logits contain non-finite values.');
  return logits as number[];
}

export function createGlinerMobileRunner(
  ort: unknown,
  session: GlinerOnnxSession | null
): {
  run(inputIds: number[], markerPositions: number[], signal?: GlinerSignal | null): Promise<number[]>;
  dispose(): Promise<void>;
} {
  if (
    typeof (ort as GlinerOrtRuntime | null | undefined)?.Tensor !== 'function' ||
    typeof session?.run !== 'function' ||
    typeof session?.release !== 'function'
  )
    throw new TypeError('GLiNER requires ONNX Runtime WebGPU and an initialized session.');

  let closed = false;
  let disposing: Promise<void> | undefined;

  return {
    async run(inputIds: number[], markerPositions: number[], signal?: GlinerSignal | null): Promise<number[]> {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The GLiNER model has been disposed.');
      signal?.throwIfAborted();
      const length = inputIds.length;
      const ids = BigInt64Array.from(inputIds, (id) => BigInt(id));
      const attentionMask = new BigInt64Array(length).fill(1n);
      const positions = BigInt64Array.from(markerPositions, (position) => BigInt(position));
      const feeds = {
        input_ids: new (ort as GlinerOrtWithTensor).Tensor('int64', ids, [1, length]),
        attention_mask: new (ort as GlinerOrtWithTensor).Tensor('int64', attentionMask, [1, length]),
        marker_positions: new (ort as GlinerOrtWithTensor).Tensor('int64', positions, [1, markerPositions.length])
      };
      let output: unknown;
      try {
        output = await (session as GlinerOnnxSession).run(feeds, ['logits']);
        signal?.throwIfAborted();
        return readLogits((output as {logits?: unknown} | null | undefined)?.logits, markerPositions.length);
      } catch (cause) {
        if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
          throw cause;
        throw fail('GLiNER ONNX inference failed.', cause);
      } finally {
        for (const tensor of Object.values(feeds)) tensor.dispose?.();
        for (const tensor of Object.values((output ?? {}) as Record<string, GlinerTensor>)) tensor.dispose?.();
      }
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve().then(async () => {
        await (session as GlinerOnnxSession).release();
        session = null;
      });
      return disposing;
    }
  };
}
