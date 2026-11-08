import {WebDecisionError} from '../kev.js';

const fail = (message, cause) => new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function readLogits(output, markerCount) {
  if (
    output?.type !== 'float32' ||
    !Array.isArray(output.dims) ||
    output.dims.length !== 2 ||
    output.dims[0] !== 1 ||
    output.dims[1] !== markerCount ||
    !ArrayBuffer.isView(output.data) ||
    output.data.length !== markerCount
  )
    throw fail('GLiNER logits have an unexpected shape.');
  const logits = Array.from(output.data);
  if (logits.some((value) => !Number.isFinite(value))) throw fail('GLiNER logits contain non-finite values.');
  return logits;
}

export function createGlinerMobileRunner(ort, session) {
  if (typeof ort?.Tensor !== 'function' || typeof session?.run !== 'function' || typeof session?.release !== 'function')
    throw new TypeError('GLiNER requires ONNX Runtime WebGPU and an initialized session.');

  let closed = false;
  let disposing;

  return {
    async run(inputIds, markerPositions, signal) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The GLiNER model has been disposed.');
      signal?.throwIfAborted();
      const length = inputIds.length;
      const ids = BigInt64Array.from(inputIds, (id) => BigInt(id));
      const attentionMask = new BigInt64Array(length).fill(1n);
      const positions = BigInt64Array.from(markerPositions, (position) => BigInt(position));
      const feeds = {
        input_ids: new ort.Tensor('int64', ids, [1, length]),
        attention_mask: new ort.Tensor('int64', attentionMask, [1, length]),
        marker_positions: new ort.Tensor('int64', positions, [1, markerPositions.length])
      };
      let output;
      try {
        output = await session.run(feeds, ['logits']);
        signal?.throwIfAborted();
        return readLogits(output?.logits, markerPositions.length);
      } catch (cause) {
        if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
        throw fail('GLiNER ONNX inference failed.', cause);
      } finally {
        for (const tensor of Object.values(feeds)) tensor.dispose?.();
        for (const tensor of Object.values(output ?? {})) tensor.dispose?.();
      }
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve().then(async () => {
        await session.release();
        session = null;
      });
      return disposing;
    }
  };
}
