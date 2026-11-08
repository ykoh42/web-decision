import {WebDecisionError} from '../kev.js';

const inferenceFailure = (cause) =>
  new WebDecisionError('INFERENCE_ERROR', 'Skejja inference failed.', cause ? {cause} : undefined);

function isLogitTensor(tensor, count) {
  return (
    tensor?.type === 'float32' &&
    Array.isArray(tensor.dims) &&
    tensor.dims.length === 2 &&
    tensor.dims[0] === 1 &&
    tensor.dims[1] === count &&
    ArrayBuffer.isView(tensor.data) &&
    tensor.data.length === count &&
    Array.from(tensor.data).every(Number.isFinite)
  );
}

export function createSkejjaRunner(ort, session) {
  if (typeof ort?.Tensor !== 'function' || typeof session?.run !== 'function' || typeof session?.release !== 'function')
    throw new TypeError('Skejja requires an ONNX Runtime Web session.');

  let closed = false;
  let disposing;

  return {
    async run(inputIds, labelCount) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const ids = BigInt64Array.from(inputIds, BigInt);
      const mask = new BigInt64Array(ids.length).fill(1n);
      const slots = new Float32Array(labelCount);
      let outputs;
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
      return Array.from(outputs.logits.data);
    },
    async releaseCachedMemory() {},
    dispose() {
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
