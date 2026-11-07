import {WebDecisionError} from '../kev.js';

const inferenceFailure = (message, cause) =>
  new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function makeBatch(rows, padId) {
  let width = 0;
  for (const row of rows) {
    if (row.length > width) width = row.length;
  }
  const ids = new BigInt64Array(rows.length * width);
  const mask = new BigInt64Array(rows.length * width);
  for (let row = 0; row < rows.length; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      ids[index] = BigInt(rows[row][column] ?? padId);
      mask[index] = BigInt(column < rows[row].length ? 1 : 0);
    }
  }
  return {ids, mask, width};
}

function isExpectedTensor(tensor, count) {
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

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

export function createTamevRunner(ort, session, tokenizer) {
  if (
    typeof ort?.Tensor !== 'function' ||
    typeof session?.run !== 'function' ||
    typeof session?.release !== 'function' ||
    typeof tokenizer?.encodeContext !== 'function' ||
    !Number.isSafeInteger(tokenizer?.special?.padId)
  )
    throw new TypeError('TAMEV requires an ONNX session and tokenizer.');

  let closed = false;
  let disposing;

  return {
    async run(contextIds, optionRows) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      if (
        !Array.isArray(contextIds) ||
        contextIds.length < 2 ||
        !Array.isArray(optionRows) ||
        optionRows.length < 2 ||
        optionRows.some((row) => !Array.isArray(row) || row.length < 2)
      )
        throw inferenceFailure('TAMEV received invalid token rows.');

      const context = makeBatch([contextIds], tokenizer.special.padId);
      const options = makeBatch(optionRows, tokenizer.special.padId);
      let outputs;
      try {
        outputs = await session.run({
          ctx_input_ids: new ort.Tensor('int64', context.ids, [1, context.width]),
          ctx_attention_mask: new ort.Tensor('int64', context.mask, [1, context.width]),
          opt_input_ids: new ort.Tensor('int64', options.ids, [1, optionRows.length, options.width]),
          opt_attention_mask: new ort.Tensor('int64', options.mask, [1, optionRows.length, options.width])
        });
      } catch (cause) {
        throw inferenceFailure('TAMEV inference failed.', cause);
      }

      const logits = outputs?.logits;
      const probabilities = outputs?.probs;
      if (!isExpectedTensor(logits, optionRows.length) || !isExpectedTensor(probabilities, optionRows.length))
        throw inferenceFailure('TAMEV returned unexpected tensors.');
      const values = Array.from(probabilities.data);
      const total = sum(values);
      if (values.some((value) => value < 0) || Math.abs(total - 1) > 1e-3)
        throw inferenceFailure('TAMEV returned invalid probabilities.');
      return {logits: Array.from(logits.data), probabilities: values};
    },
    async releaseCachedMemory() {},
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve()
        .then(() => session.release())
        .catch((cause) => {
          throw inferenceFailure('Could not release the TAMEV ONNX session.', cause);
        });
      return disposing;
    }
  };
}
