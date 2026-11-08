import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_BATCH_SIZE} from './contract.mjs';

const fail = (message, cause) => new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function makeFeeds(ort, rows, padId) {
  const batchSize = rows.length;
  const sequenceLength = Math.max(...rows.map((row) => row.inputIds.length));
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const ids = rows[rowIndex].inputIds;
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

function readLogits(tensor, rows) {
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== rows ||
    tensor.dims[1] !== 1 ||
    !ArrayBuffer.isView(tensor.data) ||
    tensor.data.length !== rows
  )
    throw fail('SafeDecide ONNX logits have an unexpected shape.');
  const logits = Array.from(tensor.data);
  if (logits.some((value) => !Number.isFinite(value))) throw fail('SafeDecide ONNX logits are not finite.');
  return logits;
}

export function createSafeDecideRunner(ort, session, padId, batchSize = SAFEDECIDE_BATCH_SIZE) {
  if (
    typeof ort?.Tensor !== 'function' ||
    typeof session?.run !== 'function' ||
    typeof session?.release !== 'function' ||
    !Number.isSafeInteger(padId) ||
    !Number.isSafeInteger(batchSize) ||
    batchSize < 1
  )
    throw new TypeError('SafeDecide runner components are required.');

  let closed = false;
  let disposing;
  return {
    async run(rows, signal) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The SafeDecide model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1) throw fail('SafeDecide needs at least one option pair.');
      const logits = new Array(rows.length);
      for (let start = 0; start < rows.length; start += batchSize) {
        signal?.throwIfAborted();
        const chunk = rows.slice(start, start + batchSize);
        const feeds = makeFeeds(ort, chunk, padId);
        let output;
        try {
          output = await session.run(feeds, ['logits']);
          const chunkLogits = readLogits(output?.logits, chunk.length);
          for (let index = 0; index < chunk.length; index++) logits[start + index] = chunkLogits[index];
        } finally {
          for (const tensor of Object.values(feeds)) tensor.dispose?.();
          for (const tensor of Object.values(output ?? {})) tensor.dispose?.();
        }
        signal?.throwIfAborted();
      }
      return logits;
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
