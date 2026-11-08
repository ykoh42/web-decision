import {WebDecisionError} from '../kev.js';
import {CRIVO_BATCH_SIZE, CRIVO_BATCH_TOKEN_BUDGET, CRIVO_PAD_MULTIPLE} from './contract.mjs';

const fail = (message, cause) => new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function makeFeeds(ort, rows) {
  const batchSize = rows.length;
  const sequenceLength =
    Math.ceil(Math.max(...rows.map((row) => row.inputIds.length)) / CRIVO_PAD_MULTIPLE) * CRIVO_PAD_MULTIPLE;
  const optionCount = Math.max(...rows.map((row) => row.optionPositions.length));
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);
  const optionPositions = new BigInt64Array(batchSize * optionCount);
  const optionMask = new Uint8Array(batchSize * optionCount);
  const qtype = new BigInt64Array(batchSize);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const row = rows[rowIndex];
    const offset = rowIndex * sequenceLength;
    for (let tokenIndex = 0; tokenIndex < row.inputIds.length; tokenIndex++) {
      inputIds[offset + tokenIndex] = BigInt(row.inputIds[tokenIndex]);
      attentionMask[offset + tokenIndex] = 1n;
    }
    for (let optionIndex = 0; optionIndex < row.optionPositions.length; optionIndex++) {
      optionPositions[rowIndex * optionCount + optionIndex] = BigInt(row.optionPositions[optionIndex]);
      optionMask[rowIndex * optionCount + optionIndex] = 1;
    }
    qtype[rowIndex] = BigInt(row.qtype);
  }

  return {
    optionCount,
    values: {
      input_ids: new ort.Tensor('int64', inputIds, [batchSize, sequenceLength]),
      attention_mask: new ort.Tensor('int64', attentionMask, [batchSize, sequenceLength]),
      opt_positions: new ort.Tensor('int64', optionPositions, [batchSize, optionCount]),
      opt_mask: new ort.Tensor('bool', optionMask, [batchSize, optionCount]),
      qtype: new ort.Tensor('int64', qtype, [batchSize])
    }
  };
}

function readLogits(output, batchSize, optionCount) {
  const tensor = output?.logits;
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== batchSize ||
    tensor.dims[1] !== optionCount ||
    !ArrayBuffer.isView(tensor.data) ||
    tensor.data.length !== batchSize * optionCount
  )
    throw fail('Crivo ONNX logits have an unexpected shape.');
  const values = Array.from(tensor.data);
  if (values.some((value) => !Number.isFinite(value))) throw fail('Crivo ONNX logits are not finite.');
  return Array.from({length: batchSize}, (_, index) =>
    Float32Array.from(values.slice(index * optionCount, (index + 1) * optionCount))
  );
}

function makeBatches(rows) {
  const order = rows
    .map((_, index) => index)
    .sort((left, right) => rows[left].inputIds.length - rows[right].inputIds.length);
  const batches = [];
  let batch = [];
  for (const index of order) {
    const nextLength = rows[index].inputIds.length;
    if (
      batch.length > 0 &&
      (batch.length >= CRIVO_BATCH_SIZE || nextLength * (batch.length + 1) > CRIVO_BATCH_TOKEN_BUDGET)
    ) {
      batches.push(batch);
      batch = [];
    }
    batch.push(index);
  }
  if (batch.length > 0) batches.push(batch);
  return batches;
}

export function createCrivoRunner(ort, session) {
  if (typeof ort?.Tensor !== 'function' || typeof session?.run !== 'function' || typeof session?.release !== 'function')
    throw new TypeError('Crivo runner components are required.');

  let closed = false;
  let disposing;
  return {
    async run(rows, signal) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Crivo model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1) throw fail('Crivo needs at least one question.');
      const results = new Array(rows.length);
      for (const batch of makeBatches(rows)) {
        signal?.throwIfAborted();
        const batchRows = batch.map((index) => rows[index]);
        const feeds = makeFeeds(ort, batchRows);
        let output;
        try {
          output = await session.run(feeds.values, ['logits']);
          const logits = readLogits(output, batchRows.length, feeds.optionCount);
          for (let index = 0; index < batch.length; index++) {
            const optionCount = batchRows[index].optionPositions.length;
            results[batch[index]] = logits[index].slice(0, optionCount);
          }
        } catch (cause) {
          if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
          throw fail('Crivo ONNX inference failed.', cause);
        } finally {
          for (const tensor of Object.values(feeds.values)) tensor.dispose?.();
          for (const tensor of Object.values(output ?? {})) tensor.dispose?.();
        }
        signal?.throwIfAborted();
      }
      return results;
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
