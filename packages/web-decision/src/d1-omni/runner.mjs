import {WebDecisionError} from '../kev.js';
import {D1_OMNI_MAX_TOKENS} from './contract.mjs';

const INPUT_NAMES = ['input_ids', 'attention_mask', 'prefix_len', 'marker_pos', 'marker_mask', 'qtype'];
const OUTPUT_NAMES = ['logits'];
const MAX_BATCH_ROWS = 8;
const MAX_BATCH_TOKENS = 65536;

const fail = (message, cause) => new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);

function validateSession(session) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (!INPUT_NAMES.every((name) => inputs.has(name)) || !OUTPUT_NAMES.every((name) => outputs.has(name)))
    throw new WebDecisionError('INVALID_ASSET', 'd1-omni ONNX graph has an unexpected interface.');
  return session;
}

function validateRow(row) {
  if (
    !Array.isArray(row?.ids) ||
    !row.ids.length ||
    row.ids.length > D1_OMNI_MAX_TOKENS ||
    row.ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 65536) ||
    !Array.isArray(row.markerPositions) ||
    row.markerPositions.length < 2 ||
    row.markerPositions.some(
      (position) =>
        !Number.isSafeInteger(position) || position < 0 || position >= row.ids.length || row.ids[position] !== 16
    ) ||
    !Number.isSafeInteger(row.question?.qtype) ||
    row.question.qtype < 0 ||
    row.question.qtype > 2 ||
    row.markerPositions.length !== row.question.options?.length
  )
    throw new WebDecisionError('INVALID_INPUT', 'd1-omni received an invalid encoded question.');
}

function nextBatch(rows, start) {
  const batch = [];
  let longest = 0;
  while (start + batch.length < rows.length && batch.length < MAX_BATCH_ROWS) {
    const row = rows[start + batch.length];
    const nextLongest = Math.max(longest, row.ids.length);
    if (batch.length > 0 && (batch.length + 1) * nextLongest > MAX_BATCH_TOKENS) break;
    longest = nextLongest;
    batch.push(row);
  }
  return batch;
}

function makeFeeds(ort, rows, padId) {
  const batchSize = rows.length;
  const sequenceLength = Math.max(...rows.map((row) => row.ids.length));
  const optionCount = Math.max(...rows.map((row) => row.markerPositions.length));
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);
  const prefixLen = new BigInt64Array(batchSize);
  const markerPositions = new BigInt64Array(batchSize * optionCount);
  const markerMask = new BigInt64Array(batchSize * optionCount);
  const qtype = new BigInt64Array(batchSize);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const row = rows[rowIndex];
    const tokenOffset = rowIndex * sequenceLength;
    const markerOffset = rowIndex * optionCount;
    qtype[rowIndex] = BigInt(row.question.qtype);
    for (let tokenIndex = 0; tokenIndex < sequenceLength; tokenIndex++) {
      const token = row.ids[tokenIndex];
      inputIds[tokenOffset + tokenIndex] = BigInt(token ?? padId);
      attentionMask[tokenOffset + tokenIndex] = BigInt(token === undefined ? 0 : 1);
    }
    for (let optionIndex = 0; optionIndex < row.markerPositions.length; optionIndex++) {
      markerPositions[markerOffset + optionIndex] = BigInt(row.markerPositions[optionIndex]);
      markerMask[markerOffset + optionIndex] = 1n;
    }
  }

  return {
    input_ids: new ort.Tensor('int64', inputIds, [batchSize, sequenceLength]),
    attention_mask: new ort.Tensor('int64', attentionMask, [batchSize, sequenceLength]),
    prefix_len: new ort.Tensor('int64', prefixLen, [batchSize]),
    marker_pos: new ort.Tensor('int64', markerPositions, [batchSize, optionCount]),
    marker_mask: new ort.Tensor('int64', markerMask, [batchSize, optionCount]),
    qtype: new ort.Tensor('int64', qtype, [batchSize])
  };
}

function readLogits(output, rows, optionCount) {
  const tensor = output?.logits;
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== rows.length ||
    tensor.dims[1] !== optionCount ||
    !ArrayBuffer.isView(tensor.data) ||
    tensor.data.length !== rows.length * optionCount
  )
    throw fail('d1-omni returned logits with an unexpected shape.');

  const logits = Array.from(tensor.data);
  if (logits.some((value) => !Number.isFinite(value))) throw fail('d1-omni returned non-finite logits.');
  return rows.map((row, index) => {
    const offset = index * optionCount;
    return logits.slice(offset, offset + row.markerPositions.length);
  });
}

export function createD1OmniRunner(ort, session, padId = 0) {
  if (
    typeof ort?.Tensor !== 'function' ||
    typeof session?.run !== 'function' ||
    typeof session?.release !== 'function' ||
    !Number.isSafeInteger(padId) ||
    padId < 0
  )
    throw new TypeError('d1-omni requires ONNX Runtime WebGPU, a session, and a valid pad ID.');
  validateSession(session);

  let closed = false;
  let disposing;

  return {
    async run(rows, signal) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The d1-omni model has been disposed.');
      if (!Array.isArray(rows) || !rows.length)
        throw new WebDecisionError('INVALID_INPUT', 'd1-omni received an invalid question batch.');
      rows.forEach(validateRow);
      const results = new Array(rows.length);
      let start = 0;
      while (start < rows.length) {
        signal?.throwIfAborted();
        const batch = nextBatch(rows, start);
        const feeds = makeFeeds(ort, batch, padId);
        let output;
        try {
          output = await session.run(feeds, OUTPUT_NAMES);
          signal?.throwIfAborted();
          const optionCount = Math.max(...batch.map((row) => row.markerPositions.length));
          const logits = readLogits(output, batch, optionCount);
          for (let index = 0; index < batch.length; index++) results[start + index] = logits[index];
        } catch (cause) {
          if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
          throw fail('d1-omni ONNX inference failed.', cause);
        } finally {
          for (const tensor of Object.values(feeds)) tensor.dispose?.();
          for (const tensor of Object.values(output ?? {})) tensor.dispose?.();
        }
        start += batch.length;
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

export const D1_OMNI_INPUT_NAMES = Object.freeze([...INPUT_NAMES]);
export const D1_OMNI_OUTPUT_NAMES = Object.freeze([...OUTPUT_NAMES]);
