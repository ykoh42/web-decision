import {WebDecisionError} from '../kev.js';
import {VERDICT_MAX_LABELS, VERDICT_MAX_TOKENS} from './contract.mjs';

export const VERDICT_INPUT_NAMES = Object.freeze(['input_ids', 'attention_mask']);
export const VERDICT_OUTPUT_NAMES = Object.freeze(['logits']);

const MAX_BATCH_ROWS = 8;
const MAX_BATCH_TOKENS = 4096;
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

function validateSession(session) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (
    !VERDICT_INPUT_NAMES.every((name) => inputs.has(name)) ||
    !VERDICT_OUTPUT_NAMES.every((name) => outputs.has(name))
  )
    throw fail('INVALID_ASSET', 'Verdict ONNX graph has an unexpected interface.');
  return session;
}

function validateRow(row) {
  if (
    !Array.isArray(row?.ids) ||
    row.ids.length < 2 ||
    row.ids.length > VERDICT_MAX_TOKENS ||
    row.ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 50370) ||
    !Array.isArray(row.candidateIds) ||
    row.candidateIds.length < 3 ||
    row.candidateIds.length > VERDICT_MAX_LABELS
  )
    throw fail('INVALID_INPUT', 'Verdict received an invalid encoded question.');
}

function nextBatch(rows, start) {
  const batch = [];
  let longest = 0;
  while (start + batch.length < rows.length && batch.length < MAX_BATCH_ROWS) {
    const row = rows[start + batch.length];
    const nextLongest = Math.max(longest, row.ids.length);
    if (batch.length && (batch.length + 1) * nextLongest > MAX_BATCH_TOKENS) break;
    longest = nextLongest;
    batch.push(row);
  }
  return batch;
}

function makeFeeds(ort, rows, padId) {
  const batchSize = rows.length;
  const sequenceLength = Math.max(...rows.map((row) => row.ids.length));
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const row = rows[rowIndex];
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

function readLogits(output, rows) {
  const tensor = output?.logits;
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== rows.length ||
    tensor.dims[1] !== VERDICT_MAX_LABELS ||
    !ArrayBuffer.isView(tensor.data) ||
    tensor.data.length !== rows.length * VERDICT_MAX_LABELS
  )
    throw fail('INFERENCE_ERROR', 'Verdict returned logits with an unexpected shape.');

  const logits = Array.from(tensor.data);
  if (logits.some((value) => !Number.isFinite(value)))
    throw fail('INFERENCE_ERROR', 'Verdict returned non-finite logits.');
  return rows.map((row, index) => {
    const offset = index * VERDICT_MAX_LABELS;
    return logits.slice(offset, offset + row.candidateIds.length);
  });
}

export function createVerdictRunner(ort, onnxSession, padId = 50283) {
  if (
    typeof ort?.Tensor !== 'function' ||
    typeof onnxSession?.run !== 'function' ||
    typeof onnxSession?.release !== 'function' ||
    !Number.isSafeInteger(padId) ||
    padId < 0 ||
    padId >= 50370
  )
    throw new TypeError('Verdict requires ONNX Runtime WebGPU, a session, and a valid pad ID.');
  let session = validateSession(onnxSession);
  let closed = false;
  let disposing;

  return {
    async run(rows, signal) {
      if (closed) throw fail('SESSION_CLOSED', 'The Verdict model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1)
        throw fail('INVALID_INPUT', 'Verdict received an invalid question batch.');
      rows.forEach(validateRow);
      const results = new Array(rows.length);
      let start = 0;
      while (start < rows.length) {
        signal?.throwIfAborted();
        const batch = nextBatch(rows, start);
        const feeds = makeFeeds(ort, batch, padId);
        let output;
        try {
          output = await session.run(feeds, VERDICT_OUTPUT_NAMES);
          signal?.throwIfAborted();
          const logits = readLogits(output, batch);
          for (let index = 0; index < batch.length; index++) results[start + index] = logits[index];
        } catch (cause) {
          if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
          throw fail('INFERENCE_ERROR', 'Verdict ONNX inference failed.', cause);
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
        const current = session;
        session = null;
        await current?.release();
      });
      return disposing;
    }
  };
}
