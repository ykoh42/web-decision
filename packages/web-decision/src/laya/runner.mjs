import {WebDecisionError} from '../kev.js';

const fail = (message, cause) => new WebDecisionError('INFERENCE_ERROR', message, cause ? {cause} : undefined);
const MAX_BATCH_ROWS = 8;
const MAX_BATCH_TOKENS = 1024;

function nextBatch(rows, start) {
  let end = start;
  let maxSequenceLength = 0;
  while (end < rows.length && end - start < MAX_BATCH_ROWS) {
    const nextLength = Math.max(maxSequenceLength, rows[end].inputIds.length);
    const nextRows = end - start + 1;
    if (nextRows > 1 && nextLength * nextRows > MAX_BATCH_TOKENS) break;
    maxSequenceLength = nextLength;
    end++;
  }
  return rows.slice(start, end);
}

function floatTensor(tensor, rows, width, label) {
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== rows ||
    tensor.dims[1] !== width ||
    !ArrayBuffer.isView(tensor.data) ||
    tensor.data.length !== rows * width
  )
    throw fail(`Laya ${label} output has an unexpected shape.`);
  const values = Array.from(tensor.data);
  if (values.some((value) => !Number.isFinite(value))) throw fail(`Laya ${label} output contains non-finite values.`);
  return values;
}

function makeFeeds(ort, rows, tokenizer) {
  const sequenceLength = Math.max(...rows.map((row) => row.inputIds.length));
  const markerCount = Math.max(...rows.map((row) => row.markerPositions.length));
  const batchSize = rows.length;
  const inputIds = new BigInt64Array(batchSize * sequenceLength);
  const attentionMask = new BigInt64Array(batchSize * sequenceLength);
  const markerPositions = new BigInt64Array(batchSize * markerCount);
  const markerMask = new Uint8Array(batchSize * markerCount);
  const qtype = new BigInt64Array(batchSize);

  for (let rowIndex = 0; rowIndex < batchSize; rowIndex++) {
    const row = rows[rowIndex];
    const tokenOffset = rowIndex * sequenceLength;
    const markerOffset = rowIndex * markerCount;
    qtype[rowIndex] = BigInt(row.qtype);
    for (let tokenIndex = 0; tokenIndex < sequenceLength; tokenIndex++) {
      inputIds[tokenOffset + tokenIndex] = BigInt(row.inputIds[tokenIndex] ?? tokenizer.special.padId);
      attentionMask[tokenOffset + tokenIndex] = BigInt(tokenIndex < row.inputIds.length ? 1 : 0);
    }
    for (let markerIndex = 0; markerIndex < row.markerPositions.length; markerIndex++) {
      markerPositions[markerOffset + markerIndex] = BigInt(row.markerPositions[markerIndex]);
      markerMask[markerOffset + markerIndex] = 1;
    }
  }

  return {
    input_ids: new ort.Tensor('int64', inputIds, [batchSize, sequenceLength]),
    attention_mask: new ort.Tensor('int64', attentionMask, [batchSize, sequenceLength]),
    marker_pos: new ort.Tensor('int64', markerPositions, [batchSize, markerCount]),
    marker_mask: new ort.Tensor('bool', markerMask, [batchSize, markerCount]),
    qtype: new ort.Tensor('int64', qtype, [batchSize])
  };
}

export function createLayaRunner(ort, session, tokenizer) {
  if (
    typeof ort?.Tensor !== 'function' ||
    typeof session?.run !== 'function' ||
    typeof session?.release !== 'function' ||
    !Number.isSafeInteger(tokenizer?.special?.padId)
  )
    throw new TypeError('Laya requires ONNX Runtime WebGPU, its session, and its tokenizer.');

  let closed = false;
  let disposing;

  return {
    async run(rows, signal) {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Laya model has been disposed.');
      if (!Array.isArray(rows) || rows.length < 1) throw fail('Laya received an invalid question batch.');
      const outputRows = [];
      for (let start = 0; start < rows.length;) {
        signal?.throwIfAborted();
        const chunk = nextBatch(rows, start);
        start += chunk.length;
        const feeds = makeFeeds(ort, chunk, tokenizer);
        let output;
        try {
          output = await session.run(feeds, ['logits', 'act_logits']);
          const width = Math.max(...chunk.map((row) => row.options.length));
          const logits = floatTensor(output?.logits, chunk.length, width, 'choice logits');
          const actionLogits = floatTensor(output?.act_logits, chunk.length, 2, 'action logits');
          for (let rowIndex = 0; rowIndex < chunk.length; rowIndex++) {
            const count = chunk[rowIndex].options.length;
            outputRows.push({
              logits: logits.slice(rowIndex * width, rowIndex * width + count),
              actLogits: actionLogits.slice(rowIndex * 2, rowIndex * 2 + 2)
            });
          }
        } finally {
          for (const tensor of Object.values(feeds)) tensor.dispose?.();
          for (const tensor of Object.values(output ?? {})) tensor.dispose?.();
        }
        signal?.throwIfAborted();
      }
      return {rows: outputRows};
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve().then(async () => {
        await session.release();
        session = null;
        tokenizer = null;
      });
      return disposing;
    }
  };
}
