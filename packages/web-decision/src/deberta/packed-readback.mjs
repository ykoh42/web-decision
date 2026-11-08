/** Diagnostic candidate: combine 3-score head outputs into one GPU readback.
 * Uses pinned TVM WebGPU internals and is not a public API.
 */
import {WebDecisionError, decodeFloat16} from '../kev.js';
import {unpackDebertaScores} from './packing.mjs';

const noFailure = Symbol('no failure');

function checkDeviceAfterCleanup(session, message) {
  if (session.lifecycle.loss || session.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

export async function runWithPackedReadback(packed) {
  let failure = noFailure,
    result,
    scope = false,
    gpuScope = false,
    readbackStarted = false,
    syncAttempted = false;
  try {
    this.checkDevice();
    this.runtime.beginScope();
    scope = true;
    this.gpu.pushErrorScope('validation');
    gpuScope = true;
    const ctx = this.runtime.lib.webGPUContext;
    if (!ctx || !Array.isArray(ctx.bufferTable)) throw new Error('Unsupported TVM WebGPU runtime.');
    const copyWithin = ctx.getDeviceAPI('deviceCopyWithinGPU');
    const tensor = (values) => {
      const t = this.runtime.empty([1, values.length], 'int32', this.device);
      t.copyFrom(Int32Array.from(values));
      return t;
    };
    const {input_ids, attention_mask, seg} = packed.inputs;
    const hidden = this.encoder(tensor(input_ids), tensor(attention_mask), ...this.encoderWeights);
    // Every 3-float16 result occupies six logical bytes. WebGPU copies require
    // four-byte alignment, so each destination slot is padded to eight bytes.
    const output = this.runtime.empty([1, packed.chunks.length * 4], 'float16', this.device),
      outputPtr = output.getDataPtr(),
      outputOffset = output.byteOffset;
    const outputBuffer = ctx.bufferTable[outputPtr];
    if (
      !outputBuffer ||
      !Number.isSafeInteger(outputOffset) ||
      outputOffset < 0 ||
      outputOffset % 4 !== 0 ||
      outputBuffer.size < outputOffset + packed.chunks.length * 8
    )
      throw new Error('Unexpected destination tensor storage.');
    for (const [i, chunk] of packed.chunks.entries()) {
      const out = this.head(hidden, tensor(seg), tensor(chunk.pair_q), tensor(chunk.pair_opt), ...this.headWeights);
      const ptr = out.getDataPtr(),
        offset = out.byteOffset,
        buffer = ctx.bufferTable[ptr];
      if (
        out.dtype !== 'float16' ||
        out.shape.length !== 2 ||
        out.shape[0] !== 1 ||
        out.shape[1] !== 3 ||
        !Number.isSafeInteger(offset) ||
        offset < 0 ||
        offset % 4 !== 0 ||
        !buffer ||
        buffer.size < offset + 8
      )
        throw new Error('Unexpected head tensor storage.');
      copyWithin(ptr, offset, outputPtr, outputOffset + i * 8, 8);
    }
    const host = this.runtime.empty([1, packed.chunks.length * 4], 'float16', this.runtime.cpu());
    readbackStarted = true;
    host.copyFrom(output);
    syncAttempted = true;
    await this.device.sync();
    this.checkDevice();
    const raw = host.toRawBytes();
    if (raw.byteLength !== packed.chunks.length * 8) throw new Error('Unexpected packed output length.');
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength),
      scores = [];
    for (const [i, chunk] of packed.chunks.entries())
      for (let j = 0; j < chunk.count; j++) scores.push(decodeFloat16(view.getUint16(i * 8 + j * 2, true)));
    result = unpackDebertaScores(packed, scores);
  } catch (cause) {
    failure = cause;
  } finally {
    if (readbackStarted && !syncAttempted) {
      try {
        await this.device.sync();
        this.checkDevice();
      } catch (cause) {
        if (failure === noFailure) failure = cause;
      }
    }
    try {
      const error = gpuScope ? await this.gpu.popErrorScope() : null;
      if (error && failure === noFailure) failure = error;
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
    if (scope && !this.lifecycle.disposed)
      try {
        this.runtime.endScope();
      } catch (cause) {
        if (failure === noFailure) failure = cause;
      }
  }
  if (failure !== noFailure)
    throw new WebDecisionError('INFERENCE_ERROR', 'DeBERTa inference failed.', {cause: failure});
  if (this.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', this.uncaptured[0]);
  // Device loss may arrive while the asynchronous error scope is closing.
  checkDeviceAfterCleanup(this, 'DeBERTa inference failed.');
  return result;
}
