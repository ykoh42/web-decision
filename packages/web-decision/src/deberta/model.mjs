import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
/** Experimental, preloaded DeBERTa session. Not yet part of the public package. */
import {WebDecisionError, decodeFloat16} from '../kev.js';
import {DEBERTA_MAX_OPTIONS, packDebertaRequest, unpackDebertaScores} from './packing.mjs';
import {finishDebertaSystemOne, prepareDebertaSystemOne} from './typed.mjs';
import {waitForAbort} from '../assets.js';

const noFailure = Symbol('no failure');

function checkDeviceAfterCleanup(session, message) {
  if (session.lifecycle.loss || session.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

async function appendChunk(session, hidden, seg, chunk, tensor, scores) {
  session.runtime.beginScope();
  let failure = noFailure;
  try {
    const out = session.head(hidden, tensor(seg), tensor(chunk.pair_q), tensor(chunk.pair_opt), ...session.headWeights);
    const host = session.runtime.empty([1, 3], 'float16', session.runtime.cpu()).copyFrom(out);
    await session.device.sync();
    session.checkDevice();
    const raw = host.toRawBytes();
    if (raw.byteLength !== 6) throw new Error('Unexpected output length.');
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    for (let i = 0; i < chunk.count; i++) scores.push(decodeFloat16(view.getUint16(i * 2, true)));
  } catch (cause) {
    failure = cause;
  } finally {
    if (!session.lifecycle.disposed) {
      try {
        session.runtime.endScope();
      } catch (cause) {
        if (failure === noFailure) failure = cause;
      }
    }
  }
  if (failure !== noFailure) throw failure;
}

export class ExperimentalDebertaSession {
  /** The caller supplies an initialized runtime with an open root scope and checked weights. */
  constructor({runtime, device, gpu, encoder, head, encoderWeights, headWeights, tokenizer, uncaptured, lifecycle}) {
    Object.assign(this, {
      runtime,
      device,
      gpu,
      encoder,
      head,
      encoderWeights,
      headWeights,
      tokenizer,
      uncaptured,
      lifecycle
    });
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }
  async decide(request, options = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFields(options);
    request = validateRequestFields(request, Infinity, Infinity, Infinity, DEBERTA_MAX_OPTIONS, Infinity, {
      dataPropertiesOnly: true
    });
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    let packed;
    try {
      packed = packDebertaRequest(this.tokenizer, request);
    } catch (cause) {
      throw new WebDecisionError('INVALID_INPUT', 'Invalid DeBERTa request.', {cause});
    }
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const schedule = () => {
      if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const task = this.queue.then(() => {
        signal?.throwIfAborted();
        return this.run(packed);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbort(schedule, signal) : schedule();
  }
  async systemOne(state, questions, options = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const prepared = prepareDebertaSystemOne(state, questions);
    const result = await this.decide(prepared.request, options);
    return finishDebertaSystemOne(prepared, result);
  }
  checkDevice() {
    if (this.lifecycle.loss || this.lifecycle.disposed) throw new Error('The GPU device was lost.');
    if (this.uncaptured.length) throw new Error(this.uncaptured[0]);
  }
  async run(packed) {
    let failure = noFailure,
      result,
      scope = false,
      gpuScope = false;
    try {
      this.checkDevice();
      this.runtime.beginScope();
      scope = true;
      this.gpu.pushErrorScope('validation');
      gpuScope = true;
      const tensor = (values) => {
        const t = this.runtime.empty([1, values.length], 'int32', this.device);
        t.copyFrom(Int32Array.from(values));
        return t;
      };
      const {input_ids, attention_mask, seg} = packed.inputs;
      const hidden = this.encoder(tensor(input_ids), tensor(attention_mask), ...this.encoderWeights),
        scores = [];
      for (const chunk of packed.chunks) await appendChunk(this, hidden, seg, chunk, tensor, scores);
      result = unpackDebertaScores(packed, scores);
    } catch (cause) {
      failure = cause;
    } finally {
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
  async releaseCachedMemory() {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const task = this.queue.then(async () => {
      let failure = noFailure,
        scope = false,
        gpuScope = false;
      try {
        this.checkDevice();
        this.runtime.beginScope();
        scope = true;
        this.gpu.pushErrorScope('validation');
        gpuScope = true;
        this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await this.device.sync();
        this.checkDevice();
      } catch (cause) {
        failure = cause;
      } finally {
        try {
          const e = gpuScope ? await this.gpu.popErrorScope() : null;
          if (e && failure === noFailure) failure = e;
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
        throw new WebDecisionError('INFERENCE_ERROR', 'Could not release DeBERTa cached memory.', {cause: failure});
      if (this.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', this.uncaptured[0]);
      checkDeviceAfterCleanup(this, 'Could not release DeBERTa cached memory.');
    });
    this.queue = task.catch(() => {});
    return task;
  }
  async dispose() {
    if (this.disposal) return this.disposal;
    this.closed = true;
    this.disposal = (async () => {
      await this.queue;
      try {
        if (!this.lifecycle.disposed) {
          try {
            if (!this.lifecycle.loss) this.runtime.endScope();
          } finally {
            this.runtime.dispose();
          }
        }
      } finally {
        this.encoderWeights = [];
        this.headWeights = [];
        this.runtime = this.device = this.gpu = this.encoder = this.head = this.tokenizer = this.lifecycle = null;
        this.uncaptured = [];
      }
    })();
    return this.disposal;
  }
}
