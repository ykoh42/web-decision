import {validateDecisionFields, validateRequestFields} from './request-fields.mjs';
import {KEV_MAX_OPTIONS, WebDecisionError, decodeFloat16, packKevRequest, unpackKevScores} from './kev.js';
import {assertKevAttentionFits} from './kev-gpu.js';
import {waitForAbort} from './assets.js';
import {finishKevSystemOne, prepareKevSystemOne} from './kev/system-one.mjs';

const noFailure = Symbol('no failure');

function checkDeviceAfterCleanup(session, message) {
  if (session.lifecycle.loss || session.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

async function inferDecisionModel(model, request, options = {}, packing = {}) {
  if (model.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
  const signal = options?.signal;
  signal?.throwIfAborted();
  validateDecisionFields(options);
  request = validateRequestFields(
    request,
    model.limits?.maxQuestions,
    model.limits?.maxInputCharacters,
    model.limits?.maxInputTokens,
    model.limits?.maxOptions ?? KEV_MAX_OPTIONS
  );
  if (model.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
  // Pack synchronously before entering the queue so later caller mutations do
  // not affect an already-submitted request.
  const packed = packKevRequest(model.tokenizer, request, model.limits, packing);
  if (model.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
  const schedule = () => {
    if (model.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const task = model.queue.then(() => {
      signal?.throwIfAborted();
      return model.run(packed);
    });
    model.queue = task.catch(() => {});
    return task;
  };
  // Cancellation stops the caller waiting, but the internal queue must
  // still wait for any GPU work already in progress before using the device.
  return signal ? waitForAbort(schedule, signal) : schedule();
}

export class DecisionModel {
  constructor(
    runtime,
    device,
    gpu,
    fn,
    weights,
    tokenizer,
    limits,
    uncaptured,
    lifecycle,
    supportsSystemOne = false,
    modelId = 'kev-0.6b-q4f16'
  ) {
    Object.assign(this, {
      runtime,
      device,
      gpu,
      fn,
      weights,
      tokenizer,
      limits,
      uncaptured,
      lifecycle,
      supportsSystemOne,
      modelId
    });
    this.closed = false;
    this.queue = Promise.resolve();
    this.disposal = null;
  }

  async decide(request, options = {}) {
    return inferDecisionModel(this, request, options);
  }

  async systemOne(state, questions, options = {}) {
    if (!this.supportsSystemOne)
      throw new WebDecisionError(
        'UNSUPPORTED_ENVIRONMENT',
        'Typed System One input is unavailable for this model route.'
      );
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    options?.signal?.throwIfAborted();
    validateDecisionFields(options);
    const prepared = prepareKevSystemOne(state, questions, this.limits?.maxInputCharacters);
    const result = await inferDecisionModel(this, prepared.request, options, prepared.packing);
    return finishKevSystemOne(prepared, result, this.modelId);
  }

  async run(packed) {
    if (this.lifecycle.loss || this.lifecycle.disposed) {
      throw new WebDecisionError('INFERENCE_ERROR', 'The GPU device was lost. Load the model again.');
    }
    if (this.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', this.uncaptured[0]);
    const n = packed.ids.length;
    assertKevAttentionFits(n, this.gpu?.limits);
    let result,
      failure = noFailure,
      scopePushed = false,
      scopeStarted = false;
    try {
      this.runtime.beginScope();
      scopeStarted = true;
      this.gpu.pushErrorScope('validation');
      scopePushed = true;
      const ids = this.runtime.empty([1, n], 'int32', this.device);
      const mask = this.runtime.empty([1, n], 'int32', this.device);
      ids.copyFrom(Int32Array.from(packed.ids));
      mask.copyFrom(new Int32Array(n).fill(1));
      const y = this.fn(ids, mask, ...this.weights);
      const host = this.runtime.empty([1, n], 'float16', this.runtime.cpu()).copyFrom(y);
      await this.device.sync();
      if (this.lifecycle.loss || this.lifecycle.disposed) throw new Error('The GPU device was lost.');
      const raw = host.toRawBytes();
      if (raw.byteLength !== n * 2) throw new Error('Unexpected output length.');
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      result = unpackKevScores(
        Array.from({length: n}, (_, i) => decodeFloat16(view.getUint16(i * 2, true))),
        packed
      );
    } catch (cause) {
      failure = cause;
    } finally {
      try {
        const gpuError = scopePushed ? await this.gpu.popErrorScope() : null;
        if (gpuError && failure === noFailure) failure = gpuError;
      } catch (cause) {
        if (failure === noFailure) failure = cause;
      }
      if (scopeStarted && !this.lifecycle.disposed) {
        try {
          this.runtime.endScope();
        } catch (cause) {
          if (failure === noFailure) failure = cause;
        }
      }
    }
    if (failure !== noFailure)
      throw new WebDecisionError('INFERENCE_ERROR', 'Model inference failed.', {cause: failure});
    if (this.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', this.uncaptured[0]);
    // Device loss may arrive while the asynchronous error scope is closing.
    checkDeviceAfterCleanup(this, 'Model inference failed.');
    return result;
  }

  /** Release unused TVM allocator buffers after earlier queued requests finish. */
  async releaseCachedMemory() {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const task = this.queue.then(async () => {
      if (this.lifecycle.loss || this.lifecycle.disposed) {
        throw new WebDecisionError('INFERENCE_ERROR', 'The GPU device was lost. Load the model again.');
      }
      if (this.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', this.uncaptured[0]);
      let failure = noFailure,
        scopeStarted = false,
        scopePushed = false;
      try {
        this.runtime.beginScope();
        scopeStarted = true;
        this.gpu.pushErrorScope('validation');
        scopePushed = true;
        this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await this.device.sync();
        if (this.lifecycle.loss || this.lifecycle.disposed) throw new Error('The GPU device was lost.');
      } catch (cause) {
        failure = cause;
      } finally {
        try {
          const gpuError = scopePushed ? await this.gpu.popErrorScope() : null;
          if (gpuError && failure === noFailure) failure = gpuError;
        } catch (cause) {
          if (failure === noFailure) failure = cause;
        }
        if (scopeStarted && !this.lifecycle.disposed) {
          try {
            this.runtime.endScope();
          } catch (cause) {
            if (failure === noFailure) failure = cause;
          }
        }
      }
      if (failure !== noFailure)
        throw new WebDecisionError('INFERENCE_ERROR', 'Could not release cached runtime memory.', {cause: failure});
      if (this.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', this.uncaptured[0]);
      checkDeviceAfterCleanup(this, 'Could not release cached runtime memory.');
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
        // Do not retain 394 tensor wrappers and the runtime after closing.
        this.weights = [];
        this.fn = this.runtime = this.device = this.gpu = this.tokenizer = this.lifecycle = null;
        this.uncaptured = [];
      }
    })();
    return this.disposal;
  }
}
