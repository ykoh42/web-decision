// Isolated session integration against the experimental five-input graph ABI.
// The diagnostic graph is separate; no public loader or model ID is provided here.
import {waitForAbort} from '../abort.js';
import {checkGpuDevice, createCausalInputs} from '../gpu-session-utils.js';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields} from '../request-fields.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';

// Failed disposal must retain the whole session, including weight wrappers,
// until the realm is discarded. A guarded runtime alone may not own them.
const retainedSessions = new Set();

function validateQuestion(question, maxTokens) {
  const ids = question?.inputIds;
  if (
    !Array.isArray(ids) ||
    ids.length < 2 ||
    ids.length > maxTokens ||
    Array.from(ids).some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320) ||
    !Number.isSafeInteger(question.lastIndex) ||
    question.lastIndex !== ids.length - 1
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Imajev tokens or final position.');
}

function validatePrepared(prepared, maxTokens) {
  if (!Array.isArray(prepared?.questions) || prepared.questions.length < 1 || prepared.questions.length > 8)
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Imajev question count.');
  for (const question of prepared.questions) validateQuestion(question, maxTokens);
}

function checkAfterScope(session, message) {
  try {
    checkGpuDevice(session);
  } catch (cause) {
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause});
  }
}

/** Experiment only. The caller must install the Strands disposal guard after
 * tracking the runtime and before initWebGPU registers TVM's loss callback. */
export class ImajevSession {
  constructor({runtime, device, gpu, fn, weights, processor, maxInputTokens, uncaptured, lifecycle, disposalGuard}) {
    if (
      !processor ||
      typeof processor.prepare !== 'function' ||
      typeof processor.finish !== 'function' ||
      !Number.isSafeInteger(maxInputTokens) ||
      maxInputTokens < 2 ||
      maxInputTokens > 4096 ||
      !disposalGuard ||
      typeof disposalGuard.retain !== 'function' ||
      typeof disposalGuard.blocked !== 'boolean'
    )
      throw new WebDecisionError('INVALID_ASSET', 'Invalid Imajev processor or graph token limit.');
    Object.assign(this, {
      runtime,
      device,
      gpu,
      fn,
      weights,
      processor,
      maxInputTokens,
      uncaptured,
      lifecycle,
      disposalGuard
    });
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }

  async decide(request, options = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    const signal = options?.signal;
    if (signal !== undefined) validateH2oSignal(signal);
    signal?.throwIfAborted();
    validateDecisionFields(options);
    // Snapshot and validate before enqueue, even if another request is running.
    const prepared = this.processor.prepare(request);
    validatePrepared(prepared, this.maxInputTokens);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const schedule = () => {
      if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const task = this.queue.then(() => {
        signal?.throwIfAborted();
        return this.run(prepared, signal);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  async run(prepared, signal) {
    const rows = [];
    for (const question of prepared.questions) {
      signal?.throwIfAborted();
      validateQuestion(question, this.maxInputTokens);
      const row = await withStrandsScope(this, 'Imajev inference failed.', async (lease) => {
        const {ids, positions, mask, valid} = createCausalInputs(this, question.inputIds);
        const last = this.runtime.empty([1], 'int32', this.device).copyFrom(Int32Array.of(question.lastIndex));
        signal?.throwIfAborted();
        const output = this.fn(ids, positions, mask, valid, last, ...this.weights);
        if (
          output?.dtype !== 'float32' ||
          !Array.isArray(output.shape) ||
          output.shape.length !== 2 ||
          output.shape[0] !== 1 ||
          output.shape[1] !== 256
        )
          throw new Error('The Imajev graph output does not match [1, 256] FP32.');
        const host = this.runtime.empty([1, 256], 'float32', this.runtime.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(output);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        checkGpuDevice(this);
        const readout = host.toArray();
        if (
          !(readout instanceof Float32Array) ||
          readout.length !== 256 ||
          (typeof SharedArrayBuffer !== 'undefined' && readout.buffer instanceof SharedArrayBuffer) ||
          readout.some((value) => !Number.isFinite(value))
        )
          throw new Error('The Imajev graph returned an invalid 256-score FP32 row.');
        // Copy before this scope ends or another question reuses host storage.
        return new Float32Array(readout);
      });
      // Loss can occur while async GPU error scopes are being popped.
      checkAfterScope(this, 'Imajev inference failed.');
      rows.push(row);
    }
    signal?.throwIfAborted();
    return this.processor.finish(prepared, rows);
  }

  async releaseCachedMemory() {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    const task = this.queue.then(async () => {
      await withStrandsScope(this, 'Could not release Imajev cached memory.', async (lease) => {
        this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await lease.syncWithoutReadback();
      });
      checkAfterScope(this, 'Could not release Imajev cached memory.');
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
        assertUsable(this);
        if (this.lifecycle.loss || this.lifecycle.disposed) throw new Error('The GPU device was lost before disposal.');
        this.runtime.endScope();
        if (this.lifecycle.loss || this.lifecycle.disposed || this.disposalGuard.blocked)
          throw new Error('The GPU device changed during disposal.');
        this.runtime.dispose();
        if (this.lifecycle.loss || this.disposalGuard.blocked)
          throw new Error('The GPU device changed during disposal.');
      } catch (cause) {
        retainedSessions.add(this);
        this.disposalGuard.retain();
        throw new WebDecisionError('INFERENCE_ERROR', 'Could not safely dispose the Imajev session.', {cause});
      }
      this.weights = [];
      this.runtime = this.device = this.gpu = this.fn = this.processor = this.lifecycle = null;
      this.disposalGuard = null;
      this.uncaptured = [];
    })();
    return this.disposal;
  }
}
