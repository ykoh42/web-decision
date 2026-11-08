import {WebDecisionError} from '../kev.js';
import {checkGpuDevice} from '../gpu-session-utils.js';

// A failed drain cannot safely free WASM destinations or the owning runtime.
// Keep them alive until this JavaScript realm is discarded.
const quarantine = new Set();
const noFailure = Symbol('no failure');

export function assertUsable(runner) {
  if (runner.poisoned || runner.disposalGuard?.blocked)
    throw new WebDecisionError('INFERENCE_ERROR', 'The Strands session has unresolved GPU work.');
}

function poison(runner, lease) {
  runner.poisoned = true;
  observeUnexpected(lease?.context);
  observeUnexpected(lease?.currentContext());
  quarantine.add({runner, lease});
}

// Attaching a rejection handler does not certify that a readback is safe or
// settled; it only prevents an unobserved native Promise rejection.
function observeUnexpected(context) {
  try {
    const descriptor = context && Object.getOwnPropertyDescriptor(context, 'pendingGPUToCPUCopy');
    if (descriptor && Object.hasOwn(descriptor, 'value') && descriptor.value instanceof Promise)
      Promise.prototype.then.call(descriptor.value, undefined, () => {});
  } catch {}
}

function pendingField(context) {
  const descriptor = context && Object.getOwnPropertyDescriptor(context, 'pendingGPUToCPUCopy');
  if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.writable) {
    observeUnexpected(context);
    throw Error('Unrecognized TVM readback state');
  }
  return descriptor.value;
}

class ReadbackLease {
  constructor(runner) {
    this.runner = runner;
    this.context = runner.runtime?.lib?.webGPUContext;
    this.unknown = false;
    this.prepared = false;
    this.settled = false;
    this.succeeded = false;
  }
  currentContext() {
    return this.runner.runtime?.lib?.webGPUContext;
  }
  assertEmpty() {
    if (this.currentContext() !== this.context || pendingField(this.context) !== null) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      throw Error('Unexpected pending TVM readback');
    }
  }
  assertInitial() {
    try {
      this.assertEmpty();
    } catch (error) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      this.unknown = true;
      throw error;
    }
  }
  prepareReadback() {
    try {
      this.assertEmpty();
      if (this.prepared) throw Error('Readback already prepared');
      this.prepared = true;
    } catch (error) {
      this.unknown = true;
      throw error;
    }
  }
  captureReadback() {
    try {
      if (this.currentContext() !== this.context || this.readback || !this.prepared)
        throw Error('Readback context changed');
      this.readback = pendingField(this.context);
      if (!(this.readback instanceof Promise)) throw Error('Missing TVM readback promise');
      this.observed = this.readback.then(
        () => {
          this.settled = true;
          this.succeeded = true;
        },
        () => {
          this.settled = true;
        }
      );
    } catch (error) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      this.unknown = true;
      throw error;
    }
  }
  async sync() {
    if (this.currentContext() !== this.context || pendingField(this.context) !== this.readback || !this.readback) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      this.unknown = true;
      throw Error('Readback state changed before sync');
    }
    await this.runner.device.sync();
    if (
      !this.settled ||
      !this.succeeded ||
      this.currentContext() !== this.context ||
      pendingField(this.context) !== null
    ) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      this.unknown = true;
      throw Error('Readback did not settle during sync');
    }
  }
  async syncWithoutReadback() {
    try {
      this.assertEmpty();
    } catch (error) {
      this.unknown = true;
      throw error;
    }
    await this.runner.device.sync();
    try {
      this.assertEmpty();
    } catch (error) {
      this.unknown = true;
      throw error;
    }
  }
  async recover(open) {
    // The captured promise includes the WASM write. A sync rejection alone
    // does not establish that this promise has settled.
    if (this.observed && !this.settled) await this.observed;
    if (!open || this.unknown) return;
    try {
      if (this.runner.lifecycle.loss || this.runner.lifecycle.disposed || this.currentContext() !== this.context)
        throw Error('Device state changed during drain');
      const pending = pendingField(this.context);
      if (pending !== null && pending !== this.readback) {
        observeUnexpected(this.context);
        throw Error('Unexpected readback during drain');
      }
      await this.runner.device.sync();
      this.assertEmpty();
    } catch {
      this.unknown = true;
    }
  }
}

function closeRuntimeScope(runner, lease, open, failure) {
  if (!open || runner.lifecycle.disposed) return failure;
  try {
    runner.runtime.endScope();
  } catch (cause) {
    poison(runner, lease);
    if (failure === noFailure) failure = cause;
  }
  return failure;
}

async function closeScopes(runner, lease, count, open, failure) {
  while (count-- > 0) {
    try {
      const error = await runner.gpu.popErrorScope();
      if (error && failure === noFailure) failure = error;
    } catch (cause) {
      poison(runner, lease);
      if (failure === noFailure) failure = cause;
    }
  }
  return closeRuntimeScope(runner, lease, open, failure);
}

function assertCompleted(runner, message, failure) {
  if (failure !== noFailure) throw new WebDecisionError('INFERENCE_ERROR', message, {cause: failure});
  if (runner.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', runner.uncaptured[0]);
  if (runner.lifecycle.loss || runner.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

/** Strands owns one readback per row in the pinned TVM runtime. */
export async function withStrandsScope(runner, message, operation) {
  assertUsable(runner);
  let failure = noFailure,
    result,
    open = false,
    gpuScopes = 0;
  const lease = new ReadbackLease(runner);
  try {
    checkGpuDevice(runner);
    lease.assertInitial();
    runner.runtime.beginScope();
    open = true;
    runner.gpu.pushErrorScope('out-of-memory');
    gpuScopes++;
    runner.gpu.pushErrorScope('validation');
    gpuScopes++;
    result = await operation(lease);
    checkGpuDevice(runner);
  } catch (cause) {
    failure = cause;
    await lease.recover(open);
  }
  if (lease.unknown) {
    poison(runner, lease);
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: failure === noFailure ? undefined : failure});
  }
  failure = await closeScopes(runner, lease, gpuScopes, open, failure);
  assertCompleted(runner, message, failure);
  return result;
}
