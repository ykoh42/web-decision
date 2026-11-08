// Experimental runner against the pinned TVM JS runtime. This file does not
// instantiate WASM or request a GPU device.
import {WebDecisionError} from '../kev.js';
import {checkGpuDevice} from '../gpu-session-utils.js';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
// A failed root-scope release or disposal leaves ownership uncertain. Keep the
// runner, including its tensor handles, alive until the JavaScript realm ends.
const retained = new Set();
const shapeEq = (a, b) =>
  Array.isArray(a) && a.length === b.length && b.every((x, i) => Object.hasOwn(a, i) && a[i] === x);
const typedArrayPrototype = Object.getPrototypeOf(Int32Array.prototype);
const typedArrayLength = Object.getOwnPropertyDescriptor(typedArrayPrototype, 'length').get;
const typedArrayBuffer = Object.getOwnPropertyDescriptor(typedArrayPrototype, 'buffer').get;
const typedArrayTag = Object.getOwnPropertyDescriptor(typedArrayPrototype, Symbol.toStringTag).get;
const sharedByteLength =
  typeof SharedArrayBuffer === 'function' &&
  Object.getOwnPropertyDescriptor(SharedArrayBuffer.prototype, 'byteLength').get;

function isSharedBuffer(buffer) {
  if (!sharedByteLength) return false;
  try {
    sharedByteLength.call(buffer);
    return true;
  } catch {
    return false;
  }
}

function copyTypedData(Type, data) {
  try {
    return new Type(data);
  } catch (error) {
    if (error instanceof TypeError) throw invalid('Invalid H2O graph inputs.');
    throw error;
  }
}

function snapshotInput(input, index) {
  if (!input || typeof input !== 'object') throw invalid('Invalid H2O graph inputs.');
  // Read each field once. Copy typed data without user-provided iterators or
  // slice methods before inspecting shape elements, which may themselves be getters.
  const dtype = input.dtype,
    shape = input.shape,
    data = input.data;
  const Type = index === 2 || index === 3 ? Uint16Array : Int32Array;
  const expectedTag = index === 2 || index === 3 ? 'Uint16Array' : 'Int32Array';
  if (!ArrayBuffer.isView(data) || typedArrayTag.call(data) !== expectedTag) throw invalid('Invalid H2O graph inputs.');
  const length = typedArrayLength.call(data);
  const limit = index === 2 ? 512 * 512 : index === 4 ? 1 : 512;
  if (length > limit || isSharedBuffer(typedArrayBuffer.call(data))) throw invalid('Invalid H2O graph inputs.');
  const copy = copyTypedData(Type, data);
  const dimensions = index === 2 ? 4 : index === 4 ? 1 : 2;
  if (!Array.isArray(shape) || shape.length !== dimensions) throw invalid('Invalid H2O graph inputs.');
  const shapeCopy = [];
  for (let i = 0; i < dimensions; i++) {
    if (!Object.hasOwn(shape, i)) throw invalid('Invalid H2O graph inputs.');
    shapeCopy.push(shape[i]);
  }
  return {dtype, shape: shapeCopy, data: copy};
}

function snapshotInputs(inputs) {
  if (!Array.isArray(inputs) || inputs.length !== 5) throw invalid('Expected five H2O graph inputs.');
  const snapshot = [];
  for (let i = 0; i < 5; i++) {
    if (!Object.hasOwn(inputs, i)) throw invalid('Expected five H2O graph inputs.');
    snapshot.push(snapshotInput(inputs[i], i));
  }
  const [ids, positions, mask, valid, slot] = snapshot;
  const n = ids.data.length;
  if (
    !Number.isSafeInteger(n) ||
    n < 2 ||
    n > 512 ||
    ids.dtype !== 'int32' ||
    !shapeEq(ids.shape, [1, n]) ||
    !(ids.data instanceof Int32Array) ||
    ids.data.some((x) => x < 0 || x >= 248320) ||
    positions?.dtype !== 'int32' ||
    !shapeEq(positions.shape, [1, n]) ||
    !(positions.data instanceof Int32Array) ||
    positions.data.length !== n ||
    positions.data.some((x, i) => x !== i) ||
    mask?.dtype !== 'float16' ||
    !shapeEq(mask.shape, [1, 1, n, n]) ||
    !(mask.data instanceof Uint16Array) ||
    mask.data.length !== n * n ||
    mask.data.some((x) => x !== 0) ||
    valid?.dtype !== 'float16' ||
    !shapeEq(valid.shape, [1, n]) ||
    !(valid.data instanceof Uint16Array) ||
    valid.data.length !== n ||
    valid.data.some((x) => x !== 0x3c00) ||
    slot?.dtype !== 'int32' ||
    !shapeEq(slot.shape, [1]) ||
    !(slot.data instanceof Int32Array) ||
    slot.data.length !== 1 ||
    slot.data[0] !== n - 1
  )
    throw invalid('Invalid H2O graph inputs.');
  return snapshot;
}

function tensorFromInput(runner, {dtype, shape, data}) {
  const tensor = runner.runtime.empty(shape, dtype, runner.device);
  if (dtype === 'int32') return tensor.copyFrom(data);
  // GPU writes are aligned to four bytes even for odd FP16 element counts.
  const raw = new Uint8Array(Math.ceil((data.length * 2) / 4) * 4),
    view = new DataView(raw.buffer);
  for (let i = 0; i < data.length; i++) view.setUint16(i * 2, data[i], true);
  const context = runner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof tensor.byteOffset !== 'number')
    throw new Error('Pinned TVM aligned upload API unavailable.');
  context.copyRawBytesToBuffer(raw, tensor.getDataPtr(), tensor.byteOffset, raw.length);
  return tensor;
}

/** Owns a previously initialized root scope and exactly 682 ordered tensors. */
export class H2oTvmRunner {
  constructor({
    runtime,
    device,
    gpu,
    fn,
    weights,
    lifecycle,
    disposalGuard,
    uncaptured,
    teardownObserver,
    teardownTimeoutMs
  }) {
    if (
      !runtime ||
      !device ||
      !gpu ||
      typeof fn !== 'function' ||
      !Array.isArray(weights) ||
      weights.length !== 682 ||
      Array.from({length: 682}, (_, i) => i).some((i) => !Object.hasOwn(weights, i) || !weights[i]) ||
      !lifecycle ||
      !disposalGuard ||
      !Array.isArray(uncaptured) ||
      typeof teardownObserver?.beforeDispose !== 'function' ||
      typeof teardownObserver?.afterDispose !== 'function' ||
      !Number.isSafeInteger(teardownTimeoutMs) ||
      teardownTimeoutMs < 1
    )
      throw new TypeError('An owned pinned TVM runtime and 682 ordered tensors are required.');
    Object.assign(this, {
      runtime,
      device,
      gpu,
      fn,
      weights: [...weights],
      lifecycle,
      disposalGuard,
      uncaptured,
      teardownObserver,
      teardownTimeoutMs
    });
    this.busy = false;
    this.closed = false;
    this.disposal = null;
  }

  async exclusive(operation) {
    assertUsable(this);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    if (this.busy) throw invalid('Concurrent direct runner use is not supported.');
    this.busy = true;
    try {
      return await operation();
    } finally {
      this.busy = false;
    }
  }

  async run(inputs) {
    const snapshot = snapshotInputs(inputs);
    return this.exclusive(() =>
      withStrandsScope(this, 'H2O inference failed.', async (lease) => {
        const args = snapshot.map((input) => tensorFromInput(this, input));
        const output = this.fn(...args, ...this.weights);
        if (output?.dtype !== 'float32' || !shapeEq(output.shape, [1, 1, 255]))
          throw new Error('Unexpected H2O graph output.');
        const host = this.runtime.empty([1, 1, 255], 'float32', this.runtime.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(output);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        checkGpuDevice(this);
        const data = host.toArray();
        if (!(data instanceof Float32Array) || data.length !== 255 || !data.every(Number.isFinite))
          throw new Error('Invalid H2O score readback.');
        return data;
      })
    );
  }

  async releaseCachedMemory() {
    return this.exclusive(() =>
      withStrandsScope(this, 'Could not release H2O cached memory.', async (lease) => {
        this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await lease.syncWithoutReadback();
      })
    );
  }

  dispose() {
    if (this.disposal) return this.disposal;
    if (this.busy) return Promise.reject(invalid('Cannot dispose while direct runner work is active.'));
    this.closed = true;
    this.disposal = Promise.resolve().then(async () => {
      try {
        const checkBeforeDispose = () => {
          assertUsable(this);
          if (this.lifecycle.disposed || this.lifecycle.loss || this.uncaptured.length)
            throw new Error('GPU device failed before H2O disposal.');
        };
        await this.teardownObserver.beforeDispose();
        checkBeforeDispose();
        this.runtime.endScope();
        await this.teardownObserver.beforeDispose();
        checkBeforeDispose();
        this.runtime.dispose();
        await this.teardownObserver.afterDispose(this.teardownTimeoutMs);
        if (this.lifecycle.loss || this.disposalGuard.blocked || this.uncaptured.length || this.poisoned)
          throw new Error('GPU device failed during H2O disposal.');
        this.weights = [];
        this.runtime = this.device = this.gpu = this.fn = this.lifecycle = this.teardownObserver = null;
        this.uncaptured = [];
      } catch (cause) {
        this.poisoned = true;
        retained.add(this);
        try {
          this.disposalGuard?.retain();
        } catch {
          retainStrandsResources(this.runtime, this.gpu);
        }
        if (cause instanceof WebDecisionError) throw cause;
        throw new WebDecisionError('INFERENCE_ERROR', 'Could not safely dispose H2O runtime.', {cause});
      }
    });
    return this.disposal;
  }
}
