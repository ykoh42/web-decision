// Isolated fixed-shape host. The caller owns and verifies model assets/runtime.
import {validatePacked} from './adapter.mjs';
import {WebDecisionError} from '../kev.js';
import {checkGpuDevice, createCausalInputs} from '../gpu-session-utils.js';
import {assertUsable, withNariScope} from './scope.mjs';
import {runCheckpointedStages} from './checkpoints.mjs';

const invalid = () => new WebDecisionError('INVALID_INPUT', 'Invalid fixed-shape Nari input.');
const badAsset = () => new WebDecisionError('INVALID_ASSET', 'Invalid Nari parameter binding.');
const retained = new Set();
const isDense = (value) =>
  Array.isArray(value) && Array.from({length: value.length}, (_, i) => Object.hasOwn(value, i)).every(Boolean);

function shapeMatches(tensor, shape, dtype, device) {
  return (
    tensor &&
    tensor.dtype === dtype &&
    isDense(tensor.shape) &&
    tensor.shape.length === shape.length &&
    shape.every((n, i) => tensor.shape[i] === n) &&
    tensor.device &&
    tensor.device.deviceType === device.deviceType &&
    tensor.device.deviceId === device.deviceId &&
    tensor.isView !== true
  );
}

/** Validate before producing any narrowing typed array or touching the GPU. */
export function snapshotToqInput(value) {
  const keys = ['inputIds', 'optPos', 'optQ', 'optType', 'slotPos', 'nLetters', 'options'];
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    !Reflect.ownKeys(value).every((k) => typeof k === 'string' && keys.includes(k)) ||
    !keys.every((k) => Object.hasOwn(value, k) && isDense(value[k])) ||
    !value.options.every(isDense)
  )
    throw invalid();
  validatePacked(value);
  const snapshot = {
    inputIds: [...value.inputIds],
    optPos: [...value.optPos],
    optQ: [...value.optQ],
    optType: [...value.optType],
    slotPos: [...value.slotPos],
    nLetters: [...value.nLetters],
    options: value.options.map((x) => [...x])
  };
  const total = validatePacked(snapshot),
    questions = snapshot.options.length,
    tokens = snapshot.inputIds.length;
  if (tokens < 6 || tokens > 361 || total < 2 || total > 64 || questions < 1 || questions > 8) throw invalid();
  // validatePacked checks vocabulary, strictly ordered positions, ownership,
  // type=choice and the 26/27 slot rule before Int32Array conversion.
  for (const key of ['inputIds', 'optPos', 'optQ', 'optType', 'slotPos', 'nLetters'])
    if (snapshot[key].some((v) => !Number.isSafeInteger(v) || v < -2147483648 || v > 2147483647)) throw invalid();
  return snapshot;
}

function uploadOptionMask(runner, count) {
  return runner.runtime.empty([1, count], 'int32', runner.device).copyFrom(new Int32Array(count).fill(1));
}

function outputTensor(value, shape) {
  if (
    !value ||
    value.dtype !== 'float32' ||
    !isDense(value.shape) ||
    value.shape.length !== shape.length ||
    !shape.every((n, i) => value.shape[i] === n)
  )
    throw new Error('Unexpected Nari output ABI.');
  return value;
}

function checkedReadout(host, count) {
  const value = host.toArray();
  if (
    !(value instanceof Float32Array) ||
    value.length !== count ||
    (typeof SharedArrayBuffer !== 'undefined' && value.buffer instanceof SharedArrayBuffer)
  )
    throw new Error('Invalid Nari readback.');
  return Float32Array.from(value);
}

export class NariToqRunner {
  #weights;
  #letterRows;
  #bindings;
  constructor({
    runtime,
    device,
    gpu,
    fn,
    parameters,
    bindings,
    letterRows,
    lifecycle,
    uncaptured,
    disposalGuard,
    teardownObserver,
    teardownTimeoutMs,
    diagnostic,
    progress,
    expectedBindings = 730
  }) {
    if (
      !runtime ||
      device?.deviceType !== 15 ||
      !Number.isSafeInteger(device.deviceId) ||
      device.deviceId < 0 ||
      typeof device.sync !== 'function' ||
      !gpu ||
      typeof gpu.pushErrorScope !== 'function' ||
      typeof gpu.popErrorScope !== 'function' ||
      typeof fn !== 'function' ||
      !lifecycle ||
      !Array.isArray(uncaptured) ||
      !disposalGuard ||
      typeof disposalGuard.retain !== 'function' ||
      typeof disposalGuard.blocked !== 'boolean' ||
      typeof teardownObserver?.beforeDispose !== 'function' ||
      typeof teardownObserver?.afterDispose !== 'function' ||
      !Number.isSafeInteger(teardownTimeoutMs) ||
      teardownTimeoutMs < 1 ||
      (progress !== undefined && typeof progress !== 'function') ||
      (expectedBindings !== 730 && expectedBindings !== 606) ||
      !isDense(parameters) ||
      parameters.length !== expectedBindings ||
      !isDense(bindings) ||
      bindings.length !== expectedBindings ||
      !shapeMatches(letterRows, [2, 26, 2560], 'float32', device)
    )
      throw badAsset();
    const weights = [],
      expected = [];
    for (let i = 0; i < bindings.length; i++) {
      const row = parameters[i],
        want = bindings[i],
        tensor = row?.tensor;
      if (
        !want ||
        !isDense(want.shape) ||
        !want.shape.length ||
        !row ||
        row.target !== want.target ||
        row.graphArgument !== want.graphArgument ||
        !shapeMatches(tensor, want.shape, want.dtype, device)
      )
        throw badAsset();
      weights.push(tensor);
      expected.push({shape: [...want.shape], dtype: want.dtype});
    }
    Object.assign(this, {
      runtime,
      device,
      gpu,
      fn,
      lifecycle,
      uncaptured,
      disposalGuard,
      teardownObserver,
      teardownTimeoutMs,
      diagnostic,
      progress
    });
    this.#weights = weights;
    this.#bindings = expected;
    this.#letterRows = letterRows;
    this.busy = false;
    this.closed = false;
  }

  async run(value) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    if (this.busy) throw new WebDecisionError('INFERENCE_ERROR', 'Concurrent runner use is unsupported.');
    const packed = snapshotToqInput(value),
      count = packed.optPos.length,
      questions = packed.options.length;
    if (
      !shapeMatches(this.#letterRows, [2, 26, 2560], 'float32', this.device) ||
      this.#weights.some(
        (tensor, i) => !shapeMatches(tensor, this.#bindings[i].shape, this.#bindings[i].dtype, this.device)
      )
    )
      throw badAsset();
    this.busy = true;
    try {
      return await withNariScope(this, 'Nari inference failed.', async (lease) => {
        let ids, positions, mask, valid, optPos, optMask, optType, slots, head, letters, a, b, headScores, letterScores;
        await runCheckpointedStages({
          gpu: this.gpu,
          retain: () => this.disposalGuard.retain(),
          stages: {
            inputs: {
              submit: async () => {
                this.diagnostic?.phase('graph-input');
                await this.progress?.();
                checkGpuDevice(this);
                ({ids, positions, mask, valid} = createCausalInputs(this, packed.inputIds));
                optPos = this.runtime.empty([1, count], 'int32', this.device).copyFrom(Int32Array.from(packed.optPos));
                optMask = uploadOptionMask(this, count);
                optType = this.runtime
                  .empty([1, count], 'int32', this.device)
                  .copyFrom(Int32Array.from(packed.optType));
                slots = this.runtime
                  .empty([1, questions], 'int32', this.device)
                  .copyFrom(Int32Array.from(packed.slotPos));
              },
              sync: async () => {
                await lease.syncWithoutReadback();
                checkGpuDevice(this);
              }
            },
            graph: {
              submit: async () => {
                this.diagnostic?.phase('graph-dispatch');
                await this.progress?.();
                checkGpuDevice(this);
                const tuple = this.fn(
                  ids,
                  positions,
                  mask,
                  valid,
                  optPos,
                  optMask,
                  optType,
                  slots,
                  this.#letterRows,
                  ...this.#weights
                );
                this.diagnostic?.phase('graph-returned');
                await this.progress?.();
                checkGpuDevice(this);
                if (!tuple || typeof tuple.size !== 'function' || tuple.size() !== 2 || typeof tuple.get !== 'function')
                  throw new Error('Unexpected Nari output tuple.');
                head = outputTensor(tuple.get(0), [1, count]);
                letters = outputTensor(tuple.get(1), [1, questions, 26]);
              },
              sync: async () => {
                await lease.syncWithoutReadback();
                checkGpuDevice(this);
              }
            },
            head: {
              submit: async () => {
                this.diagnostic?.phase('head-readback');
                await this.progress?.();
                checkGpuDevice(this);
                a = this.runtime.empty([1, count], 'float32', this.runtime.cpu());
                lease.prepareReadback();
                try {
                  a.copyFrom(head);
                } finally {
                  lease.captureReadback();
                }
              },
              sync: async () => {
                await lease.sync();
                checkGpuDevice(this);
                headScores = checkedReadout(a, count);
                lease.nextReadback();
              }
            },
            letter: {
              submit: async () => {
                this.diagnostic?.phase('letter-readback');
                await this.progress?.();
                checkGpuDevice(this);
                b = this.runtime.empty([1, questions, 26], 'float32', this.runtime.cpu());
                lease.prepareReadback();
                try {
                  b.copyFrom(letters);
                } finally {
                  lease.captureReadback();
                }
              },
              sync: async () => {
                await lease.sync();
                checkGpuDevice(this);
                letterScores = checkedReadout(b, questions * 26);
              }
            }
          }
        });
        return {headScores, letterScores};
      });
    } finally {
      this.busy = false;
    }
  }

  async releaseCachedMemory() {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    if (this.busy) throw new WebDecisionError('INFERENCE_ERROR', 'Concurrent runner use is unsupported.');
    this.busy = true;
    try {
      return await withNariScope(this, 'Could not release Nari memory.', async (lease) => {
        this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await lease.syncWithoutReadback();
      });
    } finally {
      this.busy = false;
    }
  }

  async dispose() {
    if (this.disposal) return this.disposal;
    if (this.busy) throw new WebDecisionError('INFERENCE_ERROR', 'Cannot dispose a running Nari runner.');
    this.closed = true;
    this.disposal = Promise.resolve().then(async () => {
      try {
        const check = () => {
          assertUsable(this);
          if (this.lifecycle.loss || this.lifecycle.disposed || this.uncaptured.length)
            throw Error('GPU device failed before Nari disposal.');
        };
        await this.teardownObserver.beforeDispose();
        check();
        this.runtime.endScope();
        await this.teardownObserver.beforeDispose();
        check();
        this.runtime.dispose();
        await this.teardownObserver.afterDispose(this.teardownTimeoutMs);
        if (this.lifecycle.loss || this.disposalGuard.blocked || this.uncaptured.length || this.poisoned)
          throw Error('GPU device failed during Nari disposal.');
        this.#weights = [];
        this.#bindings = [];
        this.#letterRows = null;
        this.runtime = this.device = this.gpu = this.fn = this.lifecycle = this.teardownObserver = null;
        this.uncaptured = [];
      } catch (cause) {
        this.poisoned = true;
        retained.add(this);
        try {
          this.disposalGuard.retain();
        } catch {}
        throw new WebDecisionError('INFERENCE_ERROR', 'Could not dispose Nari runtime.', {cause});
      }
    });
    return this.disposal;
  }
}
