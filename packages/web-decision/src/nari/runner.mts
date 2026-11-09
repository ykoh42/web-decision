// Isolated fixed-shape host. The caller owns and verifies model assets/runtime.
import {validatePacked} from './adapter.mjs';
import {WebDecisionError} from '../kev.js';
import {checkGpuDevice, createCausalInputs} from '../gpu-session-utils.js';
import {assertUsable, withNariScope} from './scope.mjs';
import {runCheckpointedStages} from './checkpoints.mjs';
import type {NariRuntimeDevice, NariRuntimeTensor, NariTvmRuntime} from './runtime.mjs';
import type {NariReadbackLease} from './scope.mjs';

interface NariPackedInput {
  readonly inputIds: readonly number[];
  readonly optPos: readonly number[];
  readonly optQ: readonly number[];
  readonly optType: readonly number[];
  readonly slotPos: readonly number[];
  readonly nLetters: readonly number[];
  readonly options: readonly (readonly string[])[];
}

interface NariSnapshot {
  inputIds: number[];
  optPos: number[];
  optQ: number[];
  optType: number[];
  slotPos: number[];
  nLetters: number[];
  options: string[][];
}

interface NariShapeTensor {
  readonly dtype: unknown;
  readonly shape: unknown;
  readonly device?: {readonly deviceType: unknown; readonly deviceId: unknown} | null | undefined;
  readonly isView?: unknown;
}

interface NariRunnerTensor extends NariRuntimeTensor {
  readonly device?: {readonly deviceType: unknown; readonly deviceId: unknown} | null | undefined;
  readonly isView?: unknown;
  copyFrom(value: unknown): NariRunnerTensor;
  toArray(): unknown;
}

interface NariRunnerRuntime extends NariTvmRuntime {
  cpu(): unknown;
  endScope(): unknown;
  getGlobalFunc(name: string): () => unknown;
  empty(shape: number[], dtype: string, device: unknown): NariRunnerTensor;
}

interface NariRunnerGpu {
  pushErrorScope(filter: 'out-of-memory' | 'validation'): unknown;
  popErrorScope(): Promise<unknown>;
}

interface NariLifecycle {
  readonly loss: unknown;
  readonly disposed: boolean;
}

interface NariDisposalGuard {
  readonly blocked: boolean;
  retain(): void;
}

interface NariTeardownObserver {
  beforeDispose(): Promise<unknown>;
  afterDispose(timeoutMs: number): Promise<unknown>;
}

interface NariDiagnostic {
  phase(value: string): void;
}

interface NariBinding {
  readonly target: unknown;
  readonly graphArgument: unknown;
  readonly shape: unknown;
  readonly dtype: unknown;
}

interface NariParameter {
  readonly target: unknown;
  readonly graphArgument: unknown;
  readonly tensor: unknown;
}

interface NariRunnerOptions {
  readonly runtime: unknown;
  readonly device: NariRuntimeDevice;
  readonly gpu: NariRunnerGpu;
  readonly fn: (...args: unknown[]) => unknown;
  readonly parameters: unknown;
  readonly bindings: unknown;
  readonly letterRows: unknown;
  readonly lifecycle: NariLifecycle;
  readonly uncaptured: unknown;
  readonly disposalGuard: NariDisposalGuard;
  readonly teardownObserver: NariTeardownObserver;
  readonly teardownTimeoutMs: number;
  readonly diagnostic?: NariDiagnostic | undefined;
  readonly progress?: (() => unknown) | undefined;
  readonly expectedBindings?: number | undefined;
}

interface NariExpectedBinding {
  readonly shape: unknown[];
  readonly dtype: unknown;
}

interface NariOutputTuple {
  size(): number;
  get(index: number): unknown;
}

const invalid = () => new WebDecisionError('INVALID_INPUT', 'Invalid fixed-shape Nari input.');
const badAsset = () => new WebDecisionError('INVALID_ASSET', 'Invalid Nari parameter binding.');
const retained = new Set<NariToqRunner>();
const isDense = (value: unknown): value is unknown[] =>
  Array.isArray(value) && Array.from({length: value.length}, (_, i) => Object.hasOwn(value, i)).every(Boolean);

function shapeMatches(tensor: unknown, shape: readonly unknown[], dtype: unknown, device: NariRuntimeDevice): unknown {
  return (
    tensor &&
    (tensor as NariShapeTensor).dtype === dtype &&
    isDense((tensor as NariShapeTensor).shape) &&
    ((tensor as NariShapeTensor).shape as unknown[]).length === shape.length &&
    shape.every((n, i) => ((tensor as NariShapeTensor).shape as unknown[])[i] === n) &&
    (tensor as NariShapeTensor).device &&
    (tensor as NariShapeTensor).device!.deviceType === device.deviceType &&
    (tensor as NariShapeTensor).device!.deviceId === device.deviceId &&
    (tensor as NariShapeTensor).isView !== true
  );
}

/** Validate before producing any narrowing typed array or touching the GPU. */
export function snapshotToqInput(value: unknown): NariSnapshot {
  const keys = ['inputIds', 'optPos', 'optQ', 'optType', 'slotPos', 'nLetters', 'options'];
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    !Reflect.ownKeys(value).every((k) => typeof k === 'string' && keys.includes(k)) ||
    !keys.every((k) => Object.hasOwn(value, k) && isDense((value as Record<string, unknown>)[k])) ||
    !(value as {readonly options: unknown[]}).options.every(isDense)
  )
    throw invalid();
  validatePacked(value as NariPackedInput);
  const snapshot = {
    inputIds: [...(value as NariPackedInput).inputIds],
    optPos: [...(value as NariPackedInput).optPos],
    optQ: [...(value as NariPackedInput).optQ],
    optType: [...(value as NariPackedInput).optType],
    slotPos: [...(value as NariPackedInput).slotPos],
    nLetters: [...(value as NariPackedInput).nLetters],
    options: (value as NariPackedInput).options.map((x) => [...x])
  };
  const total = validatePacked(snapshot),
    questions = snapshot.options.length,
    tokens = snapshot.inputIds.length;
  if (tokens < 6 || tokens > 361 || total < 2 || total > 64 || questions < 1 || questions > 8) throw invalid();
  // validatePacked checks vocabulary, strictly ordered positions, ownership,
  // type=choice and the 26/27 slot rule before Int32Array conversion.
  for (const key of ['inputIds', 'optPos', 'optQ', 'optType', 'slotPos', 'nLetters'] as const)
    if (snapshot[key].some((v) => !Number.isSafeInteger(v) || v < -2147483648 || v > 2147483647)) throw invalid();
  return snapshot;
}

function uploadOptionMask(runner: NariToqRunner, count: number): NariRunnerTensor {
  return runner.runtime.empty([1, count], 'int32', runner.device).copyFrom(new Int32Array(count).fill(1));
}

function outputTensor(value: unknown, shape: readonly unknown[]): NariRunnerTensor {
  if (
    !value ||
    (value as NariShapeTensor).dtype !== 'float32' ||
    !isDense((value as NariShapeTensor).shape) ||
    ((value as NariShapeTensor).shape as unknown[]).length !== shape.length ||
    !shape.every((n, i) => ((value as NariShapeTensor).shape as unknown[])[i] === n)
  )
    throw new Error('Unexpected Nari output ABI.');
  return value as NariRunnerTensor;
}

function checkedReadout(host: NariRunnerTensor, count: number): Float32Array {
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
  #weights: NariRunnerTensor[];
  #letterRows: NariRunnerTensor | null;
  #bindings: NariExpectedBinding[];
  declare runtime: NariRunnerRuntime;
  declare device: NariRuntimeDevice;
  declare gpu: NariRunnerGpu;
  declare fn: (...args: unknown[]) => unknown;
  declare lifecycle: NariLifecycle;
  declare uncaptured: string[];
  declare disposalGuard: NariDisposalGuard;
  declare teardownObserver: NariTeardownObserver;
  declare teardownTimeoutMs: number;
  declare diagnostic: NariDiagnostic | undefined;
  declare progress: (() => unknown) | undefined;
  declare busy: boolean;
  declare closed: boolean;
  declare poisoned: boolean | undefined;
  declare disposal: Promise<void> | undefined;

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
  }: NariRunnerOptions) {
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
    const weights: NariRunnerTensor[] = [],
      expected: NariExpectedBinding[] = [];
    for (let i = 0; i < bindings.length; i++) {
      const row = parameters[i] as NariParameter | undefined,
        want = bindings[i] as NariBinding | undefined,
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
      weights.push(tensor as NariRunnerTensor);
      expected.push({shape: [...(want.shape as unknown[])], dtype: want.dtype});
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
    this.#letterRows = letterRows as NariRunnerTensor;
    this.busy = false;
    this.closed = false;
  }

  async run(value: unknown): Promise<{headScores: Float32Array; letterScores: Float32Array}> {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    if (this.busy) throw new WebDecisionError('INFERENCE_ERROR', 'Concurrent runner use is unsupported.');
    const packed = snapshotToqInput(value),
      count = packed.optPos.length,
      questions = packed.options.length;
    if (
      !shapeMatches(this.#letterRows, [2, 26, 2560], 'float32', this.device) ||
      this.#weights.some(
        (tensor, i) => !shapeMatches(tensor, this.#bindings[i]!.shape, this.#bindings[i]!.dtype, this.device)
      )
    )
      throw badAsset();
    this.busy = true;
    try {
      return await withNariScope(this, 'Nari inference failed.', async (lease: NariReadbackLease) => {
        let ids!: ReturnType<typeof createCausalInputs>['ids'],
          positions!: ReturnType<typeof createCausalInputs>['positions'],
          mask!: ReturnType<typeof createCausalInputs>['mask'],
          valid!: ReturnType<typeof createCausalInputs>['valid'],
          optPos!: NariRunnerTensor,
          optMask!: NariRunnerTensor,
          optType!: NariRunnerTensor,
          slots!: NariRunnerTensor,
          head!: NariRunnerTensor,
          letters!: NariRunnerTensor,
          a!: NariRunnerTensor,
          b!: NariRunnerTensor,
          headScores!: Float32Array,
          letterScores!: Float32Array;
        await runCheckpointedStages({
          gpu: this.gpu,
          retain: () => this.disposalGuard.retain(),
          stages: {
            inputs: {
              submit: async () => {
                this.diagnostic?.phase('graph-input');
                await this.progress?.();
                checkGpuDevice(this);
                ({ids, positions, mask, valid} = createCausalInputs(
                  this as unknown as Parameters<typeof createCausalInputs>[0],
                  packed.inputIds
                ));
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
                if (
                  !tuple ||
                  typeof (tuple as NariOutputTuple).size !== 'function' ||
                  (tuple as NariOutputTuple).size() !== 2 ||
                  typeof (tuple as NariOutputTuple).get !== 'function'
                )
                  throw new Error('Unexpected Nari output tuple.');
                head = outputTensor((tuple as NariOutputTuple).get(0), [1, count]);
                letters = outputTensor((tuple as NariOutputTuple).get(1), [1, questions, 26]);
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

  async releaseCachedMemory(): Promise<void> {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    if (this.busy) throw new WebDecisionError('INFERENCE_ERROR', 'Concurrent runner use is unsupported.');
    this.busy = true;
    try {
      return await withNariScope(this, 'Could not release Nari memory.', async (lease: NariReadbackLease) => {
        this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await lease.syncWithoutReadback();
      });
    } finally {
      this.busy = false;
    }
  }

  async dispose(): Promise<void> {
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
        this.runtime = this.device = this.gpu = this.fn = this.lifecycle = this.teardownObserver = null as never;
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
