import {WebDecisionError} from '../kev.js';
import {checkGpuDevice} from '../gpu-session-utils.js';

interface StrandsRuntime {
  readonly lib?: {readonly webGPUContext?: unknown} | null | undefined;
  beginScope?(): unknown;
  endScope?(): unknown;
}

interface StrandsDevice {
  sync(): unknown;
}

interface StrandsGpu {
  pushErrorScope(filter: 'out-of-memory' | 'validation'): unknown;
  popErrorScope(): unknown;
}

interface StrandsLifecycle {
  readonly loss: unknown;
  readonly disposed: unknown;
}

interface UsableRunner {
  poisoned?: boolean | undefined;
  readonly disposalGuard?: {readonly blocked?: unknown} | null | undefined;
}

interface StrandsScopeRunner extends UsableRunner {
  readonly runtime: StrandsRuntime | null | undefined;
  readonly device: unknown;
  readonly gpu: unknown;
  readonly lifecycle: StrandsLifecycle | null | undefined;
  readonly uncaptured: readonly unknown[] | undefined;
}

interface QuarantinedScope {
  readonly runner: StrandsScopeRunner;
  readonly lease: ReadbackLease;
}

// A failed drain cannot safely free WASM destinations or the owning runtime.
// Keep them alive until this JavaScript realm is discarded.
const quarantine = new Set<QuarantinedScope>();
const noFailure = Symbol('no failure');

export function assertUsable(runner: unknown): void {
  if ((runner as UsableRunner).poisoned || (runner as UsableRunner).disposalGuard?.blocked)
    throw new WebDecisionError('INFERENCE_ERROR', 'The Strands session has unresolved GPU work.');
}

function poison(runner: StrandsScopeRunner, lease: ReadbackLease): void {
  runner.poisoned = true;
  observeUnexpected(lease?.context);
  observeUnexpected(lease?.currentContext());
  quarantine.add({runner, lease});
}

// Attaching a rejection handler does not certify that a readback is safe or
// settled; it only prevents an unobserved native Promise rejection.
function observeUnexpected(context: unknown): void {
  try {
    const descriptor = (context && Object.getOwnPropertyDescriptor(context as object, 'pendingGPUToCPUCopy')) as
      | {readonly value?: unknown; readonly writable?: boolean}
      | null
      | undefined;
    if (descriptor && Object.hasOwn(descriptor, 'value') && descriptor.value instanceof Promise)
      Promise.prototype.then.call(descriptor.value as Promise<unknown>, undefined, () => {});
  } catch {}
}

function pendingField(context: unknown): unknown {
  const descriptor = (context && Object.getOwnPropertyDescriptor(context as object, 'pendingGPUToCPUCopy')) as
    | {readonly value?: unknown; readonly writable?: boolean}
    | null
    | undefined;
  if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.writable) {
    observeUnexpected(context);
    throw Error('Unrecognized TVM readback state');
  }
  return descriptor.value;
}

class ReadbackLease {
  declare runner: StrandsScopeRunner;
  declare context: unknown;
  declare unknown: boolean;
  declare prepared: boolean;
  declare settled: boolean;
  declare succeeded: boolean;
  declare readback: Promise<unknown> | undefined;
  declare observed: Promise<void> | undefined;

  constructor(runner: StrandsScopeRunner) {
    this.runner = runner;
    this.context = runner.runtime?.lib?.webGPUContext;
    this.unknown = false;
    this.prepared = false;
    this.settled = false;
    this.succeeded = false;
  }
  currentContext(): unknown {
    return this.runner.runtime?.lib?.webGPUContext;
  }
  assertEmpty(): void {
    if (this.currentContext() !== this.context || pendingField(this.context) !== null) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      throw Error('Unexpected pending TVM readback');
    }
  }
  assertInitial(): void {
    try {
      this.assertEmpty();
    } catch (error) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      this.unknown = true;
      throw error;
    }
  }
  prepareReadback(): void {
    try {
      this.assertEmpty();
      if (this.prepared) throw Error('Readback already prepared');
      this.prepared = true;
    } catch (error) {
      this.unknown = true;
      throw error;
    }
  }
  captureReadback(): void {
    try {
      if (this.currentContext() !== this.context || this.readback || !this.prepared)
        throw Error('Readback context changed');
      this.readback = pendingField(this.context) as Promise<unknown>;
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
  async sync(): Promise<void> {
    if (this.currentContext() !== this.context || pendingField(this.context) !== this.readback || !this.readback) {
      observeUnexpected(this.context);
      observeUnexpected(this.currentContext());
      this.unknown = true;
      throw Error('Readback state changed before sync');
    }
    await (this.runner.device as StrandsDevice).sync();
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
  async syncWithoutReadback(): Promise<void> {
    try {
      this.assertEmpty();
    } catch (error) {
      this.unknown = true;
      throw error;
    }
    await (this.runner.device as StrandsDevice).sync();
    try {
      this.assertEmpty();
    } catch (error) {
      this.unknown = true;
      throw error;
    }
  }
  async recover(open: boolean): Promise<void> {
    // The captured promise includes the WASM write. A sync rejection alone
    // does not establish that this promise has settled.
    if (this.observed && !this.settled) await this.observed;
    if (!open || this.unknown) return;
    try {
      if (this.runner.lifecycle!.loss || this.runner.lifecycle!.disposed || this.currentContext() !== this.context)
        throw Error('Device state changed during drain');
      const pending = pendingField(this.context);
      if (pending !== null && pending !== this.readback) {
        observeUnexpected(this.context);
        throw Error('Unexpected readback during drain');
      }
      await (this.runner.device as StrandsDevice).sync();
      this.assertEmpty();
    } catch {
      this.unknown = true;
    }
  }
}

function closeRuntimeScope(runner: StrandsScopeRunner, lease: ReadbackLease, open: boolean, failure: unknown): unknown {
  if (!open || runner.lifecycle!.disposed) return failure;
  try {
    runner.runtime!.endScope!();
  } catch (cause) {
    poison(runner, lease);
    if (failure === noFailure) failure = cause;
  }
  return failure;
}

async function closeScopes(
  runner: StrandsScopeRunner,
  lease: ReadbackLease,
  count: number,
  open: boolean,
  failure: unknown
): Promise<unknown> {
  while (count-- > 0) {
    try {
      const error = await (runner.gpu as StrandsGpu).popErrorScope();
      if (error && failure === noFailure) failure = error;
    } catch (cause) {
      poison(runner, lease);
      if (failure === noFailure) failure = cause;
    }
  }
  return closeRuntimeScope(runner, lease, open, failure);
}

function assertCompleted(runner: StrandsScopeRunner, message: string, failure: unknown): void {
  if (failure !== noFailure) throw new WebDecisionError('INFERENCE_ERROR', message, {cause: failure});
  if ((runner.uncaptured as readonly unknown[]).length)
    throw new WebDecisionError('INFERENCE_ERROR', (runner.uncaptured as readonly unknown[])[0] as string);
  if (runner.lifecycle!.loss || runner.lifecycle!.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

/** Strands owns one readback per row in the pinned TVM runtime. */
export async function withStrandsScope<Result>(
  runner: StrandsScopeRunner,
  message: string,
  operation: (lease: ReadbackLease) => Result | PromiseLike<Result>
): Promise<Result> {
  assertUsable(runner);
  let failure: unknown = noFailure,
    result: Result | undefined,
    open = false,
    gpuScopes = 0;
  const lease = new ReadbackLease(runner);
  try {
    checkGpuDevice(runner);
    lease.assertInitial();
    runner.runtime!.beginScope!();
    open = true;
    (runner.gpu as StrandsGpu).pushErrorScope('out-of-memory');
    gpuScopes++;
    (runner.gpu as StrandsGpu).pushErrorScope('validation');
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
  return result as Result;
}
