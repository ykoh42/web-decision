import {WebDecisionError} from '../kev.js';
import {checkGpuDevice} from '../gpu-session-utils.js';

interface NariScopeRunner {
  poisoned?: boolean | undefined;
  readonly disposalGuard?: {readonly blocked?: unknown} | null | undefined;
  readonly runtime: {
    readonly lib?: {readonly webGPUContext?: unknown} | null | undefined;
    beginScope(): unknown;
    endScope(): unknown;
  };
  readonly device: {sync(): unknown};
  readonly gpu: {
    pushErrorScope(filter: 'out-of-memory' | 'validation'): unknown;
    popErrorScope(): unknown;
  };
  readonly lifecycle: {readonly loss: unknown; readonly disposed: unknown};
  readonly uncaptured: readonly unknown[];
}

export interface NariReadbackLease {
  prepareReadback(): void;
  captureReadback(): void;
  sync(): Promise<void>;
  nextReadback(): void;
  syncWithoutReadback(): Promise<void>;
}

// A failed drain cannot safely free WASM destinations or the owning runtime.
// Keep them alive until this JavaScript realm is discarded.
const quarantine = new Set<{readonly runner: NariScopeRunner; readonly lease: ReadbackLease | undefined}>();
const noFailure: unique symbol = Symbol('no failure');

export function assertUsable(runner: unknown): void {
  if ((runner as NariScopeRunner).poisoned || (runner as NariScopeRunner).disposalGuard?.blocked)
    throw new WebDecisionError('INFERENCE_ERROR', 'The Nari session has unresolved GPU work.');
}

function poison(runner: NariScopeRunner, lease: ReadbackLease | undefined): void {
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
      | {readonly value?: unknown}
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
  declare runner: NariScopeRunner;
  declare context: unknown;
  declare unknown: boolean;
  declare prepared: boolean;
  declare settled: boolean;
  declare succeeded: boolean;
  declare readback: Promise<unknown> | null | undefined;
  declare observed: Promise<void> | null | undefined;

  constructor(runner: NariScopeRunner) {
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
  // Only after the preceding readback has fully settled may the second start.
  nextReadback(): void {
    try {
      if (!this.prepared || !this.readback || !this.settled || !this.succeeded)
        throw Error('First readback is not complete');
      this.assertEmpty();
      this.prepared = false;
      this.settled = false;
      this.succeeded = false;
      this.readback = null;
      this.observed = null;
    } catch (error) {
      this.unknown = true;
      throw error;
    }
  }
  async syncWithoutReadback(): Promise<void> {
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
  async recover(open: boolean): Promise<void> {
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

function closeRuntimeScope(
  runner: NariScopeRunner,
  lease: ReadbackLease,
  open: boolean,
  failure: typeof noFailure
): typeof noFailure {
  if (!open || runner.lifecycle.disposed) return failure;
  try {
    runner.runtime.endScope();
  } catch (cause) {
    poison(runner, lease);
    if (failure === noFailure) failure = cause as typeof noFailure;
  }
  return failure;
}

async function closeScopes(
  runner: NariScopeRunner,
  lease: ReadbackLease,
  count: number,
  open: boolean,
  failure: typeof noFailure
): Promise<typeof noFailure> {
  while (count-- > 0) {
    try {
      const error = await runner.gpu.popErrorScope();
      if (error && failure === noFailure) failure = error as typeof noFailure;
    } catch (cause) {
      poison(runner, lease);
      if (failure === noFailure) failure = cause as typeof noFailure;
    }
  }
  return closeRuntimeScope(runner, lease, open, failure);
}

function assertCompleted(runner: NariScopeRunner, message: string, failure: typeof noFailure): void {
  if (failure !== noFailure) throw new WebDecisionError('INFERENCE_ERROR', message, {cause: failure as unknown});
  if (runner.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', runner.uncaptured[0] as string);
  if (runner.lifecycle.loss || runner.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

/** Nari owns sequential readbacks in the pinned TVM runtime. */
export async function withNariScope<Value>(
  runner: unknown,
  message: string,
  operation: (lease: NariReadbackLease) => Value | PromiseLike<Value>
): Promise<Awaited<Value>> {
  assertUsable(runner);
  let failure: typeof noFailure = noFailure,
    result!: Awaited<Value>,
    open = false,
    gpuScopes = 0;
  const lease = new ReadbackLease(runner as NariScopeRunner);
  try {
    checkGpuDevice(runner as NariScopeRunner);
    lease.assertInitial();
    (runner as NariScopeRunner).runtime.beginScope();
    open = true;
    (runner as NariScopeRunner).gpu.pushErrorScope('out-of-memory');
    gpuScopes++;
    (runner as NariScopeRunner).gpu.pushErrorScope('validation');
    gpuScopes++;
    result = await operation(lease);
    checkGpuDevice(runner as NariScopeRunner);
  } catch (cause) {
    failure = cause as typeof noFailure;
    await lease.recover(open);
  }
  if (lease.unknown) {
    poison(runner as NariScopeRunner, lease);
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: failure === noFailure ? undefined : failure});
  }
  failure = await closeScopes(runner as NariScopeRunner, lease, gpuScopes, open, failure);
  assertCompleted(runner as NariScopeRunner, message, failure);
  return result;
}
