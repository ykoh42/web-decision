import {WebDecisionError} from './kev.js';
import {retainStrandsResources} from './strands/disposal-guard.mjs';

interface RuntimeLifecycle {
  readonly disposed: boolean;
  readonly loss: unknown;
}

interface FailedLoad {
  readonly loadWatch?: {close(): unknown} | null | undefined;
  readonly runtime?: {dispose(): unknown; endScope?: () => unknown} | null | undefined;
  readonly scopeOpen?: boolean | undefined;
  readonly lifecycle?: RuntimeLifecycle | null | undefined;
  readonly gpuDevice?: {destroy(): unknown} | null | undefined;
}

/** Track TVM's automatic device-loss disposal and make disposal idempotent. */
export function trackRuntime(runtime: {dispose(...args: unknown[]): unknown}, gpu: {lost: PromiseLike<unknown>}) {
  let disposed = false;
  let loss: unknown = null;
  const lossController = new AbortController();
  const dispose = runtime.dispose.bind(runtime);
  runtime.dispose = (...args: unknown[]) => {
    if (disposed) return;
    disposed = true;
    return dispose(...args);
  };
  // Register before TVM's own listener. A device destroyed by a normal
  // runtime.dispose() must not turn a deliberate close into a lost device.
  gpu.lost.then(
    (info) => {
      if (!disposed) {
        loss = info;
        lossController.abort(new Error('The GPU device was lost.'));
      }
    },
    (error: unknown) => {
      if (!disposed) {
        loss = {message: String(error)};
        lossController.abort(new Error('The GPU device was lost.'));
      }
    }
  );
  return {
    lossSignal: lossController.signal,
    get disposed() {
      return disposed;
    },
    get loss() {
      return loss;
    }
  };
}

/** Refuse to return a model whose device was lost or runtime was disposed while loading. */
export function assertLoadActive(lifecycle: RuntimeLifecycle): void {
  if (lifecycle.loss || lifecycle.disposed)
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'The GPU device was lost while loading the model.');
}

/** Attempt a drain on failure; a rejected sync does not prove work has settled. */
export async function trySyncBeforeCleanup(device: {sync(): unknown}, lifecycle: RuntimeLifecycle): Promise<void> {
  if (lifecycle.loss || lifecycle.disposed) return;
  try {
    await device.sync();
  } catch {}
}

/** Close the load watcher and attempt scope, runtime, and device cleanup. */
export function cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice}: FailedLoad): void {
  loadWatch?.close();
  try {
    if (runtime && scopeOpen && !lifecycle?.disposed && !lifecycle?.loss) (runtime as {endScope(): unknown}).endScope();
  } catch {}
  try {
    runtime?.dispose();
  } catch {}
  try {
    gpuDevice?.destroy();
  } catch {
    retainStrandsResources(runtime, gpuDevice);
  }
}
