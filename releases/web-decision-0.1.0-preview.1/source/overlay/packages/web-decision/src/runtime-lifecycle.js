import {WebDecisionError} from './kev.js';
import {retainStrandsResources} from './strands/disposal-guard.mjs';

/** Track TVM's automatic device-loss disposal and make disposal idempotent. */
export function trackRuntime(runtime, gpu) {
  let disposed = false;
  let loss = null;
  const dispose = runtime.dispose.bind(runtime);
  runtime.dispose = (...args) => {
    if (disposed) return;
    disposed = true;
    return dispose(...args);
  };
  // Register before TVM's own listener. A device destroyed by a normal
  // runtime.dispose() must not turn a deliberate close into a lost device.
  gpu.lost.then(
    (info) => {
      if (!disposed) loss = info;
    },
    (error) => {
      if (!disposed) loss = {message: String(error)};
    }
  );
  return {
    get disposed() {
      return disposed;
    },
    get loss() {
      return loss;
    }
  };
}

/** Refuse to return a model whose device was lost or runtime was disposed while loading. */
export function assertLoadActive(lifecycle) {
  if (lifecycle.loss || lifecycle.disposed)
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'The GPU device was lost while loading the model.');
}

/** Attempt a drain on failure; a rejected sync does not prove work has settled. */
export async function trySyncBeforeCleanup(device, lifecycle) {
  if (lifecycle.loss || lifecycle.disposed) return;
  try {
    await device.sync();
  } catch {}
}

/** Close the load watcher and attempt scope, runtime, and device cleanup. */
export function cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice}) {
  loadWatch?.close();
  try {
    if (runtime && scopeOpen && !lifecycle?.disposed && !lifecycle?.loss) runtime.endScope();
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
