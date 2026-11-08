// The pinned TVM runtime calls this.dispose() independently on device loss.
// Retain uncertain resources until the JavaScript realm is discarded.
const retained = new Set<{runtime: unknown; gpu: unknown}>();

export function retainStrandsResources(runtime: unknown, gpu: unknown): void {
  retained.add({runtime, gpu});
}

/** Install after trackRuntime and before initWebGPU registers its loss callback. */
export function guardStrandsDisposal(
  runtime: {dispose(...args: unknown[]): unknown},
  gpu: {lost?: PromiseLike<unknown> | undefined} | null | undefined,
  lifecycle: {disposed: boolean; loss: unknown}
) {
  const trackedDispose = runtime.dispose;
  if (typeof trackedDispose !== 'function' || typeof gpu?.lost?.then !== 'function')
    throw new TypeError('A tracked runtime and device loss promise are required');
  let blocked = false,
    disposing = false;
  function retain() {
    if (!blocked) retainStrandsResources(runtime, gpu);
    blocked = true;
  }
  const onLoss = () => {
    try {
      if (!lifecycle.disposed) retain();
    } catch {
      retain();
    }
  };
  gpu.lost.then(onLoss, onLoss);
  runtime.dispose = (...args: unknown[]) => {
    try {
      if (blocked || lifecycle.loss) {
        retain();
        return;
      }
    } catch {
      retain();
      return;
    }
    if (disposing) return;
    disposing = true;
    try {
      return Reflect.apply(trackedDispose, runtime, args);
    } finally {
      disposing = false;
    }
  };
  return {
    retain,
    get blocked() {
      return blocked;
    }
  };
}
