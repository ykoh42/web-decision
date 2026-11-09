import {trackRuntime} from '../runtime-lifecycle.js';
import {guardStrandsDisposal, retainStrandsResources} from '../strands/disposal-guard.mjs';
import {H2oTvmRunner} from './runner.mjs';
import {observeH2oTeardown, H2O_TEARDOWN_TIMEOUT_MS} from './teardown.mjs';

export type H2oGpu = {
  lost: PromiseLike<unknown>;
  addEventListener?: (
    type: 'uncapturederror',
    listener: (event: {error?: {message?: unknown} | null | undefined}) => void
  ) => void;
};
export type H2oTensor = {
  readonly byteOffset?: unknown;
  copyFrom(value: unknown): unknown;
  getDataPtr(): unknown;
};
export type H2oRuntime = {
  dispose(...args: unknown[]): unknown;
  initWebGPU(gpu: H2oGpu): unknown;
  readonly lib?: {readonly webGPUContext?: unknown} | null | undefined;
  beginScope(): unknown;
  endScope(): unknown;
  webgpu(): unknown;
  createVirtualMachine(device: unknown): {getFunction(name: string): unknown};
  empty(shape: number[], dtype: string, device: unknown): H2oTensor;
};
export type H2oRuntimeOwner<Runtime extends H2oRuntime = H2oRuntime, Gpu extends H2oGpu = H2oGpu> = {
  runtime: Runtime;
  gpu: Gpu;
  device: unknown;
  fn: (...args: unknown[]) => unknown;
  lifecycle: ReturnType<typeof trackRuntime>;
  disposalGuard: ReturnType<typeof guardStrandsDisposal>;
  uncaptured: unknown[];
  teardownObserver: ReturnType<typeof observeH2oTeardown>;
  teardownTimeoutMs: number;
  phase: 'prepared' | 'loading' | 'attached' | 'failed';
  uploadedWeights?: readonly H2oTensor[] | null;
};
export type H2oRunner = {
  run(inputs: unknown[]): Promise<unknown>;
  releaseCachedMemory(): Promise<unknown>;
  dispose(): Promise<unknown>;
};
type H2oRunnerConstructor = new (
  options: Omit<H2oRuntimeOwner, 'uploadedWeights'> & {
    uploadedWeights?: H2oRuntimeOwner['uploadedWeights'];
    weights: readonly H2oTensor[];
  }
) => H2oRunner;
type InitializeH2oRuntimeOptions = {
  teardownTimeoutMs?: number | undefined;
  setupWebGPUContext?: ((context: unknown) => unknown) | undefined;
};

/** Protect the pinned runtime before its WebGPU loss callback is installed. */
export function initializeH2oRuntime<Runtime extends H2oRuntime, Gpu extends H2oGpu>(
  runtime: Runtime,
  gpu: Gpu,
  {teardownTimeoutMs = H2O_TEARDOWN_TIMEOUT_MS, setupWebGPUContext}: InitializeH2oRuntimeOptions = {}
): H2oRuntimeOwner<Runtime, Gpu> {
  let lifecycle: ReturnType<typeof trackRuntime> | undefined,
    disposalGuard: ReturnType<typeof guardStrandsDisposal> | undefined;
  try {
    if (
      !Number.isSafeInteger(teardownTimeoutMs) ||
      teardownTimeoutMs < 1 ||
      teardownTimeoutMs > H2O_TEARDOWN_TIMEOUT_MS
    )
      throw new TypeError('Invalid H2O teardown timeout.');
    if (setupWebGPUContext !== undefined && typeof setupWebGPUContext !== 'function')
      throw new TypeError('Invalid WebGPU context setup callback.');
    const teardownObserver = observeH2oTeardown(gpu);
    lifecycle = trackRuntime(runtime, gpu);
    disposalGuard = guardStrandsDisposal(runtime, gpu, lifecycle);
    if (typeof gpu.addEventListener !== 'function') throw new TypeError('WebGPU error listener unavailable.');
    const uncaptured: unknown[] = [];
    gpu.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
    });
    runtime.initWebGPU(gpu);
    setupWebGPUContext?.(runtime.lib?.webGPUContext);
    runtime.beginScope();
    const device = runtime.webgpu();
    const fn = runtime.createVirtualMachine(device).getFunction('main') as (...args: unknown[]) => unknown;
    if (typeof fn !== 'function') throw new TypeError('Missing H2O VM main.');
    return {
      runtime,
      gpu,
      device,
      fn,
      lifecycle,
      disposalGuard,
      uncaptured,
      teardownObserver,
      teardownTimeoutMs,
      phase: 'prepared'
    };
  } catch (error) {
    // Partial initialization may have registered work or a loss callback.
    if (disposalGuard) disposalGuard.retain();
    else retainStrandsResources(runtime, gpu);
    throw error;
  }
}

/** Transfer an initialized root scope to one runner, once. */
export function attachH2oRunner(
  owner: H2oRuntimeOwner | null | undefined,
  weights: readonly H2oTensor[] | undefined
): H2oRunner {
  if (owner?.phase !== 'prepared' || !owner.uploadedWeights || weights !== owner.uploadedWeights)
    throw new Error('Runtime is not available for H2O.');
  try {
    if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.disposalGuard.blocked || owner.uncaptured.length)
      throw new Error('Runtime failed before H2O attachment.');
    const runner = new (H2oTvmRunner as unknown as H2oRunnerConstructor)({...owner, weights: weights!});
    owner.phase = 'attached';
    owner.uploadedWeights = null;
    return runner;
  } catch (error) {
    owner.phase = 'failed';
    owner.disposalGuard.retain();
    throw error;
  }
}
