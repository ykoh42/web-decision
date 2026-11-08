import {trackRuntime} from '../runtime-lifecycle.js';
import {guardStrandsDisposal, retainStrandsResources} from '../strands/disposal-guard.mjs';
import {H2oTvmRunner} from './runner.mjs';
import {observeH2oTeardown, H2O_TEARDOWN_TIMEOUT_MS} from './teardown.mjs';

/** Protect the pinned runtime before its WebGPU loss callback is installed. */
export function initializeH2oRuntime(
  runtime,
  gpu,
  {teardownTimeoutMs = H2O_TEARDOWN_TIMEOUT_MS, setupWebGPUContext} = {}
) {
  let lifecycle, disposalGuard;
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
    const uncaptured = [];
    gpu.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
    });
    runtime.initWebGPU(gpu);
    setupWebGPUContext?.(runtime.lib?.webGPUContext);
    runtime.beginScope();
    const device = runtime.webgpu();
    const fn = runtime.createVirtualMachine(device).getFunction('main');
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
export function attachH2oRunner(owner, weights) {
  if (owner?.phase !== 'prepared' || !owner.uploadedWeights || weights !== owner.uploadedWeights)
    throw new Error('Runtime is not available for H2O.');
  try {
    if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.disposalGuard.blocked || owner.uncaptured.length)
      throw new Error('Runtime failed before H2O attachment.');
    const runner = new H2oTvmRunner({...owner, weights});
    owner.phase = 'attached';
    owner.uploadedWeights = null;
    return runner;
  } catch (error) {
    owner.phase = 'failed';
    owner.disposalGuard.retain();
    throw error;
  }
}
