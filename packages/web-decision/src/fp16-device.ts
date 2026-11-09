import {waitForAbort} from './assets.js';
import {WebDecisionError} from './kev.js';
import {retainStrandsResources} from './strands/disposal-guard.mjs';

function destroyDevice(device: GPUDevice | null | undefined): void {
  if (!device) return;
  try {
    device.destroy();
  } catch {
    retainStrandsResources(undefined, device);
  }
}

/** Request a shader-f16 device with the limits required by a pinned graph. */
export async function requestFp16Device(
  gpu: GPU,
  signal: AbortSignal | null | undefined,
  modelName: string,
  requiredLimits: Record<string, number>
): Promise<GPUDevice> {
  let device: GPUDevice | undefined;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => gpu.requestAdapter({powerPreference: 'high-performance'}), signal);
    signal?.throwIfAborted();
    if (!adapter) throw new Error('No compatible WebGPU adapter was found.');
    if (!adapter.features.has('shader-f16'))
      throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'This model requires WebGPU shader-f16 support.');
    for (const [name, needed] of Object.entries(requiredLimits)) {
      const available = (adapter.limits as unknown as Record<string, unknown>)[name];
      if (typeof available !== 'number' || !Number.isFinite(available) || available < needed)
        throw new WebDecisionError(
          'UNSUPPORTED_ENVIRONMENT',
          `${modelName} needs WebGPU ${name} of at least ${needed}.`
        );
    }
    // requestDevice cannot be aborted; destroy a device returned after cancellation.
    device = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits}),
      signal,
      destroyDevice
    );
    signal?.throwIfAborted();
    return device;
  } catch (cause) {
    destroyDevice(device);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', `Could not initialize a compatible ${modelName} WebGPU device.`, {
      cause
    });
  }
}
