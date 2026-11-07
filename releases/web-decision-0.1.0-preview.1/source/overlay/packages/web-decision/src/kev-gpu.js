import {destroyGpuDevice, requestGpuAdapter, requestGpuDevice} from './abortable-gpu-device.js';

/** Experimental limits for the pinned Kev 0.6B Q4 WebGPU build only. */
export const KEV_REQUIRED_LIMITS = Object.freeze({
  maxBufferSize: 256 * 1024 * 1024,
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupStorageSize: 16 * 1024,
  maxStorageBuffersPerShaderStage: 8,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 8,
  maxComputeWorkgroupSizeZ: 1,
  maxComputeWorkgroupsPerDimension: 65535
});

/** Return an adapter/device pair compatible with the current loader's interface. */
export async function detectKevGPUDevice(gpu = globalThis.navigator?.gpu, signal) {
  signal?.throwIfAborted();
  if (!gpu) return undefined;
  const adapter = await requestGpuAdapter(gpu, signal);
  signal?.throwIfAborted();
  if (!adapter) throw new Error('No compatible WebGPU adapter was found.');
  // Let the caller emit its existing UNSUPPORTED_ENVIRONMENT diagnostic
  // without requesting a device that could never run this shader-f16 model.
  if (!adapter.features.has('shader-f16')) {
    signal?.throwIfAborted();
    return {adapter, device: undefined, adapterInfo: adapter.info};
  }
  for (const [name, required] of Object.entries(KEV_REQUIRED_LIMITS)) {
    const available = adapter.limits[name];
    if (typeof available !== 'number' || !Number.isFinite(available) || available < required)
      throw new Error(`WebGPU ${name} requires ${required}; adapter reports ${available}.`);
  }
  // Dynamic sequence buffers can exceed the model's minimum device limits.
  const requiredLimits = {
    ...KEV_REQUIRED_LIMITS,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
  };
  const device = await requestGpuDevice(
    adapter,
    {
      requiredLimits,
      requiredFeatures: ['shader-f16']
    },
    signal
  );
  try {
    signal?.throwIfAborted();
    return {adapter, device, adapterInfo: adapter.info};
  } catch (cause) {
    destroyGpuDevice(device);
    signal?.throwIfAborted();
    throw cause;
  }
}
