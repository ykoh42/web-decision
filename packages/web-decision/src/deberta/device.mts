import {destroyGpuDevice, requestGpuAdapter, requestGpuDevice} from '../abortable-gpu-device.js';

/** Conservative WebGPU limits for the pinned DeBERTa Q4F16 split build only. */
export const DEBERTA_REQUIRED_LIMITS = Object.freeze({
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

/** This does not guarantee sufficient memory or correctness on an untested GPU. */
export async function detectDebertaGPUDevice(
  gpu: GPU | null | undefined = globalThis.navigator?.gpu,
  signal?: AbortSignal
) {
  signal?.throwIfAborted();
  if (!gpu) return undefined;
  const adapter = await requestGpuAdapter(gpu, signal);
  signal?.throwIfAborted();
  if (!adapter) throw new Error('No compatible WebGPU adapter was found.');
  // The caller produces the established unsupported-environment error.
  if (!adapter.features.has('shader-f16')) return {adapter, device: undefined, adapterInfo: adapter.info};
  for (const [name, required] of Object.entries(DEBERTA_REQUIRED_LIMITS)) {
    const available = (adapter.limits as unknown as Record<string, unknown>)[name];
    if (typeof available !== 'number' || !Number.isFinite(available) || available < required) {
      throw new Error(`WebGPU ${name} requires ${required}; adapter reports ${available}.`);
    }
  }
  const device = await requestGpuDevice(
    adapter,
    {requiredLimits: {...DEBERTA_REQUIRED_LIMITS}, requiredFeatures: ['shader-f16']} as GPUDeviceDescriptor,
    signal
  );
  try {
    signal?.throwIfAborted();
    return {adapter, device, adapterInfo: adapter.info};
  } catch (cause) {
    destroyGpuDevice(device);
    throw cause;
  }
}
