import {destroyGpuDevice, requestGpuAdapter, requestGpuDevice} from './abortable-gpu-device.js';
import {WebDecisionError} from './kev.js';

const KEV_ATTENTION_HEADS = 16n;
const KEV_ATTENTION_SCORE_BYTES = 2n;

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

export function requiredKevAttentionBytes(sequenceLength) {
  if (!Number.isSafeInteger(sequenceLength) || sequenceLength < 0)
    throw new RangeError('sequenceLength must be a non-negative safe integer.');
  const tokens = BigInt(sequenceLength);
  return tokens * tokens * KEV_ATTENTION_HEADS * KEV_ATTENTION_SCORE_BYTES;
}

export function assertKevAttentionFits(sequenceLength, limits) {
  const required = requiredKevAttentionBytes(sequenceLength);
  const maxBufferSize = limits?.maxBufferSize;
  const maxStorageBufferBindingSize = limits?.maxStorageBufferBindingSize;
  if (
    !Number.isSafeInteger(maxBufferSize) ||
    !Number.isSafeInteger(maxStorageBufferBindingSize) ||
    maxBufferSize < 0 ||
    maxStorageBufferBindingSize < 0
  )
    return;

  const limit = BigInt(Math.min(maxBufferSize, maxStorageBufferBindingSize));
  if (required > limit) {
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `This request needs a ${required.toLocaleString('en-US')}-byte Kev attention buffer, but this WebGPU device allows ${limit.toLocaleString('en-US')} bytes per buffer. No input was truncated.`
    );
  }
}

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
