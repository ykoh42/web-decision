const TVM_WORKGROUP_STORAGE_BYTES = 32 * 1024;
const COMPUTE_INVOCATION_CANDIDATES = [1024, 512, 256];

type GPUProvider = Pick<GPU, 'requestAdapter'>;

export async function detectGPUDevice(
  powerPreference: NonNullable<GPURequestAdapterOptions['powerPreference']> = 'high-performance',
  gpu: GPUProvider | undefined = typeof navigator === 'undefined' ? undefined : navigator.gpu
): Promise<{adapter: GPUAdapter; device: GPUDevice} | undefined> {
  if (!gpu) return undefined;

  const adapter = await gpu.requestAdapter({powerPreference});
  if (!adapter) return undefined;

  const limits = adapter.limits;
  if (limits.maxComputeWorkgroupStorageSize < TVM_WORKGROUP_STORAGE_BYTES) {
    throw new RangeError(
      `TVM WebGPU requires ${TVM_WORKGROUP_STORAGE_BYTES} bytes of workgroup storage; this adapter provides ${limits.maxComputeWorkgroupStorageSize}.`
    );
  }

  const maxComputeInvocationsPerWorkgroup = COMPUTE_INVOCATION_CANDIDATES.find(
    (candidate) => candidate <= limits.maxComputeInvocationsPerWorkgroup
  );
  const requiredLimits: GPUDeviceDescriptor['requiredLimits'] = {
    maxBufferSize: limits.maxBufferSize,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: TVM_WORKGROUP_STORAGE_BYTES,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    ...(maxComputeInvocationsPerWorkgroup === undefined ? {} : {maxComputeInvocationsPerWorkgroup})
  };
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('shader-f16')) requiredFeatures.push('shader-f16');
  if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');

  const device = await adapter.requestDevice({requiredFeatures, requiredLimits});
  return {adapter, device};
}
