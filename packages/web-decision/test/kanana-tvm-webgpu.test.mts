import assert from 'node:assert/strict';
import test from 'node:test';
import {detectGPUDevice} from '../src/kanana/tvm-webgpu.ts';

function makeGPU(limits: Record<string, number>, supportedFeatures: string[] = []) {
  const descriptors: GPUDeviceDescriptor[] = [];
  const device = {limits} as unknown as GPUDevice;
  const adapter = {
    limits,
    features: new Set(supportedFeatures),
    async requestDevice(descriptor: GPUDeviceDescriptor) {
      descriptors.push(descriptor);
      return device;
    }
  } as unknown as GPUAdapter;
  const options: GPURequestAdapterOptions[] = [];
  const gpu = {
    async requestAdapter(requestOptions?: GPURequestAdapterOptions) {
      options.push(requestOptions ?? {});
      return adapter;
    }
  } as unknown as GPU;

  return {descriptors, device, gpu, options};
}

test('requests the adapter-supported TVM limits and features through WebGPU', async () => {
  const limits = {
    maxBufferSize: 1_073_741_824,
    maxStorageBufferBindingSize: 805_306_368,
    maxComputeWorkgroupStorageSize: 32_768,
    maxStorageBuffersPerShaderStage: 9,
    maxComputeInvocationsPerWorkgroup: 512
  };
  const {descriptors, device, gpu, options} = makeGPU(limits, ['shader-f16', 'subgroups']);

  const detected = await detectGPUDevice('high-performance', gpu);

  assert.equal(detected?.device, device);
  assert.deepEqual(options, [{powerPreference: 'high-performance'}]);
  assert.deepEqual(descriptors, [
    {
      requiredFeatures: ['shader-f16', 'subgroups'],
      requiredLimits: {
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: 32_768,
        maxStorageBuffersPerShaderStage: 9,
        maxComputeInvocationsPerWorkgroup: 512
      }
    }
  ]);
});

test('returns undefined when the browser has no compatible adapter', async () => {
  const gpu = {requestAdapter: async () => null} as unknown as GPU;

  assert.equal(await detectGPUDevice('high-performance', gpu), undefined);
});

test('reports an adapter workgroup-storage limit below TVM requirements', async () => {
  const {gpu} = makeGPU({
    maxBufferSize: 1_073_741_824,
    maxStorageBufferBindingSize: 805_306_368,
    maxComputeWorkgroupStorageSize: 16_384,
    maxStorageBuffersPerShaderStage: 9,
    maxComputeInvocationsPerWorkgroup: 512
  });

  await assert.rejects(detectGPUDevice('high-performance', gpu), (error: unknown) => {
    assert.ok(error instanceof RangeError);
    assert.match(error.message, /32768 bytes of workgroup storage/u);
    return true;
  });
});
