import assert from 'node:assert/strict';
import test from 'node:test';
import {KEV08_LARGEST_BUFFER} from '../src/kev08/contract.js';
import {requestKev08Device} from '../src/kev08/device.js';

test('Kev 0.8B requests its required FP16 feature and buffer limits', async () => {
  const limits = {
    maxBufferSize: KEV08_LARGEST_BUFFER,
    maxStorageBufferBindingSize: KEV08_LARGEST_BUFFER,
    maxStorageBuffersPerShaderStage: 8
  };
  const device = {destroy() {}};
  let requested: GPUDeviceDescriptor | undefined;
  const gpu = {
    requestAdapter: async () => ({
      features: new Set(['shader-f16']),
      limits,
      requestDevice: async (descriptor: GPUDeviceDescriptor) => {
        requested = descriptor;
        return device;
      }
    })
  } as unknown as GPU;

  assert.equal(await requestKev08Device(gpu), device);
  assert.deepEqual(requested?.requiredFeatures, ['shader-f16']);
  assert.deepEqual(requested?.requiredLimits, limits);
});

test('Kev 0.8B rejects an adapter lacking shader-f16', async () => {
  const gpu = {
    requestAdapter: async () => ({features: new Set(), limits: {}})
  } as unknown as GPU;
  await assert.rejects(requestKev08Device(gpu), {code: 'UNSUPPORTED_ENVIRONMENT'});
});
