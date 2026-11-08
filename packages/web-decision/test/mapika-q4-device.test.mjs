import test from 'node:test';
import assert from 'node:assert/strict';
import {MAPIKA_Q4_LARGEST_BUFFER} from '../src/mapika-q4/contract.js';
import {requestMapikaQ4Device} from '../src/mapika-q4/device.js';

function fixture() {
  const state = {destroyed: 0, requests: 0};
  const device = {
    destroy() {
      state.destroyed++;
    }
  };
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {
      maxBufferSize: MAPIKA_Q4_LARGEST_BUFFER,
      maxStorageBufferBindingSize: MAPIKA_Q4_LARGEST_BUFFER,
      maxStorageBuffersPerShaderStage: 8
    },
    async requestDevice(descriptor) {
      state.requests++;
      state.descriptor = descriptor;
      return device;
    }
  };
  return {state, device, adapter, gpu: {requestAdapter: async () => adapter}};
}

test('requests the packed embedding limit and shader-f16', async () => {
  const {state, device, gpu, adapter} = fixture();
  assert.equal(await requestMapikaQ4Device(gpu), device);
  assert.deepEqual(state.descriptor.requiredFeatures, ['shader-f16']);
  assert.deepEqual(state.descriptor.requiredLimits, adapter.limits);
  assert.ok(MAPIKA_Q4_LARGEST_BUFFER < 128 * 1024 * 1024);
});

test('rejects missing shader-f16 or insufficient limits before requesting a device', async () => {
  for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxStorageBuffersPerShaderStage']) {
    const {state, adapter, gpu} = fixture();
    adapter.limits[name]--;
    await assert.rejects(requestMapikaQ4Device(gpu), {code: 'UNSUPPORTED_ENVIRONMENT', message: new RegExp(name)});
    assert.equal(state.requests, 0);
  }
  const {state, adapter, gpu} = fixture();
  adapter.features.clear();
  await assert.rejects(requestMapikaQ4Device(gpu), {code: 'UNSUPPORTED_ENVIRONMENT'});
  assert.equal(state.requests, 0);
});

test('destroys a device that arrives after cancellation', async () => {
  const {state, adapter, gpu, device} = fixture(),
    controller = new AbortController(),
    reason = Error('cancel');
  let resolveDevice;
  adapter.requestDevice = () =>
    new Promise((resolve) => {
      resolveDevice = resolve;
    });
  const promise = requestMapikaQ4Device(gpu, controller.signal);
  while (!resolveDevice) await new Promise((resolve) => setImmediate(resolve));
  controller.abort(reason);
  resolveDevice(device);
  await assert.rejects(promise, (error) => error === reason);
  assert.equal(state.destroyed, 1);
});
