import test from 'node:test';
import assert from 'node:assert/strict';
import {MAPIKA_LARGEST_BUFFER} from '../src/mapika/contract.js';
import {requestMapikaDevice} from '../src/mapika/device.js';

type FakeAdapter = {
  features: Set<string>;
  limits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxStorageBuffersPerShaderStage: number;
  };
  requestDevice(descriptor: GPUDeviceDescriptor): Promise<GPUDevice | null>;
};
type AdapterFixture = {
  destroyed: number;
  calls: number;
  device: GPUDevice;
  descriptor?: GPUDeviceDescriptor | undefined;
  value: FakeAdapter;
  gpu: GPU;
};

function adapter() {
  const result = {destroyed: 0, calls: 0} as AdapterFixture;
  result.device = {
    destroy() {
      result.destroyed++;
    }
  } as unknown as GPUDevice;
  result.value = {
    features: new Set(['shader-f16']),
    limits: {
      maxBufferSize: MAPIKA_LARGEST_BUFFER,
      maxStorageBufferBindingSize: MAPIKA_LARGEST_BUFFER,
      maxStorageBuffersPerShaderStage: 8
    },
    async requestDevice(descriptor) {
      result.calls++;
      result.descriptor = descriptor;
      return result.device;
    }
  } as FakeAdapter;
  result.gpu = {requestAdapter: async () => result.value} as unknown as GPU;
  return result;
}

test('requests the large FP16 buffer limits and shader-f16', async () => {
  const value = adapter();
  assert.equal(await requestMapikaDevice(value.gpu), value.device);
  assert.deepEqual(value.descriptor!.requiredFeatures, ['shader-f16']);
  assert.deepEqual(value.descriptor!.requiredLimits, value.value.limits);
});

test('rejects missing features and insufficient adapter limits before creating a device', async () => {
  for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxStorageBuffersPerShaderStage'] as const) {
    const value = adapter();
    value.value.limits[name]--;
    await assert.rejects(requestMapikaDevice(value.gpu), {code: 'UNSUPPORTED_ENVIRONMENT', message: new RegExp(name)});
    assert.equal(value.calls, 0);
  }
  const value = adapter();
  value.value.features.clear();
  await assert.rejects(requestMapikaDevice(value.gpu), {code: 'UNSUPPORTED_ENVIRONMENT'});
  assert.equal(value.calls, 0);
});

test('preserves cancellation while requestDevice is pending and destroys the late device', async () => {
  const value = adapter(),
    controller = new AbortController(),
    reason = Error('stop');
  let resolveDevice: ((device: GPUDevice | PromiseLike<GPUDevice>) => void) | undefined;
  value.value.requestDevice = () =>
    new Promise((resolve) => {
      resolveDevice = resolve;
    });
  const result = requestMapikaDevice(value.gpu, controller.signal);
  while (!resolveDevice) await new Promise<void>((resolve) => setImmediate(resolve));
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  assert.equal(value.destroyed, 0);
  resolveDevice!(value.device);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(value.destroyed, 1);
});

test('observes late requestDevice rejection after cancellation', async () => {
  const value = adapter(),
    controller = new AbortController(),
    reason = Error('stop');
  let rejectDevice: ((reason?: unknown) => void) | undefined;
  value.value.requestDevice = () => new Promise((_, reject) => (rejectDevice = reject));
  const result = requestMapikaDevice(value.gpu, controller.signal);
  while (!rejectDevice) await new Promise<void>((resolve) => setImmediate(resolve));
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  rejectDevice!(Error('late device failure'));
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(value.destroyed, 0);
});

test('preserves cancellation if destroying a late device throws', async () => {
  const value = adapter(),
    controller = new AbortController(),
    reason = Error('stop');
  let resolveDevice: ((device: GPUDevice | PromiseLike<GPUDevice>) => void) | undefined,
    destroys = 0;
  value.value.requestDevice = () => new Promise((resolve) => (resolveDevice = resolve));
  const result = requestMapikaDevice(value.gpu, controller.signal);
  while (!resolveDevice) await new Promise<void>((resolve) => setImmediate(resolve));
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  resolveDevice!({
    destroy() {
      destroys++;
      throw Error('destroy failed');
    }
  } as unknown as GPUDevice);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(destroys, 1);
});

test('destroys exactly once if cancellation follows device resolution', async () => {
  const value = adapter(),
    controller = new AbortController(),
    reason = Error('stop');
  let resolveDevice: ((device: GPUDevice | PromiseLike<GPUDevice>) => void) | undefined;
  value.value.requestDevice = () => new Promise((resolve) => (resolveDevice = resolve));
  const result = requestMapikaDevice(value.gpu, controller.signal);
  while (!resolveDevice) await new Promise<void>((resolve) => setImmediate(resolve));
  resolveDevice!(value.device);
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(result, (error) => error === reason);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(value.destroyed, 1);
});

test('preserves a synchronous requestDevice failure', async () => {
  const value = adapter(),
    failure = Error('request threw');
  value.value.requestDevice = () => {
    throw failure;
  };
  await assert.rejects(
    requestMapikaDevice(value.gpu, new AbortController().signal),
    (error) =>
      (error as {code?: unknown; cause?: unknown}).code === 'MODEL_LOAD_FAILED' &&
      (error as {code?: unknown; cause?: unknown}).cause === failure
  );
});

test('reports unavailable adapters and retains the cause of device initialization failures', async () => {
  await assert.rejects(requestMapikaDevice({requestAdapter: async () => null} as unknown as GPU), {
    code: 'MODEL_LOAD_FAILED'
  });
  const value = adapter(),
    failure = Error('request denied');
  value.value.requestDevice = async () => {
    throw failure;
  };
  await assert.rejects(
    requestMapikaDevice(value.gpu),
    (error) =>
      (error as {code?: unknown; cause?: unknown}).code === 'MODEL_LOAD_FAILED' &&
      (error as {code?: unknown; cause?: unknown}).cause === failure
  );
});
