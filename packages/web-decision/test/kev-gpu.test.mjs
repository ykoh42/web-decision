import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KEV_REQUIRED_LIMITS,
  assertKevAttentionFits,
  detectKevGPUDevice,
  requiredKevAttentionBytes
} from '../src/kev-gpu.js';
import {DecisionModel} from '../src/model.js';
function mockAdapter({limits = KEV_REQUIRED_LIMITS, features = ['shader-f16', 'subgroups'], failure} = {}) {
  let calls = 0,
    request;
  const device = {kind: 'device'},
    adapter = {
      limits,
      features: new Set(features),
      info: {vendor: 'test'},
      async requestDevice(options) {
        calls++;
        request = options;
        if (failure) throw failure;
        return device;
      }
    };
  return {
    adapter,
    device,
    get calls() {
      return calls;
    },
    get request() {
      return request;
    }
  };
}
test('accepts an adapter with the audited limits and does not require optional subgroups', async () => {
  const x = mockAdapter(),
    requests = [];
  const result = await detectKevGPUDevice({
    requestAdapter: async (options) => {
      requests.push(options);
      return x.adapter;
    }
  });
  assert.equal(result.device, x.device);
  assert.deepEqual(requests, [{powerPreference: 'high-performance'}]);
  assert.deepEqual(x.request.requiredLimits, KEV_REQUIRED_LIMITS);
  assert.deepEqual(x.request.requiredFeatures, ['shader-f16']);
});
test('requests larger adapter buffer limits while keeping fixed compute requirements', async () => {
  const limits = {
      ...KEV_REQUIRED_LIMITS,
      maxBufferSize: KEV_REQUIRED_LIMITS.maxBufferSize * 2,
      maxStorageBufferBindingSize: KEV_REQUIRED_LIMITS.maxStorageBufferBindingSize * 2
    },
    x = mockAdapter({limits});
  const result = await detectKevGPUDevice({requestAdapter: async () => x.adapter});

  assert.equal(result.device, x.device);
  assert.deepEqual(x.request.requiredLimits, limits);
  assert.deepEqual(x.request.requiredFeatures, ['shader-f16']);
  assert.equal(
    x.request.requiredLimits.maxComputeWorkgroupStorageSize,
    KEV_REQUIRED_LIMITS.maxComputeWorkgroupStorageSize
  );
  assert.equal(
    x.request.requiredLimits.maxStorageBuffersPerShaderStage,
    KEV_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage
  );
  assert.equal(
    x.request.requiredLimits.maxComputeInvocationsPerWorkgroup,
    KEV_REQUIRED_LIMITS.maxComputeInvocationsPerWorkgroup
  );
  assert.equal(x.request.requiredLimits.maxComputeWorkgroupSizeX, KEV_REQUIRED_LIMITS.maxComputeWorkgroupSizeX);
  assert.equal(x.request.requiredLimits.maxComputeWorkgroupSizeY, KEV_REQUIRED_LIMITS.maxComputeWorkgroupSizeY);
  assert.equal(x.request.requiredLimits.maxComputeWorkgroupSizeZ, KEV_REQUIRED_LIMITS.maxComputeWorkgroupSizeZ);
  assert.equal(
    x.request.requiredLimits.maxComputeWorkgroupsPerDimension,
    KEV_REQUIRED_LIMITS.maxComputeWorkgroupsPerDimension
  );
});
test('reports missing f16 without creating a device', async () => {
  const x = mockAdapter({features: []});
  const result = await detectKevGPUDevice({requestAdapter: async () => x.adapter});
  assert.equal(result.adapter, x.adapter);
  assert.equal(result.device, undefined);
  assert.equal(x.calls, 0);
});
test('rejects an insufficient limit before creating a device', async () => {
  const x = mockAdapter({limits: {...KEV_REQUIRED_LIMITS, maxStorageBuffersPerShaderStage: 7}});
  await assert.rejects(
    detectKevGPUDevice({requestAdapter: async () => x.adapter}),
    /maxStorageBuffersPerShaderStage requires 8/
  );
  assert.equal(x.calls, 0);
});
test('handles unavailable adapter and preserves requestDevice errors', async () => {
  assert.equal(await detectKevGPUDevice(null), undefined);
  await assert.rejects(detectKevGPUDevice({requestAdapter: async () => null}), /No compatible/);
  const failure = Error('device denied'),
    x = mockAdapter({failure});
  await assert.rejects(detectKevGPUDevice({requestAdapter: async () => x.adapter}), (e) => e === failure);
});

test('calculates the exact FP16 attention buffer at the native sequence boundary', () => {
  assert.equal(requiredKevAttentionBytes(8191), 2_146_959_392n);
  assert.equal(requiredKevAttentionBytes(8192), 2_147_483_648n);
  assert.equal(requiredKevAttentionBytes(0), 0n);
});

test('reports a device buffer limit without imposing a model token cap', () => {
  const limits = {
    maxBufferSize: 2_147_483_644,
    maxStorageBufferBindingSize: 2_147_483_644
  };
  assert.doesNotThrow(() => assertKevAttentionFits(8191, limits));
  assert.throws(() => assertKevAttentionFits(8192, limits), {
    code: 'DEVICE_RESOURCE_LIMIT',
    message:
      'This request needs a 2,147,483,648-byte Kev attention buffer, but this WebGPU device allows 2,147,483,644 bytes per buffer. No input was truncated.'
  });
  assert.doesNotThrow(() =>
    assertKevAttentionFits(8192, {
      maxBufferSize: 4_294_967_292,
      maxStorageBufferBindingSize: 4_294_967_292
    })
  );
});

test('checks device capacity before starting the TVM graph', async () => {
  let runtimeCalls = 0,
    graphCalls = 0;
  const model = new DecisionModel(
    {
      beginScope() {
        runtimeCalls++;
      }
    },
    {},
    {
      limits: {
        maxBufferSize: 2_147_483_644,
        maxStorageBufferBindingSize: 2_147_483_644
      }
    },
    () => {
      graphCalls++;
    },
    [],
    {},
    {},
    [],
    {loss: false, disposed: false}
  );

  await assert.rejects(model.run({ids: new Array(8192)}), {code: 'DEVICE_RESOURCE_LIMIT'});
  assert.equal(runtimeCalls, 0);
  assert.equal(graphCalls, 0);
});

test('pre-aborted requests preserve the reason without requesting an adapter', async () => {
  const controller = new AbortController(),
    reason = Error('cancelled');
  controller.abort(reason);
  let called = false;
  await assert.rejects(
    detectKevGPUDevice(
      {
        requestAdapter() {
          called = true;
        }
      },
      controller.signal
    ),
    (e) => e === reason
  );
  assert.equal(called, false);
  await assert.rejects(detectKevGPUDevice(null, controller.signal), (e) => e === reason);
});

test('pending adapter cancellation returns immediately and observes late rejection', {timeout: 1000}, async () => {
  const controller = new AbortController(),
    reason = Error('cancel adapter');
  let rejectAdapter;
  const request = detectKevGPUDevice(
    {
      requestAdapter: () =>
        new Promise((_, reject) => {
          rejectAdapter = reject;
        })
    },
    controller.signal
  );
  controller.abort(reason);
  await assert.rejects(request, (e) => e === reason);
  rejectAdapter(Error('late adapter failure'));
  await new Promise((resolve) => setImmediate(resolve));
});

test('pending device cancellation returns immediately and destroys the late device once', {timeout: 1000}, async () => {
  const controller = new AbortController(),
    reason = Error('cancel device');
  let resolveDevice,
    destroyCount = 0;
  const x = mockAdapter();
  x.adapter.requestDevice = () =>
    new Promise((resolve) => {
      resolveDevice = resolve;
    });
  const request = detectKevGPUDevice({requestAdapter: async () => x.adapter}, controller.signal);
  while (!resolveDevice) await new Promise((resolve) => setImmediate(resolve));
  controller.abort(reason);
  await assert.rejects(request, (e) => e === reason);
  resolveDevice({
    destroy() {
      destroyCount++;
    }
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(destroyCount, 1);
});

test('pending device cancellation observes a later rejection', {timeout: 1000}, async () => {
  const controller = new AbortController(),
    reason = Error('cancel device');
  let rejectDevice;
  const x = mockAdapter();
  x.adapter.requestDevice = () =>
    new Promise((_, reject) => {
      rejectDevice = reject;
    });
  const request = detectKevGPUDevice({requestAdapter: async () => x.adapter}, controller.signal);
  while (!rejectDevice) await new Promise((resolve) => setImmediate(resolve));
  controller.abort(reason);
  await assert.rejects(request, (e) => e === reason);
  rejectDevice(Error('late device failure'));
  await new Promise((resolve) => setImmediate(resolve));
});

test('an adapter arriving after cancellation is not used to request a device', {timeout: 1000}, async () => {
  const controller = new AbortController(),
    reason = Error('cancel adapter');
  let resolveAdapter;
  const x = mockAdapter();
  const request = detectKevGPUDevice(
    {
      requestAdapter: () =>
        new Promise((resolve) => {
          resolveAdapter = resolve;
        })
    },
    controller.signal
  );
  controller.abort(reason);
  await assert.rejects(request, (e) => e === reason);
  resolveAdapter(x.adapter);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(x.calls, 0);
});

test('a late device destroy failure stays observed after cancellation', {timeout: 1000}, async () => {
  const controller = new AbortController(),
    reason = Error('cancel device');
  let resolveDevice,
    destroyCount = 0;
  const x = mockAdapter();
  x.adapter.requestDevice = () =>
    new Promise((resolve) => {
      resolveDevice = resolve;
    });
  const request = detectKevGPUDevice({requestAdapter: async () => x.adapter}, controller.signal);
  while (!resolveDevice) await new Promise((resolve) => setImmediate(resolve));
  controller.abort(reason);
  await assert.rejects(request, (e) => e === reason);
  resolveDevice({
    destroy() {
      destroyCount++;
      throw Error('destroy failed');
    }
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(destroyCount, 1);
});

test('cancellation between device helper completion and detector continuation destroys the device once', async () => {
  const controller = new AbortController(),
    reason = Error('cancel device handoff');
  let destroyCount = 0;
  const device = {
      destroy() {
        destroyCount++;
      }
    },
    x = mockAdapter();
  x.adapter.requestDevice = () => {
    queueMicrotask(() => queueMicrotask(() => queueMicrotask(() => controller.abort(reason))));
    return Promise.resolve(device);
  };
  await assert.rejects(
    detectKevGPUDevice({requestAdapter: async () => x.adapter}, controller.signal),
    (e) => e === reason
  );
  assert.equal(destroyCount, 1);
});

test('device ownership is released if constructing the result fails', async () => {
  const failure = Error('adapter info failed'),
    x = mockAdapter();
  let destroyCount = 0;
  x.adapter.requestDevice = async () => ({
    destroy() {
      destroyCount++;
    }
  });
  Object.defineProperty(x.adapter, 'info', {
    get() {
      throw failure;
    }
  });
  await assert.rejects(detectKevGPUDevice({requestAdapter: async () => x.adapter}), (e) => e === failure);
  assert.equal(destroyCount, 1);
});
