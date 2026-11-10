import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KEV_REQUIRED_LIMITS,
  assertKevAttentionFits,
  detectKevGPUDevice,
  requiredKevAttentionBytes
} from '../src/kev-gpu.js';
import {DecisionModel} from '../src/model.js';
interface TestLimits {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupStorageSize: number;
  maxStorageBuffersPerShaderStage: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeWorkgroupsPerDimension: number;
}
interface TestDevice {
  kind?: string;
  destroy?: () => void;
}
interface TestDeviceDescriptor {
  requiredLimits: TestLimits;
  requiredFeatures: string[];
}
interface TestAdapter {
  limits: TestLimits;
  features: Set<string>;
  info: {vendor: string};
  requestDevice(options: TestDeviceDescriptor): Promise<TestDevice>;
}
interface TestAdapterOptions {
  limits?: TestLimits;
  features?: string[];
  failure?: unknown;
}

const DEFAULT_ADAPTER_LIMITS: TestLimits = {
  ...KEV_REQUIRED_LIMITS,
  maxBufferSize: 256 * 1024 * 1024,
  maxStorageBufferBindingSize: 128 * 1024 * 1024
};

function mockAdapter({
  limits = DEFAULT_ADAPTER_LIMITS,
  features = ['shader-f16', 'subgroups'],
  failure
}: TestAdapterOptions = {}) {
  let calls = 0,
    request: TestDeviceDescriptor | undefined;
  const device: TestDevice = {kind: 'device'},
    adapter: TestAdapter = {
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
    requests: Array<GPURequestAdapterOptions | undefined> = [];
  const result = await detectKevGPUDevice({
    requestAdapter: async (options: GPURequestAdapterOptions | undefined) => {
      requests.push(options);
      return x.adapter;
    }
  } as unknown as GPU);
  assert.equal(result!.device, x.device);
  assert.deepEqual(requests, [{powerPreference: 'high-performance'}]);
  assert.deepEqual(x.request!.requiredLimits, {
    ...KEV_REQUIRED_LIMITS,
    maxBufferSize: x.adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: x.adapter.limits.maxStorageBufferBindingSize
  });
  assert.equal(x.request!.requiredLimits.maxComputeWorkgroupSizeX, 256);
  assert.deepEqual(x.request!.requiredFeatures, ['shader-f16']);
});
test('requests adapter buffer maxima instead of the WebGPU device defaults', async () => {
  const limits = {
      ...DEFAULT_ADAPTER_LIMITS,
      maxBufferSize: 2 * 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 2 * 1024 * 1024 * 1024,
      maxComputeWorkgroupStorageSize: KEV_REQUIRED_LIMITS.maxComputeWorkgroupStorageSize * 2,
      maxStorageBuffersPerShaderStage: KEV_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage * 2,
      maxComputeInvocationsPerWorkgroup: KEV_REQUIRED_LIMITS.maxComputeInvocationsPerWorkgroup * 2,
      maxComputeWorkgroupSizeX: KEV_REQUIRED_LIMITS.maxComputeWorkgroupSizeX * 2,
      maxComputeWorkgroupSizeY: KEV_REQUIRED_LIMITS.maxComputeWorkgroupSizeY * 2,
      maxComputeWorkgroupSizeZ: KEV_REQUIRED_LIMITS.maxComputeWorkgroupSizeZ * 2,
      maxComputeWorkgroupsPerDimension: KEV_REQUIRED_LIMITS.maxComputeWorkgroupsPerDimension * 2
    },
    x = mockAdapter({limits});
  const result = await detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU);

  assert.equal(result!.device, x.device);
  assert.deepEqual(x.request!.requiredLimits, {
    ...KEV_REQUIRED_LIMITS,
    maxBufferSize: limits.maxBufferSize,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize
  });
  assert.deepEqual(x.request!.requiredFeatures, ['shader-f16']);
});
test('reports missing f16 without creating a device', async () => {
  const x = mockAdapter({features: []});
  const result = await detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU);
  assert.equal(result!.adapter, x.adapter);
  assert.equal(result!.device, undefined);
  assert.equal(x.calls, 0);
});
test('rejects an insufficient compute limit before creating a device', async () => {
  const x = mockAdapter({limits: {...DEFAULT_ADAPTER_LIMITS, maxStorageBuffersPerShaderStage: 7}});
  await assert.rejects(
    detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU),
    /maxStorageBuffersPerShaderStage requires 8/
  );
  assert.equal(x.calls, 0);
});
test('keeps small-input support when adapter storage bindings are below 256 MiB', async () => {
  const x = mockAdapter(),
    result = await detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU);
  assert.equal(result!.device, x.device);
  assert.equal(x.request!.requiredLimits.maxStorageBufferBindingSize, 128 * 1024 * 1024);
  assert.doesNotThrow(() => assertKevAttentionFits(2048, x.adapter.limits));
  assert.throws(() => assertKevAttentionFits(2049, x.adapter.limits), {code: 'DEVICE_RESOURCE_LIMIT'});
});
test('handles unavailable adapter and preserves requestDevice errors', async () => {
  assert.equal(await detectKevGPUDevice(null), undefined);
  await assert.rejects(detectKevGPUDevice({requestAdapter: async () => null} as unknown as GPU), /No compatible/);
  const failure = Error('device denied'),
    x = mockAdapter({failure});
  await assert.rejects(
    detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU),
    (e) => e === failure
  );
});

test('preserves dense-score memory requirements for the legacy artifact', () => {
  assert.equal(requiredKevAttentionBytes(8191), 2_146_959_392n);
  assert.equal(requiredKevAttentionBytes(8192), 2_147_483_648n);
  assert.equal(requiredKevAttentionBytes(0), 0n);
});

test('uses linear FP32 scratch and a 128-lane workgroup for the direct TVM kernel', async () => {
  const x = mockAdapter({
    limits: {
      ...DEFAULT_ADAPTER_LIMITS,
      maxBufferSize: 2 * 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 2 * 1024 * 1024 * 1024
    }
  });
  await detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU, undefined, 'linear-scratch');

  assert.equal(x.request!.requiredLimits.maxComputeWorkgroupSizeX, 128);
  assert.equal(requiredKevAttentionBytes(8192, 'linear-scratch'), 67_108_864n);
  assert.doesNotThrow(() => assertKevAttentionFits(8192, x.request!.requiredLimits, 'linear-scratch'));
});

test('reports per-buffer capacity without truncating either attention profile', () => {
  const limits = {
    maxBufferSize: 268_435_455,
    maxStorageBufferBindingSize: 268_435_455
  };
  assert.doesNotThrow(() => assertKevAttentionFits(32_767, limits, 'linear-scratch'));
  assert.throws(() => assertKevAttentionFits(32_768, limits, 'linear-scratch'), {
    code: 'DEVICE_RESOURCE_LIMIT'
  });
  assert.doesNotThrow(() =>
    assertKevAttentionFits(
      32_768,
      {
        maxBufferSize: 268_435_456,
        maxStorageBufferBindingSize: 268_435_456
      },
      'linear-scratch'
    )
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
    } as unknown as ConstructorParameters<typeof DecisionModel>[0],
    {} as ConstructorParameters<typeof DecisionModel>[1],
    {
      limits: {
        maxBufferSize: 268_435_455,
        maxStorageBufferBindingSize: 268_435_455
      }
    } as unknown as ConstructorParameters<typeof DecisionModel>[2],
    () => {
      graphCalls++;
    },
    [] as ConstructorParameters<typeof DecisionModel>[4],
    {} as ConstructorParameters<typeof DecisionModel>[5],
    {} as ConstructorParameters<typeof DecisionModel>[6],
    [] as ConstructorParameters<typeof DecisionModel>[7],
    {loss: false, disposed: false} as unknown as ConstructorParameters<typeof DecisionModel>[8]
  );

  // oxlint-disable-next-line unicorn/no-new-array -- Keep the native sparse-array boundary fixture intact.
  await assert.rejects(model.run({ids: new Array(32768)} as unknown as Parameters<typeof model.run>[0]), {
    code: 'DEVICE_RESOURCE_LIMIT'
  });
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
      } as unknown as GPU,
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
  let rejectAdapter!: (reason?: unknown) => void;
  const request = detectKevGPUDevice(
    {
      requestAdapter: () =>
        new Promise<TestAdapter>((_, reject) => {
          rejectAdapter = reject;
        })
    } as unknown as GPU,
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
  let resolveDevice!: (device: TestDevice) => void,
    destroyCount = 0;
  const x = mockAdapter();
  x.adapter.requestDevice = () =>
    new Promise<TestDevice>((resolve) => {
      resolveDevice = resolve;
    });
  const request = detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU, controller.signal);
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
  let rejectDevice!: (reason?: unknown) => void;
  const x = mockAdapter();
  x.adapter.requestDevice = () =>
    new Promise<TestDevice>((_, reject) => {
      rejectDevice = reject;
    });
  const request = detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU, controller.signal);
  while (!rejectDevice) await new Promise((resolve) => setImmediate(resolve));
  controller.abort(reason);
  await assert.rejects(request, (e) => e === reason);
  rejectDevice(Error('late device failure'));
  await new Promise((resolve) => setImmediate(resolve));
});

test('an adapter arriving after cancellation is not used to request a device', {timeout: 1000}, async () => {
  const controller = new AbortController(),
    reason = Error('cancel adapter');
  let resolveAdapter!: (adapter: TestAdapter) => void;
  const x = mockAdapter();
  const request = detectKevGPUDevice(
    {
      requestAdapter: () =>
        new Promise<TestAdapter>((resolve) => {
          resolveAdapter = resolve;
        })
    } as unknown as GPU,
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
  let resolveDevice!: (device: TestDevice) => void,
    destroyCount = 0;
  const x = mockAdapter();
  x.adapter.requestDevice = () =>
    new Promise<TestDevice>((resolve) => {
      resolveDevice = resolve;
    });
  const request = detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU, controller.signal);
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
    detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU, controller.signal),
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
  await assert.rejects(
    detectKevGPUDevice({requestAdapter: async () => x.adapter} as unknown as GPU),
    (e) => e === failure
  );
  assert.equal(destroyCount, 1);
});
