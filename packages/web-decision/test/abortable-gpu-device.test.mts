import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import test from 'node:test';
import {requestGpuAdapter, requestGpuDevice} from '../src/abortable-gpu-device.js';

type ResolveContext = {parentURL?: string};
type ResolveResult = {shortCircuit?: boolean; url: string};
type RegisterHooks = (hooks: {
  resolve(
    specifier: string,
    context: ResolveContext,
    nextResolve: (specifier: string, context: ResolveContext) => ResolveResult
  ): ResolveResult;
}) => {deregister(): void};
const registerHooks = (nodeModule as typeof nodeModule & {registerHooks?: RegisterHooks}).registerHooks;

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
}

const nextTurn = () => new Promise((resolve) => setImmediate(resolve));

test('requests the adapter and device with unchanged descriptors', async () => {
  const adapter = {},
    device = {destroy() {}},
    descriptor = {requiredFeatures: ['shader-f16'], requiredLimits: {maxBufferSize: 123}};
  let adapterOptions: unknown, deviceOptions: unknown;
  const gpu = {
    requestAdapter(options: {powerPreference: 'high-performance'}) {
      adapterOptions = options;
      return Promise.resolve(adapter);
    }
  };
  const supported = {
    requestDevice(options: typeof descriptor) {
      deviceOptions = options;
      return Promise.resolve(device);
    }
  };
  assert.equal(await requestGpuAdapter(gpu), adapter);
  assert.deepEqual(adapterOptions, {powerPreference: 'high-performance'});
  assert.equal(await requestGpuDevice(supported, descriptor), device);
  assert.equal(deviceOptions, descriptor);
});

test('abort during an adapter request returns the reason without requesting a device', async () => {
  const controller = new AbortController(),
    reason = Error('adapter stop'),
    pending = deferred<unknown>();
  let calls = 0;
  const result = requestGpuAdapter(
    {
      requestAdapter() {
        calls++;
        return pending.promise;
      }
    },
    controller.signal
  );
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  pending.resolve({
    requestDevice() {
      assert.fail('late adapter must not request a device');
    }
  });
  await nextTurn();
  assert.equal(calls, 1);
});

test('a cancelled adapter request observes a late rejection', async () => {
  const controller = new AbortController(),
    reason = Error('adapter stop'),
    pending = deferred<unknown>();
  const result = requestGpuAdapter({requestAdapter: () => pending.promise}, controller.signal);
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  pending.reject(Error('late adapter failure'));
  await nextTurn();
});

test('abort after adapter resolution but before helper return preserves the reason', async () => {
  const controller = new AbortController(),
    reason = Error('adapter stop'),
    pending = deferred<unknown>();
  const result = requestGpuAdapter({requestAdapter: () => pending.promise}, controller.signal);
  pending.resolve({});
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(result, (error) => error === reason);
});

test('abort during a device request destroys a late device once', async () => {
  const controller = new AbortController(),
    reason = Error('device stop'),
    pending = deferred<{destroy(): void}>();
  let destroyed = 0;
  const result = requestGpuDevice({requestDevice: () => pending.promise}, {}, controller.signal);
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  pending.resolve({
    destroy() {
      destroyed++;
    }
  });
  await nextTurn();
  assert.equal(destroyed, 1);
});

test('a cancelled device request observes a late rejection', async () => {
  const controller = new AbortController(),
    reason = Error('device stop'),
    pending = deferred<{destroy(): void}>();
  const result = requestGpuDevice({requestDevice: () => pending.promise}, {}, controller.signal);
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  pending.reject(Error('late device failure'));
  await nextTurn();
});

test('abort after device resolution but before helper return destroys the device once', async () => {
  const controller = new AbortController(),
    reason = Error('device stop'),
    pending = deferred<{destroy(): void}>();
  let destroyed = 0;
  const result = requestGpuDevice({requestDevice: () => pending.promise}, {}, controller.signal);
  pending.resolve({
    destroy() {
      destroyed++;
    }
  });
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(result, (error) => error === reason);
  await nextTurn();
  assert.equal(destroyed, 1);
});

test('pre-aborted requests do not start and ordinary errors are preserved', async () => {
  const controller = new AbortController(),
    reason = Error('already stopped'),
    failure = Error('request failed');
  controller.abort(reason);
  const never = () => assert.fail('request must not start');
  await assert.rejects(requestGpuAdapter({requestAdapter: never}, controller.signal), (error) => error === reason);
  await assert.rejects(requestGpuDevice({requestDevice: never}, {}, controller.signal), (error) => error === reason);
  const fail = () => {
    throw failure;
  };
  await assert.rejects(requestGpuAdapter({requestAdapter: fail}), (error) => error === failure);
  await assert.rejects(requestGpuDevice({requestDevice: fail}, {}), (error) => error === failure);
});

test(
  'failed late destroy retains the exact device and preserves the abort reason',
  {skip: typeof registerHooks !== 'function'},
  async () => {
    const helper = new URL('../src/abortable-gpu-device.ts?retention-test', import.meta.url).href;
    const fakeGuard = `data:text/javascript,${encodeURIComponent('export function retainStrandsResources(runtime, gpu) { globalThis.__retainedGpu = {runtime, gpu}; }')}`;
    const hooks = registerHooks!({
      resolve(specifier, context, nextResolve) {
        if (context.parentURL === helper && specifier === './strands/disposal-guard.mjs')
          return {url: fakeGuard, shortCircuit: true};
        return nextResolve(specifier, context);
      }
    });
    let tested: typeof import('../src/abortable-gpu-device.js');
    try {
      tested = await import(helper);
    } finally {
      hooks.deregister();
    }
    const controller = new AbortController(),
      reason = Error('device stop'),
      pending = deferred<{destroy(): void}>(),
      device = {
        destroy() {
          throw Error('destroy failed');
        }
      };
    const retained = globalThis as typeof globalThis & {__retainedGpu?: {runtime: unknown; gpu: unknown}};
    try {
      const result = tested.requestGpuDevice({requestDevice: () => pending.promise}, {}, controller.signal);
      controller.abort(reason);
      await assert.rejects(result, (error) => error === reason);
      pending.resolve(device);
      await nextTurn();
      assert.equal(retained.__retainedGpu?.gpu, device);
      assert.equal(retained.__retainedGpu?.runtime, undefined);
    } finally {
      delete retained.__retainedGpu;
    }
  }
);
