import assert from 'node:assert/strict';
import {setImmediate, setTimeout} from 'node:timers/promises';

type Load = (options: object) => Promise<unknown>;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return {promise, resolve};
}

export async function checkGemma4V1StartupCancellation(load: Load, modelId: string) {
  const prior = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const limits = {maxBufferSize: 268_435_456, maxStorageBufferBindingSize: 268_435_456};
  try {
    for (const phase of ['adapter', 'device'] as const) {
      const entered = deferred<void>();
      const adapterResult = deferred<object>();
      const deviceResult = deferred<object>();
      let deviceRequests = 0;
      let deviceDestroyed = 0;
      const fakeDevice = {destroy: () => deviceDestroyed++};
      const fakeAdapter = {
        limits,
        requestDevice() {
          deviceRequests++;
          if (phase !== 'device') return fakeDevice;
          entered.resolve();
          return deviceResult.promise;
        }
      };
      Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: {
          gpu: {
            requestAdapter() {
              if (phase !== 'adapter') return fakeAdapter;
              entered.resolve();
              return adapterResult.promise;
            }
          }
        }
      });
      const controller = new AbortController();
      const reason = Error('cancel at ' + phase);
      const result = load({modelId, cache: false, signal: controller.signal}).then(
        () => 'unexpected success',
        (error: unknown) => error
      );
      const first = await Promise.race([entered.promise, result]);
      assert.equal(first, undefined, 'Load entered ' + phase);
      controller.abort(reason);
      const observed = await Promise.race([result, setTimeout(100, 'still pending')]);
      adapterResult.resolve(fakeAdapter);
      deviceResult.resolve(fakeDevice);
      assert.equal(await result, reason);
      await setImmediate();
      assert.equal(observed, reason, 'Cancellation must settle before the delayed ' + phase);
      assert.equal(deviceRequests, phase === 'adapter' ? 0 : 1);
      assert.equal(deviceDestroyed, phase === 'adapter' ? 0 : 1);
    }

    const reason = Error('adapter unavailable');
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        gpu: {
          requestAdapter: async () => {
            throw reason;
          }
        }
      }
    });
    await assert.rejects(load({modelId, cache: false}), {code: 'DEVICE_RESOURCE_LIMIT', cause: reason});
  } finally {
    if (prior) Object.defineProperty(globalThis, 'navigator', prior);
    else Reflect.deleteProperty(globalThis, 'navigator');
  }
}
