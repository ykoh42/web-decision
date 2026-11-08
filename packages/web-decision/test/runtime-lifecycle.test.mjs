import test from 'node:test';
import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import {cleanupFailedLoad, trackRuntime} from '../src/runtime-lifecycle.js';

test('failed-load cleanup closes watcher, scope, runtime, then device', () => {
  const steps = [];
  cleanupFailedLoad({
    loadWatch: {close: () => steps.push('watch')},
    runtime: {endScope: () => steps.push('scope'), dispose: () => steps.push('runtime')},
    scopeOpen: true,
    lifecycle: {loss: null, disposed: false},
    gpuDevice: {destroy: () => steps.push('device')}
  });
  assert.deepEqual(steps, ['watch', 'scope', 'runtime', 'device']);
});

test(
  'failed device destruction retains runtime and device without hiding cancellation',
  {skip: typeof nodeModule.registerHooks !== 'function'},
  async () => {
    const moduleUrl = new URL('../src/runtime-lifecycle.js?retention-test', import.meta.url).href;
    const fakeGuard = `data:text/javascript,${encodeURIComponent('export function retainStrandsResources(runtime, gpu) { globalThis.__failedLoadRetained = {runtime, gpu}; }')}`;
    const hooks = nodeModule.registerHooks({
      resolve(specifier, context, nextResolve) {
        if (context.parentURL === moduleUrl && specifier === './strands/disposal-guard.mjs')
          return {url: fakeGuard, shortCircuit: true};
        return nextResolve(specifier, context);
      }
    });
    let tested;
    try {
      tested = await import(moduleUrl);
    } finally {
      hooks.deregister();
    }
    const steps = [],
      controller = new AbortController(),
      reason = Error('cancelled'),
      runtime = {endScope: () => steps.push('scope'), dispose: () => steps.push('runtime')},
      gpuDevice = {
        destroy() {
          steps.push('device');
          throw Error('destroy failed');
        }
      };
    controller.abort(reason);
    try {
      assert.throws(
        () => {
          try {
            controller.signal.throwIfAborted();
          } catch {
            tested.cleanupFailedLoad({
              loadWatch: {close: () => steps.push('watch')},
              runtime,
              scopeOpen: true,
              lifecycle: {loss: null, disposed: false},
              gpuDevice
            });
            controller.signal.throwIfAborted();
          }
        },
        (error) => error === reason
      );
      assert.deepEqual(steps, ['watch', 'scope', 'runtime', 'device']);
      assert.equal(globalThis.__failedLoadRetained?.runtime, runtime);
      assert.equal(globalThis.__failedLoadRetained?.gpu, gpuDevice);
      tested.cleanupFailedLoad({gpuDevice});
      assert.equal(globalThis.__failedLoadRetained?.runtime, undefined);
      assert.equal(globalThis.__failedLoadRetained?.gpu, gpuDevice);
    } finally {
      delete globalThis.__failedLoadRetained;
    }
  }
);

test('records device loss and makes TVM and caller disposal share one cleanup', async () => {
  let resolveLoss,
    calls = 0;
  const gpu = {
    lost: new Promise((resolve) => {
      resolveLoss = resolve;
    })
  };
  const runtime = {
    dispose() {
      calls++;
    }
  };
  const state = trackRuntime(runtime, gpu);
  resolveLoss({reason: 'unknown', message: 'device lost'});
  await gpu.lost;
  assert.equal(state.loss.reason, 'unknown');
  runtime.dispose(); // Simulates TVM's own device-loss handler.
  runtime.dispose(); // Simulates a caller closing the session.
  assert.equal(calls, 1);
  assert.equal(state.disposed, true);
});

test('normal disposal does not misclassify the device destroy it causes', async () => {
  let resolveLoss,
    calls = 0;
  const gpu = {
    lost: new Promise((resolve) => {
      resolveLoss = resolve;
    })
  };
  const runtime = {
    dispose() {
      calls++;
      resolveLoss({reason: 'destroyed'});
    }
  };
  const state = trackRuntime(runtime, gpu);
  runtime.dispose();
  await gpu.lost;
  assert.equal(state.loss, null);
  assert.equal(calls, 1);
});

test('a loss promise rejection does not become an unhandled rejection', async () => {
  let rejectLoss;
  const gpu = {
    lost: new Promise((_, reject) => {
      rejectLoss = reject;
    })
  };
  const runtime = {dispose() {}};
  const state = trackRuntime(runtime, gpu);
  rejectLoss(new Error('test failure'));
  await gpu.lost.catch(() => {});
  assert.match(state.loss.message, /test failure/);
});
