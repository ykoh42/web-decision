import test from 'node:test';
import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import {cleanupFailedLoad, trackRuntime} from '../src/runtime-lifecycle.js';

type DeviceLoss = {reason: string; message?: string};
type LossDetails = {message?: string; reason?: string};
type TestRuntime = {dispose(): void; endScope?(): void};
type TestGpuDevice = {destroy(): void};
type TestGpuLossSource = {lost: Promise<DeviceLoss>};
type TestLifecycle = {disposed: boolean; loss: LossDetails | null; lossSignal: AbortSignal};
type FailedLoad = {
  gpuDevice?: TestGpuDevice;
  lifecycle?: Pick<TestLifecycle, 'disposed' | 'loss'>;
  loadWatch?: {close(): void};
  runtime?: TestRuntime;
  scopeOpen?: boolean;
};
type CleanupFailedLoadInput = {
  gpuDevice: TestGpuDevice | undefined;
  lifecycle: Pick<TestLifecycle, 'disposed' | 'loss'> | undefined;
  loadWatch: {close(): void} | undefined;
  runtime: TestRuntime | undefined;
  scopeOpen: boolean | undefined;
};
type CleanupFailedLoad = (input: CleanupFailedLoadInput) => void;
type LifecycleModule = typeof import('../src/runtime-lifecycle.js');
type ModuleResolveContext = {parentURL?: string};
type ModuleResolveResult = {shortCircuit?: boolean; url: string};
type ModuleRegisterHooks = (hooks: {
  resolve(
    specifier: string,
    context: ModuleResolveContext,
    nextResolve: (specifier: string, context: ModuleResolveContext) => ModuleResolveResult
  ): ModuleResolveResult;
}) => {deregister(): void};
type RetainedResources = {gpu: TestGpuDevice | undefined; runtime: TestRuntime | undefined};

declare global {
  var __failedLoadRetained: RetainedResources | undefined;
}

const registerHooks = (nodeModule as typeof nodeModule & {registerHooks?: ModuleRegisterHooks}).registerHooks;

function invokeCleanup(cleanup: CleanupFailedLoad, input: FailedLoad): void {
  cleanup({
    gpuDevice: input.gpuDevice,
    lifecycle: input.lifecycle,
    loadWatch: input.loadWatch,
    runtime: input.runtime,
    scopeOpen: input.scopeOpen
  });
}

function cleanupTestLoad(input: FailedLoad): void {
  invokeCleanup(cleanupFailedLoad, input);
}

function trackTestRuntime(runtime: TestRuntime, gpu: TestGpuLossSource): TestLifecycle {
  return trackRuntime(runtime, gpu) as TestLifecycle;
}

test('failed-load cleanup closes watcher, scope, runtime, then device', () => {
  const steps: string[] = [];
  const failedLoad: FailedLoad = {
    loadWatch: {close: () => steps.push('watch')},
    runtime: {endScope: () => steps.push('scope'), dispose: () => steps.push('runtime')},
    scopeOpen: true,
    lifecycle: {loss: null, disposed: false},
    gpuDevice: {destroy: () => steps.push('device')}
  };
  cleanupTestLoad(failedLoad);
  assert.deepEqual(steps, ['watch', 'scope', 'runtime', 'device']);
});

test(
  'failed device destruction retains runtime and device without hiding cancellation',
  {skip: typeof registerHooks !== 'function'},
  async () => {
    const moduleUrl = new URL('../src/runtime-lifecycle.js?retention-test', import.meta.url).href;
    const fakeGuard = `data:text/javascript,${encodeURIComponent('export function retainStrandsResources(runtime, gpu) { globalThis.__failedLoadRetained = {runtime, gpu}; }')}`;
    const hooks = registerHooks!({
      resolve(specifier, context, nextResolve) {
        if (context.parentURL === moduleUrl && specifier === './strands/disposal-guard.mjs')
          return {url: fakeGuard, shortCircuit: true};
        return nextResolve(specifier, context);
      }
    });
    let tested: LifecycleModule | undefined;
    try {
      tested = (await import(moduleUrl)) as LifecycleModule;
    } finally {
      hooks.deregister();
    }
    assert.ok(tested);
    const cleanupTestedLoad = tested.cleanupFailedLoad as CleanupFailedLoad;
    const steps: string[] = [],
      controller = new AbortController(),
      reason = Error('cancelled'),
      runtime: TestRuntime = {endScope: () => steps.push('scope'), dispose: () => steps.push('runtime')},
      gpuDevice: TestGpuDevice = {
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
            invokeCleanup(cleanupTestedLoad, {
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
      invokeCleanup(cleanupTestedLoad, {gpuDevice});
      assert.equal(globalThis.__failedLoadRetained?.runtime, undefined);
      assert.equal(globalThis.__failedLoadRetained?.gpu, gpuDevice);
    } finally {
      Reflect.deleteProperty(globalThis, '__failedLoadRetained');
    }
  }
);

test('records device loss and makes TVM and caller disposal share one cleanup', async () => {
  let resolveLoss: (value: DeviceLoss) => void = () => {
    throw Error('the GPU loss promise has no listener');
  };
  let calls = 0;
  const gpu = {
    lost: new Promise<DeviceLoss>((resolve) => {
      resolveLoss = (value) => resolve(value);
    })
  };
  const runtime: TestRuntime = {
    dispose() {
      calls++;
    }
  };
  const state = trackTestRuntime(runtime, gpu);
  resolveLoss({reason: 'unknown', message: 'device lost'});
  await gpu.lost;
  assert.ok(state.loss);
  assert.equal(state.loss.reason, 'unknown');
  assert.equal(state.lossSignal.aborted, true);
  assert.match((state.lossSignal.reason as Error).message, /GPU device was lost/);
  runtime.dispose(); // Simulates TVM's own device-loss handler.
  runtime.dispose(); // Simulates a caller closing the session.
  assert.equal(calls, 1);
  assert.equal(state.disposed, true);
});

test('normal disposal does not misclassify the device destroy it causes', async () => {
  let resolveLoss: (value: DeviceLoss) => void = () => {
    throw Error('the GPU loss promise has no listener');
  };
  let calls = 0;
  const gpu = {
    lost: new Promise<DeviceLoss>((resolve) => {
      resolveLoss = (value) => resolve(value);
    })
  };
  const runtime: TestRuntime = {
    dispose() {
      calls++;
      resolveLoss({reason: 'destroyed'});
    }
  };
  const state = trackTestRuntime(runtime, gpu);
  runtime.dispose();
  await gpu.lost;
  assert.equal(state.loss, null);
  assert.equal(state.lossSignal.aborted, false);
  assert.equal(calls, 1);
});

test('a loss promise rejection does not become an unhandled rejection', async () => {
  let rejectLoss: (reason: Error) => void = () => {
    throw Error('the GPU loss promise has no listener');
  };
  const gpu = {
    lost: new Promise<DeviceLoss>((_resolve, reject) => {
      rejectLoss = (reason) => reject(reason);
    })
  };
  const runtime: TestRuntime = {dispose() {}, endScope() {}};
  const state = trackTestRuntime(runtime, gpu);
  rejectLoss(new Error('test failure'));
  await gpu.lost.catch(() => {});
  assert.ok(state.loss);
  assert.ok(state.loss.message);
  assert.match(state.loss.message, /test failure/);
  assert.equal(state.lossSignal.aborted, true);
});
