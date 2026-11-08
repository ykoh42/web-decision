import assert from 'node:assert/strict';
import test from 'node:test';
import {setupKananaTvmRuntime} from '../src/kanana/tvm-runtime.mjs';

// These mocks test lifecycle ordering only. They do not emulate or validate WebGPU.
function mockRuntime(events, overrides = {}) {
  const vm = {kind: 'mock-vm'};
  const runtime = {
    initWebGPU(device) {
      events.push(['initWebGPU', device]);
    },
    beginScope() {
      events.push('beginScope');
    },
    webgpu() {
      events.push('webgpu');
      return {kind: 'mock-webgpu-device'};
    },
    createVirtualMachine(device) {
      events.push(['createVirtualMachine', device]);
      return vm;
    },
    endScope() {
      events.push('endScope');
    },
    dispose() {
      events.push('dispose');
    },
    ...overrides
  };
  return {runtime, vm};
}

function mockDependencies(events, runtime, onInstantiate = () => {}) {
  return {
    createPolyfillWASI() {
      events.push('createPolyfillWASI');
      return {kind: 'mock-wasi'};
    },
    async instantiate(bytes, wasi, logger) {
      events.push('instantiate');
      onInstantiate(bytes, wasi, logger);
      return runtime;
    }
  };
}

test('mocked TVM runtime setup orders initialization and disposes once', async () => {
  const events = [];
  const device = {kind: 'mock-gpu-device'};
  const {runtime, vm} = mockRuntime(events);
  const bytes = Uint8Array.from([1, 2, 3]).buffer;
  const dependencies = mockDependencies(events, runtime, (providedBytes, wasi, logger) => {
    assert.equal(providedBytes, bytes);
    assert.deepEqual(wasi, {kind: 'mock-wasi'});
    assert.equal(typeof logger, 'function');
  });

  const setup = await setupKananaTvmRuntime({...dependencies, device, bytes});
  assert.equal(setup.runtime, runtime);
  assert.equal(setup.vm, vm);
  assert.deepEqual(events, [
    'createPolyfillWASI',
    'instantiate',
    ['initWebGPU', device],
    'beginScope',
    'webgpu',
    ['createVirtualMachine', {kind: 'mock-webgpu-device'}]
  ]);

  setup.dispose();
  setup.dispose();
  assert.deepEqual(events.slice(-2), ['endScope', 'dispose']);
  assert.equal(events.filter((event) => event === 'dispose').length, 1);
});

test('mocked setup copies only a byte view subrange before instantiation', async () => {
  const events = [];
  const {runtime} = mockRuntime(events);
  const backing = Uint8Array.from([8, 4, 5, 9]);
  const bytes = backing.subarray(1, 3);
  const dependencies = mockDependencies(events, runtime, (providedBytes) => {
    assert.ok(providedBytes instanceof ArrayBuffer);
    assert.deepEqual([...new Uint8Array(providedBytes)], [4, 5]);
    assert.notEqual(providedBytes, backing.buffer);
  });

  const setup = await setupKananaTvmRuntime({...dependencies, device: {}, bytes});
  setup.dispose();
});

test('mocked setup reuses a full byte view without copying its backing buffer', async () => {
  const events = [];
  const {runtime} = mockRuntime(events);
  const bytes = Uint8Array.from([4, 5]);
  const dependencies = mockDependencies(events, runtime, (providedBytes) => {
    assert.equal(providedBytes, bytes.buffer);
  });

  const setup = await setupKananaTvmRuntime({...dependencies, device: {}, bytes});
  setup.dispose();
});

test('mocked WebGPU initialization failure disposes the instantiated runtime', async () => {
  const events = [];
  const failure = new Error('mock initWebGPU failure');
  const {runtime} = mockRuntime(events, {
    initWebGPU() {
      events.push('initWebGPU failed');
      throw failure;
    }
  });
  const dependencies = mockDependencies(events, runtime);

  await assert.rejects(
    setupKananaTvmRuntime({...dependencies, device: {}, bytes: new ArrayBuffer(1)}),
    (error) => error === failure
  );
  assert.deepEqual(events, ['createPolyfillWASI', 'instantiate', 'initWebGPU failed', 'dispose']);
});

test('mocked VM creation failure closes the scope and disposes the runtime', async () => {
  const events = [];
  const failure = new Error('mock VM creation failure');
  const {runtime} = mockRuntime(events, {
    createVirtualMachine() {
      events.push('createVirtualMachine failed');
      throw failure;
    }
  });
  const dependencies = mockDependencies(events, runtime);

  await assert.rejects(
    setupKananaTvmRuntime({...dependencies, device: {}, bytes: new ArrayBuffer(1)}),
    (error) => error === failure
  );
  assert.deepEqual(events.slice(-3), ['createVirtualMachine failed', 'endScope', 'dispose']);
});

test('mocked disposal still releases the runtime when ending scope throws and remains idempotent', async () => {
  const events = [];
  const scopeFailure = new Error('mock endScope failure');
  const {runtime} = mockRuntime(events, {
    endScope() {
      events.push('endScope failed');
      throw scopeFailure;
    }
  });
  const dependencies = mockDependencies(events, runtime);
  const setup = await setupKananaTvmRuntime({...dependencies, device: {}, bytes: new ArrayBuffer(1)});

  assert.throws(
    () => setup.dispose(),
    (error) => error === scopeFailure
  );
  setup.dispose();
  assert.deepEqual(events.slice(-2), ['endScope failed', 'dispose']);
});

test('mocked setup preserves setup and cleanup failures together', async () => {
  const events = [];
  const setupFailure = new Error('mock VM creation failure');
  const scopeFailure = new Error('mock scope cleanup failure');
  const runtimeFailure = new Error('mock runtime cleanup failure');
  const {runtime} = mockRuntime(events, {
    createVirtualMachine() {
      events.push('createVirtualMachine failed');
      throw setupFailure;
    },
    endScope() {
      events.push('endScope failed');
      throw scopeFailure;
    },
    dispose() {
      events.push('dispose failed');
      throw runtimeFailure;
    }
  });
  const dependencies = mockDependencies(events, runtime);

  await assert.rejects(setupKananaTvmRuntime({...dependencies, device: {}, bytes: new ArrayBuffer(1)}), (error) => {
    assert.ok(error instanceof Error);
    assert.equal(error.cause, setupFailure);
    assert.ok(error.cleanupError instanceof AggregateError);
    assert.equal(error.cleanupError.cause, scopeFailure);
    assert.deepEqual(error.cleanupError.errors, [scopeFailure, runtimeFailure]);
    return true;
  });
  assert.deepEqual(events.slice(-3), ['createVirtualMachine failed', 'endScope failed', 'dispose failed']);
});

test('mocked argument and instantiation failures do not leak a runtime', async () => {
  const noCalls = [];
  const dependencies = mockDependencies(noCalls, mockRuntime(noCalls).runtime);
  await assert.rejects(
    setupKananaTvmRuntime({...dependencies, device: {}, bytes: null}),
    /ArrayBuffer or an ArrayBuffer view/
  );
  await assert.rejects(
    setupKananaTvmRuntime({instantiate: null, createPolyfillWASI: () => ({}), device: {}, bytes: new ArrayBuffer(1)}),
    /TVM instantiate\(\) is required/
  );
  await assert.rejects(
    setupKananaTvmRuntime({instantiate: () => {}, createPolyfillWASI: null, device: {}, bytes: new ArrayBuffer(1)}),
    /TVM createPolyfillWASI\(\) is required/
  );
  await assert.rejects(
    setupKananaTvmRuntime({...dependencies, device: null, bytes: new ArrayBuffer(1)}),
    /A WebGPU device is required/
  );

  const failure = new Error('mock instantiate failure');
  const instantiateFailure = {
    createPolyfillWASI() {
      noCalls.push('createPolyfillWASI');
      return {};
    },
    async instantiate() {
      noCalls.push('instantiate failed');
      throw failure;
    }
  };
  await assert.rejects(
    setupKananaTvmRuntime({...instantiateFailure, device: {}, bytes: new ArrayBuffer(1)}),
    (error) => error === failure
  );
  assert.deepEqual(noCalls, ['createPolyfillWASI', 'instantiate failed']);
});
