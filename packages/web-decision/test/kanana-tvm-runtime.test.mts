import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdir, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import test, {after} from 'node:test';
import type {
  KananaTvmSetupOptions,
  TvmRuntimeLike,
  setupKananaTvmRuntime as setupKananaTvmRuntimeSource
} from '../src/kanana/tvm-runtime.js';

type MockVm = {readonly kind: 'mock-vm'};
type MockWasi = {readonly kind: 'mock-wasi'};
type MockRuntime = TvmRuntimeLike<MockVm>;
type RuntimeEvent = string | readonly [string, unknown];
type RuntimeSetupOptions = KananaTvmSetupOptions<MockWasi, MockRuntime>;

interface MockInstantiateObserver {
  (bytes: ArrayBuffer, wasi: MockWasi, logger: (message: string) => void): void;
}

const require = createRequire(import.meta.url);
const compiler = join(dirname(require.resolve('typescript/package.json')), 'bin/tsc');
const packageRoot = fileURLToPath(new URL('..', import.meta.url));
const coverageDirectory = join(packageRoot, 'coverage');
await mkdir(coverageDirectory, {recursive: true});
const outputDirectory = await mkdtemp(join(coverageDirectory, 'kanana-tvm-'));
await writeFile(join(outputDirectory, 'package.json'), '{"type":"module"}\n');

try {
  execFileSync(
    process.execPath,
    [
      compiler,
      '--project',
      join(packageRoot, 'tsconfig.kanana-runtime.json'),
      '--noEmit',
      'false',
      '--sourceMap',
      '--outDir',
      outputDirectory
    ],
    {cwd: packageRoot, stdio: 'inherit'}
  );
} catch (error) {
  await rm(outputDirectory, {recursive: true, force: true});
  throw error;
}

const runtimeModule: {setupKananaTvmRuntime: typeof setupKananaTvmRuntimeSource} = await import(
  pathToFileURL(join(outputDirectory, 'tvm-runtime.js')).href
);
const {setupKananaTvmRuntime} = runtimeModule;
after(() => rm(outputDirectory, {recursive: true, force: true}));

// These mocks test lifecycle ordering only. They do not emulate or validate WebGPU.
function mockRuntime(
  events: RuntimeEvent[],
  overrides: Partial<MockRuntime> = {}
): {readonly runtime: MockRuntime; readonly vm: MockVm} {
  const vm: MockVm = {kind: 'mock-vm'};
  const runtime: MockRuntime = {
    initWebGPU(device: object): void {
      events.push(['initWebGPU', device]);
    },
    beginScope(): void {
      events.push('beginScope');
    },
    webgpu(): object {
      events.push('webgpu');
      return {kind: 'mock-webgpu-device'};
    },
    createVirtualMachine(device: object): MockVm {
      events.push(['createVirtualMachine', device]);
      return vm;
    },
    endScope(): void {
      events.push('endScope');
    },
    dispose(): void {
      events.push('dispose');
    },
    ...overrides
  };
  return {runtime, vm};
}

function mockDependencies(
  events: RuntimeEvent[],
  runtime: MockRuntime,
  onInstantiate: MockInstantiateObserver = () => {}
): Pick<RuntimeSetupOptions, 'instantiate' | 'createPolyfillWASI'> {
  return {
    createPolyfillWASI(): MockWasi {
      events.push('createPolyfillWASI');
      return {kind: 'mock-wasi'};
    },
    async instantiate(bytes: ArrayBuffer, wasi: MockWasi, logger: (message: string) => void): Promise<MockRuntime> {
      events.push('instantiate');
      onInstantiate(bytes, wasi, logger);
      return runtime;
    }
  };
}

test('mocked TVM runtime setup orders initialization and disposes once', async () => {
  const events: RuntimeEvent[] = [];
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
  const events: RuntimeEvent[] = [];
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
  const events: RuntimeEvent[] = [];
  const {runtime} = mockRuntime(events);
  const bytes = Uint8Array.from([4, 5]);
  const dependencies = mockDependencies(events, runtime, (providedBytes) => {
    assert.equal(providedBytes, bytes.buffer);
  });

  const setup = await setupKananaTvmRuntime({...dependencies, device: {}, bytes});
  setup.dispose();
});

test('mocked WebGPU initialization failure disposes the instantiated runtime', async () => {
  const events: RuntimeEvent[] = [];
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
  const events: RuntimeEvent[] = [];
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
  const events: RuntimeEvent[] = [];
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
  const events: RuntimeEvent[] = [];
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

  await assert.rejects(
    setupKananaTvmRuntime({...dependencies, device: {}, bytes: new ArrayBuffer(1)}),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.equal(error.cause, setupFailure);
      assert.ok((error as Error & {cleanupError: unknown}).cleanupError instanceof AggregateError);
      assert.equal((error as Error & {cleanupError: AggregateError}).cleanupError.cause, scopeFailure);
      assert.deepEqual((error as Error & {cleanupError: AggregateError}).cleanupError.errors, [
        scopeFailure,
        runtimeFailure
      ]);
      return true;
    }
  );
  assert.deepEqual(events.slice(-3), ['createVirtualMachine failed', 'endScope failed', 'dispose failed']);
});

test('mocked argument and instantiation failures do not leak a runtime', async () => {
  const noCalls: RuntimeEvent[] = [];
  const dependencies = mockDependencies(noCalls, mockRuntime(noCalls).runtime);
  // These calls deliberately pass malformed values to exercise the runtime guards.
  await assert.rejects(
    setupKananaTvmRuntime({...dependencies, device: {}, bytes: null} as unknown as RuntimeSetupOptions),
    /ArrayBuffer or an ArrayBuffer view/
  );
  await assert.rejects(
    setupKananaTvmRuntime({
      instantiate: null,
      createPolyfillWASI: () => ({}),
      device: {},
      bytes: new ArrayBuffer(1)
    } as unknown as RuntimeSetupOptions),
    /TVM instantiate\(\) is required/
  );
  await assert.rejects(
    setupKananaTvmRuntime({
      instantiate: () => {},
      createPolyfillWASI: null,
      device: {},
      bytes: new ArrayBuffer(1)
    } as unknown as RuntimeSetupOptions),
    /TVM createPolyfillWASI\(\) is required/
  );
  await assert.rejects(
    setupKananaTvmRuntime({...dependencies, device: null, bytes: new ArrayBuffer(1)} as unknown as RuntimeSetupOptions),
    /A WebGPU device is required/
  );

  const failure = new Error('mock instantiate failure');
  const instantiateFailure = {
    createPolyfillWASI(): MockWasi {
      noCalls.push('createPolyfillWASI');
      return {} as unknown as MockWasi;
    },
    async instantiate(): Promise<MockRuntime> {
      noCalls.push('instantiate failed');
      throw failure;
    }
  };
  await assert.rejects(
    setupKananaTvmRuntime({...instantiateFailure, device: {}, bytes: new ArrayBuffer(1)}),
    (error: unknown) => error === failure
  );
  assert.deepEqual(noCalls, ['createPolyfillWASI', 'instantiate failed']);
});
