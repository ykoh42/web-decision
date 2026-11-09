import assert from 'node:assert/strict';
import {after, test} from 'node:test';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {trackRuntime} from '../src/runtime-lifecycle.js';
import {guardStrandsDisposal} from '../src/strands/disposal-guard.mjs';

// Vite transforms the pinned source in-process; middleware mode listens on no port.
const server = await createServer({
  root: fileURLToPath(new URL('../../../', import.meta.url)),
  configFile: false,
  logLevel: 'silent',
  appType: 'custom',
  optimizeDeps: {noDiscovery: true},
  server: {middlewareMode: true}
});
const {Instance} = await server.ssrLoadModule('/.toolchains/apache-tvm-src/web/src/runtime.ts');
after(() => server.close());

Object.assign(globalThis, {
  GPUBufferUsage: {MAP_READ: 1, COPY_DST: 2},
  GPUMapMode: {READ: 1}
});

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(reason?: unknown): void;
}

function deferred<T = void>(): Deferred<T> {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
}

interface FakeGpuContext {
  bufferTable: Array<{size: number}>;
  deviceCopyFromGPU(...args: number[]): void;
  pendingGPUToCPUCopy: Promise<unknown> | null | undefined;
}

interface FakeDevice {
  lost: PromiseLike<unknown>;
  addEventListener(): void;
  pushErrorScope(): void;
  popErrorScope(): Promise<null>;
  createBuffer(descriptor: {size: number}): {
    size: number;
    mapAsync(): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  };
  createCommandEncoder(): {
    copyBufferToBuffer(): void;
    finish(): Record<string, never>;
  };
  queue: {
    submit(): void;
    writeBuffer(): void;
    onSubmittedWorkDone(): Promise<void>;
  };
}

interface RuntimeUnderTest {
  dispose(...args: unknown[]): unknown;
  cacheState: {dispose(): void};
  ctx: {dispose(): void};
  lib: {dispose(): void; webGPUContext?: FakeGpuContext};
  memory: {storeRawBytes(): void};
  registerFunc(): void;
  registerAsyncServerFunc(): void;
  asyncifyHandler: {enabled(): boolean};
  initWebGPU(device: FakeDevice): void;
}

type Guard = ReturnType<typeof guardStrandsDisposal>;

interface Fixture {
  runtime: RuntimeUnderTest;
  lifecycle: ReturnType<typeof trackRuntime>;
  device: FakeDevice;
  lost: Deferred<{reason: string}>;
  map: Deferred<void>;
  events: string[];
  guard: Guard | null;
}

type GuardedFixture = Omit<Fixture, 'guard'> & {guard: Guard};

function fixture(options: {deferOnLoss: true}): GuardedFixture;
function fixture(options?: {deferOnLoss?: false}): Fixture;
function fixture({deferOnLoss = false}: {deferOnLoss?: boolean} = {}): Fixture {
  const lost = deferred<{reason: string}>(),
    map = deferred<void>(),
    events: string[] = [];
  let freed = false;
  const device: FakeDevice = {
    lost: lost.promise,
    addEventListener() {},
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    },
    createBuffer({size}) {
      return {
        size,
        mapAsync: () => map.promise,
        getMappedRange: () => Uint8Array.of(1, 2, 3, 4).buffer,
        unmap() {},
        destroy() {}
      };
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer() {},
        finish() {
          return {};
        }
      };
    },
    queue: {submit() {}, writeBuffer() {}, onSubmittedWorkDone: () => Promise.resolve()}
  };
  // Call the actual pinned prototype methods with fake infrastructure. In particular,
  // fake ctx/lib disposal records rather than actually freeing WASM or GPU resources.
  const runtime = Object.create(Instance.prototype) as RuntimeUnderTest;
  runtime.cacheState = {
    dispose() {
      events.push('cache-dispose');
    }
  };
  runtime.ctx = {
    dispose() {
      events.push('scope-dispose');
    }
  };
  runtime.lib = {
    dispose() {
      freed = true;
      events.push('lib-dispose');
    }
  };
  runtime.memory = {
    storeRawBytes() {
      events.push(freed ? 'write-after-dispose' : 'write-alive');
    }
  };
  runtime.registerFunc = () => {};
  runtime.registerAsyncServerFunc = () => {};
  runtime.asyncifyHandler = {enabled: () => false};
  const lifecycle = trackRuntime(runtime, device);
  const guard = deferOnLoss ? guardStrandsDisposal(runtime, device, lifecycle) : null;
  runtime.initWebGPU(device);
  return {runtime, lifecycle, device, lost, map, events, guard};
}

function submitReadback(f: Fixture): Promise<unknown> {
  const context = f.runtime.lib.webGPUContext;
  if (!context) throw new Error('Expected the pinned WebGPU context.');
  context.bufferTable.push({size: 64});
  context.deviceCopyFromGPU(1, 0, 20, 4);
  if (!context.pendingGPUToCPUCopy) throw new Error('Expected a pending GPU readback.');
  return context.pendingGPUToCPUCopy;
}

async function fireLoss(f: Fixture): Promise<void> {
  // The real pinned callback logs loss; suppress only that expected fixture log.
  const original = console.error;
  console.error = () => {};
  try {
    f.lost.resolve({reason: 'fixture-loss'});
    await new Promise<void>((resolve) => setImmediate(resolve));
  } finally {
    console.error = original;
  }
}

test('pinned loss callback disposes the tracked runtime before a delayed readback writes', async () => {
  const f = fixture(),
    readback = submitReadback(f);
  await fireLoss(f);
  assert.equal(f.lifecycle.disposed, true);
  f.map.resolve();
  await readback;
  assert.ok(f.events.includes('write-after-dispose'));
});

test('Strands loss guard defers pinned automatic disposal and preserves the readback target', async () => {
  const f = fixture({deferOnLoss: true}),
    readback = submitReadback(f);
  await fireLoss(f);
  assert.equal(f.lifecycle.disposed, false);
  assert.equal(f.guard.blocked, true);
  f.runtime.dispose();
  f.map.resolve();
  await readback;
  assert.ok(f.events.includes('write-alive'));
  assert.equal(f.events.includes('lib-dispose'), false);
});

test('Strands loss guard preserves normal tracked disposal and idempotence', () => {
  const f = fixture({deferOnLoss: true});
  f.runtime.dispose();
  f.runtime.dispose();
  assert.equal(f.lifecycle.disposed, true);
  assert.deepEqual(f.events, ['cache-dispose', 'scope-dispose', 'lib-dispose']);
});

test('normal disposal remains idempotent after the resulting device loss', async () => {
  const f = fixture({deferOnLoss: true});
  f.runtime.dispose();
  await fireLoss(f);
  assert.equal(f.guard.blocked, false);
  assert.deepEqual(f.events, ['cache-dispose', 'scope-dispose', 'lib-dispose']);
});

test('a reentrant dispose cannot recursively invoke tracked disposal', () => {
  const lost = deferred(),
    events: string[] = [];
  const runtime = {
    dispose() {
      events.push('dispose');
      runtime.dispose();
    }
  };
  const device = {lost: lost.promise};
  const lifecycle = trackRuntime(runtime, device);
  guardStrandsDisposal(runtime, device, lifecycle);
  runtime.dispose();
  assert.deepEqual(events, ['dispose']);
  assert.equal(lifecycle.disposed, true);
});

test('loss reported after explicit disposal does not reenter disposal or retain the runtime', async () => {
  const f = fixture({deferOnLoss: true});
  f.runtime.dispose();
  await fireLoss(f);
  assert.equal(f.lifecycle.loss, null);
  assert.equal(f.guard.blocked, false);
  assert.deepEqual(f.events, ['cache-dispose', 'scope-dispose', 'lib-dispose']);
});

test('a rejected delayed map remains observed while the loss guard retains the runtime', async () => {
  const f = fixture({deferOnLoss: true}),
    readback = submitReadback(f);
  const rejection = assert.rejects(readback, /fake map failure/);
  await fireLoss(f);
  f.map.reject(new Error('fake map failure'));
  await rejection;
  assert.equal(f.lifecycle.disposed, false);
  assert.equal(f.guard.blocked, true);
  assert.equal(
    f.events.some((event) => event.startsWith('write-')),
    false
  );
});
