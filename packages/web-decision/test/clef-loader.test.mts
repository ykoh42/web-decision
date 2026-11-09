import test from 'node:test';
import assert from 'node:assert/strict';
import {loadFakeClefCandidate, validateClefOptions, validateClefAdapter} from '../src/clef/loader.mjs';
import {CLEF_LIMITS, CLEF_MODEL} from '../src/clef/contract.mjs';
import {uploadFakeClefParameters} from '../src/clef/upload.mjs';
import type {ClefTensorEntry} from '../src/clef/types.js';

const options = () => ({model: CLEF_MODEL, assetsUrl: 'https://example.test/assets'});
type FakeDependencies = Parameters<typeof loadFakeClefCandidate>[1];
type FakeGpuApi = NonNullable<FakeDependencies['gpu']>;
type FakeAdapter = NonNullable<Awaited<ReturnType<FakeGpuApi['requestAdapter']>>>;
type FakeGpuDevice = Awaited<ReturnType<FakeAdapter['requestDevice']>>;
type FakeRuntime = Awaited<ReturnType<FakeDependencies['instantiate']>>;
type FakeSource = Awaited<ReturnType<FakeDependencies['openSource']>>;
type FakeOwner = ReturnType<FakeDependencies['initialize']>;
type FakeTokenizerBundle = Awaited<ReturnType<FakeDependencies['prepareTokenizer']>>;
type FakeSessionRunner = ReturnType<FakeDependencies['runner']>;
type FixtureDependencies = FakeDependencies & {gpu: FakeGpuApi};

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

const tick = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));

function fixture(): {
  deps: FixtureDependencies;
  owner: FakeOwner;
  runtime: FakeRuntime;
  gpuDevice: FakeGpuDevice;
  loss: Deferred<unknown>;
  events: string[];
} {
  const loss = deferred(),
    events: string[] = [];
  const gpuDevice: FakeGpuDevice = {
    lost: loss.promise,
    destroy() {
      events.push('gpu.destroy');
    }
  };
  const tensor = {
    byteOffset: 0,
    dtype: 'float32',
    shape: [],
    copyFrom() {},
    getDataPtr() {
      return 1;
    },
    toArray() {
      return [];
    }
  };
  const runtime: FakeRuntime = {
    lib: {webGPUContext: {}},
    empty() {
      return tensor;
    },
    cpu() {
      return {};
    },
    dispose() {
      events.push('runtime.dispose');
    },
    initWebGPU() {},
    beginScope() {},
    endScope() {},
    webgpu() {
      return {};
    },
    createVirtualMachine() {
      return {getFunction: () => () => tensor};
    }
  };
  const owner: FakeOwner = {
    runtime,
    device: {},
    gpu: gpuDevice,
    fn: () => tensor,
    teardownObserver: null,
    teardownTimeoutMs: 1,
    lifecycle: {loss: null, disposed: false},
    disposalGuard: {
      blocked: false,
      retain() {
        events.push('retain');
      }
    },
    uncaptured: []
  };
  const source: FakeSource = {
    wasm: new Uint8Array(),
    tokenizer: {},
    tokenizerConfig: {},
    entries: [],
    inputBanks: [],
    outputBanks: [],
    async readRange() {
      return new ArrayBuffer(0);
    }
  };
  const deps: FixtureDependencies = {
    fake: true,
    gpu: {
      async requestAdapter() {
        const adapter: FakeAdapter = {
          features: new Set(['shader-f16']),
          info: {isFallbackAdapter: false},
          limits: CLEF_LIMITS,
          async requestDevice() {
            return gpuDevice;
          }
        };
        return adapter;
      }
    },
    async openSource() {
      return source;
    },
    async prepareTokenizer() {
      const bundle: FakeTokenizerBundle = {tokenizer: {encode: () => ({ids: []})}, controlIds: []};
      return bundle;
    },
    async instantiate() {
      return runtime;
    },
    initialize() {
      return owner;
    },
    async upload() {
      return [];
    },
    runner() {
      return {} as FakeSessionRunner;
    },
    session() {
      return {ok: true};
    }
  };
  return {deps, owner, runtime, gpuDevice, loss, events};
}

test('load options reject unsupported advertised features before acquiring GPU', () => {
  assert.equal(validateClefOptions(options()).base.href, 'https://example.test/assets/');
  for (const option of [
    {cache: true},
    {offline: true},
    {maxPrefetchBytes: 1},
    {onProgress() {}},
    {debertaReadback: 'x'}
  ])
    assert.throws(() => validateClefOptions({...options(), ...option}), {code: 'INVALID_INPUT'});
});

test('successful fake load attaches the session and abort before device prevents request', async () => {
  const {deps} = fixture();
  assert.deepEqual(await loadFakeClefCandidate(options(), deps), {ok: true});
  const abort = new AbortController(),
    waiting = deferred<void>(),
    entered = deferred<void>(),
    next = fixture();
  let requested = false;
  next.deps.beforeDevice = async () => {
    entered.resolve();
    await waiting.promise;
  };
  next.deps.gpu.requestAdapter = async () => {
    const adapter: FakeAdapter = {
      features: new Set(['shader-f16']),
      info: {isFallbackAdapter: false},
      limits: CLEF_LIMITS,
      requestDevice() {
        requested = true;
        return Promise.resolve(next.gpuDevice);
      }
    };
    return adapter;
  };
  const result = loadFakeClefCandidate({...options(), signal: abort.signal}, next.deps);
  await entered.promise;
  abort.abort();
  waiting.resolve();
  await assert.rejects(result, {name: 'AbortError'});
  assert.equal(requested, false);
});

test('late device after cancellation is destroyed and upload abort retains initialized owner', async () => {
  const abort = new AbortController(),
    waiting = deferred<FakeGpuDevice>(),
    entered = deferred<void>(),
    next = fixture();
  next.deps.gpu.requestAdapter = async () => ({
    features: new Set(['shader-f16']),
    info: {isFallbackAdapter: false},
    limits: CLEF_LIMITS,
    requestDevice() {
      entered.resolve();
      return waiting.promise;
    }
  });
  const result = loadFakeClefCandidate({...options(), signal: abort.signal}, next.deps);
  await entered.promise;
  abort.abort();
  await assert.rejects(result, {name: 'AbortError'});
  waiting.resolve(next.gpuDevice);
  await tick();
  assert.deepEqual(next.events, ['gpu.destroy']);
  const current = fixture(),
    upload = deferred(),
    started = deferred(),
    controller = new AbortController();
  current.deps.upload = async () => {
    started.resolve();
    await upload.promise;
    return [];
  };
  const pending = loadFakeClefCandidate({...options(), signal: controller.signal}, current.deps);
  await started.promise;
  controller.abort();
  await tick();
  assert.deepEqual(current.events, []);
  upload.resolve();
  await assert.rejects(pending, {name: 'AbortError'});
  assert.deepEqual(current.events, ['retain']);
});

test('abort during upload hashing retains the initialized owner without issuing a write', async (t) => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  const entered = deferred(),
    hash = deferred<ArrayBuffer>(),
    current = fixture(),
    controller = new AbortController();
  const reason = Error('hash cancelled'),
    bytes = Uint8Array.of(0, 60, 0, 64);
  let writes = 0;
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: {
      subtle: {
        digest() {
          entered.resolve();
          return hash.promise;
        }
      }
    }
  });
  t.after(() =>
    original ? Object.defineProperty(globalThis, 'crypto', original) : Reflect.deleteProperty(globalThis, 'crypto')
  );
  const entry: ClefTensorEntry = {
    position: 17,
    target: 'x',
    path: 'fixture.pending',
    dtype: 'float16',
    shape: [2],
    bytes: 4,
    sha256: '00'.repeat(32),
    chunks: [{offset: 0, length: 4, sha256: '00'.repeat(32)}]
  };
  current.deps.upload = (_owner, _source, signal) =>
    uploadFakeClefParameters([entry], {
      fake: true,
      signal,
      createTensor: () => ({}),
      readRange: async () => bytes,
      writeRange: () => {
        writes++;
      }
    });
  const load = loadFakeClefCandidate({...options(), signal: controller.signal}, current.deps);
  const outcome = load.then(
    () => 'resolved',
    (error) => error
  );
  await entered.promise;
  controller.abort(reason);
  const observed = await Promise.race([outcome, new Promise((resolve) => setTimeout(() => resolve('pending'), 50))]);
  hash.resolve(new Uint8Array(32).buffer);
  await tick();
  assert.equal(observed, reason);
  assert.equal(writes, 0);
  assert.deepEqual(current.events, ['retain']);
});

test('device loss during upload retains the owner instead of attaching a session', async () => {
  const current = fixture(),
    upload = deferred(),
    started = deferred();
  let sessionCalls = 0;
  current.deps.upload = async () => {
    started.resolve();
    await upload.promise;
    return [];
  };
  current.deps.session = () => {
    sessionCalls++;
    return {};
  };
  const pending = loadFakeClefCandidate(options(), current.deps);
  await started.promise;
  current.loss.resolve({message: 'device lost'});
  await tick();
  upload.resolve();
  await assert.rejects(pending, {code: 'MODEL_LOAD_FAILED'});
  assert.deepEqual(current.events, ['retain']);
  assert.equal(sessionCalls, 0);
});

test('adapter fallback status must explicitly identify a non-fallback GPU', () => {
  const adapter = {features: new Set(['shader-f16']), limits: CLEF_LIMITS, requestDevice() {}};
  assert.throws(() => validateClefAdapter(adapter), {code: 'UNSUPPORTED_ENVIRONMENT'});
  assert.throws(() => validateClefAdapter({...adapter, info: {isFallbackAdapter: true}}), {
    code: 'UNSUPPORTED_ENVIRONMENT'
  });
  assert.doesNotThrow(() => validateClefAdapter({...adapter, info: {isFallbackAdapter: false}}));
});
