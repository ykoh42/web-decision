import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {FLASH_LIMITS, FLASH_MODEL, loadFakeFlashCandidate, validateFlashOptions} from '../src/pplx/loader.mjs';
import {uploadFlashParameters} from '../src/pplx/upload-parameters.mjs';
import type {FlashQ4CandidateEntry, FlashTensor} from '../src/pplx/types.js';

type FakeDependencies = Parameters<typeof loadFakeFlashCandidate>[1];
type FakeRuntime = Awaited<ReturnType<FakeDependencies['instantiate']>>;
type FakeOwner = ReturnType<FakeDependencies['initialize']>;
type FakeDevice = Parameters<FakeDependencies['initialize']>[1];
type FakeGpu = NonNullable<FakeDependencies['gpu']>;
interface FakeAdapterLimits extends Record<string, number> {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxStorageBuffersPerShaderStage: number;
}

type FakeAdapter = {
  features: Set<string>;
  limits: FakeAdapterLimits;
  isFallbackAdapter?: unknown;
  requestDevice: NonNullable<Awaited<ReturnType<FakeGpu['requestAdapter']>>>['requestDevice'];
};

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(reason?: unknown): void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
}

function fakeTensor(): FlashTensor {
  return {
    byteOffset: 0,
    dtype: 'uint8',
    shape: [],
    copyFrom() {},
    getDataPtr() {
      return 1;
    },
    toArray() {
      return [];
    }
  };
}

function fixture(): {
  events: string[];
  device: FakeDevice;
  adapter: FakeAdapter;
  owner: FakeOwner;
  deps: FakeDependencies;
  options: {model: typeof FLASH_MODEL; assetsUrl: string};
} {
  const events: string[] = [];
  const device: FakeDevice = {
    lost: new Promise(() => {}),
    destroy() {
      events.push('destroy');
    }
  };
  const adapter: FakeAdapter = {
    features: new Set(['shader-f16']),
    limits: {...FLASH_LIMITS},
    isFallbackAdapter: false,
    async requestDevice() {
      events.push('device');
      return device;
    }
  };
  const tensor = fakeTensor();
  const runtime: FakeRuntime = {
    lib: {webGPUContext: {copyRawBytesToBuffer() {}}},
    empty() {
      return tensor;
    },
    cpu() {
      return {};
    },
    dispose() {
      events.push('dispose-runtime');
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
    device,
    gpu: device,
    fn: () => tensor,
    lifecycle: {loss: null, disposed: false},
    disposalGuard: {
      blocked: false,
      retain() {
        events.push('retain');
      }
    },
    uncaptured: [],
    teardownObserver: null,
    teardownTimeoutMs: 100
  };
  const deps: FakeDependencies = {
    fake: true,
    gpu: {
      async requestAdapter() {
        events.push('adapter');
        return adapter;
      }
    },
    async openSource() {
      events.push('source');
      return {
        manifest: {state: 'pplx-q4-joint-assets-pending', entries: []},
        wasm: Uint8Array.of(1),
        tokenizer: {},
        tokenizerConfig: {},
        async readRange() {
          return new ArrayBuffer(0);
        }
      };
    },
    prepareTokenizer() {
      events.push('tokenizer');
      return {
        tokenizer: {encode: () => ({ids: []})},
        config: {codes: [], markerId: 0, escapedTokens: [], addedTokens: [], addedTokenIds: [], maxInputTokens: 1}
      };
    },
    async instantiate() {
      events.push('instantiate');
      return runtime;
    },
    initialize() {
      events.push('initialize');
      return owner;
    },
    async upload() {
      events.push('upload');
      return ['weight'];
    },
    runner(value) {
      events.push('runner');
      return {
        ...value,
        run: async () => [],
        dispose() {},
        releaseCachedMemory() {}
      };
    },
    session(value) {
      events.push('session');
      return value;
    }
  };
  return {
    events,
    device,
    adapter,
    owner,
    deps,
    options: {model: FLASH_MODEL, assetsUrl: 'https://example.test/assets'}
  };
}

test('loads verified metadata, device and ordered weights before creating the session', async () => {
  const f = fixture();
  const value = await loadFakeFlashCandidate(f.options, f.deps);
  assert.deepEqual((value as {runner: {weights: unknown[]}}).runner.weights, ['weight']);
  assert.deepEqual(f.events, [
    'adapter',
    'source',
    'tokenizer',
    'device',
    'instantiate',
    'initialize',
    'upload',
    'runner',
    'session'
  ]);
});

test('rejects unsupported options and adapters before opening assets', async () => {
  const f = fixture();
  let accessed = false;
  const getter = {...f.options};
  Object.defineProperty(getter, 'cache', {
    get() {
      accessed = true;
      return false;
    }
  });
  for (const options of [
    getter,
    {...f.options, cache: true},
    {...f.options, offline: true},
    {...f.options, onProgress() {}},
    {...f.options, assetsUrl: 'file:///tmp/'}
  ])
    assert.throws(() => validateFlashOptions(options), {code: 'INVALID_INPUT'});
  assert.equal(accessed, false);
  for (const change of ['no-gpu', 'no-f16', 'fallback', 'limit']) {
    const sample = fixture();
    if (change === 'no-gpu') sample.deps.gpu = undefined;
    if (change === 'no-f16') sample.adapter.features.clear();
    if (change === 'fallback') sample.adapter.isFallbackAdapter = true;
    if (change === 'limit') sample.adapter.limits.maxBufferSize--;
    await assert.rejects(loadFakeFlashCandidate(sample.options, sample.deps), {code: 'UNSUPPORTED_ENVIRONMENT'});
    assert.equal(sample.events.includes('source'), false);
  }
});

test('aborting a pending device request destroys the device when it resolves', async () => {
  const f = fixture(),
    controller = new AbortController(),
    request = deferred<FakeDevice>();
  let requestStarted = false;
  f.adapter.requestDevice = () => {
    requestStarted = true;
    return request.promise;
  };
  const pending = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!requestStarted) await nextTurn();
  controller.abort();
  await assert.rejects(pending, {name: 'AbortError'});
  assert.ok(!f.events.includes('destroy'));
  request.resolve(f.device);
  await nextTurn();
  assert.ok(f.events.includes('destroy'));
  assert.ok(!f.events.includes('initialize'));
});

test('a late device whose destruction fails remains retained after cancellation', async () => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('cancelled while requesting device'),
    retained: Array<{runtime: unknown; device: unknown}> = [],
    request = deferred<FakeDevice>();
  let requestStarted = false,
    destroys = 0;
  f.deps.retainResources = (runtime, device) => retained.push({runtime, device});
  f.device.destroy = () => {
    destroys++;
    throw new Error('device destruction failed');
  };
  f.adapter.requestDevice = () => {
    requestStarted = true;
    return request.promise;
  };
  const pending = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!requestStarted) await nextTurn();
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  request.resolve(f.device);
  await nextTurn();
  assert.equal(destroys, 1);
  assert.deepEqual(retained, [{runtime: undefined, device: f.device}]);
  assert.ok(!f.events.includes('initialize'));
});

test('failed device destruction after a load error retains the device', async () => {
  const f = fixture(),
    retained: Array<{runtime: unknown; device: unknown}> = [];
  let destroys = 0;
  f.deps.retainResources = (runtime, device) => retained.push({runtime, device});
  f.device.destroy = () => {
    destroys++;
    throw new Error('device destruction failed');
  };
  f.deps.instantiate = async () => {
    throw new Error('instantiate failed');
  };
  await assert.rejects(loadFakeFlashCandidate(f.options, f.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.equal(destroys, 1);
  assert.deepEqual(retained, [{runtime: undefined, device: f.device}]);
  assert.ok(!f.events.includes('initialize'));
});

test('a late device request rejection is observed after cancellation', async () => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('cancelled while requesting device'),
    request = deferred<FakeDevice>();
  let requestStarted = false;
  f.adapter.requestDevice = () => {
    requestStarted = true;
    return request.promise;
  };
  const pending = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!requestStarted) await nextTurn();
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  request.reject(new Error('late request failure'));
  await nextTurn();
  assert.ok(!f.events.includes('initialize'));
});

test('device loss during upload retains the initialized runtime', async () => {
  const f = fixture();
  const loss = deferred<unknown>(),
    started = deferred<void>();
  Object.assign(f.device, {lost: loss.promise});
  f.deps.upload = async (_owner, _source, signal) =>
    new Promise<unknown[]>((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      started.resolve(undefined);
    });
  const pending = loadFakeFlashCandidate(f.options, f.deps);
  await started.promise;
  loss.resolve({reason: 'unknown'});
  await assert.rejects(pending, {code: 'MODEL_LOAD_FAILED'});
  assert.ok(f.events.includes('retain'));
  assert.ok(!f.events.includes('session'));
});

test('abort during the real upload hash retains the owner and observes late failure', async (t) => {
  const f = fixture();
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const entries: FlashQ4CandidateEntry[] = Array.from({length: 682}, (_, index) => ({
    position: index + 5,
    shape: [1],
    dtype: 'uint32',
    bindingKind: 'q4-candidate',
    outputPath: `weight-${index}`,
    outputBytes: 4,
    sourceOffset: index * 4,
    conversion: 'none',
    ranges: [{offset: 0, bytes: 4, sha256}]
  }));
  const started = deferred<void>(),
    digest = deferred<ArrayBuffer>();
  t.mock.method(crypto.subtle, 'digest', () => {
    started.resolve(undefined);
    return digest.promise;
  });
  f.deps.upload = (_owner, _source, signal) =>
    uploadFlashParameters(
      {state: 'pplx-q4-joint-assets-pending', entries},
      {
        signal,
        createTensor: () => fakeTensor(),
        readRange: () => bytes.slice().buffer,
        writeRange: () => f.events.push('write')
      }
    );
  const controller = new AbortController();
  const reason = Error('cancel upload hash');
  const job = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  await started.promise;
  controller.abort(reason);
  await assert.rejects(job, (error) => error === reason);
  assert.ok(f.events.includes('retain'));
  assert.equal(f.events.includes('write'), false);
  assert.equal(f.events.includes('session'), false);
  digest.reject(Error('late digest failure'));
  await nextTurn();
});
