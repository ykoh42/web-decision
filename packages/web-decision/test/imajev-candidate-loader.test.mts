import assert from 'node:assert/strict';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {IMAJEV_MODEL} from '../src/imajev/contract.mjs';
import {IMAJEV_LIMITS, loadFakeImajevCandidate, validateImajevOptions} from '../src/imajev/candidate-loader.mjs';

type FakeDependencies = Parameters<typeof loadFakeImajevCandidate>[1];
type FakeRuntime = Awaited<ReturnType<FakeDependencies['instantiate']>>;
type FakeSource = Awaited<ReturnType<FakeDependencies['openSource']>>;
type FakeOwner = ReturnType<ReturnType<FakeDependencies['ownership']>['initialize']>;
type FakeProcessor = ReturnType<FakeDependencies['processor']>;
type MutableOwner = Omit<FakeOwner, 'blocked' | 'lost'> & {blocked: boolean; lost: unknown};
type FixtureDependencies = Omit<FakeDependencies, 'gpu'> & {gpu: GPU | undefined};

interface FakeGpuDevice {
  lost: Promise<unknown>;
  destroy(): void;
}

interface FakeGpuAdapter {
  features: Set<string>;
  limits: {
    maxStorageBuffersPerShaderStage: number;
    maxStorageBufferBindingSize: number;
    maxBufferSize: number;
  };
  requestDevice(): Promise<FakeGpuDevice>;
}

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

function asGpuDeviceApi(device: FakeGpuDevice): GPUDevice {
  return device as unknown as GPUDevice;
}

function asGpuAdapterApi(adapter: FakeGpuAdapter): GPUAdapter {
  return adapter as unknown as GPUAdapter;
}

function fixture(): {
  events: string[];
  device: FakeGpuDevice;
  adapter: FakeGpuAdapter;
  source: FakeSource;
  owner: MutableOwner;
  deps: FixtureDependencies;
  options: {model: typeof IMAJEV_MODEL; assetsUrl: string};
} {
  const events: string[] = [];
  const device: FakeGpuDevice = {
    lost: new Promise(() => {}),
    destroy() {
      events.push('destroy');
    }
  };
  const adapter: FakeGpuAdapter = {
    features: new Set(['shader-f16']),
    limits: {...IMAJEV_LIMITS},
    async requestDevice() {
      events.push('device');
      return device;
    }
  };
  const sourceHash = '0'.repeat(64);
  const bindings: FakeSource['plan']['bindings'] = Array.from({length: 1082}, (_, index) => {
    const argument = `argument-${index}`;
    if (index < 1080)
      return {
        argument,
        target: `target-${index}`,
        dtype: 'float32',
        shape: [1],
        bytes: 4,
        offset: 0,
        path: 'fixture.pending',
        fileSha256: sourceHash,
        rangeSha256: sourceHash,
        fileBytes: 4,
        chunks: [{offset: 0, bytes: 4, sha256: sourceHash}]
      };
    return {
      argument,
      target: index === 1080 ? 'body.rotary_emb.inv_freq' : 'body.rotary_emb.original_inv_freq',
      dtype: 'float32',
      shape: [32],
      sha256: 'ef51962de52263845c53966465eb225b734f9bf294e3e382229ec4630404077f',
      bytesHex: '0'.repeat(256)
    };
  });
  const source: FakeSource = {
    plan: {bindings, maxInputTokens: 128},
    readRange: async () => Uint8Array.of(1),
    async readAsset(name) {
      events.push(name);
      return Uint8Array.of(1);
    },
    close() {
      events.push('close');
    }
  };
  const tensor = {
    byteOffset: 0,
    dtype: 'float32',
    shape: [],
    getDataPtr() {
      return 1;
    }
  };
  const runtime: FakeRuntime = {
    dispose() {
      events.push('late-dispose');
    },
    initWebGPU() {},
    beginScope() {},
    webgpu() {
      return {deviceType: 0, deviceId: 0, sync() {}};
    },
    createVirtualMachine() {
      return {getFunction: () => () => tensor};
    },
    empty() {
      return tensor;
    }
  };
  const owner: MutableOwner = {
    runtime,
    gpu: asGpuDeviceApi(device),
    device: {deviceType: 0, deviceId: 0, sync() {}},
    fn: () => tensor,
    lifecycle: {disposed: false, loss: null},
    uncaptured: [],
    disposalGuard: {blocked: false, retain() {}},
    teardownObserver: {beforeDispose: async () => {}, afterDispose: async () => {}},
    teardownTimeoutMs: 1,
    phase: 'prepared',
    blocked: false,
    lost: false,
    resources: [],
    retain() {
      events.push('retain');
    },
    hold() {
      return {release() {}};
    }
  };
  const processor: FakeProcessor = {
    prepare: () => ({tokenCount: 0, questions: []}),
    finish: () => ({tokenCount: 0, answers: []})
  };
  const gpu: GPU = Object.assign(new EventTarget(), {
    wgslLanguageFeatures: new Set<string>(),
    getPreferredCanvasFormat: () => 'rgba8unorm' as GPUTextureFormat,
    async requestAdapter() {
      events.push('adapter');
      return asGpuAdapterApi(adapter);
    }
  });
  const deps: FixtureDependencies = {
    fake: true,
    gpu,
    ownership: () => ({
      initialize() {
        events.push('initialize');
        return owner;
      },
      fail(_runtime, gpu) {
        events.push('fail');
        if (events.includes('initialize')) owner.retain();
        else gpu?.destroy();
      }
    }),
    async openSource() {
      events.push('source');
      return source;
    },
    processor() {
      events.push('processor');
      return processor;
    },
    async instantiate() {
      events.push('instantiate');
      return runtime;
    },
    async upload() {
      events.push('upload');
      return ['weights'];
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
    source,
    owner,
    deps,
    options: {model: IMAJEV_MODEL, assetsUrl: 'https://example.test/assets/'}
  };
}

test('candidate orchestrates verified source, device, upload and session in order', async () => {
  const f = fixture();
  const result = await loadFakeImajevCandidate(f.options, f.deps);
  const session = result as {maxInputTokens: number; weights: unknown[]};
  assert.equal(session.maxInputTokens, 128);
  assert.deepEqual(session.weights, ['weights']);
  assert.ok(f.events.indexOf('processor') < f.events.indexOf('device'));
  assert.ok(f.events.indexOf('initialize') < f.events.indexOf('upload'));
  assert.ok(f.events.indexOf('upload') < f.events.indexOf('session'));
  assert.equal(f.events.at(-1), 'close');
});

test('invalid options and insufficient device limits reject before assets are opened', async () => {
  const f = fixture();
  for (const update of [{cache: true}, {offline: true}, {onProgress() {}}, {assetsUrl: 'file:///tmp/'}])
    assert.throws(() => validateImajevOptions({...f.options, ...update}), {code: 'INVALID_INPUT'});
  f.adapter.limits.maxBufferSize--;
  await assert.rejects(loadFakeImajevCandidate(f.options, f.deps), {code: 'UNSUPPORTED_ENVIRONMENT'});
  assert.equal(f.events.includes('source'), false);
});

test('rejects unsafe options and asset URLs without invoking getters', () => {
  const f = fixture();
  let accessed = false;
  const getter = {...f.options};
  Object.defineProperty(getter, 'cache', {
    get() {
      accessed = true;
      return false;
    }
  });
  const inherited = Object.assign(Object.create({custom: true}), f.options);
  for (const options of [
    null,
    [],
    getter,
    inherited,
    {...f.options, [Symbol('extra')]: true},
    ...['https://user:pass@example.test/', 'https://example.test/?key=value', 'https://example.test/#part'].map(
      (assetsUrl) => ({...f.options, assetsUrl})
    )
  ]) {
    assert.throws(() => validateImajevOptions(options), {code: 'INVALID_INPUT'});
  }
  assert.equal(accessed, false);
  assert.equal(
    validateImajevOptions({...f.options, assetsUrl: 'https://example.test/assets'}).base.pathname,
    '/assets/'
  );
});

test('rejects unsupported GPUs before opening assets', async () => {
  for (const change of ['no-gpu', 'no-adapter', 'no-f16', 'missing-limit']) {
    const f = fixture();
    if (change === 'no-gpu') f.deps.gpu = undefined;
    if (change === 'no-adapter') f.deps.gpu!.requestAdapter = async () => null;
    if (change === 'no-f16') f.adapter.features.clear();
    if (change === 'missing-limit') Reflect.deleteProperty(f.adapter.limits, 'maxBufferSize');
    await assert.rejects(loadFakeImajevCandidate(f.options, f.deps), {code: 'UNSUPPORTED_ENVIRONMENT'});
    assert.equal(f.events.includes('source'), false);
  }
});

test('closes the source and preserves the failure when loading cannot complete', async () => {
  for (const stage of ['asset', 'device', 'runtime', 'blocked', 'session']) {
    const f = fixture();
    const failure = new Error(`failed ${stage}`);
    if (stage === 'asset')
      f.source.readAsset = async () => {
        throw failure;
      };
    if (stage === 'device')
      f.adapter.requestDevice = async () => {
        throw failure;
      };
    if (stage === 'runtime')
      f.deps.instantiate = async () => {
        throw failure;
      };
    if (stage === 'blocked') f.owner.blocked = true;
    if (stage === 'session')
      f.deps.session = () => {
        throw failure;
      };
    await assert.rejects(loadFakeImajevCandidate(f.options, f.deps), (error) => {
      const failureRecord = error as {code?: unknown; cause?: unknown};
      assert.equal(failureRecord.code, 'MODEL_LOAD_FAILED');
      if (stage !== 'blocked') assert.equal(failureRecord.cause, failure);
      return true;
    });
    assert.ok(f.events.includes('close'));
    assert.ok(f.events.includes('fail'));
    if (stage === 'blocked' || stage === 'session') assert.ok(f.events.includes('retain'));
  }
});

test('abort while device request is pending destroys the late device and closes source', async () => {
  const f = fixture(),
    controller = new AbortController(),
    request = deferred<FakeGpuDevice>();
  let requestStarted = false;
  f.adapter.requestDevice = () => {
    requestStarted = true;
    return request.promise;
  };
  const result = loadFakeImajevCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!requestStarted) await nextTurn();
  controller.abort();
  await assert.rejects(result, {name: 'AbortError'});
  assert.ok(f.events.includes('close'));
  assert.equal(f.events.includes('destroy'), false);
  request.resolve(f.device);
  await nextTurn();
  assert.ok(f.events.includes('destroy'));
  assert.ok(f.events.includes('close'));
});

test('abort during device request observes a late rejection', async () => {
  const f = fixture();
  const controller = new AbortController();
  const reason = Error('cancel device request');
  const request = deferred<FakeGpuDevice>();
  let requestStarted = false;
  f.adapter.requestDevice = () => {
    requestStarted = true;
    return request.promise;
  };
  const result = loadFakeImajevCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!requestStarted) await nextTurn();
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  assert.ok(f.events.includes('close'));
  request.reject(Error('late device failure'));
  await nextTurn();
  assert.equal(f.events.includes('instantiate'), false);
});

test('abort during small asset reads closes the source and observes late failures', async () => {
  const f = fixture();
  const reads: Deferred<Uint8Array<ArrayBuffer>>[] = [];
  f.source.readAsset = () => {
    const read = deferred<Uint8Array<ArrayBuffer>>();
    reads.push(read);
    return read.promise;
  };
  const controller = new AbortController();
  const reason = Error('cancel asset reads');
  const job = loadFakeImajevCandidate({...f.options, signal: controller.signal}, f.deps);
  while (reads.length < 5) await nextTurn();
  controller.abort(reason);
  await assert.rejects(job, (error) => error === reason);
  assert.ok(f.events.includes('close'));
  assert.ok(f.events.includes('fail'));
  assert.equal(f.events.includes('device'), false);
  reads[0]!.resolve(Uint8Array.of(1));
  for (const read of reads.slice(1)) read.reject(Error('late asset failure'));
  await nextTurn();
});

test('device loss during upload aborts loading and retains initialized resources', async () => {
  const f = fixture();
  const loss = deferred<unknown>(),
    started = deferred<void>();
  f.device.lost = loss.promise;
  f.deps.upload = async (_owner, _source, {signal}) =>
    new Promise<unknown[]>((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      started.resolve(undefined);
    });
  const result = loadFakeImajevCandidate(f.options, f.deps);
  await started.promise;
  loss.resolve({reason: 'unknown'});
  await assert.rejects(result, {code: 'MODEL_LOAD_FAILED'});
  assert.ok(f.events.includes('retain'));
  assert.ok(f.events.includes('close'));
  assert.equal(f.events.includes('session'), false);
});
