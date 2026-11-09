import assert from 'node:assert/strict';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {
  MAPIKA4_LIMITS,
  MAPIKA4_MODEL,
  loadFakeMapika4Candidate,
  validateMapika4Options
} from '../src/mapika4b/loader.mjs';
import {MAPIKA4_DLIGHT_ALL_MODEL, MAPIKA4_DLIGHT_MODEL} from '../src/mapika4b/contract.mjs';

type FakeModel = typeof MAPIKA4_MODEL | typeof MAPIKA4_DLIGHT_MODEL | typeof MAPIKA4_DLIGHT_ALL_MODEL;
type FakeDevice = {lost: Promise<unknown>; destroy(): void};
type FakeAdapter = {
  features: Set<string>;
  limits: {maxBufferSize: number; maxStorageBufferBindingSize: number; maxStorageBuffersPerShaderStage: number};
  isFallbackAdapter?: boolean;
  requestDevice(): Promise<FakeDevice>;
};
type FakeSource = {wasm: Uint8Array; tokenizer: unknown; tokenizerConfig: unknown};
type FakeOwner = {
  lifecycle: {loss: unknown; disposed: boolean};
  uncaptured: unknown[];
  disposalGuard: {blocked: boolean; retain(): void};
};
type FakeRunner = {weights: string[]};
type FakeDependencies = {
  fake: true;
  gpu: {requestAdapter(): Promise<FakeAdapter>} | undefined;
  openSource(base: URL, options: {signal: AbortSignal | undefined; model: FakeModel}): Promise<FakeSource>;
  prepareTokenizer(data: unknown, config: unknown): {tokenizer: unknown; config: unknown};
  instantiate(wasm: Uint8Array): Promise<{dispose(): void}>;
  initialize(runtime: {dispose(): void}, gpu: FakeDevice): FakeOwner;
  upload(owner: FakeOwner, source: FakeSource, signal: AbortSignal): Promise<string[]>;
  runner(value: {weights: string[]}): FakeRunner;
  session(value: {runner: FakeRunner; tokenizer?: unknown; config?: unknown}): {runner: FakeRunner};
};
type FakeFixture = {
  events: string[];
  device: FakeDevice;
  adapter: FakeAdapter;
  owner: FakeOwner;
  deps: FakeDependencies;
  options: {model: typeof MAPIKA4_MODEL; assetsUrl: string};
};

function fixture(): FakeFixture {
  const events: string[] = [];
  const device: FakeDevice = {
    lost: new Promise(() => {}),
    destroy() {
      events.push('destroy');
    }
  };
  const adapter: FakeAdapter = {
    features: new Set(['shader-f16']),
    limits: {...MAPIKA4_LIMITS},
    async requestDevice() {
      events.push('device');
      return device;
    }
  };
  const owner: FakeOwner = {
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    disposalGuard: {
      blocked: false,
      retain() {
        events.push('retain');
      }
    }
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
      return {wasm: Uint8Array.of(1), tokenizer: {}, tokenizerConfig: {}};
    },
    prepareTokenizer() {
      events.push('tokenizer');
      return {tokenizer: {}, config: {}};
    },
    async instantiate() {
      events.push('instantiate');
      return {
        dispose() {
          events.push('dispose-runtime');
        }
      };
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
      return value;
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
    options: {model: MAPIKA4_MODEL, assetsUrl: 'https://example.test/assets'}
  };
}

test('loads verified metadata, device and ordered weights before creating the session', async () => {
  const f = fixture();
  const value = (await loadFakeMapika4Candidate(f.options, f.deps)) as {runner: FakeRunner};
  assert.deepEqual(value.runner.weights, ['weight']);
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

test('the optimized model selection is forwarded to the asset source', async () => {
  for (const model of [MAPIKA4_DLIGHT_MODEL, MAPIKA4_DLIGHT_ALL_MODEL]) {
    const f = fixture();
    let selected: FakeModel | undefined;
    const original = f.deps.openSource;
    f.deps.openSource = async (base, options) => {
      selected = options.model;
      return original(base, options);
    };
    await loadFakeMapika4Candidate({...f.options, model}, f.deps);
    assert.equal(selected, model);
  }
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
    assert.throws(() => validateMapika4Options(options), {code: 'INVALID_INPUT'});
  assert.equal(accessed, false);
  for (const change of ['no-gpu', 'no-f16', 'fallback', 'limit']) {
    const sample = fixture();
    if (change === 'no-gpu') sample.deps.gpu = undefined;
    if (change === 'no-f16') sample.adapter.features.clear();
    if (change === 'fallback') sample.adapter.isFallbackAdapter = true;
    if (change === 'limit') sample.adapter.limits.maxBufferSize--;
    await assert.rejects(loadFakeMapika4Candidate(sample.options, sample.deps), {code: 'UNSUPPORTED_ENVIRONMENT'});
    assert.equal(sample.events.includes('source'), false);
  }
});

test('aborting a pending device request destroys the device when it resolves', async () => {
  const f = fixture(),
    controller = new AbortController();
  let resolveDevice: ((device: FakeDevice | PromiseLike<FakeDevice>) => void) | undefined;
  f.adapter.requestDevice = () =>
    new Promise((resolve) => {
      resolveDevice = resolve;
    });
  const pending = loadFakeMapika4Candidate({...f.options, signal: controller.signal}, f.deps);
  while (!resolveDevice) await nextTurn();
  controller.abort();
  await assert.rejects(pending, {name: 'AbortError'});
  assert.ok(!f.events.includes('destroy'));
  resolveDevice!(f.device);
  await nextTurn();
  assert.ok(f.events.includes('destroy'));
  assert.ok(!f.events.includes('initialize'));
});

test('device loss during upload retains the initialized runtime', async () => {
  const f = fixture();
  let lose!: (reason: unknown) => void, entered!: () => void;
  f.device.lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  f.deps.upload = async (_owner, _source, signal) =>
    new Promise<string[]>((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      entered();
    });
  const pending = loadFakeMapika4Candidate(f.options, f.deps);
  await started;
  lose({reason: 'unknown'});
  await assert.rejects(pending, {code: 'MODEL_LOAD_FAILED'});
  assert.ok(f.events.includes('retain'));
  assert.ok(!f.events.includes('session'));
});
