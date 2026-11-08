import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {FLASH_LIMITS, FLASH_MODEL, loadFakeFlashCandidate, validateFlashOptions} from '../src/pplx/loader.mjs';
import {uploadFlashParameters} from '../src/pplx/upload-parameters.mjs';

function fixture() {
  const events = [];
  const device = {
    lost: new Promise(() => {}),
    destroy() {
      events.push('destroy');
    }
  };
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {...FLASH_LIMITS},
    async requestDevice() {
      events.push('device');
      return device;
    }
  };
  const owner = {
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    disposalGuard: {
      blocked: false,
      retain() {
        events.push('retain');
      }
    }
  };
  const deps = {
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
    options: {model: FLASH_MODEL, assetsUrl: 'https://example.test/assets'}
  };
}

test('loads verified metadata, device and ordered weights before creating the session', async () => {
  const f = fixture();
  const value = await loadFakeFlashCandidate(f.options, f.deps);
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
    controller = new AbortController();
  let resolveDevice;
  f.adapter.requestDevice = () =>
    new Promise((resolve) => {
      resolveDevice = resolve;
    });
  const pending = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!resolveDevice) await nextTurn();
  controller.abort();
  await assert.rejects(pending, {name: 'AbortError'});
  assert.ok(!f.events.includes('destroy'));
  resolveDevice(f.device);
  await nextTurn();
  assert.ok(f.events.includes('destroy'));
  assert.ok(!f.events.includes('initialize'));
});

test('a late device whose destruction fails remains retained after cancellation', async () => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('cancelled while requesting device'),
    retained = [];
  let resolveDevice,
    destroys = 0;
  f.deps.retainResources = (runtime, device) => retained.push({runtime, device});
  f.device.destroy = () => {
    destroys++;
    throw new Error('device destruction failed');
  };
  f.adapter.requestDevice = () => new Promise((resolve) => (resolveDevice = resolve));
  const pending = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!resolveDevice) await nextTurn();
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  resolveDevice(f.device);
  await nextTurn();
  assert.equal(destroys, 1);
  assert.deepEqual(retained, [{runtime: undefined, device: f.device}]);
  assert.ok(!f.events.includes('initialize'));
});

test('failed device destruction after a load error retains the device', async () => {
  const f = fixture(),
    retained = [];
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
    reason = new Error('cancelled while requesting device');
  let rejectDevice;
  f.adapter.requestDevice = () => new Promise((_resolve, reject) => (rejectDevice = reject));
  const pending = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!rejectDevice) await nextTurn();
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  rejectDevice(new Error('late request failure'));
  await nextTurn();
  assert.ok(!f.events.includes('initialize'));
});

test('device loss during upload retains the initialized runtime', async () => {
  const f = fixture();
  let lose, entered;
  f.device.lost = new Promise((resolve) => {
    lose = resolve;
  });
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  f.deps.upload = async (_owner, _source, signal) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      entered();
    });
  const pending = loadFakeFlashCandidate(f.options, f.deps);
  await started;
  lose({reason: 'unknown'});
  await assert.rejects(pending, {code: 'MODEL_LOAD_FAILED'});
  assert.ok(f.events.includes('retain'));
  assert.ok(!f.events.includes('session'));
});

test('abort during the real upload hash retains the owner and observes late failure', async (t) => {
  const f = fixture();
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const entries = Array.from({length: 682}, (_, index) => ({
    position: index + 5,
    shape: [1],
    dtype: 'uint32',
    bindingKind: 'q4-candidate',
    outputBytes: 4,
    ranges: [{offset: 0, bytes: 4, sha256}]
  }));
  let entered, rejectHash;
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  t.mock.method(crypto.subtle, 'digest', () => {
    entered();
    return new Promise((_resolve, reject) => {
      rejectHash = reject;
    });
  });
  f.deps.upload = (_owner, _source, signal) =>
    uploadFlashParameters(
      {state: 'pplx-q4-joint-assets-pending', entries},
      {
        signal,
        createTensor: () => ({}),
        readRange: () => bytes.slice().buffer,
        writeRange: () => f.events.push('write')
      }
    );
  const controller = new AbortController();
  const reason = Error('cancel upload hash');
  const job = loadFakeFlashCandidate({...f.options, signal: controller.signal}, f.deps);
  await started;
  controller.abort(reason);
  await assert.rejects(job, (error) => error === reason);
  assert.ok(f.events.includes('retain'));
  assert.equal(f.events.includes('write'), false);
  assert.equal(f.events.includes('session'), false);
  rejectHash(Error('late digest failure'));
  await nextTurn();
});
