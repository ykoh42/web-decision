import assert from 'node:assert/strict';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {IMAJEV_MODEL} from '../src/imajev/contract.mjs';
import {IMAJEV_LIMITS, loadFakeImajevCandidate, validateImajevOptions} from '../src/imajev/candidate-loader.mjs';

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
    limits: {...IMAJEV_LIMITS},
    async requestDevice() {
      events.push('device');
      return device;
    }
  };
  const source = {
    plan: {bindings: Array.from({length: 1082}, () => ({})), maxInputTokens: 128},
    async readAsset(name) {
      events.push(name);
      return Uint8Array.of(1);
    },
    close() {
      events.push('close');
    }
  };
  const owner = {
    phase: 'prepared',
    blocked: false,
    lost: false,
    uncaptured: [],
    retain() {
      events.push('retain');
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
      return {};
    },
    async instantiate() {
      events.push('instantiate');
      return {
        dispose() {
          events.push('late-dispose');
        }
      };
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
  assert.equal(result.maxInputTokens, 128);
  assert.deepEqual(result.weights, ['weights']);
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
    if (change === 'no-adapter') f.deps.gpu.requestAdapter = async () => null;
    if (change === 'no-f16') f.adapter.features.clear();
    if (change === 'missing-limit') delete f.adapter.limits.maxBufferSize;
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
      assert.equal(error.code, 'MODEL_LOAD_FAILED');
      if (stage !== 'blocked') assert.equal(error.cause, failure);
      return true;
    });
    assert.ok(f.events.includes('close'));
    assert.ok(f.events.includes('fail'));
    if (stage === 'blocked' || stage === 'session') assert.ok(f.events.includes('retain'));
  }
});

test('abort while device request is pending destroys the late device and closes source', async () => {
  const f = fixture(),
    controller = new AbortController();
  let resolveDevice;
  f.adapter.requestDevice = () =>
    new Promise((resolve) => {
      resolveDevice = resolve;
    });
  const result = loadFakeImajevCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!resolveDevice) await nextTurn();
  controller.abort();
  await assert.rejects(result, {name: 'AbortError'});
  assert.ok(f.events.includes('close'));
  assert.equal(f.events.includes('destroy'), false);
  resolveDevice(f.device);
  await nextTurn();
  assert.ok(f.events.includes('destroy'));
  assert.ok(f.events.includes('close'));
});

test('abort during device request observes a late rejection', async () => {
  const f = fixture();
  const controller = new AbortController();
  const reason = Error('cancel device request');
  let rejectDevice;
  f.adapter.requestDevice = () =>
    new Promise((_resolve, reject) => {
      rejectDevice = reject;
    });
  const result = loadFakeImajevCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!rejectDevice) await nextTurn();
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  assert.ok(f.events.includes('close'));
  rejectDevice(Error('late device failure'));
  await nextTurn();
  assert.equal(f.events.includes('instantiate'), false);
});

test('abort during small asset reads closes the source and observes late failures', async () => {
  const f = fixture();
  const reads = [];
  f.source.readAsset = () =>
    new Promise((resolve, reject) => {
      reads.push({resolve, reject});
    });
  const controller = new AbortController();
  const reason = Error('cancel asset reads');
  const job = loadFakeImajevCandidate({...f.options, signal: controller.signal}, f.deps);
  while (reads.length < 5) await nextTurn();
  controller.abort(reason);
  await assert.rejects(job, (error) => error === reason);
  assert.ok(f.events.includes('close'));
  assert.ok(f.events.includes('fail'));
  assert.equal(f.events.includes('device'), false);
  reads[0].resolve(Uint8Array.of(1));
  for (const read of reads.slice(1)) read.reject(Error('late asset failure'));
  await nextTurn();
});

test('device loss during upload aborts loading and retains initialized resources', async () => {
  const f = fixture();
  let lose, entered;
  f.device.lost = new Promise((resolve) => {
    lose = resolve;
  });
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  f.deps.upload = async (_owner, _source, {signal}) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      entered();
    });
  const result = loadFakeImajevCandidate(f.options, f.deps);
  await started;
  lose({reason: 'unknown'});
  await assert.rejects(result, {code: 'MODEL_LOAD_FAILED'});
  assert.ok(f.events.includes('retain'));
  assert.ok(f.events.includes('close'));
  assert.equal(f.events.includes('session'), false);
});
