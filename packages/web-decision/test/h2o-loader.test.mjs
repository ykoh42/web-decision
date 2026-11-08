import test from 'node:test';
import assert from 'node:assert/strict';
import {runInNewContext} from 'node:vm';
import {
  H2O_CANDIDATE_MODEL_ID,
  loadFakeH2oCandidate,
  loadH2oCandidate,
  PRELIMINARY_LIMITS
} from '../src/h2o/loader.mjs';
import {selectModelFamily} from '../src/model-family.js';
import {WebDecisionError} from '../src/kev.js';

function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return {promise, resolve, reject};
}
function fixture() {
  const lost = deferred(),
    events = [],
    uploaded = [];
  const device = {
    lost: lost.promise,
    destroy() {
      events.push('destroy');
    }
  };
  const source = {
    stage: Uint8Array.of(1),
    manifest: Uint8Array.of(2),
    wasm: Uint8Array.of(3),
    tokenizer: Uint8Array.of(4),
    tokenizerConfig: Uint8Array.of(5),
    tokenizerContract: Uint8Array.of(6),
    inventory: {synthetic: true},
    readRange() {
      return this.inventory;
    },
    async close() {
      events.push('source-close');
    }
  };
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {...PRELIMINARY_LIMITS},
    async requestDevice(request) {
      events.push('device');
      assert.deepEqual(request.requiredLimits, PRELIMINARY_LIMITS);
      return device;
    }
  };
  const runtime = {
    dispose() {
      events.push('runtime-dispose');
    }
  };
  const owner = {
    disposalGuard: {
      retain() {
        events.push('retain');
      }
    }
  };
  const runner = {name: 'runner'},
    session = {name: 'session'},
    plan = {name: 'plan'};
  const deps = {
    fake: true,
    gpu: {
      async requestAdapter() {
        events.push('adapter');
        return adapter;
      }
    },
    async openSource({base}) {
      events.push('source');
      assert.equal(base.href, 'https://example.invalid/assets/');
      return source;
    },
    async preparePlan(stage, manifest) {
      events.push('plan');
      assert.equal(stage[0], 1);
      assert.equal(manifest[0], 2);
      return plan;
    },
    async verifyWasm(wasm) {
      events.push('wasm');
      assert.equal(wasm[0], 3);
      assert.notEqual(wasm, source.wasm);
      return wasm;
    },
    async prepareTokenizer(tokens, config, contract) {
      events.push('tokenizer');
      assert.deepEqual([tokens[0], config[0], contract[0]], [4, 5, 6]);
      return {tokenizer: {}, addedTokens: []};
    },
    async createRuntime(wasm) {
      events.push('runtime');
      assert.equal(wasm[0], 3);
      return runtime;
    },
    initialize(rt, gpu) {
      events.push('initialize');
      assert.equal(rt, runtime);
      assert.equal(gpu, device);
      return owner;
    },
    async upload(args) {
      events.push('upload');
      uploaded.push(args);
      return [];
    },
    attach(actual, weights) {
      events.push('attach');
      assert.equal(actual, owner);
      assert.deepEqual(weights, []);
      return runner;
    },
    async createSession(args) {
      events.push('session');
      assert.equal(args.runner, runner);
      return session;
    },
    makeHasher() {}
  };
  return {lost, events, uploaded, device, source, adapter, runtime, owner, runner, session, deps};
}
const options = {assetsUrl: 'https://example.invalid/assets', cache: false};

for (const [method, event, later, signalIndex] of [
  ['preparePlan', 'plan', ['wasm', 'tokenizer', 'device'], 3],
  ['verifyWasm', 'wasm', ['tokenizer', 'device'], 1],
  ['prepareTokenizer', 'tokenizer', ['device'], 3]
]) {
  test(`abort during ${method} closes the source and observes late rejection`, async () => {
    const f = fixture(),
      started = deferred(),
      pending = deferred(),
      controller = new AbortController(),
      reason = new Error('synthetic cancellation'),
      unhandled = [];
    const onUnhandled = (error) => unhandled.push(error);
    process.on('unhandledRejection', onUnhandled);
    try {
      f.deps[method] = (...args) => {
        f.events.push(event);
        assert.equal(args[signalIndex], controller.signal);
        started.resolve();
        return pending.promise;
      };
      const task = loadFakeH2oCandidate({...options, signal: controller.signal}, f.deps);
      await started.promise;
      controller.abort(reason);
      await assert.rejects(task, (error) => error === reason);
      assert.ok(f.events.includes('source-close'));
      assert.ok(later.every((name) => !f.events.includes(name)));
      pending.reject(new Error('late synthetic rejection'));
      await new Promise((resolve) => setImmediate(resolve));
      assert.deepEqual(unhandled, []);
      assert.ok(later.every((name) => !f.events.includes(name)));
    } finally {
      process.removeListener('unhandledRejection', onUnhandled);
    }
  });
}

test('abort during plan skips later work even if the plan resolves late', async () => {
  const f = fixture(),
    started = deferred(),
    pending = deferred(),
    controller = new AbortController(),
    reason = new Error('synthetic cancellation');
  f.deps.preparePlan = () => {
    started.resolve();
    return pending.promise;
  };
  const task = loadFakeH2oCandidate({...options, signal: controller.signal}, f.deps);
  await started.promise;
  controller.abort(reason);
  await assert.rejects(task, (error) => error === reason);
  pending.resolve({});
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(f.events.includes('source-close'));
  assert.ok(['wasm', 'tokenizer', 'device'].every((name) => !f.events.includes(name)));
});

test('fake glue orders checks and passes pinned source and range reader through', async () => {
  const f = fixture();
  assert.equal(await loadFakeH2oCandidate(options, f.deps), f.session);
  assert.deepEqual(f.events, [
    'adapter',
    'source',
    'plan',
    'wasm',
    'tokenizer',
    'device',
    'runtime',
    'initialize',
    'upload',
    'attach',
    'session',
    'source-close'
  ]);
  assert.equal(f.uploaded[0].readRange(), f.source.inventory);
  assert.equal(f.uploaded[0].inventory, f.source.inventory);
  assert.equal(f.uploaded[0].base.href, 'https://example.invalid/assets/');
  assert.equal(f.events.includes('destroy'), false);
});

test('GPU upload failures during load use the model-load error code', async () => {
  const f = fixture();
  f.deps.upload = async () => {
    f.events.push('upload-failed');
    throw new WebDecisionError('INFERENCE_ERROR', 'GPU lost during weight upload.');
  };
  await assert.rejects(loadFakeH2oCandidate(options, f.deps), (error) => {
    assert.equal(error.code, 'MODEL_LOAD_FAILED');
    assert.equal(error.cause.code, 'INFERENCE_ERROR');
    return true;
  });
  assert.ok(f.events.includes('retain'));
  assert.ok(f.events.includes('source-close'));
  assert.equal(f.events.includes('attach'), false);
});

test('device guard runs immediately before device request and rejects cleanly', async () => {
  const f = fixture();
  f.deps.beforeDevice = async () => {
    f.events.push('device-guard');
  };
  await loadFakeH2oCandidate(options, f.deps);
  assert.ok(f.events.indexOf('device-guard') < f.events.indexOf('device'));
  assert.equal(f.events[f.events.indexOf('device-guard') + 1], 'device');
  const denied = fixture();
  denied.deps.beforeDevice = async () => {
    throw Error('threshold');
  };
  await assert.rejects(loadFakeH2oCandidate(options, denied.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.equal(denied.events.includes('device'), false);
  assert.equal(denied.events.includes('source-close'), true);
});

test('unsupported options and insufficient adapter fail before source or device', async () => {
  for (const extra of [
    {offline: true},
    {cache: true},
    {maxPrefetchBytes: 1},
    {debertaReadback: 'serial'},
    {model: 'h2o'},
    {model: 'kev-0.8b-f16'},
    {model: null},
    {device: 'cpu'},
    {onProgress() {}}
  ]) {
    const f = fixture();
    await assert.rejects(loadFakeH2oCandidate({...options, ...extra}, f.deps), {code: 'INVALID_INPUT'});
    assert.deepEqual(f.events, []);
  }
  for (const bad of ['shader', 'buffer', 'binding', 'storage']) {
    const f = fixture();
    if (bad === 'shader') f.adapter.features.clear();
    if (bad === 'buffer') f.adapter.limits.maxBufferSize--;
    if (bad === 'binding') f.adapter.limits.maxStorageBufferBindingSize--;
    if (bad === 'storage') f.adapter.limits.maxStorageBuffersPerShaderStage--;
    await assert.rejects(loadFakeH2oCandidate(options, f.deps), {code: 'UNSUPPORTED_ENVIRONMENT'});
    assert.deepEqual(f.events, ['adapter']);
  }
});

test('real loader rejects unsupported options before GPU or network use', async () => {
  const runtimeBindings = {Tokenizer: class {}, instantiate: async () => ({}), createPolyfillWASI: () => ({})};
  for (const unsupported of [
    {cache: true},
    {offline: true},
    {maxPrefetchBytes: 1},
    {onProgress() {}},
    {debertaReadback: 'packed'}
  ]) {
    await assert.rejects(
      loadH2oCandidate({...options, model: H2O_CANDIDATE_MODEL_ID, ...unsupported}, {runtimeBindings}),
      {code: 'INVALID_INPUT'}
    );
  }
});

test('inherited, non-enumerable and accessor options fail before executing getters', async () => {
  let getterCalls = 0;
  const inherited = Object.assign(Object.create({model: 'h2o'}), {assetsUrl: options.assetsUrl});
  const inheritedGetter = Object.assign(
    Object.create({
      get model() {
        getterCalls++;
        return 'h2o';
      }
    }),
    {assetsUrl: options.assetsUrl}
  );
  const hiddenUnsupported = {assetsUrl: options.assetsUrl};
  Object.defineProperty(hiddenUnsupported, 'onProgress', {value() {}, enumerable: false});
  const hiddenCache = {assetsUrl: options.assetsUrl};
  Object.defineProperty(hiddenCache, 'cache', {value: true, enumerable: false});
  const ownGetter = {};
  Object.defineProperty(ownGetter, 'assetsUrl', {
    get() {
      getterCalls++;
      return options.assetsUrl;
    },
    enumerable: true
  });
  const modelGetter = {assetsUrl: options.assetsUrl};
  Object.defineProperty(modelGetter, 'model', {
    get() {
      getterCalls++;
      return H2O_CANDIDATE_MODEL_ID;
    },
    enumerable: true
  });
  const symbol = {[Symbol('unknown')]: true, assetsUrl: options.assetsUrl};
  for (const input of [inherited, inheritedGetter, hiddenUnsupported, hiddenCache, ownGetter, modelGetter, symbol]) {
    const f = fixture();
    await assert.rejects(loadFakeH2oCandidate(input, f.deps), {code: 'INVALID_INPUT'});
    assert.deepEqual(f.events, []);
  }
  assert.equal(getterCalls, 0);
});

test('ordinary and null-prototype data-property options remain accepted', async () => {
  assert.equal(H2O_CANDIDATE_MODEL_ID, 'experimental-h2o-lightning-4b-q4f16');
  assert.equal(selectModelFamily({model: H2O_CANDIDATE_MODEL_ID}), 'h2o');
  const ordinary = {...options, model: H2O_CANDIDATE_MODEL_ID, offline: false, maxPrefetchBytes: 0};
  const nullProto = Object.assign(Object.create(null), ordinary);
  for (const input of [ordinary, nullProto]) {
    const f = fixture();
    assert.equal(await loadFakeH2oCandidate(input, f.deps), f.session);
  }
  const explicitUndefined = {...ordinary, onProgress: undefined, debertaReadback: undefined};
  const f = fixture();
  assert.equal(await loadFakeH2oCandidate(explicitUndefined, f.deps), f.session);
});

test('cross-realm ordinary and null-prototype options accept supported values', async () => {
  const controller = new AbortController();
  const ordinary = runInNewContext(
    `({assetsUrl: 'https://example.invalid/assets', cache: false,
    offline: false, maxPrefetchBytes: 0, signal})`,
    {signal: controller.signal}
  );
  const nullProto = runInNewContext(
    `Object.assign(Object.create(null), {assetsUrl: 'https://example.invalid/assets'})`
  );
  for (const input of [ordinary, nullProto]) {
    const f = fixture();
    assert.equal(await loadFakeH2oCandidate(input, f.deps), f.session);
  }
});

test('cross-realm inherited and hidden unsupported fields or accessors are rejected', async () => {
  const input = runInNewContext(`(() => {
    const url = 'https://example.invalid/assets';
    const inherited = Object.assign(Object.create({model: 'h2o'}), {assetsUrl: url});
    const hidden = {assetsUrl: url}; Object.defineProperty(hidden, 'onProgress', {value() {}});
    const symbol = {assetsUrl: url}; symbol[Symbol('x')] = true;
    const getter = {}; Object.defineProperty(getter, 'assetsUrl', {get() {throw Error('must not run')}});
    const polluted = Object.assign(Object.create(Object.assign(Object.create(null), {cache: true})), {assetsUrl: url});
    return [inherited, hidden, symbol, getter, polluted];
  })()`);
  for (const options of input) {
    const f = fixture();
    await assert.rejects(loadFakeH2oCandidate(options, f.deps), {code: 'INVALID_INPUT'});
    assert.deepEqual(f.events, []);
  }
  const f = fixture();
  await assert.rejects(loadFakeH2oCandidate({...options, signal: Object.create(AbortSignal.prototype)}, f.deps), {
    code: 'INVALID_INPUT'
  });
});

test('unexpected inherited properties on Object.prototype do not become options', async () => {
  let calls = 0;
  Object.defineProperty(Object.prototype, 'model', {
    configurable: true,
    get() {
      calls++;
      return 'h2o';
    }
  });
  try {
    const f = fixture();
    await assert.rejects(loadFakeH2oCandidate(options, f.deps), {code: 'INVALID_INPUT'});
    assert.deepEqual(f.events, []);
    assert.equal(calls, 0);
    const isolated = fixture();
    assert.equal(
      await loadFakeH2oCandidate(Object.assign(Object.create(null), options), isolated.deps),
      isolated.session
    );
  } finally {
    delete Object.prototype.model;
  }
});

test('range reader keeps source receiver while upload is active', async () => {
  const f = fixture();
  f.source.readRange = function (key) {
    assert.equal(this, f.source);
    return this.inventory[key];
  };
  f.deps.upload = async ({readRange}) => {
    assert.equal(readRange('synthetic'), true);
    return [];
  };
  assert.equal(await loadFakeH2oCandidate(options, f.deps), f.session);
});

test('abort while device request is pending destroys the late device', async () => {
  const f = fixture(),
    pending = deferred(),
    controller = new AbortController();
  f.adapter.requestDevice = () => {
    f.events.push('device');
    return pending.promise;
  };
  const task = loadFakeH2oCandidate({...options, signal: controller.signal}, f.deps);
  while (!f.events.includes('device')) await new Promise((resolve) => setImmediate(resolve));
  controller.abort();
  pending.resolve(f.device);
  await assert.rejects(task, {name: 'AbortError'});
  assert.ok(f.events.includes('destroy'));
  assert.ok(f.events.includes('source-close'));
  assert.equal(f.events.includes('runtime'), false);
});

test('initialization failure retains runtime rather than disposing partial GPU state', async () => {
  const f = fixture();
  f.deps.initialize = () => {
    f.events.push('initialize-failed');
    throw new Error('failure');
  };
  await assert.rejects(loadFakeH2oCandidate(options, f.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.ok(f.events.includes('source-close'));
  assert.equal(f.events.includes('runtime-dispose'), false);
  assert.equal(f.events.includes('destroy'), false);
});

test('abort after owner exists retains it and closes source', async () => {
  const f = fixture(),
    controller = new AbortController();
  f.deps.upload = async () => {
    f.events.push('upload');
    controller.abort();
    throw controller.signal.reason;
  };
  await assert.rejects(loadFakeH2oCandidate({...options, signal: controller.signal}, f.deps), {name: 'AbortError'});
  assert.ok(f.events.includes('retain'));
  assert.ok(f.events.includes('source-close'));
  assert.equal(f.events.includes('runtime-dispose'), false);
  assert.equal(f.events.includes('destroy'), false);
});

test('abort while source opens closes a late source', async () => {
  const f = fixture(),
    pending = deferred(),
    controller = new AbortController();
  f.deps.openSource = () => {
    f.events.push('source');
    return pending.promise;
  };
  const task = loadFakeH2oCandidate({...options, signal: controller.signal}, f.deps);
  while (!f.events.includes('source')) await new Promise((resolve) => setImmediate(resolve));
  controller.abort();
  await assert.rejects(task, {name: 'AbortError'});
  pending.resolve(f.source);
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(f.events.includes('source-close'));
});

test('device loss while runtime factory is pending disposes the late uninitialized runtime', async () => {
  const f = fixture(),
    pending = deferred();
  f.deps.createRuntime = () => {
    f.events.push('runtime');
    return pending.promise;
  };
  const task = loadFakeH2oCandidate(options, f.deps);
  while (!f.events.includes('runtime')) await new Promise((resolve) => setImmediate(resolve));
  f.lost.resolve({reason: 'fake loss'});
  await assert.rejects(task, {code: 'MODEL_LOAD_FAILED'});
  assert.ok(f.events.includes('destroy'));
  assert.ok(f.events.includes('source-close'));
  pending.resolve(f.runtime);
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(f.events.includes('runtime-dispose'));
  assert.equal(f.events.includes('initialize'), false);
});

test('verification and use observe a snapshot of every small source buffer', async () => {
  const f = fixture(),
    pending = deferred();
  f.deps.preparePlan = async (stage, manifest) => {
    f.events.push('plan');
    await pending.promise;
    assert.deepEqual([...stage], [1]);
    assert.deepEqual([...manifest], [2]);
    return {name: 'plan'};
  };
  const task = loadFakeH2oCandidate(options, f.deps);
  while (!f.events.includes('plan')) await new Promise((resolve) => setImmediate(resolve));
  for (const name of ['stage', 'manifest', 'wasm', 'tokenizer', 'tokenizerConfig', 'tokenizerContract'])
    f.source[name][0] = 99;
  pending.resolve();
  assert.equal(await task, f.session);
});

test('DLight model snapshot is bound to the selected registration before any async work', async () => {
  const model = 'experimental-h2o-lightning-4b-q4f16-dlight';
  const f = fixture(),
    input = {...options, model};
  f.deps.expectedModel = model;
  f.source.candidateMetadata = Uint8Array.of(7);
  const originalOpen = f.deps.openSource;
  f.deps.openSource = async (args) => {
    input.model = H2O_CANDIDATE_MODEL_ID;
    return originalOpen(args);
  };
  assert.equal(await loadFakeH2oCandidate(input, f.deps), f.session);
  const mismatch = fixture();
  mismatch.deps.expectedModel = model;
  await assert.rejects(loadFakeH2oCandidate({...options, model: H2O_CANDIDATE_MODEL_ID}, mismatch.deps), {
    code: 'INVALID_INPUT'
  });
  assert.deepEqual(mismatch.events, []);
});

test('runtime receives only the owned bytes returned by WASM verification', async () => {
  const f = fixture(),
    owned = Uint8Array.of(7, 8, 9);
  f.deps.verifyWasm = async () => owned;
  f.deps.createRuntime = async (wasm) => {
    assert.equal(wasm, owned);
    return f.runtime;
  };
  assert.equal(await loadFakeH2oCandidate(options, f.deps), f.session);
});

test('real H2O loader rejects model and expected registration mismatch before device access', async () => {
  const runtimeBindings = {Tokenizer: class {}, instantiate: async () => ({}), createPolyfillWASI: () => ({})};
  const dlight = 'experimental-h2o-lightning-4b-q4f16-dlight';
  for (const [model, expectedModel] of [
    [dlight, H2O_CANDIDATE_MODEL_ID],
    [H2O_CANDIDATE_MODEL_ID, dlight],
    [undefined, dlight]
  ]) {
    await assert.rejects(loadH2oCandidate({...options, model}, {runtimeBindings, expectedModel}), {
      code: 'INVALID_INPUT'
    });
  }
});

test('real H2O loader resolves empty delimiters before a synthetic inventory fetch', async (t) => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator'),
    requests = [];
  let adapterRequests = 0,
    deviceRequests = 0;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          adapterRequests++;
          return {
            features: new Set(['shader-f16']),
            limits: PRELIMINARY_LIMITS,
            requestDevice() {
              deviceRequests++;
            }
          };
        }
      }
    }
  });
  t.after(() => (previous ? Object.defineProperty(globalThis, 'navigator', previous) : delete globalThis.navigator));
  t.mock.method(globalThis, 'fetch', async (url) => {
    requests.push(String(url));
    throw Error('synthetic fetch stop');
  });
  const runtimeBindings = {Tokenizer: class {}, instantiate: async () => ({}), createPolyfillWASI: () => ({})};
  for (const model of [
    H2O_CANDIDATE_MODEL_ID,
    'experimental-h2o-lightning-4b-q4f16-dlight',
    'experimental-h2o-lightning-4b-q4f16-dual'
  ]) {
    for (const suffix of ['?', '#', '?#']) {
      await assert.rejects(
        loadH2oCandidate(
          {model, assetsUrl: `https://example.invalid/assets/${suffix}`, cache: false},
          {runtimeBindings, expectedModel: model}
        ),
        {code: 'ASSET_LOAD_FAILED'}
      );
    }
  }
  assert.deepEqual(requests, Array(9).fill('https://example.invalid/assets/evidence/inventory.json'));
  assert.equal(adapterRequests, 9);
  assert.equal(deviceRequests, 0);
});
