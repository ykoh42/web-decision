import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import test from 'node:test';
import {KEV_DELIMITERS} from '../src/kev.js';
import {describeModel} from '../src/model-registry.js';
import {describeStableModel} from '../src/stable-models.js';

// Exercise the actual Kev loader with small fake asset, GPU and runtime boundaries.
const supportsHooks = typeof nodeModule.registerHooks === 'function';
let loadDecisionModel;
let installFakeBundledAssetsHook = () => null;
if (supportsHooks) {
  const loaderUrls = new Set([
    new URL('../src/index.js', import.meta.url).href,
    new URL('../src/kev-loader.js', import.meta.url).href
  ]);
  const abortUrl = new URL('../src/abort.js', import.meta.url).href;
  const fakeSource = `
    export {waitForAbort} from ${JSON.stringify(abortUrl)};
    const fixture = () => globalThis.__fakeKevLoader;
    export class Tokenizer {}
    export function createPolyfillWASI() { return {}; }
    export function instantiate() { fixture().instantiations++; fixture().started(); return fixture().pending; }
    export async function detectKevGPUDevice() { return {device: fixture().gpu, adapter: {features: new Set(['shader-f16'])}}; }
    export const CACHE_NAME = 'fake';
    export function parseJson(value) { return value; }
    export async function readManifestResponse() { return fixture().manifest; }
    export async function readKev06ManifestResponse() { return fixture().manifest; }
    export const EXPERIMENTAL_KEV_MODEL = 'experimental-kev-0.6b-q4f16-upstream';
    export function createKevReconstructedSource(overrides = {}) {
      fixture().bundledUrls = overrides.bundledUrls;
      return {
        async prepare(options) {
          fixture().events.push('prepare');
          options.signal?.throwIfAborted();
          if (!fixture().gpuAvailable) throw Object.assign(new Error('WebGPU unavailable'), {code: 'UNSUPPORTED_ENVIRONMENT'});
          return {base: 'https://kev-assets.web-decision.invalid/experimental-kev-upstream/', signal: options.signal, cache: null, response: {}, maxPrefetchBytes: 0};
        },
        async fetch(url, {signal} = {}) {
          if (!fixture().bundledUrls) {
            const module = await import('./kev06/bundled-assets.js');
            fixture().bundledUrls = module.experimentalKevBundledUrls;
          }
          signal?.throwIfAborted();
          return new Response(new Uint8Array([1]));
        },
        setProgress() {},
        dispose() {}
      };
    }
    export async function readExperimentalKevManifest() { return fixture().manifest; }
    export async function commitCachedManifest() {}
    export async function prepareModelSource(options) { return {signal: options.signal, response: {}, cache: null}; }
    export function createModelAssetReader({base, fetchImpl, signal}) {
      return {
        async read(name, readSignal = signal) {
          if (fetchImpl && !fixture().warmCache)
            await fetchImpl(new URL(name, base).href, {signal: readSignal});
          return new ArrayBuffer(1);
        },
        isCacheComplete() { return fixture().warmCache; },
        setProgressPhase() {}
      };
    }
    export async function readAssetBatch(names, read, signal) {
      await Promise.all(names.map((name) => read(name, signal)));
      return [{}, {}, fixture().modelConfig, {}, fixture().cacheManifest];
    }
    export async function uploadModelShards() { fixture().uploads++; }
    export class DecisionModel {
      constructor(runtime, device, gpu, fn, weights, tokenizer, limits) { this.runtime = runtime; this.gpu = gpu; this.limits = limits; }
      dispose() { this.runtime.dispose(); this.gpu.destroy(); }
    }
    const loadFake = (family, options) => {
      fixture().selectedFamily = family;
      fixture().selectedOptions = options;
      return {family};
    };
    export async function loadExperimentalDeberta(options) { return loadFake('deberta', options); }
    export async function loadExperimentalMapika(options) { return loadFake('mapika', options); }
    export async function loadExperimentalMapikaQ4(options) { return loadFake('mapika-q4', options); }
    export async function loadExperimentalKev08(options) { return loadFake('kev08', options); }
    export async function loadExperimentalStrandsFromSource(options) { return loadFake('strands', options); }
    export async function loadTamev(options) {
      const model = loadFake('tamev', options);
      model.dispose = async () => {
        fixture().tamevDisposals++;
        if (fixture().tamevDisposeFails) throw new Error('fake TAMEV disposal failed');
      };
      const mode = fixture().tamevSystemOneMode;
      if (mode === 'accessor') {
        Object.defineProperty(model, 'systemOne', {get: () => () => ({model: options.model, answers: {}})});
      } else if (mode === 'nonconfigurable') {
        Object.defineProperty(model, 'systemOne', {
          configurable: false,
          enumerable: true,
          value: async () => ({model: options.model, answers: {}}),
          writable: false
        });
      } else if (mode === 'nonfunction') {
        model.systemOne = 'not a function';
      } else if (mode !== 'missing') {
        model.systemOne = async () => ({model: options.model, answers: {}});
      }
      return model;
    }
  `;
  const fakeModule = `data:text/javascript;base64,${Buffer.from(fakeSource).toString('base64')}`;
  const fakeBundledAssets = `
    globalThis.__fakeKevLoader.helperLoads++;
    globalThis.__fakeKevLoader.events.push('helper');
    export const experimentalKevBundledUrls = Object.freeze({
      'model.wasm': 'data:application/wasm;base64,AA==',
      'tokenizer.json': 'data:application/json;base64,e30=',
      'tokenizer_config.json': 'data:application/json;base64,e30=',
      'model_config.json': 'data:application/json;base64,e30=',
      'weights/tensor-cache.json': 'data:application/json;base64,e30='
    });
  `;
  const fakeBundledModule = `data:text/javascript;base64,${Buffer.from(fakeBundledAssets).toString('base64')}`;
  installFakeBundledAssetsHook = () =>
    nodeModule.registerHooks({
      resolve(specifier, context, nextResolve) {
        if (context.parentURL === fakeModule && specifier === './kev06/bundled-assets.js')
          return {url: fakeBundledModule, shortCircuit: true};
        return nextResolve(specifier, context);
      }
    });
  const mocked = new Set([
    '@huggingface/tokenizers',
    './kev-gpu.js',
    './assets.js',
    './kev06/contract.js',
    './kev06/upstream-source.mjs',
    './model.js',
    './asset-batch.js',
    './model-source.js',
    './model-assets.js',
    './deberta/loader.mjs',
    './mapika/loader.js',
    './mapika-q4/loader.js',
    './kev08/loader.js',
    './strands/loader.mjs',
    './tamev/loader.mjs'
  ]);
  const hooks = nodeModule.registerHooks({
    resolve(specifier, context, nextResolve) {
      if (loaderUrls.has(context.parentURL) && (mocked.has(specifier) || specifier.includes('/.toolchains/')))
        return {url: fakeModule, shortCircuit: true};
      return nextResolve(specifier, context);
    }
  });
  try {
    ({loadDecisionModel} = await import('../src/index.js'));
  } finally {
    hooks.deregister();
  }
}

function setup(t) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, '__fakeKevLoader');
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, '__fakeKevLoader', previous);
    else delete globalThis.__fakeKevLoader;
  });
  let resolveRuntime, rejectRuntime, markStarted, lose;
  const pending = new Promise((resolve, reject) => {
    resolveRuntime = resolve;
    rejectRuntime = reject;
  });
  const started = new Promise((resolve) => {
    markStarted = resolve;
  });
  const lost = new Promise((resolve) => {
    lose = resolve;
  });
  const state = {disposed: 0, destroyed: 0, ended: 0, initialized: 0, instantiations: 0, uploads: 0};
  const gpu = {
    lost,
    addEventListener() {},
    destroy() {
      state.destroyed++;
    }
  };
  const runtime = {
    dispose() {
      state.disposed++;
    },
    initWebGPU() {
      state.initialized++;
    },
    webgpu() {
      return {};
    },
    beginScope() {},
    endScope() {
      state.ended++;
    },
    createVirtualMachine() {
      return {
        getFunction() {
          return () => {};
        }
      };
    }
  };
  const records = Array.from({length: 394}, (_, i) => ({
    name: `param_${i}`,
    dtype: 'uint32',
    format: 'raw',
    shape: [1],
    byteOffset: i * 4,
    nbytes: 4
  }));
  const fake = {
    ...state,
    helperLoads: 0,
    events: [],
    gpuAvailable: true,
    warmCache: false,
    tamevSystemOneMode: 'method',
    tamevDisposeFails: false,
    tamevDisposals: 0,
    gpu,
    pending,
    started: markStarted,
    manifest: {
      files: {'weights/shard.bin': {bytes: 394 * 4}},
      parameterCount: 394,
      maxStateTokens: 8192,
      maxBranchTokens: 8192
    },
    modelConfig: {kev: {delimiter_ids: KEV_DELIMITERS}},
    cacheManifest: {
      metadata: {numParams: 394},
      records: [{dataPath: 'shard.bin', format: 'raw-shard', nbytes: 394 * 4, records}]
    }
  };
  Object.defineProperty(globalThis, '__fakeKevLoader', {configurable: true, value: fake});
  return {state, fake, runtime, started, resolveRuntime, rejectRuntime, lose};
}

test(
  'Kev loader abort or device loss during instantiation releases the GPU and disposes the late runtime',
  {skip: !supportsHooks},
  async (t) => {
    for (const kind of ['abort', 'loss']) {
      await t.test(kind, async (t) => {
        const f = setup(t);
        const controller = new AbortController();
        const reason = new Error('caller cancelled');
        const loading = loadDecisionModel({signal: controller.signal});
        await f.started;
        if (kind === 'abort') controller.abort(reason);
        else f.lose({message: 'device lost'});
        await assert.rejects(loading, (error) =>
          kind === 'abort' ? error === reason : error.code === 'MODEL_LOAD_FAILED'
        );
        assert.equal(f.state.destroyed, 1);
        assert.equal(f.state.disposed, 0);
        f.resolveRuntime(f.runtime);
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(f.state.disposed, 1);
        assert.equal(f.state.initialized, 0);
        assert.equal(f.state.ended, 0);
        assert.equal(f.fake.uploads, 0);
      });
    }
  }
);

test('Kev loader handles a runtime rejection after cancellation', {skip: !supportsHooks}, async (t) => {
  const f = setup(t),
    controller = new AbortController(),
    reason = new Error('cancelled');
  const loading = loadDecisionModel({signal: controller.signal});
  await f.started;
  controller.abort(reason);
  await assert.rejects(loading, (error) => error === reason);
  f.rejectRuntime(new Error('late failure'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.state.destroyed, 1);
  assert.equal(f.state.disposed, 0);
});

test('Kev loader transfers a successful runtime to the model', {skip: !supportsHooks}, async (t) => {
  const f = setup(t);
  const loading = loadDecisionModel({});
  await f.started;
  f.resolveRuntime(f.runtime);
  const model = await loading;
  assert.equal(model.runtime, f.runtime);
  assert.equal(model.description, describeModel('kev-0.6b-q4f16'));
  assert.equal(Object.getOwnPropertyDescriptor(model, 'description').writable, false);
  assert.equal(f.state.initialized, 1);
  assert.equal(f.fake.uploads, 1);
  assert.equal(model.limits.maxInputTokens, undefined);
  assert.equal(model.limits.maxStateTokens, 8192);
  assert.equal(model.limits.maxBranchTokens, 8192);
  assert.equal(f.state.disposed, 0);
  assert.equal(f.state.destroyed, 0);
  model.dispose();
  assert.equal(f.state.disposed, 1);
  assert.equal(f.state.destroyed, 1);
});

test(
  'experimental upstream Kev imports its immutable data URLs lazily after prepare on a cache miss',
  {skip: !supportsHooks},
  async (t) => {
    const f = setup(t);
    const helperHook = installFakeBundledAssetsHook();
    t.after(() => helperHook?.deregister());
    const loading = loadDecisionModel({model: 'experimental-kev-0.6b-q4f16-upstream'});
    await f.started;
    f.resolveRuntime(f.runtime);
    const model = await loading;
    assert.equal(f.fake.helperLoads, 1);
    assert.deepEqual(f.fake.events, ['prepare', 'helper']);
    assert.deepEqual(
      Object.keys(f.fake.bundledUrls).sort(),
      ['model.wasm', 'model_config.json', 'tokenizer.json', 'tokenizer_config.json', 'weights/tensor-cache.json'].sort()
    );
    assert.equal(f.fake.bundledUrls['model.wasm'], 'data:application/wasm;base64,AA==');
    assert.equal(model.description.model, 'experimental-kev-0.6b-q4f16-upstream');
    model.dispose();
  }
);

test(
  'experimental Kev skips its helper chunk for pre-abort, unavailable WebGPU, and warm offline cache',
  {
    skip: !supportsHooks
  },
  async (t) => {
    for (const kind of ['pre-abort', 'no-gpu', 'warm-offline']) {
      await t.test(kind, async (t) => {
        const f = setup(t);
        const helperHook = installFakeBundledAssetsHook();
        t.after(() => helperHook?.deregister());
        const options = {model: 'experimental-kev-0.6b-q4f16-upstream'};
        if (kind === 'pre-abort') {
          const controller = new AbortController();
          const reason = new Error('cancelled before load');
          controller.abort(reason);
          options.signal = controller.signal;
          await assert.rejects(loadDecisionModel(options), (error) => error === reason);
          assert.deepEqual(f.fake.events, ['prepare']);
        } else if (kind === 'no-gpu') {
          f.fake.gpuAvailable = false;
          await assert.rejects(loadDecisionModel(options), {code: 'UNSUPPORTED_ENVIRONMENT'});
          assert.deepEqual(f.fake.events, ['prepare']);
        } else {
          f.fake.warmCache = true;
          options.offline = true;
          const loading = loadDecisionModel(options);
          await f.started;
          f.resolveRuntime(f.runtime);
          const model = await loading;
          assert.deepEqual(f.fake.events, ['prepare']);
          model.dispose();
        }
        assert.equal(f.fake.helperLoads, 0);
      });
    }
  }
);

test('Kev accepts an optional caller-selected aggregate token cap', {skip: !supportsHooks}, async (t) => {
  const f = setup(t);
  const loading = loadDecisionModel({maxInputTokens: 2048});
  await f.started;
  f.resolveRuntime(f.runtime);
  const model = await loading;
  assert.equal(model.limits.maxInputTokens, 2048);
  model.dispose();
});

test(
  'Kev rejects invalid caller-selected aggregate token caps before loading assets',
  {skip: !supportsHooks},
  async (t) => {
    for (const maxInputTokens of [-1, 1.5, Infinity, '2048']) {
      await t.test(String(maxInputTokens), async (t) => {
        const f = setup(t);
        await assert.rejects(loadDecisionModel({maxInputTokens}), {code: 'INVALID_INPUT'});
        assert.equal(f.state.initialized, 0);
        assert.equal(f.fake.uploads, 0);
      });
    }
  }
);

test('model identity is captured once before asynchronous loading', {skip: !supportsHooks}, async (t) => {
  const f = setup(t);
  let selected = 'kev-0.6b-q4f16',
    reads = 0;
  const loading = loadDecisionModel({
    get model() {
      reads++;
      return selected;
    }
  });
  selected = 'kev-0.6b-reflexes-q4f16';
  f.resolveRuntime(f.runtime);
  const model = await loading;
  assert.equal(reads, 1);
  assert.equal(model.description.model, 'kev-0.6b-q4f16');
  assert.equal(model.description, describeModel('kev-0.6b-q4f16'));
  model.dispose();
});

test('each non-Kev loader returns its own matching immutable description', {skip: !supportsHooks}, async (t) => {
  for (const [id, family] of [
    ['open-jev-deberta-v3-large-q4f16', 'deberta'],
    ['mapika-0.8b-f16', 'mapika'],
    ['mapika-0.8b-q4f16', 'mapika-q4'],
    ['kev-0.8b-f16', 'kev08'],
    ['experimental-strands-qwen-2b-fp16', 'strands']
  ]) {
    await t.test(id, async (t) => {
      const f = setup(t);
      let reads = 0;
      const options = {
        assetsUrl: '/test/',
        get model() {
          reads++;
          return id;
        }
      };
      const model = await loadDecisionModel(options);
      assert.equal(reads, 1);
      assert.equal(f.fake.selectedFamily, family);
      assert.equal(model.description, describeModel(id));
      assert.equal(Object.getOwnPropertyDescriptor(model, 'description').writable, false);
      if (family === 'strands') assert.equal(f.fake.selectedOptions.assetsUrl, '/test/');
    });
  }
});

test(
  'stable model IDs route through pinned implementations and expose stable identity',
  {skip: !supportsHooks},
  async (t) => {
    for (const [modelId, routeModel, assetsUrl] of [
      ['tamev-micro-minilm-int8', 'experimental-tamev-micro-minilm-int8', undefined],
      ['tamev-nano-tinybert-int8', 'experimental-tamev-nano-tinybert-int8', 'https://mirror.example.test/nano/'],
      ['tamev-nano-flash-tinybert-int8', 'experimental-tamev-nano-flash-tinybert-int8', undefined]
    ]) {
      await t.test(modelId, async (t) => {
        const f = setup(t);
        const model = await loadDecisionModel({modelId, ...(assetsUrl ? {assetsUrl} : {})});

        assert.equal(f.fake.selectedFamily, 'tamev');
        assert.equal(f.fake.selectedOptions.model, routeModel);
        assert.equal(f.fake.selectedOptions.assetsUrl, assetsUrl);
        assert.equal(Object.hasOwn(f.fake.selectedOptions, 'modelId'), false);
        assert.equal(model.description, describeStableModel(modelId));
        assert.equal(model.description.model, modelId);
        assert.deepEqual(await model.systemOne(), {model: modelId, answers: {}});
        assert.equal(Object.getOwnPropertyDescriptor(model, 'systemOne').enumerable, true);
        await model.dispose();
        assert.equal(f.fake.tamevDisposals, 1);
      });
    }
  }
);

test('unknown stable IDs and bare string IDs never fall through to default Kev', {skip: !supportsHooks}, async (t) => {
  const f = setup(t);
  f.resolveRuntime(f.runtime);
  await assert.rejects(loadDecisionModel({modelId: 'experimental-strands-qwen-2b-fp16'}), {
    code: 'INVALID_INPUT'
  });
  await assert.rejects(loadDecisionModel({modelId: 'not-a-model'}), {code: 'INVALID_INPUT'});
  await assert.rejects(loadDecisionModel(Object.create({modelId: 'unknown-model'})), {code: 'INVALID_INPUT'});
  await assert.rejects(
    loadDecisionModel({modelId: 'tamev-micro-minilm-int8', model: 'experimental-tamev-micro-minilm-int8'}),
    {code: 'INVALID_INPUT'}
  );
  let modelIdReads = 0;
  const accessorOptions = Object.defineProperty({}, 'modelId', {
    get() {
      modelIdReads++;
      return 'tamev-micro-minilm-int8';
    }
  });
  await assert.rejects(loadDecisionModel(accessorOptions), {code: 'INVALID_INPUT'});
  assert.equal(modelIdReads, 0);
  await assert.rejects(loadDecisionModel('kev-0.6b-q4f16'), {code: 'INVALID_INPUT'});
  assert.equal(f.state.initialized, 0);
  assert.equal(f.state.instantiations, 0);
  assert.equal(f.fake.selectedFamily, undefined);
});

test(
  'stable TAMEV adaptation disposes sessions when the public method cannot be wrapped',
  {skip: !supportsHooks},
  async (t) => {
    for (const mode of ['missing', 'accessor', 'nonfunction', 'nonconfigurable']) {
      await t.test(mode, async (t) => {
        const f = setup(t);
        f.fake.tamevSystemOneMode = mode;
        if (mode === 'nonconfigurable') {
          await assert.rejects(loadDecisionModel({modelId: 'tamev-micro-minilm-int8'}), TypeError);
        } else {
          await assert.rejects(loadDecisionModel({modelId: 'tamev-micro-minilm-int8'}), {
            code: 'MODEL_LOAD_FAILED'
          });
        }
        assert.equal(f.fake.tamevDisposals, 1);
      });
    }
  }
);

test(
  'stable TAMEV cleanup preserves the original adaptation error when dispose fails',
  {skip: !supportsHooks},
  async (t) => {
    const f = setup(t);
    f.fake.tamevSystemOneMode = 'missing';
    f.fake.tamevDisposeFails = true;
    await assert.rejects(loadDecisionModel({modelId: 'tamev-micro-minilm-int8'}), {
      code: 'MODEL_LOAD_FAILED',
      message: 'Stable TAMEV model has no System One method.'
    });
    assert.equal(f.fake.tamevDisposals, 1);
  }
);
