import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import test from 'node:test';
import {KEV_DELIMITERS} from '../src/kev.js';
import {describeModel} from '../src/model-registry.js';

// Exercise the actual Kev loader with small fake asset, GPU and runtime boundaries.
const supportsHooks = typeof nodeModule.registerHooks === 'function';
let loadDecisionModel;
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
    export function instantiate() { fixture().started(); return fixture().pending; }
    export async function detectKevGPUDevice() { return {device: fixture().gpu, adapter: {features: new Set(['shader-f16'])}}; }
    export const CACHE_NAME = 'fake';
    export function parseJson(value) { return value; }
    export async function readManifestResponse() { return fixture().manifest; }
    export async function readKev06ManifestResponse() { return fixture().manifest; }
    export async function commitCachedManifest() {}
    export async function prepareModelSource(options) { return {signal: options.signal, response: {}, cache: null}; }
    export function createModelAssetReader() { return {read() {}, isCacheComplete() { return false; }}; }
    export async function readAssetBatch() { return [{}, {}, fixture().modelConfig, {}, fixture().cacheManifest]; }
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
  `;
  const fakeModule = `data:text/javascript;base64,${Buffer.from(fakeSource).toString('base64')}`;
  const mocked = new Set([
    '@huggingface/tokenizers',
    './kev-gpu.js',
    './assets.js',
    './kev06/contract.js',
    './model.js',
    './asset-batch.js',
    './model-source.js',
    './model-assets.js',
    './deberta/loader.mjs',
    './mapika/loader.js',
    './mapika-q4/loader.js',
    './kev08/loader.js',
    './strands/loader.mjs'
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
  const state = {disposed: 0, destroyed: 0, ended: 0, initialized: 0, uploads: 0};
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
