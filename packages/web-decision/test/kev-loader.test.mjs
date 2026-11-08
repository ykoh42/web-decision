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
    new URL('../src/index.ts', import.meta.url).href,
    new URL('../src/kev-loader.js', import.meta.url).href,
    new URL('../src/kev-loader.ts', import.meta.url).href
  ]);
  const fakeModule = new URL('./fixtures/kev-loader/loader.ts', import.meta.url).href;
  const fakeBundledModule = new URL('./fixtures/kev-loader/kev06/bundled-assets.ts', import.meta.url).href;
  const fakeStableBundledModule = new URL('./fixtures/kev-loader/kev06/stable-bundled-assets.ts', import.meta.url).href;
  test('uses file URLs for Kev loader mocks so coverage can normalize source maps', () => {
    for (const moduleUrl of [fakeModule, fakeBundledModule, fakeStableBundledModule]) {
      assert.equal(new URL(moduleUrl).protocol, 'file:');
    }
  });
  installFakeBundledAssetsHook = () =>
    nodeModule.registerHooks({
      resolve(specifier, context, nextResolve) {
        if (context.parentURL === fakeModule && specifier === './kev06/bundled-assets.js')
          return {url: fakeBundledModule, shortCircuit: true};
        if (context.parentURL === fakeModule && specifier === './kev06/stable-bundled-assets.js')
          return {url: fakeStableBundledModule, shortCircuit: true};
        return nextResolve(specifier, context);
      }
    });
  const mocked = new Set([
    '@huggingface/tokenizers',
    './kev-gpu.js',
    './assets.js',
    './runtime/tvm.js',
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
    stableHelperLoads: 0,
    events: [],
    candidateSourceCreates: 0,
    stableSourceCreates: 0,
    stableSourceDisposals: 0,
    stableFetchUrls: [],
    assetFetches: [],
    externalPrepares: 0,
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
    modelConfig: {max_position_embeddings: 32768, kev: {delimiter_ids: KEV_DELIMITERS}},
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
  'Kev loader rejects model config with different pinned position embedding metadata',
  {skip: !supportsHooks},
  async (t) => {
    const f = setup(t);
    f.fake.modelConfig.max_position_embeddings = 32_767;
    await assert.rejects(loadDecisionModel({}), {
      code: 'INVALID_ASSET',
      message: 'Unexpected Kev position embedding metadata.'
    });
    assert.equal(f.state.initialized, 0);
    assert.equal(f.fake.uploads, 0);
  }
);

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

test(
  'Kev accepts a caller-selected aggregate token cap above the position table size',
  {skip: !supportsHooks},
  async (t) => {
    const f = setup(t);
    const loading = loadDecisionModel({maxInputTokens: 32_769});
    await f.started;
    f.resolveRuntime(f.runtime);
    const model = await loading;
    assert.equal(model.limits.maxInputTokens, 32_769);
    model.dispose();
  }
);

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

test('stable Kev modelId loads its pinned source without an assetsUrl', {skip: !supportsHooks}, async (t) => {
  const f = setup(t);
  const helperHook = installFakeBundledAssetsHook();
  t.after(() => helperHook?.deregister());
  const loading = loadDecisionModel({modelId: 'kev-0.6b-q4f16'});
  await f.started;
  f.resolveRuntime(f.runtime);
  const model = await loading;

  assert.equal(f.fake.stableSourceCreates, 1);
  assert.equal(f.fake.candidateSourceCreates, 0);
  assert.equal(f.fake.externalPrepares, 0);
  assert.equal(f.fake.stablePrepareOptions.model, 'kev-0.6b-q4f16');
  assert.equal(Object.hasOwn(f.fake.stablePrepareOptions, 'assetsUrl'), false);
  assert.equal(f.fake.stableHelperLoads, 1);
  assert.equal(f.fake.helperLoads, 0);
  assert.equal(f.fake.stableFetchUrls.length, 5);
  assert.ok(
    f.fake.stableFetchUrls.every((url) => url.startsWith('https://kev-assets.web-decision.invalid/kev-0.6b-q4f16/'))
  );
  assert.equal(model.description, describeStableModel('kev-0.6b-q4f16'));
  model.dispose();
  assert.equal(f.fake.stableSourceDisposals, 1);
});

test(
  'stable Kev modelId rejects assetsUrl while the legacy asset-pack route remains available',
  {skip: !supportsHooks},
  async (t) => {
    for (const options of [
      {modelId: 'kev-0.6b-q4f16', assetsUrl: 'https://mirror.example.test/kev/'},
      {modelId: 'kev-0.6b-q4f16', assetsUrl: undefined}
    ]) {
      await t.test(
        `rejects assetsUrl ${options.assetsUrl === undefined ? 'when undefined' : 'when supplied'}`,
        async (t) => {
          const f = setup(t);
          await assert.rejects(loadDecisionModel(options), {
            code: 'INVALID_INPUT',
            message: 'assetsUrl is not supported for the stable Kev model ID.'
          });
          assert.equal(f.fake.stableSourceCreates, 0);
          assert.equal(f.fake.externalPrepares, 0);
        }
      );
    }

    await t.test('keeps the legacy asset-pack route', async (t) => {
      const f = setup(t);
      const options = {model: 'kev-0.6b-q4f16', assetsUrl: 'https://legacy.example.test/kev/'};
      const loading = loadDecisionModel(options);
      await f.started;
      f.resolveRuntime(f.runtime);
      const model = await loading;
      assert.equal(f.fake.stableSourceCreates, 0);
      assert.equal(f.fake.externalPrepares, 1);
      assert.equal(f.fake.externalOptions.assetsUrl, options.assetsUrl);
      assert.equal(model.description.model, 'kev-0.6b-q4f16');
    });
  }
);

test(
  'stable Kev source avoids helper loading for a warm offline cache and honors pre-abort',
  {skip: !supportsHooks},
  async (t) => {
    for (const kind of ['warm-offline', 'pre-abort']) {
      await t.test(kind, async (t) => {
        const f = setup(t);
        const helperHook = installFakeBundledAssetsHook();
        t.after(() => helperHook?.deregister());
        if (kind === 'warm-offline') {
          f.fake.warmCache = true;
          const loading = loadDecisionModel({modelId: 'kev-0.6b-q4f16', offline: true});
          await f.started;
          f.resolveRuntime(f.runtime);
          const model = await loading;
          assert.equal(f.fake.stableSourceCreates, 1);
          assert.deepEqual(f.fake.assetFetches, []);
          assert.deepEqual(f.fake.stableFetchUrls, []);
          assert.equal(f.fake.stableHelperLoads, 0);
          model.dispose();
        } else {
          const controller = new AbortController();
          const reason = new Error('cancel stable model load');
          controller.abort(reason);
          await assert.rejects(
            loadDecisionModel({modelId: 'kev-0.6b-q4f16', signal: controller.signal}),
            (error) => error === reason
          );
          assert.equal(f.fake.stableSourceCreates, 1);
          assert.equal(f.fake.stableHelperLoads, 0);
          assert.equal(f.fake.externalPrepares, 0);
          assert.equal(f.state.instantiations, 0);
        }
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
