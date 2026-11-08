import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import {test, type TestContext} from 'node:test';
import {KEV_DELIMITERS} from '../src/kev.js';
import {describeModel} from '../src/model-registry.js';
import {describeStableModel} from '../src/stable-models.js';

type ResolveContext = {readonly parentURL?: string};
type ResolveResult = {readonly url: string; readonly shortCircuit?: boolean};
type ResolveHook = (
  specifier: string,
  context: ResolveContext,
  nextResolve: (specifier: string, context: ResolveContext) => ResolveResult
) => ResolveResult;
type HookHandle = {deregister(): void};
type RegisterHooks = (hooks: {resolve: ResolveHook}) => HookHandle;

type FakeRuntime = {
  dispose(): void;
  initWebGPU(): void;
  webgpu(): object;
  beginScope(): void;
  endScope(): void;
  createVirtualMachine(): {getFunction(): () => void};
};
type FakeGpu = {
  readonly lost: Promise<{readonly message: string}>;
  addEventListener(): void;
  destroy(): void;
};
type LoaderState = {
  disposed: number;
  destroyed: number;
  ended: number;
  initialized: number;
  instantiations: number;
  uploads: number;
};
type TensorRecord = {
  readonly name: string;
  readonly dtype: string;
  readonly format: string;
  readonly shape: readonly number[];
  readonly byteOffset: number;
  readonly nbytes: number;
};
type LoaderFixture = LoaderState & {
  helperLoads: number;
  stableHelperLoads: number;
  events: string[];
  candidateSourceCreates: number;
  stableSourceCreates: number;
  stableSourceDisposals: number;
  stableFetchUrls: string[];
  assetFetches: string[];
  externalPrepares: number;
  gpuAvailable: boolean;
  warmCache: boolean;
  tamevSystemOneMode: string;
  tamevDisposeFails: boolean;
  tamevDisposals: number;
  readonly gpu: FakeGpu;
  readonly pending: Promise<FakeRuntime>;
  readonly started: () => void;
  readonly manifest: {
    readonly files: Readonly<Record<string, {readonly bytes: number}>>;
    readonly parameterCount: number;
    readonly maxStateTokens: number;
    readonly maxBranchTokens: number;
  };
  modelConfig: {
    max_position_embeddings: number;
    readonly kev: {readonly delimiter_ids: typeof KEV_DELIMITERS};
  };
  readonly cacheManifest: {
    readonly metadata: {readonly numParams: number};
    readonly records: readonly {
      readonly dataPath: string;
      readonly format: string;
      readonly nbytes: number;
      readonly records: readonly TensorRecord[];
    }[];
  };
  bundledUrls: Readonly<Record<string, string>> | undefined;
  stableBundledUrls: Readonly<Record<string, string>> | undefined;
  selectedFamily: string | undefined;
  selectedOptions: Readonly<Record<string, unknown>> | undefined;
  stablePrepareOptions: Readonly<Record<string, unknown>> | undefined;
  externalOptions: Readonly<Record<string, unknown>> | undefined;
};
type TestDescription = {readonly model: string};
type TestModel = {
  readonly description: TestDescription;
  readonly runtime?: object;
  readonly limits?: Readonly<Record<string, unknown>>;
  readonly dispose?: () => unknown;
  readonly systemOne?: (...args: unknown[]) => unknown;
};
type LoadDecisionModel = (options: unknown) => Promise<TestModel>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isRawLoader(value: unknown): value is (options: unknown) => Promise<unknown> {
  return typeof value === 'function';
}

function isTestDescription(value: unknown): value is TestDescription {
  return isRecord(value) && typeof value.model === 'string';
}

function isDisposal(value: unknown): value is () => unknown {
  return typeof value === 'function';
}

function isSystemOne(value: unknown): value is (...args: unknown[]) => unknown {
  return typeof value === 'function';
}

function isTestModel(value: unknown): value is TestModel {
  return (
    isRecord(value) &&
    isTestDescription(value.description) &&
    (value.runtime === undefined || isRecord(value.runtime)) &&
    (value.limits === undefined || isRecord(value.limits)) &&
    (value.dispose === undefined || isDisposal(value.dispose)) &&
    (value.systemOne === undefined || isSystemOne(value.systemOne))
  );
}

function requireModel(value: unknown): TestModel {
  if (!isTestModel(value)) throw new TypeError('Kev loader returned a value outside the test model boundary.');
  return value;
}

function modelLimits(model: TestModel): Readonly<Record<string, unknown>> {
  if (!model.limits) throw new TypeError('Kev model did not expose its limits.');
  return model.limits;
}

function disposeModel(model: TestModel): unknown {
  if (!model.dispose) throw new TypeError('Model did not expose dispose().');
  return model.dispose();
}

async function callSystemOne(model: TestModel): Promise<unknown> {
  if (!model.systemOne) throw new TypeError('Stable TAMEV model did not expose systemOne().');
  return model.systemOne();
}

function requireOptions(value: Readonly<Record<string, unknown>> | undefined): Readonly<Record<string, unknown>> {
  if (!value) throw new TypeError('Mock loader did not record its options.');
  return value;
}

function requireBundledUrls(value: Readonly<Record<string, string>> | undefined): Readonly<Record<string, string>> {
  if (!value) throw new TypeError('Mock loader did not load bundled assets.');
  return value;
}

function requirePropertyDescriptor(value: object, property: PropertyKey): PropertyDescriptor {
  const descriptor = Object.getOwnPropertyDescriptor(value, property);
  if (!descriptor) throw new TypeError(`Expected own property ${String(property)}.`);
  return descriptor;
}

function hasErrorCode(error: unknown, code: string): boolean {
  return isRecord(error) && error.code === code;
}

type FakeLoaderGlobal = typeof globalThis & {__fakeKevLoader?: LoaderFixture};

// Exercise the actual Kev loader with small fake asset, GPU and runtime boundaries.
const registerHooks = (nodeModule as unknown as {readonly registerHooks?: RegisterHooks}).registerHooks;
const supportsHooks = typeof registerHooks === 'function';
let loadDecisionModel: LoadDecisionModel = async () => {
  throw new Error('The Node.js module registerHooks API is unavailable.');
};
let installFakeBundledAssetsHook: () => HookHandle | null = () => null;
if (registerHooks) {
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
    registerHooks({
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
  const hooks = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (
        context.parentURL !== undefined &&
        loaderUrls.has(context.parentURL) &&
        (mocked.has(specifier) || specifier.includes('/.toolchains/'))
      )
        return {url: fakeModule, shortCircuit: true};
      return nextResolve(specifier, context);
    }
  });
  try {
    const indexModule: unknown = await import('../src/index.js');
    const rawLoader = isRecord(indexModule) ? indexModule.loadDecisionModel : undefined;
    if (!isRawLoader(rawLoader)) throw new TypeError('The Kev index module did not export loadDecisionModel().');
    loadDecisionModel = async (options) => requireModel(await rawLoader(options));
  } finally {
    hooks.deregister();
  }
}

function setup(t: TestContext): {
  readonly state: LoaderState;
  readonly fake: LoaderFixture;
  readonly runtime: FakeRuntime;
  readonly started: Promise<void>;
  readonly resolveRuntime: (runtime: FakeRuntime) => void;
  readonly rejectRuntime: (reason: unknown) => void;
  readonly lose: (info: {readonly message: string}) => void;
} {
  const fakeGlobal = globalThis as FakeLoaderGlobal;
  const previous = Object.getOwnPropertyDescriptor(fakeGlobal, '__fakeKevLoader');
  t.after(() => {
    if (previous) Object.defineProperty(fakeGlobal, '__fakeKevLoader', previous);
    else delete fakeGlobal.__fakeKevLoader;
  });
  let resolveRuntime: (runtime: FakeRuntime) => void = () => {
    throw new Error('Runtime promise has not been initialized.');
  };
  let rejectRuntime: (reason: unknown) => void = () => {
    throw new Error('Runtime promise has not been initialized.');
  };
  let markStarted: () => void = () => {
    throw new Error('Instantiation signal has not been initialized.');
  };
  let lose: (info: {readonly message: string}) => void = () => {
    throw new Error('GPU loss signal has not been initialized.');
  };
  const pending = new Promise<FakeRuntime>((resolve, reject) => {
    resolveRuntime = resolve;
    rejectRuntime = reject;
  });
  const started = new Promise<void>((resolve) => {
    markStarted = resolve;
  });
  const lost = new Promise<{readonly message: string}>((resolve) => {
    lose = resolve;
  });
  const state: LoaderState = {
    disposed: 0,
    destroyed: 0,
    ended: 0,
    initialized: 0,
    instantiations: 0,
    uploads: 0
  };
  const gpu: FakeGpu = {
    lost,
    addEventListener() {},
    destroy() {
      state.destroyed++;
    }
  };
  const runtime: FakeRuntime = {
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
  const records: TensorRecord[] = Array.from({length: 394}, (_, i) => ({
    name: `param_${i}`,
    dtype: 'uint32',
    format: 'raw',
    shape: [1],
    byteOffset: i * 4,
    nbytes: 4
  }));
  const fake: LoaderFixture = {
    ...state,
    helperLoads: 0,
    stableHelperLoads: 0,
    events: [],
    candidateSourceCreates: 0,
    stableSourceCreates: 0,
    stableSourceDisposals: 0,
    stableFetchUrls: [],
    assetFetches: [],
    bundledUrls: undefined,
    stableBundledUrls: undefined,
    selectedFamily: undefined,
    selectedOptions: undefined,
    stablePrepareOptions: undefined,
    externalOptions: undefined,
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
  Object.defineProperty(fakeGlobal, '__fakeKevLoader', {configurable: true, value: fake});
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
        await assert.rejects(loading, (error: unknown) =>
          kind === 'abort' ? error === reason : hasErrorCode(error, 'MODEL_LOAD_FAILED')
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
  await assert.rejects(loading, (error: unknown) => error === reason);
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
  assert.equal(requirePropertyDescriptor(model, 'description').writable, false);
  assert.equal(f.state.initialized, 1);
  assert.equal(f.fake.uploads, 1);
  assert.equal(modelLimits(model).maxInputTokens, undefined);
  assert.equal(modelLimits(model).maxStateTokens, 8192);
  assert.equal(modelLimits(model).maxBranchTokens, 8192);
  assert.equal(f.state.disposed, 0);
  assert.equal(f.state.destroyed, 0);
  disposeModel(model);
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
    const bundledUrls = requireBundledUrls(f.fake.bundledUrls);
    assert.deepEqual(
      Object.keys(bundledUrls).sort(),
      ['model.wasm', 'model_config.json', 'tokenizer.json', 'tokenizer_config.json', 'weights/tensor-cache.json'].sort()
    );
    assert.equal(bundledUrls['model.wasm'], 'data:application/wasm;base64,AA==');
    assert.equal(model.description.model, 'experimental-kev-0.6b-q4f16-upstream');
    disposeModel(model);
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
        const options: Record<string, unknown> = {model: 'experimental-kev-0.6b-q4f16-upstream'};
        if (kind === 'pre-abort') {
          const controller = new AbortController();
          const reason = new Error('cancelled before load');
          controller.abort(reason);
          options.signal = controller.signal;
          await assert.rejects(loadDecisionModel(options), (error: unknown) => error === reason);
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
          disposeModel(model);
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

test('Kev accepts an optional caller-selected aggregate token cap', {skip: !supportsHooks}, async (t) => {
  const f = setup(t);
  const loading = loadDecisionModel({maxInputTokens: 2048});
  await f.started;
  f.resolveRuntime(f.runtime);
  const model = await loading;
  assert.equal(modelLimits(model).maxInputTokens, 2048);
  disposeModel(model);
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
  disposeModel(model);
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
      assert.equal(requirePropertyDescriptor(model, 'description').writable, false);
      if (family === 'strands') assert.equal(requireOptions(f.fake.selectedOptions).assetsUrl, '/test/');
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
        const selectedOptions = requireOptions(f.fake.selectedOptions);
        assert.equal(selectedOptions.model, routeModel);
        assert.equal(selectedOptions.assetsUrl, assetsUrl);
        assert.equal(Object.hasOwn(selectedOptions, 'modelId'), false);
        assert.equal(model.description, describeStableModel(modelId));
        assert.equal(model.description.model, modelId);
        assert.deepEqual(await callSystemOne(model), {model: modelId, answers: {}});
        assert.equal(requirePropertyDescriptor(model, 'systemOne').enumerable, true);
        await disposeModel(model);
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
  const stablePrepareOptions = requireOptions(f.fake.stablePrepareOptions);
  assert.equal(stablePrepareOptions.model, 'kev-0.6b-q4f16');
  assert.equal(Object.hasOwn(stablePrepareOptions, 'assetsUrl'), false);
  assert.equal(f.fake.stableHelperLoads, 1);
  assert.equal(f.fake.helperLoads, 0);
  assert.equal(f.fake.stableFetchUrls.length, 5);
  assert.ok(
    f.fake.stableFetchUrls.every((url) => url.startsWith('https://kev-assets.web-decision.invalid/kev-0.6b-q4f16/'))
  );
  assert.equal(model.description, describeStableModel('kev-0.6b-q4f16'));
  disposeModel(model);
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
      assert.equal(requireOptions(f.fake.externalOptions).assetsUrl, options.assetsUrl);
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
          disposeModel(model);
        } else {
          const controller = new AbortController();
          const reason = new Error('cancel stable model load');
          controller.abort(reason);
          await assert.rejects(
            loadDecisionModel({modelId: 'kev-0.6b-q4f16', signal: controller.signal}),
            (error: unknown) => error === reason
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
