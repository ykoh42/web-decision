import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import * as nodeModule from 'node:module';
import test, {type TestContext} from 'node:test';
import {
  EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
  EXPERIMENTAL_KEV_UPSTREAM_REVISION,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH
} from '../src/kev06/upstream-contract.js';
import {KEV06_MANIFEST_SHA256, readKev06ManifestResponse} from '../src/kev06/contract.js';
import type {ModelProgress} from '../src/curated/types.js';

const bundledAssetsUrl = new URL('../src/kev06/bundled-assets.js', import.meta.url).href;
const stableBundledAssetsUrl = new URL('../src/kev06/stable-bundled-assets.js', import.meta.url).href;
const upstreamSourceUrl = new URL('../src/kev06/upstream-source.mts', import.meta.url).href;
const inlineAssets: Readonly<Record<string, string>> = Object.freeze({
  'model.wasm': 'data:application/wasm;base64,AQID',
  'tokenizer.json': `data:application/json;base64,${Buffer.from('{"tokens":[]}').toString('base64')}`,
  'tokenizer_config.json': 'data:application/json,%7B%22lowercase%22%3Atrue%7D',
  'model_config.json': `data:application/json;base64,${Buffer.from('{"hidden_size":4}').toString('base64')}`,
  'weights/tensor-cache.json': 'data:application/json,%7B%22records%22%3A%5B%5D%7D'
});
const stableInlineAssets: Readonly<Record<string, string>> = Object.freeze({
  ...inlineAssets,
  'model.wasm': 'data:application/wasm;base64,BAUG'
});
const fixtureCache: Cache = {
  add: async () => {},
  addAll: async () => {},
  delete: async () => false,
  keys: async () => [],
  match: async () => undefined,
  matchAll: async () => [],
  put: async () => {}
};

type ResolveContext = {readonly parentURL: string | undefined};
type ResolveResult = {readonly url: string; readonly shortCircuit?: boolean};
type NextResolve = (specifier: string, context?: Partial<ResolveContext>) => ResolveResult | Promise<ResolveResult>;
type TestResolveHook = (
  specifier: string,
  context: ResolveContext,
  nextResolve: NextResolve
) => ResolveResult | Promise<ResolveResult>;
type RegisteredHooks = {deregister(): void};
type RegisterHooks = (hooks: {readonly resolve: TestResolveHook}) => RegisteredHooks;
type KevSourceOverrides = NonNullable<Parameters<typeof createKevReconstructedSource>[0]>;
type FixtureState = {
  sourceReads: number;
  cacheSettings: boolean[];
  preflightGpus: unknown[];
  events: string[];
  sourceUrl?: string;
  fetchInit?: RequestInit;
};

function moduleDataUrl(source: string): string {
  return `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
}

function resolveTestAssetImport(
  specifier: string,
  context: ResolveContext,
  nextResolve: NextResolve
): ResolveResult | Promise<ResolveResult> {
  const parentURL = context.parentURL;
  if (!parentURL) return nextResolve(specifier, context);
  if ([bundledAssetsUrl, stableBundledAssetsUrl].includes(parentURL) && specifier.endsWith('?url&inline')) {
    const assetUrl = new URL(specifier.slice(0, specifier.indexOf('?')), parentURL);
    const isStableWasm = parentURL === stableBundledAssetsUrl && assetUrl.pathname.includes('/kev06-stable/');
    const assetPath = assetUrl.pathname.split(isStableWasm ? '/kev06-stable/' : '/experimental-kev-upstream/').at(-1);
    const assets = isStableWasm ? stableInlineAssets : inlineAssets;
    const value = assets[assetPath!];
    if (typeof value !== 'string') throw new Error(`No inline asset fixture for ${assetPath}.`);
    return {url: moduleDataUrl(`export default ${JSON.stringify(value)};`), shortCircuit: true};
  }
  if (parentURL === upstreamSourceUrl && specifier.endsWith('source-shard-map.json?raw')) {
    const pathUrl = new URL(specifier.slice(0, specifier.indexOf('?')), parentURL);
    const sourceMapText = readFileSync(pathUrl, 'utf8');
    return {url: moduleDataUrl(`export default ${JSON.stringify(sourceMapText)};`), shortCircuit: true};
  }
  return nextResolve(specifier, context);
}

const registerHooks = Reflect.get(nodeModule, 'registerHooks') as RegisterHooks | undefined;
const hooks = registerHooks
  ? registerHooks({resolve: resolveTestAssetImport})
  : (nodeModule.register(new URL('./kev06-asset-import-loader.mjs', import.meta.url), import.meta.url, {
      data: {bundledAssetsUrl, stableBundledAssetsUrl, upstreamSourceUrl, inlineAssets, stableInlineAssets}
    }),
    undefined);

const bundledAssets = await import('../src/kev06/bundled-assets.js');
const stableBundledAssets = await import('../src/kev06/stable-bundled-assets.js');
const upstreamSource = await import('../src/kev06/upstream-source.mjs');
hooks?.deregister?.();
const {createKev06StableReconstructedSource, createKevReconstructedSource, EXPERIMENTAL_KEV_MODEL, KEV06_STABLE_MODEL} =
  upstreamSource;

const bytes = (...values: number[]): Uint8Array => new Uint8Array(values);
const digest = (value: Uint8Array | string): string => createHash('sha256').update(value).digest('hex');
const shardName = 'weights/params_shard_0.bin';
const sourceBase = 'https://kev-assets.web-decision.invalid/experimental-kev-upstream/';

function installTestGpu(t: TestContext, gpu: unknown = {testGpu: true}): unknown {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    enumerable: true,
    value: {gpu}
  });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else Reflect.deleteProperty(globalThis, 'navigator');
  });
  return gpu;
}

function makeFixture({seed = 0, responseBytes}: {seed?: number; responseBytes?: Uint8Array} = {}) {
  const sourceBytes = bytes(0xaa + seed, 0xbb + seed, 1 + seed, 2 + seed, 3 + seed, 4 + seed);
  const shardBytes = sourceBytes.subarray(2);
  const repository = `test/kev-${seed}`;
  const revision = `revision-${seed}`;
  const sourcePath = `model-${seed}.onnx_data`;
  const target = {
    shards: [{path: shardName, bytes: shardBytes.byteLength, sha256: digest(shardBytes)}]
  };
  const mapping = {
    source: {
      repository,
      revision,
      path: sourcePath,
      bytes: sourceBytes.byteLength,
      sha256: digest(sourceBytes),
      parameterRegionStart: 2
    },
    target,
    records: [
      {sourceOffset: 2, targetOffset: 0, bytes: 2, shardIndex: 0},
      {sourceOffset: 4, targetOffset: 2, bytes: 2, shardIndex: 0}
    ]
  };
  const sourceShardMapText = JSON.stringify(mapping);
  const manifest = {
    files: {
      [shardName]: {bytes: shardBytes.byteLength, sha256: digest(shardBytes)}
    }
  };
  const state: FixtureState = {sourceReads: 0, cacheSettings: [], preflightGpus: [], events: []};
  const overrides = {
    manifest,
    sourceShardMapText,
    sourceUrl: `https://source.invalid/${sourcePath}`,
    sourceBytes: sourceBytes.byteLength,
    sourceSha256: digest(sourceBytes),
    mapSha256: digest(sourceShardMapText),
    repository,
    revision,
    sourcePath,
    parameterRegionStart: 2,
    tensorCount: mapping.records.length,
    progressChunkBytes: 2,
    preflight: async (_signal: AbortSignal | undefined, gpu: GPU) => {
      state.events.push('preflight');
      state.preflightGpus.push(gpu);
    },
    openCache: async (enabled: boolean) => {
      state.events.push('cache');
      state.cacheSettings.push(enabled);
      return fixtureCache;
    },
    fetchSource: async (url: string, init: RequestInit) => {
      state.sourceReads++;
      state.sourceUrl = url;
      state.fetchInit = init;
      const bodyBytes = responseBytes ?? sourceBytes;
      const body = new ReadableStream({
        start(controller) {
          for (let offset = 0; offset < bodyBytes.byteLength; offset += 2)
            controller.enqueue(bodyBytes.slice(offset, offset + 2));
          controller.close();
        }
      });
      return new Response(body, {
        headers: {'content-length': String(bodyBytes.byteLength)}
      });
    }
  } satisfies KevSourceOverrides;
  return {sourceBytes, shardBytes, manifest, mapping, state, overrides};
}

function makeStableFixture({
  responseBytes,
  responseStatus = 200
}: {responseBytes?: Uint8Array; responseStatus?: number} = {}) {
  const sourceBytes = bytes(0xaa, 0xbb, 1, 2, 3, 4);
  const shardBytes = sourceBytes.subarray(2);
  const target = {
    shards: [{path: shardName, bytes: shardBytes.byteLength, sha256: digest(shardBytes)}]
  };
  const mapping = {
    source: {
      repository: EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
      revision: EXPERIMENTAL_KEV_UPSTREAM_REVISION,
      path: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH,
      bytes: sourceBytes.byteLength,
      sha256: digest(sourceBytes),
      parameterRegionStart: 2
    },
    target,
    records: [
      {sourceOffset: 2, targetOffset: 0, bytes: 2, shardIndex: 0},
      {sourceOffset: 4, targetOffset: 2, bytes: 2, shardIndex: 0}
    ]
  };
  const sourceShardMapText = JSON.stringify(mapping);
  const manifest = {
    files: {
      [shardName]: {bytes: shardBytes.byteLength, sha256: digest(shardBytes)}
    }
  };
  const state: FixtureState = {sourceReads: 0, cacheSettings: [], preflightGpus: [], events: []};
  const overrides = {
    manifest,
    sourceShardMapText,
    sourceBytes: sourceBytes.byteLength,
    sourceSha256: digest(sourceBytes),
    mapSha256: digest(sourceShardMapText),
    parameterRegionStart: 2,
    tensorCount: mapping.records.length,
    progressChunkBytes: 2,
    bundledUrls: stableInlineAssets,
    preflight: async (_signal: AbortSignal | undefined, gpu: GPU) => {
      state.events.push('preflight');
      state.preflightGpus.push(gpu);
    },
    openCache: async (enabled: boolean) => {
      state.events.push('cache');
      state.cacheSettings.push(enabled);
      return fixtureCache;
    },
    fetchSource: async (url: string, init: RequestInit) => {
      state.sourceReads++;
      state.sourceUrl = url;
      state.fetchInit = init;
      const bodyBytes = responseBytes ?? sourceBytes;
      const body = new ReadableStream({
        start(controller) {
          for (let offset = 0; offset < bodyBytes.byteLength; offset += 2)
            controller.enqueue(bodyBytes.slice(offset, offset + 2));
          controller.close();
        }
      });
      return new Response(body, {
        status: responseStatus,
        headers: {'content-length': String(bodyBytes.byteLength)}
      });
    }
  } satisfies KevSourceOverrides;
  return {sourceBytes, shardBytes, manifest, mapping, state, overrides};
}

function expectDecisionError(code: string, messagePattern: RegExp): (error: unknown) => boolean {
  return (error) => {
    if (typeof error !== 'object' || error === null || !('code' in error) || !('message' in error)) return false;
    return error.code === code && messagePattern.test(String(error.message));
  };
}

test('validates the pinned map and serves real bundled-assets exports through inline fixtures', async (t: TestContext) => {
  const gpu = installTestGpu(t);
  const events: string[] = [];
  const source = createKevReconstructedSource({
    preflight: async (_signal, receivedGpu) => {
      assert.equal(receivedGpu, gpu);
      events.push('preflight');
    },
    openCache: async (enabled) => {
      assert.equal(enabled, false);
      events.push('cache');
      return null;
    }
  });

  try {
    const prepared = await source.prepare({model: EXPERIMENTAL_KEV_MODEL, cache: false});
    const preparedManifest = await prepared.response.json();
    assert.equal(prepared.base.href, sourceBase);
    assert.equal(prepared.maxPrefetchBytes, 0);
    assert.equal(preparedManifest.experimental, true);
    assert.equal(preparedManifest.checkpoint.revision, '5a827091b3d58fe4fdfc02a5a5b2ce364047d678');
    assert.deepEqual(events, ['preflight', 'cache']);

    assert.deepEqual(bundledAssets.experimentalKevBundledUrls, inlineAssets);
    assert.equal(Object.isFrozen(bundledAssets.experimentalKevBundledUrls), true);
    assert.deepEqual(Object.keys(bundledAssets.experimentalKevBundledUrls), [
      'model.wasm',
      'tokenizer.json',
      'tokenizer_config.json',
      'model_config.json',
      'weights/tensor-cache.json'
    ]);

    const wasm = await source.fetch(new URL('model.wasm', prepared.base));
    assert.equal(wasm.headers.get('content-type'), 'application/wasm');
    assert.deepEqual(new Uint8Array(await wasm.arrayBuffer()), bytes(1, 2, 3));
    const tokenizerConfig = await source.fetch(new URL('tokenizer_config.json', prepared.base));
    assert.equal(tokenizerConfig.headers.get('content-type'), 'application/json');
    assert.deepEqual(await tokenizerConfig.json(), {lowercase: true});

    const manifestText = await readFile(
      new URL('../assets/experimental-kev-upstream/manifest.json', import.meta.url),
      'utf8'
    );
    const pinnedManifest = await upstreamSource.readExperimentalKevManifest(new Response(manifestText));
    assert.equal(pinnedManifest.files['model.wasm']!.bytes, 6672833);
    await assert.rejects(
      upstreamSource.readExperimentalKevManifest(
        new Response(manifestText.replace('"experimental": true', '"experimental": false'))
      ),
      expectDecisionError('INVALID_ASSET', /verified experimental build/u)
    );
  } finally {
    await source.dispose();
  }
});

test('stable Kev defaults to its pinned manifest and stable-only WASM bundle', async (t: TestContext) => {
  installTestGpu(t);
  const source = createKev06StableReconstructedSource({
    preflight: async () => {},
    openCache: async () => null
  });

  try {
    const prepared = await source.prepare({model: KEV06_STABLE_MODEL, cache: false});
    const manifest = await readKev06ManifestResponse(prepared.response);
    assert.equal(prepared.base.href, 'https://kev-assets.web-decision.invalid/kev-0.6b-q4f16/');
    assert.equal(prepared.manifestUrl, `${prepared.base.href}manifest.json`);
    assert.equal(KEV06_MANIFEST_SHA256, '8577804e638051cb056af51fee89eb18ea5de85ccec4f196594d830c16fdc767');
    assert.equal(manifest.model, KEV06_STABLE_MODEL);
    assert.equal(manifest.parameterCount, 394);
    assert.deepEqual(manifest.files['model.wasm'], {
      sha256: '68fe0331180dc1adfb3d1e4295c06133815b3f5bfc5ad8dbd73ae66f3a02b8fb',
      bytes: 6807315
    });
    assert.deepEqual(stableBundledAssets.stableKevBundledUrls, stableInlineAssets);
    assert.notEqual(
      stableBundledAssets.stableKevBundledUrls['model.wasm'],
      bundledAssets.experimentalKevBundledUrls['model.wasm']
    );

    const wasm = await source.fetch(new URL('model.wasm', prepared.base));
    assert.equal(wasm.headers.get('content-type'), 'application/wasm');
    assert.deepEqual(new Uint8Array(await wasm.arrayBuffer()), bytes(4, 5, 6));
  } finally {
    await source.dispose();
  }
});

test('stable Kev reconstructs a shard from its pinned upstream source with its stable options', async (t: TestContext) => {
  const gpu = installTestGpu(t);
  const fixture = makeStableFixture();
  const progress: ModelProgress[] = [];
  const source = createKev06StableReconstructedSource(fixture.overrides);
  source.setProgress((event) => progress.push(event));

  try {
    const prepared = await source.prepare({model: KEV06_STABLE_MODEL, cache: false, maxPrefetchBytes: 64});
    assert.equal(prepared.base.href, 'https://kev-assets.web-decision.invalid/kev-0.6b-q4f16/');
    assert.equal(prepared.maxPrefetchBytes, 64);
    assert.deepEqual(fixture.state.events, ['preflight', 'cache']);
    assert.deepEqual(fixture.state.cacheSettings, [false]);

    const shard = await source.fetch(new URL(shardName, prepared.base));
    assert.equal(shard.headers.get('content-length'), String(fixture.shardBytes.byteLength));
    assert.deepEqual(new Uint8Array(await shard.arrayBuffer()), fixture.shardBytes);
    assert.equal(fixture.state.sourceReads, 1);
    assert.equal(
      fixture.state.sourceUrl,
      'https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/5a827091b3d58fe4fdfc02a5a5b2ce364047d678/onnx/model_q4f16.onnx_data'
    );
    assert.equal(fixture.state.fetchInit!.credentials, 'omit');
    assert.equal(fixture.state.fetchInit!.cache, 'no-store');
    assert.deepEqual(fixture.state.preflightGpus, [gpu]);
    assert.deepEqual(
      progress.map((event) => event.loadedBytes),
      [2, 4, 6]
    );
    assert.ok(progress.every((event) => event.phase === 'upstream-source' && event.totalBytes === 6));
  } finally {
    await source.dispose();
  }
});

test('stable Kev source rejects asset URLs outside its exact synthetic origin and path', async (t: TestContext) => {
  installTestGpu(t);
  const fixture = makeStableFixture();
  const source = createKev06StableReconstructedSource(fixture.overrides);
  const prepared = await source.prepare({model: KEV06_STABLE_MODEL, cache: false});
  const base = prepared.base;
  const urls = [
    `https://attacker.example.test${base.pathname}model.wasm`,
    new URL('../outside/model.wasm', base),
    `${base.href}model.wasm?variant=1`,
    `${base.href}model.wasm#fragment`,
    `${base.href}model.wasm?`,
    `${base.href}model.wasm#`
  ];

  try {
    for (const url of urls)
      await assert.rejects(source.fetch(url), expectDecisionError('INVALID_ASSET', /Unexpected stable Kev asset URL/u));
    assert.equal(fixture.state.sourceReads, 0);
  } finally {
    await source.dispose();
  }
});

test('stable Kev source rejects invalid offline settings, pre-abort and mismatched source bytes', async (t: TestContext) => {
  installTestGpu(t);
  const fixture = makeStableFixture();
  const source = createKev06StableReconstructedSource(fixture.overrides);
  await assert.rejects(
    source.prepare({model: KEV06_STABLE_MODEL, cache: false, offline: true}),
    expectDecisionError('INVALID_INPUT', /Invalid stable Kev load option values/u)
  );
  await assert.rejects(
    source.prepare({model: KEV06_STABLE_MODEL, assetsUrl: undefined}),
    expectDecisionError('INVALID_INPUT', /Unsupported stable Kev load option/u)
  );
  const invalidMap = createKev06StableReconstructedSource({...fixture.overrides, mapSha256: '0'.repeat(64)});
  await assert.rejects(
    invalidMap.prepare({model: KEV06_STABLE_MODEL, cache: false}),
    expectDecisionError('INVALID_ASSET', /source map does not match its pinned digest/u)
  );
  await invalidMap.dispose();

  const controller = new AbortController();
  const reason = new Error('cancel stable source load');
  controller.abort(reason);
  await assert.rejects(
    source.prepare({model: KEV06_STABLE_MODEL, signal: controller.signal}),
    (error) => error === reason
  );
  await source.dispose();

  const mismatch = makeStableFixture({responseBytes: bytes(0xaa, 0xbb, 7, 8, 9, 10)});
  const mismatchedSource = createKev06StableReconstructedSource(mismatch.overrides);
  const prepared = await mismatchedSource.prepare({model: KEV06_STABLE_MODEL, cache: false});
  try {
    await assert.rejects(
      mismatchedSource.fetch(new URL(shardName, prepared.base)),
      expectDecisionError('INVALID_ASSET', /failed SHA-256 verification/u)
    );
  } finally {
    await mismatchedSource.dispose();
  }

  const errorResponse = makeStableFixture({responseStatus: 404});
  const unavailableSource = createKev06StableReconstructedSource(errorResponse.overrides);
  const unavailablePrepared = await unavailableSource.prepare({model: KEV06_STABLE_MODEL, cache: false});
  try {
    await assert.rejects(
      unavailableSource.fetch(new URL(shardName, unavailablePrepared.base)),
      expectDecisionError('ASSET_LOAD_FAILED', /HTTP 404/u)
    );
  } finally {
    await unavailableSource.dispose();
  }
});

test('stable Kev abort releases an in-flight upstream response when the load is disposed', async (t: TestContext) => {
  installTestGpu(t);
  const fixture = makeStableFixture();
  let resolveReadStarted!: () => void;
  let resolveStreamCancelled!: (reason: unknown) => void;
  const readStarted = new Promise<void>((resolve) => {
    resolveReadStarted = resolve;
  });
  const streamCancelled = new Promise<unknown>((resolve) => {
    resolveStreamCancelled = resolve;
  });
  const overrides = {
    ...fixture.overrides,
    fetchSource: async (url: string, init: RequestInit) => {
      fixture.state.sourceReads++;
      fixture.state.sourceUrl = url;
      fixture.state.fetchInit = init;
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(bytes(0xaa, 0xbb));
            resolveReadStarted();
          },
          cancel(reason) {
            resolveStreamCancelled(reason);
          }
        }),
        {headers: {'content-length': String(fixture.sourceBytes.byteLength)}}
      );
    }
  };
  const source = createKev06StableReconstructedSource(overrides);
  const prepared = await source.prepare({model: KEV06_STABLE_MODEL, cache: false});
  const controller = new AbortController();
  const reason = new Error('cancel stable upstream read');
  const pending = source.fetch(new URL(shardName, prepared.base), {signal: controller.signal});
  await readStarted;
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  await source.dispose();
  assert.equal(fixture.state.fetchInit!.signal!.aborted, true);
  await streamCancelled;
});

test('reconstructs a mapped shard from one verified source read and reports source progress', async (t: TestContext) => {
  installTestGpu(t);
  const fixture = makeFixture({seed: 5});
  const source = createKevReconstructedSource(fixture.overrides);
  const progress: ModelProgress[] = [];
  upstreamSource.setExperimentalKevProgress(source, (value) => progress.push(value));

  try {
    const prepared = await source.prepare({model: EXPERIMENTAL_KEV_MODEL, cache: false});
    assert.equal(prepared.response.status, 200);
    assert.equal(prepared.manifestUrl, `${sourceBase}manifest.json`);
    assert.deepEqual(fixture.state.events, ['preflight', 'cache']);
    assert.deepEqual(fixture.state.cacheSettings, [false]);

    const url = new URL(shardName, prepared.base);
    const first = await source.fetch(url);
    assert.equal(first.headers.get('content-length'), String(fixture.shardBytes.byteLength));
    assert.deepEqual(new Uint8Array(await first.arrayBuffer()), fixture.shardBytes);
    const second = await source.fetch(url);
    assert.deepEqual(new Uint8Array(await second.arrayBuffer()), fixture.shardBytes);

    assert.equal(fixture.state.sourceReads, 1);
    assert.equal(fixture.state.sourceUrl, 'https://source.invalid/model-5.onnx_data');
    assert.equal(fixture.state.fetchInit!.credentials, 'omit');
    assert.equal(fixture.state.fetchInit!.cache, 'no-store');
    assert.deepEqual(fixture.state.preflightGpus, [globalThis.navigator.gpu]);
    assert.deepEqual(
      progress.map((event) => event.loadedBytes),
      [2, 4, 6]
    );
    assert.ok(progress.every((event) => event.phase === 'upstream-source' && event.totalBytes === 6));
  } finally {
    await source.dispose();
  }
});

test('rejects invalid source pins, load options and malformed bundled URLs', async (t: TestContext) => {
  installTestGpu(t);
  const fixture = makeFixture({seed: 9});
  const invalidMap = createKevReconstructedSource({...fixture.overrides, mapSha256: '0'.repeat(64)});
  await assert.rejects(
    invalidMap.prepare({model: EXPERIMENTAL_KEV_MODEL, cache: false}),
    expectDecisionError('INVALID_ASSET', /source map does not match/u)
  );
  await invalidMap.dispose();

  const invalidOptions = createKevReconstructedSource(fixture.overrides);
  await assert.rejects(
    invalidOptions.prepare({model: 'another-model'}),
    expectDecisionError('INVALID_INPUT', /Invalid experimental Kev load option/u)
  );
  await invalidOptions.dispose();

  assert.throws(
    () => createKevReconstructedSource({bundledUrls: {'model.wasm': 'https://example.invalid/model.wasm'}}),
    expectDecisionError('INVALID_ASSET', /missing pinned inline assets/u)
  );
  const malformedDataUrl = createKevReconstructedSource({bundledUrls: {'model.wasm': 'data:application/json,%ZZ'}});
  await assert.rejects(
    malformedDataUrl.fetch(`${sourceBase}model.wasm`),
    expectDecisionError('INVALID_ASSET', /encoding/u)
  );

  const mismatchedSource = makeFixture({seed: 13, responseBytes: bytes(0, 0, 7, 8, 9, 10)});
  const source = createKevReconstructedSource(mismatchedSource.overrides);
  await source.prepare({model: EXPERIMENTAL_KEV_MODEL, cache: false});
  try {
    await assert.rejects(
      source.fetch(`${sourceBase}${shardName}`),
      expectDecisionError('INVALID_ASSET', /failed SHA-256 verification/u)
    );
  } finally {
    await source.dispose();
  }
});
