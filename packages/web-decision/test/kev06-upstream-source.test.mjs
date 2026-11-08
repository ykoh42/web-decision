import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import * as nodeModule from 'node:module';
import test from 'node:test';

const bundledAssetsUrl = new URL('../src/kev06/bundled-assets.js', import.meta.url).href;
const upstreamSourceUrl = new URL('../src/kev06/upstream-source.mjs', import.meta.url).href;
const inlineAssets = Object.freeze({
  'model.wasm': 'data:application/wasm;base64,AQID',
  'tokenizer.json': `data:application/json;base64,${Buffer.from('{"tokens":[]}').toString('base64')}`,
  'tokenizer_config.json': 'data:application/json,%7B%22lowercase%22%3Atrue%7D',
  'model_config.json': `data:application/json;base64,${Buffer.from('{"hidden_size":4}').toString('base64')}`,
  'weights/tensor-cache.json': 'data:application/json,%7B%22records%22%3A%5B%5D%7D'
});

function moduleDataUrl(source) {
  return `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
}

function resolveTestAssetImport(specifier, context, nextResolve) {
  if (context.parentURL === bundledAssetsUrl && specifier.endsWith('?url&inline')) {
    const assetUrl = new URL(specifier.slice(0, specifier.indexOf('?')), context.parentURL);
    const assetPath = assetUrl.pathname.split('/experimental-kev-upstream/').at(-1);
    const value = inlineAssets[assetPath];
    if (typeof value !== 'string') throw new Error(`No inline asset fixture for ${assetPath}.`);
    return {url: moduleDataUrl(`export default ${JSON.stringify(value)};`), shortCircuit: true};
  }
  if (context.parentURL === upstreamSourceUrl && specifier.endsWith('source-shard-map.json?raw')) {
    const pathUrl = new URL(specifier.slice(0, specifier.indexOf('?')), context.parentURL);
    const sourceMapText = readFileSync(pathUrl, 'utf8');
    return {url: moduleDataUrl(`export default ${JSON.stringify(sourceMapText)};`), shortCircuit: true};
  }
  return nextResolve(specifier, context);
}

const hooks =
  typeof nodeModule.registerHooks === 'function'
    ? nodeModule.registerHooks({resolve: resolveTestAssetImport})
    : nodeModule.register(new URL('./kev06-asset-import-loader.mjs', import.meta.url), import.meta.url, {
        data: {bundledAssetsUrl, upstreamSourceUrl, inlineAssets}
      });

const bundledAssets = await import('../src/kev06/bundled-assets.js');
const upstreamSource = await import('../src/kev06/upstream-source.mjs');
hooks?.deregister?.();
const {createKevReconstructedSource, EXPERIMENTAL_KEV_MODEL} = upstreamSource;

const bytes = (...values) => new Uint8Array(values);
const digest = (value) => createHash('sha256').update(value).digest('hex');
const shardName = 'weights/params_shard_0.bin';
const sourceBase = 'https://kev-assets.web-decision.invalid/experimental-kev-upstream/';

function installTestGpu(t, gpu = {testGpu: true}) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    enumerable: true,
    value: {gpu}
  });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  });
  return gpu;
}

function makeFixture({seed = 0, responseBytes} = {}) {
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
  const state = {sourceReads: 0, cacheSettings: [], preflightGpus: [], events: []};
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
    preflight: async (_signal, gpu) => {
      state.events.push('preflight');
      state.preflightGpus.push(gpu);
    },
    openCache: async (enabled) => {
      state.events.push('cache');
      state.cacheSettings.push(enabled);
      return {fixtureCache: true};
    },
    fetchSource: async (url, init) => {
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
  };
  return {sourceBytes, shardBytes, manifest, mapping, state, overrides};
}

function expectDecisionError(code, messagePattern) {
  return (error) => error?.code === code && messagePattern.test(error.message);
}

test('validates the pinned map and serves real bundled-assets exports through inline fixtures', async (t) => {
  const gpu = installTestGpu(t);
  const events = [];
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
    assert.equal(pinnedManifest.files['model.wasm'].bytes, 6672833);
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

test('reconstructs a mapped shard from one verified source read and reports source progress', async (t) => {
  installTestGpu(t);
  const fixture = makeFixture({seed: 5});
  const source = createKevReconstructedSource(fixture.overrides);
  const progress = [];
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
    assert.equal(fixture.state.fetchInit.credentials, 'omit');
    assert.equal(fixture.state.fetchInit.cache, 'no-store');
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

test('rejects invalid source pins, load options and malformed bundled URLs', async (t) => {
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
