import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readFile, readdir, realpath, rm, stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {build} from 'vite';
import type {AssetCache} from '../src/assets.js';
import artifactLock from '../src/catalog/artifacts.lock.json' with {type: 'json'};
import {
  verifyPinnedQwen35VisionArtifacts,
  verifyQwen35VisionFixtureWasm
} from '../src/strands/qwen35-vision-contract.mts';
import type {Qwen35VisionArtifacts} from '../src/strands/qwen35-vision-contract.mts';
import {isPinnedQwen35VisionRopeInitializationEvidence} from '../src/strands/qwen35-vision-runner.mts';
import type {Qwen35VisionWeightBinding} from '../src/strands/qwen35-vision-weights.mts';
import {importBuiltModule} from './fixtures/import-built-module.mts';

const lockedFiles = artifactLock.byModelId['strands-decider-2b-qwen3.5-v1-2610'].manifest.files;

function hash(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

test('packages the independently built FP32 Qwen3.5 vision graph with all 297 source bindings', async () => {
  const root = new URL('../assets/qwen35/vision/', import.meta.url);
  const [fixtureBytes, wasmBytes, bindingsBytes, buildReportBytes] = await Promise.all([
    readFile(new URL('fixture.json', root)),
    readFile(new URL('qwen35-dynamic-vision.webgpu.wasm', root)),
    readFile(new URL('weight-bindings.json', root)),
    readFile(
      new URL('../../../scripts/strands/qwen35/reports/qwen35-fp32-vision-build-2026-11-21.json', import.meta.url)
    )
  ]);
  const packageManifest = JSON.parse(
    await readFile(new URL('../assets/qwen35/manifest.json', import.meta.url), 'utf8')
  ) as {readonly files: Readonly<Record<string, {readonly bytes: number; readonly sha256: string}>>};
  const fixture = JSON.parse(fixtureBytes.toString('utf8')) as Qwen35VisionArtifacts['artifact'] & {
    readonly modelSupport: string;
    readonly pins: {
      readonly base: {readonly revision: string};
      readonly checkpoint: {readonly revision: string};
    };
    readonly parameters: {
      readonly records: readonly {
        readonly bytes: number;
        readonly dtype: string;
        readonly name: string;
        readonly shape: readonly number[];
      }[];
    };
  };
  const buildReport = JSON.parse(buildReportBytes.toString('utf8')) as {
    readonly format: string;
    readonly activationPrecision: string;
    readonly replacedKernel: string;
    readonly vmSha256: string;
    readonly tvmJsArtifact: {
      readonly status: string;
      readonly wasm: {readonly bytes: number; readonly sha256: string};
    };
    readonly tvmWebgpuBuild: {readonly status: string; readonly deviceRuntimeExecuted: boolean};
  };
  const bindings = JSON.parse(bindingsBytes.toString('utf8')) as {
    readonly format: string;
    readonly bindingsSha256: string;
    readonly argumentNamesSha256: string;
    readonly bindings: readonly Qwen35VisionWeightBinding[];
  };

  const fixturePin = {bytes: fixtureBytes.byteLength, sha256: hash(fixtureBytes)};
  const wasmPin = {bytes: wasmBytes.byteLength, sha256: hash(wasmBytes)};
  assert.deepEqual(fixturePin, lockedFiles['vision/fixture.json']);
  assert.deepEqual(wasmPin, lockedFiles['vision/qwen35-dynamic-vision.webgpu.wasm']);
  assert.equal(hash(bindingsBytes), '340b604c8db09219cfb2aa79b2e930da1c5d1a0f07972263787f26a2c63dd60a');
  assert.deepEqual(packageManifest.files['vision/fixture.json'], fixturePin);
  assert.deepEqual(packageManifest.files['vision/qwen35-dynamic-vision.webgpu.wasm'], wasmPin);
  verifyQwen35VisionFixtureWasm(fixture);
  assert.equal(hash(buildReportBytes), 'fd7f9a1cbc3c2d2cee6d9c9eba3b8d2af47e396139b9fbb77d4390d563371024');
  assert.equal(buildReport.format, 'qwen35-fp32-vision-webgpu-build-v1');
  assert.equal(buildReport.activationPrecision, 'float32');
  assert.equal(fixture.activationPrecision, buildReport.activationPrecision);
  assert.equal(buildReport.replacedKernel, 'round_bf16_webgpu');
  assert.equal(buildReport.vmSha256, 'be2f5f6209bc1c79b8fd2f1eba540fe0ab21c1afd60460de893e277f54d86a5f');
  assert.equal(buildReport.tvmJsArtifact.status, 'passed');
  assert.deepEqual(buildReport.tvmJsArtifact.wasm, wasmPin);
  assert.equal(buildReport.tvmWebgpuBuild.status, 'passed');
  assert.equal(buildReport.tvmWebgpuBuild.deviceRuntimeExecuted, false);
  assert.equal(fixture.format, 'qwen35-dynamic-per-image-vision-tvmjs-fixture-v1');
  assert.equal(fixture.modelSupport, 'unverified');
  assert.equal(fixture.pins.base.revision, 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c');
  assert.equal(fixture.pins.checkpoint.revision, '216d1ccb46a9e9b036eb9187b248db98480f77d4');
  assert.equal(fixture.shapeContract.oneImagePerInvocation, true);
  assert.equal(fixture.shapeContract.patchRows.formula, '4*S');
  assert.equal(fixture.shapeContract.patchRows.minimum, 256);
  assert.equal(fixture.shapeContract.patchRows.maximum, 65_536);
  const inputs = fixture.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'USER_INPUT');
  assert.deepEqual(
    inputs.map(({name, dtype, shape, relaxShape}) => [name, dtype, shape, relaxShape]),
    [
      ['pixel_values', 'float32', [256, 1536], ['T.int64(4) * s34', 'T.int64(1536)']],
      ['grid_thw', 'int32', [1, 3], ['T.int64(1)', 'T.int64(3)']],
      ['interp_indices', 'int32', [256, 4], ['T.int64(4) * s34', 'T.int64(4)']],
      ['interp_weights', 'float32', [256, 4], ['T.int64(4) * s34', 'T.int64(4)']],
      ['position_ids_by_token', 'int32', [64, 8], ['s34', 'T.int64(8)']],
      ['cu_seqlens', 'int32', [2], ['T.int64(2)']]
    ]
  );
  assert.equal(fixture.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'PARAMETER').length, 297);
  assert.equal(
    isPinnedQwen35VisionRopeInitializationEvidence(fixture.visionRopeInitialization),
    true,
    'the packaged fixture carries the pinned vision RoPE initialization evidence'
  );
  assert.equal(bindings.format, 'qwen35-vision-pinned-source-weight-bindings-v1');
  assert.equal(bindings.bindingsSha256, 'db9bc5cad76fe05237dec5673fa761ebc8ccdd92ebe59757c6e5f0a123c510be');
  assert.equal(bindings.argumentNamesSha256, '7dbb6ce3d4e1f102736f80def6c8f2d792802f215b13905c37538e0fab40f66a');
  assert.equal(bindings.bindings.length, 297);
  assert.equal(fixture.parameters.records.length, 297);
  assert.ok(
    bindings.bindings.every(
      ({sourceDtype, conversion}) => sourceDtype === 'BF16' && conversion === 'exact-bf16-to-float32'
    )
  );

  const wordManifest = JSON.parse(
    await readFile(new URL('../assets/qwen35/pinned-word-assets-manifest.json', import.meta.url), 'utf8')
  ) as {readonly assets: {readonly base: {readonly tensors: readonly Record<string, unknown>[]}}};
  const sourceTensors = new Map(wordManifest.assets.base.tensors.map((tensor) => [tensor.name, tensor]));
  const graphParameters = fixture.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'PARAMETER');
  for (const [index, binding] of bindings.bindings.entries()) {
    const tensor = sourceTensors.get(binding.sourceName);
    const graphArgument = graphParameters[index];
    const parameter = fixture.parameters.records[index];
    assert.ok(tensor, 'source tensor is present: ' + binding.sourceName);
    assert.ok(graphArgument);
    assert.ok(parameter);
    assert.equal(binding.index, index);
    assert.equal(binding.targetName, graphArgument.name);
    assert.equal(binding.targetName, parameter.name);
    assert.equal(binding.targetDtype, graphArgument.dtype);
    assert.equal(binding.targetDtype, parameter.dtype);
    assert.deepEqual(binding.shape, graphArgument.shape);
    assert.deepEqual(binding.shape, parameter.shape);
    assert.equal(binding.elementCount * 4, parameter.bytes);
    assert.equal(tensor.dtype, 'BF16');
    assert.deepEqual(tensor.shape, binding.shape);
    assert.equal(tensor.elementCount, binding.elementCount);
    assert.equal(tensor.sourceByteCount, binding.sourceByteCount);
    assert.equal(tensor.sourcePayloadSha256, binding.sourcePayloadSha256);
  }
});

test('accepts only initialization evidence for pinned upstream FP32 RoPE buffers', () => {
  const evidence = {
    restoredBuffers: ['inv_freq', 'original_inv_freq'],
    sourceModelParameterDtype: 'bfloat16',
    bufferDtype: 'float32',
    matchesPinnedInitialization: true,
    bufferSha256: {
      inv_freq: '9cc63b858a37a16bf14e559754d6fee5f50ae74aabfce29fec9097b70902849c',
      original_inv_freq: '9cc63b858a37a16bf14e559754d6fee5f50ae74aabfce29fec9097b70902849c'
    }
  } as const;
  assert.equal(isPinnedQwen35VisionRopeInitializationEvidence(evidence), true);
  assert.equal(
    isPinnedQwen35VisionRopeInitializationEvidence({...evidence, matchesPinnedInitialization: false}),
    false
  );
  assert.equal(isPinnedQwen35VisionRopeInitializationEvidence({...evidence, restoredBuffers: ['inv_freq']}), false);
});

test('Vite library build keeps the vision graph and bindings outside JavaScript', async (t) => {
  const outDir = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-qwen35-vision-package-')));
  t.after(() => rm(outDir, {recursive: true, force: true}));

  await build({
    configFile: false,
    base: './',
    root: fileURLToPath(new URL('..', import.meta.url)),
    build: {
      outDir,
      emptyOutDir: true,
      lib: {
        entry: fileURLToPath(new URL('../src/strands/qwen35-vision-artifacts.mts', import.meta.url)),
        formats: ['es'],
        fileName: 'qwen35-vision-artifacts.js'
      },
      target: 'es2022',
      sourcemap: true
    }
  });

  const assetNames = ['fixture.json', 'qwen35-dynamic-vision.webgpu.wasm', 'weight-bindings.json'] as const;
  const files = await readdir(outDir, {recursive: true});
  const jsName = files.find((name) => String(name).endsWith('.js') || String(name).endsWith('.mjs'));
  assert.ok(jsName);
  const jsPath = join(outDir, jsName);
  const js = await readFile(jsPath, 'utf8');
  assert.ok((await stat(jsPath)).size < 32_768);
  for (const name of assetNames) assert.ok(js.includes(name));

  for (const name of assetNames) {
    assert.ok(files.includes(name));
    const bytes = await readFile(join(outDir, name));
    assert.deepEqual({bytes: bytes.byteLength, sha256: hash(bytes)}, lockedFiles[`vision/${name}`]);
  }

  const entries = new Map<string, Response>();
  const cache: AssetCache = {
    async match(key) {
      return entries.get(key)?.clone() ?? null;
    },
    async put(key, response) {
      entries.set(key, response.clone());
    }
  };
  const originalFetch = globalThis.fetch;
  const requests: string[] = [];
  globalThis.fetch = async (input) => {
    const url = new URL(String(input));
    requests.push(url.href);
    assert.equal(url.protocol, 'file:');
    url.search = '';
    url.hash = '';
    return new Response(await readFile(fileURLToPath(url)));
  };
  try {
    const bundle = (await importBuiltModule(jsPath)) as {
      openPinnedQwen35VisionArtifacts(options?: {
        cache?: AssetCache | null;
        offline?: boolean;
      }): Promise<Qwen35VisionArtifacts>;
    };

    const online = await bundle.openPinnedQwen35VisionArtifacts({cache});
    assert.equal(hash(online.fixtureBytes), lockedFiles['vision/fixture.json'].sha256);
    assert.equal(hash(online.wasmBytes), lockedFiles['vision/qwen35-dynamic-vision.webgpu.wasm'].sha256);
    assert.equal(online.bindings.length, 297);
    assert.equal(isPinnedQwen35VisionRopeInitializationEvidence(online.artifact.visionRopeInitialization), true);
    await verifyPinnedQwen35VisionArtifacts(online);
    assert.throws(
      () =>
        verifyQwen35VisionFixtureWasm({
          ...online.artifact,
          activationPrecision: online.artifact.activationPrecision === 'float32' ? 'bfloat16' : 'float32'
        }),
      /fixture precision or executable/
    );
    assert.throws(
      () =>
        verifyQwen35VisionFixtureWasm({
          ...online.artifact,
          wasm: {bytes: online.wasmBytes.length, sha256: '0'.repeat(64)}
        }),
      /fixture precision or executable/
    );
    assert.equal(requests.length, assetNames.length);
    assert.deepEqual(requests.map((request) => basename(new URL(request).pathname)).sort(), [...assetNames].sort());
    assert.equal(entries.size, assetNames.length);

    globalThis.fetch = async () => {
      assert.fail('offline Qwen vision asset reuse must not fetch');
    };
    const offline = await bundle.openPinnedQwen35VisionArtifacts({cache, offline: true});
    assert.equal(hash(offline.fixtureBytes), hash(online.fixtureBytes));
    assert.equal(hash(offline.wasmBytes), hash(online.wasmBytes));
    assert.equal(offline.bindings.length, 297);
    await verifyPinnedQwen35VisionArtifacts(offline);
    assert.equal(requests.length, assetNames.length);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
