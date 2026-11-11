import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {MODEL_CATALOG, MODEL_CATALOG_BY_ID, MODEL_IDS} from '../src/catalog/index.generated.ts';
import {getModelRegistration} from '../src/model-registry.js';
import {
  createDirectKev06ArtifactLockFromDirectory,
  createModelArtifactLock,
  validateSpec
} from '../scripts/build-model-catalog.mts';
import {DIRECT_KEV06_BASE, DIRECT_KEV06_CHECKPOINT} from '../src/kev06/direct-contract.js';

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageFile = (relativePath: string): string => path.join(packageRoot, relativePath);

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonical((value as Record<string, unknown>)[key])])
  );
}

test('catalog exposes Kev and Kanana through ID-only TVM descriptors', () => {
  assert.deepEqual(MODEL_IDS, ['kev-0.6b-q4f16', 'kanana-2-v-decide-3b-2611']);
  assert.equal(MODEL_CATALOG.length, 2);

  const kev = MODEL_CATALOG_BY_ID['kev-0.6b-q4f16'];
  assert.equal(kev, MODEL_CATALOG[0]);
  assert.equal(kev.description.adapter, 'kev06');
  assert.deepEqual(kev.description.backend.requiredFeatures, ['shader-f16']);

  const kanana = MODEL_CATALOG_BY_ID['kanana-2-v-decide-3b-2611'];
  assert.equal(kanana, MODEL_CATALOG[1]);
  assert.equal(kanana.description.adapter, 'kanana');
  assert.equal(kanana.description.backend.name, 'tvm-webgpu');
  assert.equal(kanana.description.backend.requiredLimits.maxBufferSize, 656_670_720);
  assert.equal(kanana.description.backend.requiredLimits.maxStorageBufferBindingSize, 656_670_720);
  assert.equal(kanana.description.checkpoint.revision, '12820d654ed9789a0f7f48daeee1703c3483c464');
  assert.equal(kanana.description.features.textChoice.status, 'implemented');
  assert.equal(kanana.description.features.imageChoice?.status, 'implemented');
  assert.deepEqual(kanana.description.features.imageChoice?.limits, [
    {
      resource: 'images',
      scope: 'request',
      max: 10,
      description: 'Pinned model-card maximum across images in state messages and the images field.'
    }
  ]);
  assert.equal(kanana.description.verification.currentBundleGpu, 'verified-m4-pro-chromium-157');
  for (const entry of [kev, kanana]) {
    assert.equal(Object.hasOwn(entry, 'model'), false);
    assert.equal(Object.hasOwn(entry.description, 'model'), false);
    assert.equal(Object.hasOwn(entry.description.assets, 'assetsUrl'), false);
    assert.equal(Object.hasOwn(entry.description.assets, 'manifestSha256'), true);
  }
  assert.equal(Object.hasOwn(MODEL_CATALOG_BY_ID['kev-0.6b-q4f16'].description.assets, 'directArtifactRoot'), false);
});

test('Kev catalog description retains its existing contract', () => {
  const entry = MODEL_CATALOG_BY_ID['kev-0.6b-q4f16'];
  const {adapter, ...description} = entry.description;
  const legacy = getModelRegistration(entry.modelId).description;
  const {model, ...expected} = structuredClone(legacy);
  assert.equal(model, entry.modelId);
  expected.verification.currentBundleGpu = 'not-revalidated';
  expected.verification.priorEvidence = entry.description.verification.priorEvidence!;
  expected.assets.requiredFiles = ['manifest.json', ...Object.keys(entry.artifacts.manifest.files)];

  assert.equal(adapter, 'kev06');
  assert.deepEqual(description, expected);
  assert.deepEqual(
    description.features.textChoice.limits.map(({resource, scope, min, max}) => ({resource, scope, min, max})),
    [
      {resource: 'tokens', scope: 'state', min: undefined, max: 8191},
      {resource: 'tokens', scope: 'question', min: undefined, max: 8192},
      {resource: 'options', scope: 'question', min: 1, max: 255},
      {resource: 'questions', scope: 'request', min: 1, max: undefined}
    ]
  );
});

test('catalog rejects empty, duplicate, non-TVM, and unimplemented adapter entries', async () => {
  const spec = JSON.parse(await readFile(packageFile('src/catalog/model-spec.json'), 'utf8'));
  const [model] = spec.models;

  assert.deepEqual(
    validateSpec(spec).map((entry) => entry.modelId),
    MODEL_IDS
  );
  assert.throws(() => validateSpec({...spec, models: []}), /at least one curated model/u);
  assert.throws(() => validateSpec({...spec, models: [model, model]}), /modelIds must be unique/u);
  assert.throws(
    () =>
      validateSpec({
        ...spec,
        models: [
          {
            ...model,
            description: {...model.description, backend: {...model.description.backend, name: 'onnxruntime-webgpu'}}
          }
        ]
      }),
    /must use TVM WebGPU/u
  );
  await assert.rejects(
    createModelArtifactLock({
      ...model,
      description: {...model.description, adapter: 'unregistered-adapter'}
    }),
    /No artifact recipe/u
  );
  assert.throws(
    () =>
      validateSpec({
        ...spec,
        models: [
          {
            ...model,
            description: {
              ...model.description,
              assets: {...model.description.assets, directArtifactRoot: ''}
            }
          }
        ]
      }),
    /Direct artifact roots require the direct Kev checkpoint/u
  );
});

test('direct Kev recipe locks a manifest-only remote package and rejects malformed or tampered input', async () => {
  const spec = JSON.parse(await readFile(packageFile('src/catalog/model-spec.json'), 'utf8'));
  const [sourceModel] = spec.models;
  const artifactUpstreams = [
    {
      role: 'base-config',
      repository: DIRECT_KEV06_BASE.repository,
      revision: DIRECT_KEV06_BASE.revision,
      sourcePath: 'config.json',
      sourceBytes: 727,
      sourceSha256: '504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59'
    }
  ];
  const directModel = {
    ...sourceModel,
    artifactUpstreams,
    description: {
      ...sourceModel.description,
      checkpoint: {...DIRECT_KEV06_CHECKPOINT},
      assets: {
        ...sourceModel.description.assets,
        directArtifactRoot: 'https://assets.example.test/kev-0.6b-q4f16/'
      }
    }
  };
  const fixtureDirectory = await mkdtemp(packageFile('test/fixtures/.direct-kev-package-'));
  const manifestFixturePath = path.resolve(
    packageRoot,
    '../../tools/kev-converter/reports/direct-tvm-q4-webgpu-manifest-r4.json'
  );
  const manifestBytes = await readFile(manifestFixturePath);
  const manifestFixture = JSON.parse(manifestBytes.toString('utf8'));
  const canonicalBytes = Buffer.from(JSON.stringify(canonical(manifestFixture)) + '\n');

  try {
    assert.deepEqual(validateSpec({...spec, models: [directModel]}), [directModel]);
    const {artifactUpstreams: _artifactUpstreams, ...unpinnedModel} = directModel;
    assert.throws(() => validateSpec({...spec, models: [unpinnedModel]}), /must pin its conversion inputs/u);
    await assert.rejects(
      createDirectKev06ArtifactLockFromDirectory(directModel, path.resolve(packageRoot, '..')),
      /inside the web-decision package/u
    );
    await writeFile(path.join(fixtureDirectory, 'manifest.json'), manifestBytes);
    const lock = await createDirectKev06ArtifactLockFromDirectory(directModel, fixtureDirectory);

    assert.match(lock.manifest.sourcePath, /^test\/fixtures\/.direct-kev-package-[^/]+\/manifest\.json$/u);
    assert.equal(lock.manifest.rawSha256, sha256(manifestBytes));
    assert.equal(lock.manifest.canonicalSha256, sha256(canonicalBytes));
    assert.equal(lock.manifest.canonicalBytes, canonicalBytes.byteLength);
    assert.equal(lock.manifest.experimental, true);
    assert.deepEqual(lock.manifest.files, manifestFixture.files);
    assert.deepEqual(lock.bundledRuntimeFiles, []);
    assert.equal(lock.upstream, undefined);
    assert.deepEqual(lock.upstreams, artifactUpstreams);
    assert.deepEqual(Object.keys(lock.manifest.files), Object.keys(manifestFixture.files));

    const tamperedManifest = {
      ...manifestFixture,
      files: {...manifestFixture.files, 'model.wasm': {...manifestFixture.files['model.wasm'], sha256: '0'.repeat(64)}}
    };
    await writeFile(path.join(fixtureDirectory, 'manifest.json'), JSON.stringify(tamperedManifest));
    await assert.rejects(
      createDirectKev06ArtifactLockFromDirectory(directModel, fixtureDirectory),
      /manifest does not match the verified experimental build/u
    );

    const unsafeManifest = {
      ...manifestFixture,
      files: {...manifestFixture.files, '../outside.bin': {bytes: 1, sha256: 'a'.repeat(64)}}
    };
    const unsafeCanonicalSha = sha256(Buffer.from(JSON.stringify(canonical(unsafeManifest)) + '\n'));
    await writeFile(path.join(fixtureDirectory, 'manifest.json'), JSON.stringify(unsafeManifest));
    await assert.rejects(
      createDirectKev06ArtifactLockFromDirectory(directModel, fixtureDirectory, unsafeCanonicalSha),
      /unsafe file path/u
    );

    const malformedManifest = {
      ...manifestFixture,
      files: {...manifestFixture.files, 'model.wasm': {...manifestFixture.files['model.wasm'], bytes: '4041060'}}
    };
    const malformedCanonicalSha = sha256(Buffer.from(JSON.stringify(canonical(malformedManifest)) + '\n'));
    await writeFile(path.join(fixtureDirectory, 'manifest.json'), JSON.stringify(malformedManifest));
    await assert.rejects(
      createDirectKev06ArtifactLockFromDirectory(directModel, fixtureDirectory, malformedCanonicalSha),
      /Invalid direct Kev asset metadata/u
    );

    await writeFile(path.join(fixtureDirectory, 'manifest.json'), manifestBytes);
    await writeFile(path.join(fixtureDirectory, 'undeclared.txt'), 'fixture-only file\n');
    await assert.rejects(createDirectKev06ArtifactLockFromDirectory(directModel, fixtureDirectory), /undeclared file/u);
  } finally {
    await rm(fixtureDirectory, {recursive: true, force: true});
  }
});

test('legacy Kev lock keeps the bundled runtime-file recipe', async () => {
  const spec = JSON.parse(await readFile(packageFile('src/catalog/model-spec.json'), 'utf8'));
  const [sourceModel] = spec.models;
  const {artifactUpstreams: _artifactUpstreams, ...sourceModelWithoutPrivatePins} = sourceModel;
  const bundledModel = {
    ...sourceModelWithoutPrivatePins,
    description: {
      ...sourceModel.description,
      checkpoint: {
        repository: 'onnx-community/kev-0.6b-ONNX',
        revision: '5a827091b3d58fe4fdfc02a5a5b2ce364047d678'
      }
    }
  };

  const lock = await createModelArtifactLock(bundledModel);

  assert.equal(lock.manifest.canonicalSha256, MODEL_CATALOG_BY_ID['kev-0.6b-q4f16'].artifacts.manifest.canonicalSha256);
  assert.equal(lock.bundledRuntimeFiles.length, 5);
  assert.equal(lock.upstream?.repository, 'onnx-community/kev-0.6b-ONNX');
  assert.equal(Object.hasOwn(lock, 'upstreams'), false);
});

test('artifact locks hash packaged manifests, runtime files, and model ranges', async () => {
  const lockFile = JSON.parse(await readFile(packageFile('src/catalog/artifacts.lock.json'), 'utf8'));
  assert.deepEqual(Object.keys(lockFile.byModelId), MODEL_IDS);
  for (const entry of MODEL_CATALOG) {
    const lock = entry.artifacts;
    const manifestBytes = await readFile(packageFile(lock.manifest.sourcePath));
    const manifest = JSON.parse(manifestBytes.toString('utf8'));
    const canonicalManifest = Buffer.from(JSON.stringify(canonical(manifest)) + '\n');

    if (entry.modelId === 'kanana-2-v-decide-3b-2611') {
      assert.equal(manifest.modifications.modifiedBy, 'web-decision');
      assert.match(manifest.modifications.description, /generated TVM WebGPU modules/u);
      assert.deepEqual(manifest.modifications.files, Object.keys(manifest.files));
    }

    assert.equal(lock.manifest.rawSha256, sha256(manifestBytes));
    assert.equal(lock.manifest.canonicalSha256, sha256(canonicalManifest));
    assert.equal(entry.description.assets.manifestSha256, lock.manifest.canonicalSha256);
    assert.deepEqual(entry.description.assets.requiredFiles, ['manifest.json', ...Object.keys(manifest.files)]);
    assert.deepEqual(lock.manifest.files, manifest.files);
    assert.deepEqual(lockFile.byModelId[entry.modelId], lock);
    assert.equal(Object.hasOwn(lock, 'priorHardwareEvidence'), false);

    for (const artifact of lock.bundledRuntimeFiles) {
      const bytes = await readFile(packageFile(artifact.sourcePath));
      assert.equal(artifact.bytes, bytes.byteLength);
      assert.equal(artifact.sha256, sha256(bytes));
      assert.equal(artifact.sha256, lock.manifest.files[artifact.manifestName]!.sha256);
    }

    if (lock.upstream?.shardMapPath) {
      const mapBytes = await readFile(packageFile(lock.upstream.shardMapPath));
      assert.equal(lock.upstream.shardMapSha256, sha256(mapBytes));
    }
    if (lock.upstream?.rangeManifestPath) {
      const rangeBytes = await readFile(packageFile(lock.upstream.rangeManifestPath));
      assert.equal(lock.upstream.rangeManifestSha256, sha256(rangeBytes));
    }
  }
});

test('generated files are deterministic and the ModelId union follows the curated spec', async () => {
  const typeFile = await readFile(packageFile('src/catalog/types.generated.d.ts'), 'utf8');
  const expectedIdType = "export type ModelId = '" + MODEL_IDS[0]! + "' | '" + MODEL_IDS[1]! + "';";
  assert.ok(typeFile.includes(expectedIdType));
  assert.ok(
    typeFile.includes("export type ModelIdTuple = readonly ['" + MODEL_IDS[0]! + "', '" + MODEL_IDS[1]! + "'];")
  );
  assert.ok(typeFile.includes("export type ModelAdapterId = 'kev06' | 'kanana';"));
  assert.equal(typeFile.includes("from '../index.js'"), false);
  execFileSync(process.execPath, ['--import', 'tsx', 'scripts/build-model-catalog.mts', '--check'], {
    cwd: packageRoot,
    stdio: 'pipe'
  });
});
