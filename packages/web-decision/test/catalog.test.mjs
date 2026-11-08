import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {MODEL_CATALOG, MODEL_CATALOG_BY_ID, MODEL_IDS} from '../src/catalog/index.generated.js';
import {getModelRegistration} from '../src/model-registry.js';
import {createModelArtifactLock, validateSpec} from '../scripts/build-model-catalog.mjs';

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageFile = (relativePath) => path.join(packageRoot, relativePath);

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonical(value[key])])
  );
}

test('catalog exposes one ID-only TVM descriptor with the kev06 adapter', () => {
  assert.deepEqual(MODEL_IDS, ['kev-0.6b-q4f16']);
  assert.equal(MODEL_CATALOG.length, 1);

  const entry = MODEL_CATALOG_BY_ID[MODEL_IDS[0]];
  assert.equal(entry, MODEL_CATALOG[0]);
  assert.equal(entry.modelId, MODEL_IDS[0]);
  assert.equal(entry.description.adapter, 'kev06');
  assert.equal(entry.description.backend.name, 'tvm-webgpu');
  assert.deepEqual(entry.description.backend.requiredFeatures, ['shader-f16']);
  assert.equal(entry.description.features.textChoice.status, 'implemented');
  assert.equal(entry.description.verification.currentBundleGpu, 'not-revalidated');
  assert.equal(Object.hasOwn(entry, 'model'), false);
  assert.equal(Object.hasOwn(entry.description, 'model'), false);
  assert.equal(Object.hasOwn(entry.description.assets, 'assetsUrl'), false);
  assert.equal(Object.hasOwn(entry.description.assets, 'manifestSha256'), true);
});

test('catalog description retains the existing Kev contract and narrows bundle verification', () => {
  const entry = MODEL_CATALOG[0];
  const {adapter, ...description} = entry.description;
  const legacy = getModelRegistration(entry.modelId).description;
  const {model, ...expected} = structuredClone(legacy);
  assert.equal(model, entry.modelId);
  expected.verification.currentBundleGpu = 'not-revalidated';
  expected.verification.priorEvidence = entry.description.verification.priorEvidence;
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
});

test('artifact lock hashes the manifest, bundled files, and upstream map', async () => {
  const entry = MODEL_CATALOG[0];
  const lock = entry.artifacts;
  const lockFile = JSON.parse(await readFile(packageFile('src/catalog/artifacts.lock.json'), 'utf8'));
  const manifestBytes = await readFile(packageFile(lock.manifest.sourcePath));
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const canonicalManifest = Buffer.from(JSON.stringify(canonical(manifest)) + '\n');

  assert.equal(lock.manifest.rawSha256, sha256(manifestBytes));
  assert.equal(lock.manifest.canonicalSha256, sha256(canonicalManifest));
  assert.equal(entry.description.assets.manifestSha256, lock.manifest.canonicalSha256);
  assert.deepEqual(entry.description.assets.requiredFiles, ['manifest.json', ...Object.keys(manifest.files)]);
  assert.deepEqual(lock.manifest.files, manifest.files);
  assert.equal(lock.manifest.experimental, true);
  assert.deepEqual(Object.keys(lockFile.byModelId), MODEL_IDS);
  assert.deepEqual(lockFile.byModelId[entry.modelId], lock);
  assert.equal(Object.hasOwn(lock, 'priorHardwareEvidence'), false);

  for (const artifact of lock.bundledRuntimeFiles) {
    const bytes = await readFile(packageFile(artifact.sourcePath));
    assert.equal(artifact.bytes, bytes.byteLength);
    assert.equal(artifact.sha256, sha256(bytes));
    assert.equal(artifact.sha256, lock.manifest.files[artifact.manifestName].sha256);
  }

  const mapBytes = await readFile(packageFile(lock.upstream.shardMapPath));
  assert.equal(lock.upstream.shardMapSha256, sha256(mapBytes));
  assert.equal(entry.description.verification.currentBundleGpu, 'not-revalidated');
});

test('generated files are deterministic and the ModelId union follows the curated spec', async () => {
  const typeFile = await readFile(packageFile('src/catalog/types.generated.d.ts'), 'utf8');
  const expectedIdType = "export type ModelId = '" + MODEL_IDS[0] + "';";
  assert.ok(typeFile.includes(expectedIdType));
  assert.ok(typeFile.includes("export type ModelAdapterId = 'kev06';"));
  assert.equal(typeFile.includes("from '../index.js'"), false);
  execFileSync(process.execPath, ['scripts/build-model-catalog.mjs', '--check'], {
    cwd: packageRoot,
    stdio: 'pipe'
  });
});
