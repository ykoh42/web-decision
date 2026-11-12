import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {MODEL_CATALOG, MODEL_CATALOG_BY_ID, MODEL_IDS} from '../src/catalog/index.generated.ts';
import {createModelArtifactLock, validateSpec} from '../scripts/build-model-catalog.mts';

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

test('catalog exposes Kanana through an ID-only TVM descriptor', () => {
  assert.deepEqual(MODEL_IDS, ['kanana-2-v-decide-3b-2611']);
  assert.equal(MODEL_CATALOG.length, 1);

  const kanana = MODEL_CATALOG_BY_ID['kanana-2-v-decide-3b-2611'];
  assert.equal(kanana, MODEL_CATALOG[0]);
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
  assert.equal(Object.hasOwn(kanana, 'model'), false);
  assert.equal(Object.hasOwn(kanana.description, 'model'), false);
  assert.equal(Object.hasOwn(kanana.description.assets, 'assetsUrl'), false);
  assert.equal(Object.hasOwn(kanana.description.assets, 'manifestSha256'), true);
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

    if (lock.upstream.shardMapPath) {
      const mapBytes = await readFile(packageFile(lock.upstream.shardMapPath));
      assert.equal(lock.upstream.shardMapSha256, sha256(mapBytes));
    }
    if (lock.upstream.rangeManifestPath) {
      const rangeBytes = await readFile(packageFile(lock.upstream.rangeManifestPath));
      assert.equal(lock.upstream.rangeManifestSha256, sha256(rangeBytes));
    }
  }
});

test('generated files are deterministic and the ModelId union follows the curated spec', async () => {
  const typeFile = await readFile(packageFile('src/catalog/types.generated.d.ts'), 'utf8');
  const expectedIds = MODEL_IDS.map((modelId) => "'" + modelId + "'").join(' | ');
  const expectedTuple = MODEL_IDS.map((modelId) => "'" + modelId + "'").join(', ');
  const expectedAdapters = [...new Set(MODEL_CATALOG.map(({description}) => description.adapter))]
    .map((adapter) => "'" + adapter + "'")
    .join(' | ');
  assert.ok(typeFile.includes('export type ModelId = ' + expectedIds + ';'));
  assert.ok(typeFile.includes('export type ModelIdTuple = readonly [' + expectedTuple + '];'));
  assert.ok(typeFile.includes('export type ModelAdapterId = ' + expectedAdapters + ';'));
  assert.equal(typeFile.includes("from '../index.js'"), false);
  execFileSync(process.execPath, ['--import', 'tsx', 'scripts/build-model-catalog.mts', '--check'], {
    cwd: packageRoot,
    stdio: 'pipe'
  });
});
