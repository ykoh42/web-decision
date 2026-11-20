import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile, realpath} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import artifactLock from '../../packages/web-decision/src/catalog/artifacts.lock.json' with {type: 'json'};
import modelSpec from '../../packages/web-decision/src/catalog/model-spec.json' with {type: 'json'};
import {checkConsumerTypes} from '../ci/check-types.mts';
import {findPackedContentIssues} from '../ci/packed-content.ts';
import {
  loadInstalledCatalog,
  loadPublicCatalog,
  readArchiveModelManifests,
  verifyArchiveInstall
} from './verify-release-evidence.ts';

const modelIds = ['kanana-2-v-decide-3b-2611', 'strands-decider-2b-qwen3.5-v1-2610'] as const;
type ModelRequirement = Parameters<typeof readArchiveModelManifests>[1][number];
const packageRoot = fileURLToPath(new URL('../../packages/web-decision', import.meta.url));
const licensePins = {
  'QWEN-APACHE-LICENSE.txt': '50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32',
  'STRANDS-QWEN35-LICENSE.md': 'c0d777a0fed01cceb107d91f1aef96d51737be04f2ed75b1b933286a170d7411'
} as const;

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function getModelRequirement(modelId: (typeof modelIds)[number]): ModelRequirement {
  const model = modelSpec.models.find((entry) => entry.modelId === modelId);
  assert.ok(model);
  const {manifest, bundledRuntimeFiles} = artifactLock.byModelId[modelId];
  return {
    modelId,
    backend: {
      name: 'tvm-webgpu',
      requiredFeatures: model.description.backend.requiredFeatures,
      requiredLimits: model.description.backend.requiredLimits
    },
    requiredFiles: ['manifest.json', ...Object.keys(manifest.files)].sort(),
    manifestPath: manifest.sourcePath,
    manifestSha256: manifest.canonicalSha256,
    manifestFiles: manifest.files,
    runtimeAssets: {mode: 'bundled'},
    bundledRuntimeFiles
  };
}

async function verifyCanonicalAssets(installedRoot: string, archiveFiles: readonly string[]): Promise<void> {
  const expectedPaths: string[] = [];
  for (const {manifest, bundledRuntimeFiles} of Object.values(artifactLock.byModelId)) {
    expectedPaths.push(manifest.sourcePath);
    const installed = await readFile(join(installedRoot, manifest.sourcePath));
    assert.equal(sha256(installed), manifest.rawSha256, manifest.sourcePath);
    for (const {sourcePath, bytes, sha256: digest} of bundledRuntimeFiles) {
      expectedPaths.push(sourcePath);
      const contents = await readFile(join(installedRoot, sourcePath));
      assert.equal(contents.byteLength, bytes, sourcePath);
      assert.equal(sha256(contents), digest, sourcePath);
    }
  }
  assert.deepEqual(archiveFiles.filter((path) => path.startsWith('assets/')).sort(), expectedPaths.sort());
  for (const [name, digest] of Object.entries(licensePins)) {
    assert.equal(sha256(await readFile(join(installedRoot, name))), digest, name);
  }
  const notice = await readFile(join(installedRoot, 'STRANDS-QWEN35-NOTICE'), 'utf8');
  assert.match(notice, /Copyright 2026 Alibaba Cloud/u);
  assert.match(notice, /web-decision adapted the pinned model/u);
}

async function verifyPackedContents(installedRoot: string, archiveFiles: readonly string[]): Promise<void> {
  const metadata = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8')) as {files: string[]};
  const dist = archiveFiles.filter((path) => path.startsWith('dist/'));
  assert.deepEqual(archiveFiles, [...metadata.files.filter((path) => path !== 'dist'), 'package.json', ...dist].sort());
  for (const path of archiveFiles) {
    const contents = await readFile(join(installedRoot, path));
    assert.deepEqual(findPackedContentIssues(contents), [], `Potential private content in ${path}`);
  }
}

async function verifyPublicMetadata(entryPath: string): Promise<void> {
  const library = await import(pathToFileURL(entryPath).href);
  assert.deepEqual(
    library.listModels().map(({modelId}: {modelId: string}) => modelId),
    modelIds
  );
  assert.throws(() => library.describeModel('kev-0.6b-q4f16'), {code: 'INVALID_INPUT'});
  for (const modelId of modelIds) {
    assert.equal(library.describeModel(modelId).modelId, modelId);
    await assert.rejects(library.loadDecisionModel({modelId, assetsUrl: '/mirror'}), {code: 'INVALID_INPUT'});
    const abort = new Error('Stop before model loading');
    await assert.rejects(library.loadDecisionModel({modelId, signal: AbortSignal.abort(abort)}), abort);
  }
  assert.equal(await library.clearModelCache(), false);
}

const [archivePath, ...consumers] = process.argv.slice(2);
assert.ok(archivePath && consumers.length === 2, 'Expected archive, npm consumer, and pnpm consumer paths.');
const models = modelIds.map(getModelRequirement);
const expectedCatalog = models.map(({modelId, backend, manifestSha256, requiredFiles}) => ({
  modelId,
  backend,
  manifestSha256,
  requiredFiles
}));
assert.deepEqual(await loadPublicCatalog(packageRoot, 'dist/index.js'), expectedCatalog);
for (const consumer of consumers) {
  const installedRoot = await realpath(join(consumer, 'node_modules/web-decision'));
  const archive = await verifyArchiveInstall(archivePath, installedRoot);
  assert.equal(archive.packageVersion, '0.5.0');
  assert.deepEqual(await loadInstalledCatalog(installedRoot), expectedCatalog);
  const manifests = await readArchiveModelManifests(archive, models);
  assert.deepEqual(
    [...manifests.values()].map(({path}) => path),
    models.map(({manifestPath}) => manifestPath)
  );
  await verifyCanonicalAssets(installedRoot, archive.archiveFiles);
  await verifyPackedContents(installedRoot, archive.archiveFiles);
  await verifyPublicMetadata(archive.bundlePath);
  checkConsumerTypes(consumer);
  console.log(
    JSON.stringify({
      consumer,
      archiveSha256: archive.archiveSha256,
      packageTreeSha256: archive.packageTreeSha256,
      bundleSha256: archive.bundleSha256,
      files: archive.archiveFiles.length
    })
  );
}
