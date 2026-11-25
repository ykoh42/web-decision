import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile, readdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {APPROVED_MODEL_IDS} from '../../../../packages/web-decision/src/catalog/public-catalog.generated.ts';
import {build} from 'vite';
import {verifyArchiveInstall} from '../../../release/verify-release-evidence.ts';

const reportPath = resolve(process.argv[2]!);
const report = JSON.parse(await readFile(reportPath, 'utf8'));
const archive = await verifyArchiveInstall(report.archive.archivePath, report.archive.installedRoot);
assert.deepEqual(archive, report.archive);
assert.equal(report.releaseApproved, false);
const manifest = await readFile(join(archive.installedRoot, 'dist/gemma4-v1/manifest.json'));
assert.equal(createHash('sha256').update(manifest).digest('hex'), report.manifestSha256);
assert.deepEqual(JSON.parse(manifest.toString('utf8')).files, report.assets);
const metadata = JSON.parse(await readFile(join(archive.installedRoot, 'package.json'), 'utf8'));
assert.equal(metadata.private, true);
assert.equal(metadata.publishConfig, undefined);

const consumer = dirname(dirname(archive.installedRoot));
const entry = join(consumer, 'curated-consumer.mjs');
await writeFile(
  entry,
  "export {loadDecisionModel, listModels, describeModel, WebDecisionError} from 'web-decision';\n"
);
const {loadDecisionModel, listModels, describeModel, WebDecisionError} = await import(pathToFileURL(entry).href);
const modelId = report.modelId;
assert.ok(!(APPROVED_MODEL_IDS as readonly string[]).includes(modelId));
assert.deepEqual(
  listModels().map((item: {modelId: string}) => item.modelId),
  [...APPROVED_MODEL_IDS, modelId]
);
const model = describeModel(modelId);
assert.equal(model.verification.currentBundleGpu, 'pending');
assert.equal(model.assets.manifestSha256, report.manifestSha256);
assert.deepEqual(model.assets.requiredFiles, ['manifest.json', ...Object.keys(report.assets)]);
assert.equal(model.assets.directArtifactRoot, undefined);
assert.equal(model.adapter, undefined);
assert.equal(model.features.imageChoice.status, 'not-implemented');
assert.ok(Object.isFrozen(model.features.textChoice.limits[0]));
assert.equal(model.features.textChoice.limits[0].defaultMax, 4_096);
assert.equal(model.features.textChoice.limits[0].max, 131_072);
assert.throws(() => describeModel('unknown'), {code: 'INVALID_INPUT'});

const requests: string[] = [];
const priorFetch = globalThis.fetch;
const cacheDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'caches');
const cached = new Map<string, Response>();
let cacheOpens = 0;
const root = pathToFileURL(join(archive.installedRoot, 'dist/gemma4-v1/')).href;
let bad = false;
globalThis.fetch = async (url, options) => {
  const path = String(url);
  assert.ok(path.startsWith(root), path);
  options?.signal?.throwIfAborted();
  const name = path.slice(root.length);
  assert.ok(Object.hasOwn(report.assets, name));
  requests.push(name);
  const bytes = await readFile(new URL(path));
  if (bad && name === 'graph.wasm') bytes[0] = bytes[0]! ^ 1;
  return new Response(new Uint8Array(bytes), {headers: {'Content-Length': String(bytes.length)}});
};
Object.defineProperty(globalThis, 'caches', {
  configurable: true,
  value: {
    async open() {
      cacheOpens++;
      return {
        async match(key: string) {
          return cached.get(key)?.clone();
        },
        async put(key: string, response: Response) {
          cached.set(key, response);
        }
      };
    }
  }
});

try {
  await assert.rejects(loadDecisionModel({modelId: 'unknown'}), {code: 'INVALID_INPUT'});
  await assert.rejects(loadDecisionModel({modelId, assetBase: 'https://example.org/'}), {code: 'INVALID_INPUT'});
  await assert.rejects(loadDecisionModel({modelId, maxBatch: 0}), {code: 'INVALID_INPUT'});
  await assert.rejects(loadDecisionModel({modelId, cache: false, offline: true}), {code: 'INVALID_INPUT'});
  const cancelled = new Error('cancelled before admission');
  await assert.rejects(loadDecisionModel({modelId, signal: AbortSignal.abort(cancelled)}), {
    message: cancelled.message
  });
  assert.deepEqual(requests, []);

  await assert.rejects(loadDecisionModel({modelId}), (error: unknown) => {
    assert.ok(error instanceof WebDecisionError);
    assert.equal((error as {code?: string}).code, 'DEVICE_RESOURCE_LIMIT');
    return true;
  });
  assert.equal(cacheOpens, 1);
  const names = Object.keys(report.assets).sort();
  assert.deepEqual(requests.splice(0).sort(), names);
  assert.equal(cached.size, names.length);

  const progress: {loadedBytes: number; totalBytes: number; fraction: number; phase: string}[] = [];
  await assert.rejects(
    loadDecisionModel({
      modelId,
      maxLength: 131_072,
      maxBatch: 1,
      maxBatchTokens: 8,
      usePrefixCache: false,
      onProgress: (item: (typeof progress)[number]) => progress.push(item)
    }),
    {code: 'DEVICE_RESOURCE_LIMIT'}
  );
  assert.deepEqual(requests, []);
  const totalBytes = Object.values<{bytes: number}>(report.assets).reduce((sum, item) => sum + item.bytes, 0);
  assert.equal(progress.length, names.length);
  assert.ok(progress.every((item, i) => i === 0 || item.loadedBytes > progress[i - 1]!.loadedBytes));
  assert.deepEqual(progress.at(-1), {loadedBytes: totalBytes, totalBytes, fraction: 1, phase: 'bundled-assets'});

  await assert.rejects(loadDecisionModel({modelId, offline: true}), /offline weights are incomplete/i);
  assert.deepEqual(requests, []);
  bad = true;
  await assert.rejects(loadDecisionModel({modelId, cache: false}), /integrity/i);
} finally {
  globalThis.fetch = priorFetch;
  if (cacheDescriptor) Object.defineProperty(globalThis, 'caches', cacheDescriptor);
  else Reflect.deleteProperty(globalThis, 'caches');
}

const consumerTypePath = join(consumer, 'curated-consumer.mts');
await writeFile(
  consumerTypePath,
  `import {describeModel, listModels, loadDecisionModel, type ModelId} from 'web-decision';
const v1: ModelId = 'strands-decider-e2b-gemma4-v1-2610';
const model = await loadDecisionModel({modelId: v1, maxBatch: 1, maxBatchTokens: 8, usePrefixCache: false});
const id: typeof v1 = model.modelId;
const result = await model.systemOne({state: 'test', questions: {choice: {instructions: 'Pick one.', criteria: {A: null, B: null}}}});
const resultId: typeof v1 = result.model;
const description = describeModel(resultId);
const catalogIds: readonly ModelId[] = listModels().map(item => item.modelId);
await loadDecisionModel({modelId: 'strands-decider-e2b-gemma4-v2-2611', maxLength: 8192});
// @ts-expect-error callers cannot redirect pinned assets
await loadDecisionModel({modelId: id, assetBase: 'https://example.org/'});
`
);
const require = createRequire(import.meta.url);
execFileSync(
  process.execPath,
  [
    join(dirname(require.resolve('typescript/package.json')), 'bin/tsc'),
    '--strict',
    '--exactOptionalPropertyTypes',
    '--target',
    'es2022',
    '--module',
    'nodenext',
    '--noEmit',
    consumerTypePath
  ],
  {cwd: consumer, stdio: 'inherit'}
);

await writeFile(join(consumer, 'index.html'), '<script type="module" src="/app.mjs"></script>\n');
await writeFile(
  join(consumer, 'app.mjs'),
  `import {loadDecisionModel} from 'web-decision';
globalThis.startCandidate = () => loadDecisionModel({modelId: ${JSON.stringify(modelId)}});
`
);
await build({configFile: false, root: consumer, base: '/nested/', logLevel: 'error'});
const browserAssets = Object.create(null) as Record<string, string>;
for (const name of await readdir(join(consumer, 'dist/assets'))) {
  if (!name.endsWith('.json') && !name.endsWith('.wasm')) continue;
  browserAssets[
    createHash('sha256')
      .update(await readFile(join(consumer, 'dist/assets', name)))
      .digest('hex')
  ] = name;
}
for (const [name, {sha256}] of Object.entries<{sha256: string}>(report.assets))
  assert.ok(browserAssets[sha256], `Consumer build lost the exact ${name} asset.`);

console.log(
  JSON.stringify({
    archiveSha256: archive.archiveSha256,
    catalogIds: listModels().map((item: {modelId: string}) => item.modelId),
    manifestSha256: report.manifestSha256,
    pinnedRuntimeAssets: 5,
    modelIdOnly: true,
    defaultCacheReuse: true,
    incompleteOfflineRejectedWithZeroFetches: true,
    wrongModelRejected: true,
    tamperedGraphRejected: true,
    installedTypes: true,
    browserConsumerBuild: true,
    noGpuReported: true,
    gpuExecuted: false
  })
);
