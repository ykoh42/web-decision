import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import artifactLock from '../../packages/web-decision/src/catalog/artifacts.lock.json' with {type: 'json'};
import conversion from '../../packages/web-decision/src/catalog/provenance/mapika2b-conversion.json' with {type: 'json'};
import {
  mapikaReferenceMismatches,
  mapikaReleaseCases,
  type MapikaAnswerReference,
  type MapikaReference
} from '../../tests/browser/mapika2b-evidence.mts';
import {assertEvidenceArchive, cleanSourceCommit, record, sha256} from './curated-gpu-evidence.mts';
import {verifyTvmBuildEvidence} from './tvm-build-evidence.ts';
import {collectReleaseEvidenceInputs, type ArchiveInfo} from './verify-release-evidence.ts';
import {mapikaCurrentPin, verifyMapikaCurrentSource} from './mapika-current-evidence.mts';

const workspaceRoot = fileURLToPath(new URL('../../', import.meta.url));
const packageRoot = join(workspaceRoot, 'packages/web-decision');
const modelId = 'mapika-decider-2b-v12';
const referencePins = {
  preparedSha256: 'fef0db786ff7dbd327f01cb2e7ff84448aa24231a34b2d43493d0350987734b7',
  answersSha256: 'd42c03efc1d317032868df8e6fbb88c5fe87fa1a5fb0863dd6bbadc4bdb53c7a'
};
const upstreams = artifactLock.byModelId[modelId].upstreams;

function checkpointBytes(role: 'text' | 'vision'): number {
  const pinned = upstreams.find((entry) => entry.role === role);
  assert.ok(pinned, `Unpinned Mapika ${role} checkpoint.`);
  return pinned.sourceBytes;
}

function assertAdapter(value: unknown, requiredLimits: Readonly<Record<string, number>>): void {
  const adapter = record(value, 'Mapika WebGPU adapter is missing.');
  assert.equal(adapter.isFallbackAdapter, false, 'Mapika release requires a hardware WebGPU adapter.');
  for (const [limit, minimum] of Object.entries(requiredLimits)) {
    assert.ok(
      typeof adapter[limit] === 'number' && adapter[limit] >= minimum,
      `Mapika adapter limit is too low: ${limit}`
    );
  }
}

function assertResults(value: unknown, source: MapikaReference<MapikaAnswerReference>): asserts value is unknown[] {
  assert.ok(Array.isArray(value) && value.length === mapikaReleaseCases.length, 'Mapika GPU case count differs.');
  for (const [index, id] of mapikaReleaseCases.entries()) {
    const expected = source.cases.find((item) => item.id === id);
    assert.ok(expected, `Missing pinned Mapika reference: ${id}`);
    assert.deepEqual(mapikaReferenceMismatches(value[index], expected.result), [], `Mapika result differs: ${id}`);
  }
}

function readRangeCounts(value: unknown, label: string) {
  const counts = record(value, `Missing Mapika range counts at ${label}.`);
  const number = (key: string) => {
    const item = counts[key];
    assert.ok(typeof item === 'number' && Number.isSafeInteger(item) && item >= 0, `Invalid ${label}.${key}.`);
    return item;
  };
  const textBytes = number('textBytes');
  const visionBytes = number('visionBytes');
  const requestedBytes = number('requestedBytes');
  assert.equal(requestedBytes, textBytes + visionBytes, `Mapika range bytes disagree at ${label}.`);
  return {
    textBytes,
    visionBytes,
    requestedBytes,
    textRanges: number('textRanges'),
    visionRanges: number('visionRanges')
  };
}

function assertProgressiveLoad(value: unknown): void {
  const steps = record(value, 'Missing progressive Mapika load evidence.');
  const loaded = readRangeCounts(steps.afterLoad, 'load');
  const text = readRangeCounts(steps.afterFirstText, 'first text');
  const image = readRangeCounts(steps.afterFirstImage, 'first image');
  const before = readRangeCounts(steps.beforePrepare, 'before offline preparation');
  const after = readRangeCounts(steps.afterPrepare, 'after offline preparation');
  assert.equal(steps.fullCheckpointBytes, checkpointBytes('text') + checkpointBytes('vision'));
  assert.deepEqual(steps.unboundedWeightRequests, [], 'Mapika requested an unbounded weight file.');
  assert.equal(steps.cacheStatusAfterLoad, 'incomplete');
  assert.equal(steps.cacheStatusBeforePrepare, 'incomplete');
  assert.ok(loaded.textRanges > 0 && loaded.textBytes > 0, 'Cold Mapika load must read text weights.');
  assert.equal(
    loaded.visionRanges + loaded.visionBytes + text.visionRanges + text.visionBytes,
    0,
    'Mapika requested vision weights before the first image.'
  );
  assert.ok(
    text.textBytes < checkpointBytes('text'),
    'Mapika fetched the full text checkpoint before the first result.'
  );
  assert.ok(
    image.visionRanges > 0 && image.visionBytes > 0,
    'First Mapika image did not exercise lazy vision loading.'
  );
  assert.ok(
    loaded.requestedBytes <= text.requestedBytes &&
      text.requestedBytes <= image.requestedBytes &&
      image.requestedBytes <= before.requestedBytes &&
      before.requestedBytes < after.requestedBytes,
    'Mapika request counts must show a separate offline preparation phase.'
  );
}

export function assertMapikaReleaseRecord(
  report: unknown,
  archive: ArchiveInfo,
  source: MapikaReference<MapikaAnswerReference>,
  requiredLimits: Readonly<Record<string, number>>
): void {
  const data = record(report, 'Invalid Mapika GPU evidence.');
  assert.equal(data.schemaVersion, 1);
  assert.equal(data.modelId, modelId);
  assert.equal(data.source, 'pinned-upstream', 'Mapika release must exercise the live pinned Hugging Face source.');
  assertEvidenceArchive(data.archive, archive);
  assert.deepEqual(data.references, referencePins);
  assert.deepEqual(data.cases, mapikaReleaseCases);
  assert.equal(record(data.browser, 'Missing Mapika browser evidence.').name, 'chromium');
  assert.equal(record(data.hardware, 'Missing Mapika hardware evidence.').status, 'Hardware accelerated');
  assertAdapter(data.adapter, requiredLimits);
  assertProgressiveLoad(data.progressive);
  assertResults(data.results, source);
  assert.equal(data.modelCacheStatus, 'complete');
  const offline = record(data.offline, 'Missing Mapika offline evidence.');
  assert.equal(offline.cacheStatus, 'complete');
  assert.deepEqual(offline.blockedRequests, [], 'Mapika requested model assets during offline inference.');
  assert.deepEqual(offline.results, data.results, 'Mapika cached results differ from online results.');
}

async function readReference(path: string, digest: string): Promise<Buffer> {
  const bytes = await readFile(join(workspaceRoot, path));
  assert.equal(sha256(bytes), digest, `Mapika source reference changed: ${path}`);
  return bytes;
}

async function assertConversionInputs(inputRoot = workspaceRoot, current = false): Promise<void> {
  for (const input of [...conversion.recipe, ...conversion.config]) {
    const bytes = await readFile(join(inputRoot, input.path));
    assert.equal(sha256(bytes), input.sha256, `Mapika conversion input changed: ${input.path}`);
  }
  for (const input of conversion.releaseEvidenceInputs) {
    const bytes = await readFile(join(inputRoot, input.path));
    let expected = input.sha256;
    if (current && input.path === 'tests/browser/vite.config.ts') {
      assert.equal(expected, mapikaCurrentPin.historicalViteSha256, 'Mapika historical Vite pin changed');
      expected = mapikaCurrentPin.testedViteSha256;
    }
    assert.equal(sha256(bytes), expected, `Mapika release evidence input changed: ${input.path}`);
  }
}

export async function verifyMapikaReleaseEvidence(options: {
  archivePath: string;
  installedPackagePath: string;
  reportPath: string;
  buildEvidencePath: string;
  currentTestedSource?: string;
}) {
  const testedRoot = options.currentTestedSource ? resolve(options.currentTestedSource) : workspaceRoot;
  const current = options.currentTestedSource ? await verifyMapikaCurrentSource(testedRoot, workspaceRoot) : undefined;
  const {archive, models} = await collectReleaseEvidenceInputs({
    ...options,
    workspaceRoot: testedRoot,
    sourcePackageRoot: join(testedRoot, 'packages/web-decision')
  });
  if (current)
    assert.equal(archive.archiveSha256, mapikaCurrentPin.archiveSha256, 'Unexpected current candidate archive');
  const installedMetadata = JSON.parse(await readFile(join(archive.installedRoot, 'package.json'), 'utf8')) as {
    private?: boolean;
  };
  assert.notEqual(installedMetadata.private, true, 'A private package cannot be used as Mapika release evidence.');
  const model = models.find((item) => item.modelId === modelId);
  assert.ok(model, 'The public package must expose the curated Mapika model.');
  await assertConversionInputs(testedRoot, !!current);
  await readReference('scripts/mapika/reference-fp32-prepared.json', referencePins.preparedSha256);
  const answers = await readReference('scripts/mapika/reference-fp32-answers.json', referencePins.answersSha256);
  const report = await readFile(options.reportPath);
  const source = JSON.parse(answers.toString('utf8')) as MapikaReference<MapikaAnswerReference>;
  const data = record(JSON.parse(report.toString('utf8')), 'Invalid Mapika GPU evidence.');
  assert.equal(
    data.sourceCommit,
    current?.testedCommit ?? cleanSourceCommit(workspaceRoot),
    'Mapika GPU evidence refers to a different source commit.'
  );
  assertMapikaReleaseRecord(data, archive, source, model.backend.requiredLimits);
  const build = await verifyTvmBuildEvidence(
    options.buildEvidencePath,
    [mapikaBuildModel()],
    [packageRoot, join(testedRoot, 'packages/web-decision'), archive.installedRoot]
  );
  return {
    modelId,
    archiveSha256: archive.archiveSha256,
    reportSha256: sha256(report),
    build,
    ...(current ? {current} : {})
  };
}

function mapikaBuildModel() {
  const {manifest, bundledRuntimeFiles} = artifactLock.byModelId[modelId];
  const {version, revision} = conversion.toolchain.tvm;
  return {
    modelId,
    manifestFiles: manifest.files,
    tvmVersion: version,
    tvmSourceRevision: revision,
    sourceArtifacts: Object.fromEntries(
      bundledRuntimeFiles.map(({manifestName, sourcePath}) => [manifestName, join(packageRoot, sourcePath)])
    ),
    reportRequirements: Object.fromEntries(
      Object.keys(manifest.files).map((name) => [
        name,
        {
          buildBackend: 'native-tvm',
          repository: 'Mapika/decider-2b',
          modelRevision: '7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3',
          onnxImportOrExportUsed: false
        }
      ])
    )
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [
    archivePath,
    installedPackagePath,
    reportPath,
    buildEvidencePath,
    currentOption,
    currentTestedSource,
    ...extra
  ] = process.argv.slice(2);
  assert.ok(
    archivePath && installedPackagePath && reportPath && buildEvidencePath,
    'Expected archive, installed package, GPU report and TVM build evidence.'
  );
  assert.ok(
    extra.length === 0 &&
      (currentOption === undefined || (currentOption === '--current-tested-source' && currentTestedSource)),
    'Invalid optional current source.'
  );
  const evidence = await verifyMapikaReleaseEvidence({
    archivePath,
    installedPackagePath,
    reportPath,
    buildEvidencePath,
    ...(currentTestedSource ? {currentTestedSource} : {})
  });
  console.log(JSON.stringify(evidence, null, 2));
}
