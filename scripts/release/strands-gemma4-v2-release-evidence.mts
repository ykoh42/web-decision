import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import artifactLock from '../../packages/web-decision/src/catalog/artifacts.lock.json' with {type: 'json'};
import type {ModelArtifactLock} from '../../packages/web-decision/src/catalog/types.generated.js';
import {assertEvidenceArchive, cleanSourceCommit, record, sha256} from './curated-gpu-evidence.mjs';
import {verifyTvmBuildEvidence} from './tvm-build-evidence.ts';
import {collectReleaseEvidenceInputs, type ArchiveInfo} from './verify-release-evidence.ts';
import {gitEnvironmentFor} from '../git-worktree-environment.mjs';
import {
  GEMMA_V2_CPU_REFERENCE_PATH,
  GEMMA_V2_CPU_REFERENCE_SHA256,
  validateGemmaV2CpuReferences
} from './strands-gemma4-v2-source-reference.mjs';

import {
  gemmaV2ArchiveTokenizerSha256,
  verifyGemmaV2ScorerResources
} from '../../tests/browser/gemma-v2-exact-archive-page.mjs';
import {findGemmaV2ScorerPin, gemmaV2RawProbabilityEncoding} from '../../tests/browser/gemma-v2-raw-capture.mjs';
import {
  GEMMA_V2_OFFICIAL_SUBSET_PATH,
  GEMMA_V2_OFFICIAL_SUBSET_SHA256,
  inspectGemma4V2Conformance,
  readPinnedGemma4V2Subset
} from './strands-gemma4-v2-conformance.mjs';
import {readGemmaV2PlaywrightRun} from './strands-gemma4-v2-release-browser.mjs';

const workspaceRoot = fileURLToPath(new URL('../../', import.meta.url));
const packageRoot = join(workspaceRoot, 'packages/web-decision');
const modelId = 'strands-decider-e2b-gemma4-v2-2611';
const requiredCases = ['text-1024', 'text-32768', 'noul', 'score', 'image-single', 'image-multiple'] as const;
type NumericDifference = Readonly<{path: string; actual: number; expected: number; absolute: number}>;

function compareResult(actual: unknown, expected: unknown, label: string, differences: NumericDifference[]): void {
  if (typeof expected === 'number') {
    assert.ok(typeof actual === 'number' && Number.isFinite(actual), `${label} is not a finite number.`);
    assert.ok(Number.isFinite(expected), `${label} reference is not finite.`);
    differences.push({path: label, actual, expected, absolute: Math.abs(actual - expected)});
    return;
  }
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual) && actual.length === expected.length, `${label} differs in length.`);
    expected.forEach((item, index) => compareResult(actual[index], item, `${label}[${index}]`, differences));
    return;
  }
  if (expected && typeof expected === 'object') {
    const a = record(actual, `${label} is not an object.`);
    const e = expected as Record<string, unknown>;
    assert.deepEqual(Object.keys(a).sort(), Object.keys(e).sort(), `${label} keys differ.`);
    for (const key of Object.keys(e)) compareResult(a[key], e[key], `${label}.${key}`, differences);
    return;
  }
  assert.equal(actual, expected, `${label} differs.`);
}

function compareCases(
  cases: Record<string, unknown>,
  references: Record<string, unknown>,
  differences: NumericDifference[]
): void {
  for (const id of requiredCases) {
    const sample = record(cases[id], `Missing ${id} result.`);
    const source = record(references[id], `Missing ${id} reference.`);
    assert.equal(sample.requestSha256, source.requestSha256, `${id} request differs from reference.`);
    assert.match(String(source.requestSha256), /^[a-f0-9]{64}$/u);
    assert.equal(sample.inputTokens, source.inputTokens, `${id} token count differs.`);
    assert.equal(sample.outputTokens, source.outputTokens, `${id} question count differs.`);
    assert.equal(sample.imageCount, source.imageCount, `${id} image count differs.`);
    if (id === 'text-1024') assert.equal(sample.inputTokens, 1024);
    if (id === 'text-32768') assert.equal(sample.inputTokens, 32768);
    if (id === 'image-single') assert.equal(sample.imageCount, 1);
    if (id === 'image-multiple') assert.ok(typeof sample.imageCount === 'number' && sample.imageCount >= 2);
    const expectedType = id === 'noul' ? 'noul' : id === 'score' ? 'score' : 'choice';
    assert.equal(record(source.result, `Missing ${id} reference result.`).type, expectedType);
    compareResult(sample.result, source.result, id, differences);
  }
}

/** Validate records only; production also requires the committed references and installed archive. */
export function assertGemma4V2ReleaseRecord(
  report: unknown,
  archive: ArchiveInfo,
  reference: unknown
): NumericDifference[] {
  const differences: NumericDifference[] = [];
  const data = record(report, 'Invalid Gemma v2 GPU evidence.');
  const expected = record(reference, 'Invalid Gemma v2 source reference.');
  assert.equal(expected.status, 'pinned-upstream-cpu', 'Gemma v2 requires pinned upstream CPU cases.');
  assert.equal(data.schemaVersion, 1);
  assert.equal(data.modelId, modelId);
  assert.equal(data.source, 'pinned-upstream', 'Gemma v2 must load the actual pinned upstream source.');
  assertEvidenceArchive(data.archive, archive);
  const browser = record(data.browser, 'Missing browser evidence.');
  assert.equal(browser.name, 'chromium');
  assert.ok(typeof browser.version === 'string' && browser.version.length > 0, 'Browser version is required.');
  assert.equal(record(data.hardware, 'Missing hardware evidence.').status, 'Hardware accelerated');
  const adapter = record(data.adapter, 'Missing WebGPU adapter.');
  assert.equal(adapter.isFallbackAdapter, false, 'A non-fallback WebGPU adapter is required.');
  const cases = record(data.cases, 'Missing Gemma v2 installed-package cases.');
  const references = record(expected.cases, 'Missing Gemma v2 source cases.');
  assert.deepEqual(Object.keys(cases).sort(), [...requiredCases].sort());
  assert.deepEqual(Object.keys(references).sort(), [...requiredCases].sort());
  compareCases(cases, references, differences);
  const offline = record(data.offline, 'Missing offline evidence.');
  assert.equal(offline.cacheStatus, 'complete');
  assert.deepEqual(offline.blockedRequests, [], 'Offline inference attempted a network request.');
  assert.deepEqual(offline.result, record(cases['image-single'], 'Missing image result.').result);
  assert.equal(data.cleanup, 'passed', 'GPU and session cleanup evidence is required.');
  return differences;
}

function assertRuntimeEvidence(
  data: Record<string, unknown>,
  sourceCommit: string,
  artifact: ModelArtifactLock,
  catalogSha256: string
): void {
  assert.equal(data.sourceCommit, sourceCommit, 'Gemma v2 source commit differs.');
  assert.equal(data.sourceRevision, 'c3c845782095b31d438963ac2afcdfe41b49f757');
  assert.equal(data.checkpointRevision, 'a5f32ea1bffe2e91c25c0c641d444e8115e28212');
  assert.equal(data.catalogSha256, catalogSha256, 'Gemma v2 catalog differs.');
  assert.equal(data.bundledManifestSha256, artifact.manifest.rawSha256, 'Gemma v2 manifest differs.');
  const graph = artifact.manifest.files['graph.wasm'];
  assert.ok(graph, 'Missing locked Gemma graph.');
  assert.equal(data.graphSha256, graph.sha256, 'Gemma v2 graph differs.');
  assert.deepEqual(data.assertions, {checkpointRangeObserved: true, installedGraphResponseSha256: graph.sha256});
  assert.equal(data.cpuFixtureSha256, GEMMA_V2_CPU_REFERENCE_SHA256, 'Gemma v2 CPU reference differs.');
  const hardware = record(data.hardware, 'Missing Chromium hardware evidence.');
  assert.ok(typeof hardware.renderer === 'string' && hardware.renderer.trim(), 'Missing Chromium hardware renderer.');
  const device = record(record(data.host, 'Missing physical host evidence.').device, 'Missing physical host device.');
  assert.equal(device.isVirtual, false, 'A physical GPU host is required.');
  assert.ok(typeof device.model === 'string' && device.model.trim(), 'Missing physical host model.');
  assert.equal(data.recovery, 'passed', 'Gemma image recovery evidence is required.');
}

export async function assertGemmaV2RawCapture(data: Record<string, unknown>, archive: ArchiveInfo, questions: number) {
  const conformance = record(data.conformance, 'Missing mandatory Gemma v2 raw conformance evidence.');
  assert.equal(
    conformance.probabilityEncoding,
    gemmaV2RawProbabilityEncoding,
    'Gemma v2 release evidence requires unrounded Chromium scorer probabilities.'
  );
  const capture = record(data.rawCapture, 'Missing exact Gemma scorer evidence.');
  const scorer = await findGemmaV2ScorerPin(archive.installedRoot, archive.archiveFiles);
  const {breakpointColumn, served} = capture;
  const resources = record(served, 'Missing served Gemma archive resource hashes.');
  await verifyGemmaV2ScorerResources(archive.installedRoot, archive.archiveFiles, resources, scorer.path);
  assert.ok(
    typeof breakpointColumn === 'number' &&
      Number.isSafeInteger(breakpointColumn) &&
      breakpointColumn >= scorer.columnNumber &&
      breakpointColumn < scorer.conditionEndColumn,
    'Chromium must break inside the exact installed Gemma scorer condition.'
  );
  assert.deepEqual(
    data.rawCapture,
    {
      ...scorer,
      breakpointColumn,
      frames: questions,
      tokenizerSha256: gemmaV2ArchiveTokenizerSha256,
      tokenizerServed: true,
      served: resources
    },
    'The raw Gemma capture differs from the scorer in the exact installed archive.'
  );
}

async function readCommittedBytes(path: string): Promise<Buffer> {
  const bytes = await readFile(join(workspaceRoot, path));
  const committed = execFileSync('git', ['show', `HEAD:${path}`], {
    cwd: workspaceRoot,
    env: gitEnvironmentFor(workspaceRoot)
  });
  assert.deepEqual(bytes, committed, `Gemma v2 fixture must match committed bytes: ${path}.`);
  return bytes;
}

/** Verify Gemma's evidence for one exact combined archive. Other models have separate gates. */
export async function verifyGemma4V2ReleaseEvidence(options: {
  archivePath: string;
  installedPackagePath: string;
  reportPath: string;
  buildEvidencePath: string;
  goldenPath: string;
}) {
  const sourceCommit = cleanSourceCommit(workspaceRoot);
  const {archive, models, catalogSha256} = await collectReleaseEvidenceInputs(options);
  const metadata = record(
    JSON.parse(await readFile(join(archive.installedRoot, 'package.json'), 'utf8')),
    'Invalid package.'
  );
  assert.notEqual(metadata.private, true, 'Private candidate archives are not release evidence.');
  const model = models.find((item) => item.modelId === modelId);
  assert.ok(model, 'The installed public package must expose the Gemma v2 model.');
  const lock = (artifactLock.byModelId as Record<string, ModelArtifactLock>)[modelId];
  assert.ok(lock, 'Gemma v2 release artifact lock is unavailable.');
  const [cpu, subset, golden, playwright] = await Promise.all([
    readCommittedBytes(GEMMA_V2_CPU_REFERENCE_PATH),
    readCommittedBytes(GEMMA_V2_OFFICIAL_SUBSET_PATH),
    readFile(options.goldenPath),
    readFile(options.reportPath)
  ]);
  const cases = validateGemmaV2CpuReferences(cpu);
  readPinnedGemma4V2Subset(subset);
  const {runBytes, vectors} = readGemmaV2PlaywrightRun(JSON.parse(playwright.toString('utf8')), workspaceRoot);
  const data = record(JSON.parse(runBytes.toString('utf8')), 'Invalid Gemma v2 browser-run attachment.');
  assertRuntimeEvidence(data, sourceCommit, lock, catalogSha256);
  const numericDifferences = assertGemma4V2ReleaseRecord(data, archive, {status: 'pinned-upstream-cpu', cases});
  const metrics = inspectGemma4V2Conformance(
    golden,
    vectors,
    {bytes: subset, sha256: GEMMA_V2_OFFICIAL_SUBSET_SHA256},
    'raw-fp32'
  );
  await assertGemmaV2RawCapture(data, archive, metrics.questions);
  assert.deepEqual(data.conformance, {probabilityEncoding: gemmaV2RawProbabilityEncoding, ...metrics});
  assert.ok(
    metrics.meetsUpstreamMetricThresholds,
    `Gemma v2 code probabilities fail upstream thresholds: ${JSON.stringify(metrics)}`
  );

  const {manifest, bundledRuntimeFiles} = lock;
  const sourceArtifacts = Object.fromEntries(
    bundledRuntimeFiles.map(({manifestName, sourcePath}) => [manifestName, join(packageRoot, sourcePath)])
  );
  const build = await verifyTvmBuildEvidence(
    options.buildEvidencePath,
    [
      {
        modelId,
        manifestFiles: manifest.files,
        tvmVersion: '0.27.0.post1',
        tvmSourceRevision: '7b5acb32bd271342a24130f4ef439ff6c2e52c6c',
        sourceArtifacts,
        reportRequirements: {
          'graph.wasm': {
            buildBackend: 'native-tvm',
            repository: 'StrandsAgents/strands-decider-E2B-gemma4-v2-2611',
            modelRevision: 'a5f32ea1bffe2e91c25c0c641d444e8115e28212',
            onnxImportOrExportUsed: false
          }
        }
      }
    ],
    [packageRoot, archive.installedRoot]
  );
  return {
    status: 'verified-gemma-v2-evidence',
    modelId,
    sourceCommit,
    archiveSha256: archive.archiveSha256,
    catalogSha256,
    playwrightSha256: sha256(playwright),
    probabilityEncoding: gemmaV2RawProbabilityEncoding,
    metrics,
    numericDifferences,
    build
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [archivePath, installedPackagePath, reportPath, buildEvidencePath, goldenPath] = process.argv.slice(2);
  assert.ok(
    archivePath && installedPackagePath && reportPath && buildEvidencePath && goldenPath,
    'Expected archive, installed package, Playwright JSON report, TVM build evidence, and official golden JSONL.'
  );
  console.log(
    JSON.stringify(
      await verifyGemma4V2ReleaseEvidence({
        archivePath,
        installedPackagePath,
        reportPath,
        buildEvidencePath,
        goldenPath
      }),
      null,
      2
    )
  );
}
