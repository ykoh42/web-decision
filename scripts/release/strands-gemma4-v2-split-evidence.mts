import assert from 'node:assert/strict';
import {readFile, realpath} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {archiveIdentity, cleanSourceCommit} from './curated-gpu-evidence.mjs';
import {collectReleaseEvidenceInputs} from './verify-release-evidence.ts';
import {assertSplitVerifierSource} from './strands-gemma4-v2-split-source.mjs';
import {
  corpusTitle,
  inspectSplitProducer,
  scratch,
  workspace,
  sourceCommit
} from './strands-gemma4-v2-split-producer.mjs';
import lock from '../../packages/web-decision/src/catalog/artifacts.lock.json' with {type: 'json'};
import {record, sha256} from './curated-gpu-evidence.mts';
import {verifyTvmBuildEvidence} from './tvm-build-evidence.ts';
import {
  GEMMA_V2_CPU_REFERENCE_PATH,
  GEMMA_V2_CPU_REFERENCE_SHA256,
  validateGemmaV2CpuReferences
} from './strands-gemma4-v2-source-reference.mts';
import {
  GEMMA_V2_OFFICIAL_SUBSET_PATH,
  GEMMA_V2_OFFICIAL_SUBSET_SHA256,
  GEMMA_V2_OFFICIAL_GOLDEN_SHA256,
  inspectGemma4V2Conformance
} from './strands-gemma4-v2-conformance.mts';
import {assertGemma4V2ReleaseRecord, assertGemmaV2RawCapture} from './strands-gemma4-v2-release-evidence.mts';
import {gemmaV2RawProbabilityEncoding} from '../../tests/browser/gemma-v2-raw-capture.mts';
import {verifyViteReplay} from './strands-gemma4-v2-vite-replay.mjs';
import {persistentTitle, warmTitle} from './strands-gemma4-v2-persistent-variant.mjs';
import {persistentMode, assertPersistentTrace} from './strands-gemma4-v2-persistent-trace.mjs';
import {checkedInventory} from './strands-gemma4-v2-local-checkpoint.mjs';
import {assertSplitCandidate, gemmaSplitCandidate, viteConsumerScope} from './strands-gemma4-v2-split-candidate.mjs';
import {
  assertClosedPrivateProfile,
  assertWarmTrace,
  completedWarmSeed,
  warmProfileName
} from './strands-gemma4-v2-warm-seed.mjs';

const modelId = 'strands-decider-e2b-gemma4-v2-2611';
const sourceCase = 'scripts/strands/gemma4_v2/candidate/gemma4-v2-installed-reference.spec.ts';
const tested32k = process.env.WEB_DECISION_SPLIT_TESTED_ROOT ?? '';
const archivePath = process.env.WEB_DECISION_RELEASE_ARCHIVE ?? '';
const installedPackagePath = process.env.WEB_DECISION_INSTALLED_PACKAGE ?? '';
const goldenPath = process.env.WEB_DECISION_GEMMA_V2_OFFICIAL_GOLDEN ?? '';
const archiveLock = lock.byModelId[modelId];
const parse = (bytes: Uint8Array, label: string) => record(JSON.parse(Buffer.from(bytes).toString('utf8')), label);

export async function preflight() {
  assert.ok(tested32k && archivePath && installedPackagePath && goldenPath, 'Split evidence paths are required.');
  const verifierCommit = cleanSourceCommit(workspace);
  assert.equal(cleanSourceCommit(tested32k), sourceCommit, 'The tested source checkout differs.');
  const source = assertSplitVerifierSource(workspace, sourceCommit, verifierCommit);
  const {archive, models, catalogSha256} = await collectReleaseEvidenceInputs({archivePath, installedPackagePath});
  assertSplitCandidate(sourceCommit, archiveIdentity(archive));
  const metadata = record(
    JSON.parse(await readFile(join(archive.installedRoot, 'package.json'), 'utf8')),
    'Invalid package.'
  );
  assert.notEqual(metadata.private, true, 'Private candidate archives are not release evidence.');
  assert.ok(models.some((model) => model.modelId === modelId));
  const [cpu, subset, golden] = await Promise.all([
    readFile(join(workspace, GEMMA_V2_CPU_REFERENCE_PATH)),
    readFile(join(workspace, GEMMA_V2_OFFICIAL_SUBSET_PATH)),
    readFile(goldenPath)
  ]);
  validateGemmaV2CpuReferences(cpu);
  assert.equal(sha256(cpu), GEMMA_V2_CPU_REFERENCE_SHA256);
  assert.equal(sha256(subset), GEMMA_V2_OFFICIAL_SUBSET_SHA256);
  assert.equal(sha256(golden), GEMMA_V2_OFFICIAL_GOLDEN_SHA256);
  const pins = {
    sourceCommit,
    verifierCommit,
    source,
    archive: archiveIdentity(archive),
    catalogSha256,
    manifestSha256: archiveLock.manifest.rawSha256,
    cpuFixtureSha256: GEMMA_V2_CPU_REFERENCE_SHA256,
    subsetSha256: GEMMA_V2_OFFICIAL_SUBSET_SHA256,
    graphSha256: archiveLock.manifest.files['graph.wasm'].sha256
  };
  return {pins, archive, cpu, subset, golden};
}

function exactlyOne(value: unknown, description: string) {
  assert.ok(Array.isArray(value) && value.length === 1, 'Expected one ' + description);
  return record(value[0], description);
}
function flatten(suites: unknown): unknown[] {
  assert.ok(Array.isArray(suites));
  return suites.flatMap((value) => {
    const suite = record(value, 'Playwright suite');
    const specs = suite.specs ?? [];
    assert.ok(Array.isArray(specs));
    return [...specs, ...flatten(suite.suites ?? [])];
  });
}

export function passingChromiumRun(bytes: Uint8Array, root: string, file: string, title: string) {
  const report = parse(bytes, 'Playwright report');
  assert.deepEqual(report.errors ?? [], []);
  const config = record(report.config, 'Missing Playwright configuration.');
  assert.equal(config.rootDir, resolve(root));
  const stats = record(report.stats, 'Playwright run statistics');
  assert.deepEqual([stats.expected, stats.unexpected, stats.flaky, stats.skipped], [1, 0, 0, 0]);
  assert.ok(typeof stats.startTime === 'string' && Number.isFinite(Date.parse(stats.startTime)));
  const spec = exactlyOne(flatten(report.suites), 'Playwright spec');
  assert.ok(typeof spec.file === 'string');
  assert.equal(resolve(root, spec.file), resolve(root, file));
  assert.equal(spec.title, title);
  assert.equal(spec.ok, true);
  const test = exactlyOne(spec.tests, 'Playwright result');
  assert.deepEqual([test.projectName, test.expectedStatus, test.status], ['chromium', 'passed', 'expected']);
  const run = exactlyOne(test.results, 'unretried Chromium result');
  assert.deepEqual([run.status, run.retry, run.errors ?? []], ['passed', 0, []]);
  assert.ok(typeof run.duration === 'number' && Number.isFinite(run.duration) && run.duration > 0);
  return run;
}

function stdoutRecord(run: Record<string, unknown>, prefix: string) {
  assert.ok(Array.isArray(run.stdout));
  const stdout = run.stdout.map((line) => record(line, 'Playwright stdout').text).join('');
  const entries = stdout.split(/\r?\n/u).filter((line) => line.startsWith(prefix + ' '));
  assert.equal(entries.length, 1, 'Expected one ' + prefix + ' record');
  return parse(Buffer.from(entries[0]!.slice(prefix.length + 1)), prefix);
}
export function attachment(run: Record<string, unknown>, name: string, mime: string) {
  assert.ok(Array.isArray(run.attachments) && run.attachments.length === 2, 'Both original attachments are required.');
  const entry = exactlyOne(
    run.attachments.filter((a) => record(a, 'attachment').name === name),
    name
  );
  assert.equal(entry.contentType, mime);
  assert.equal(entry.path, undefined);
  assert.ok(typeof entry.body === 'string' && entry.body.length > 0);
  const bytes = Buffer.from(entry.body, 'base64');
  assert.equal(bytes.toString('base64'), entry.body);
  return bytes;
}
function numericDeltas(actual: Record<string, unknown>, expected: Record<string, unknown>) {
  const problems = ['confidence', ...Object.keys(record(expected.probabilities, 'CPU probabilities'))];
  const observed = record(actual.probabilities, 'GPU probabilities');
  const baseline = record(expected.probabilities, 'CPU probabilities');
  assert.deepEqual(Object.keys(observed).sort(), Object.keys(baseline).sort());
  return problems.map((name) => {
    const value = name === 'confidence' ? actual.confidence : observed[name];
    const reference = name === 'confidence' ? expected.confidence : baseline[name];
    assert.ok(
      typeof value === 'number' && Number.isFinite(value) && typeof reference === 'number' && Number.isFinite(reference)
    );
    return Math.abs(value - reference);
  });
}

export async function verifyCompleted32k(
  prefix: string,
  cpu: Buffer,
  identity: {sourceCommit: string; archive: ReturnType<typeof archiveIdentity>}
) {
  const [reportBytes, startBytes, completionBytes] = await Promise.all([
    readFile(prefix + '.playwright.json'),
    readFile(prefix + '.start.json'),
    readFile(prefix + '.completed.json')
  ]);
  const start = parse(startBytes, '32K start');
  const done = parse(completionBytes, '32K completion');
  const rawCase = record(record(parse(cpu, 'CPU Fixture').cases, 'CPU long cases')['text-32768'], 'CPU 32K case');
  assert.equal(typeof rawCase.resultJson, 'string');
  const pinned = validateGemmaV2CpuReferences(cpu)['text-32768']!;
  const reference = parse(Buffer.from(rawCase.resultJson as string), 'CPU 32K result');
  const requestSha = pinned.requestSha256;
  const referenceSha = sha256(Buffer.from(rawCase.resultJson as string));
  assert.equal(referenceSha, rawCase.resultSha256);
  assert.equal(start.status, 'running-not-release-evidence');
  assert.equal(done.status, 'completed-exact-installed-run');
  assert.equal(done.releaseApproved, false);
  for (const receipt of [start, done]) {
    assert.equal(receipt.mode, 'reference');
    assert.equal(receipt.sourceCommit, identity.sourceCommit);
    assert.deepEqual(receipt.archive, identity.archive);
    assert.equal(receipt.requestSha256, requestSha);
    assert.equal(receipt.referenceSha256, referenceSha);
  }
  if (identity.archive.sha256 === gemmaSplitCandidate.archive.sha256) {
    assert.equal(start.transport, viteConsumerScope);
    assert.equal(done.transport, viteConsumerScope);
    assert.equal(done.startSha256, sha256(startBytes), 'The 32K completion refers to a different start receipt.');
    assert.deepEqual(done.harnessPins, start.harnessPins, 'The 32K harness pins differ between receipts.');
  }
  assert.equal(done.playwrightReportSha256, sha256(reportBytes));
  const run = passingChromiumRun(
    reportBytes,
    join(tested32k, dirname(sourceCase)),
    'gemma4-v2-installed-reference.spec.ts',
    'installed archive collects one pinned upstream request'
  );
  const raw = stdoutRecord(run, 'INSTALLED_REFERENCE_RAW');
  const summary = stdoutRecord(run, 'INSTALLED_REFERENCE');
  for (const event of [raw, summary]) {
    assert.equal(event.archiveSha256, identity.archive.sha256);
    assert.equal(event.bundleSha256, identity.archive.bundleSha256);
    assert.equal(event.packageVersion, identity.archive.packageVersion);
    assert.equal(event.requestSha256, requestSha);
    assert.equal(event.referenceSha256, referenceSha);
    assert.deepEqual(event.reference, reference);
  }
  const actual = record(raw.actual, '32K installed output');
  assert.deepEqual(summary.actual, actual);
  assert.equal(actual.model, modelId);
  assert.deepEqual(actual.usage, {input_tokens: 32768, output_tokens: 1});
  const answers = record(actual.answers, '32K answers');
  assert.deepEqual(Object.keys(answers), ['q']);
  const answer = record(answers.q, '32K choice');
  const expected = record(record(reference.answers, 'CPU answers').q, 'CPU choice');
  assert.deepEqual(Object.keys(answer).sort(), Object.keys(expected).sort());
  assert.equal(answer.type, 'choice');
  assert.equal(answer.choice, expected.choice);
  const hardware = record(raw.hardware, '32K browser GPU hardware');
  const adapter = record(raw.adapter, '32K hardware adapter');
  const browser = record(raw.browser, '32K browser identity');
  assert.equal(hardware.status, 'Hardware accelerated');
  assert.ok(typeof hardware.renderer === 'string' && hardware.renderer.trim());
  assert.equal(adapter.isFallbackAdapter, false);
  assert.equal(browser.name, 'chromium');
  assert.ok(typeof browser.version === 'string' && browser.version);
  assert.equal(summary.tolerance, null);
  const maxDelta = Math.max(...numericDeltas(answer, expected));
  assert.equal(summary.maxDelta, maxDelta);
  return {
    playwrightSha256: sha256(reportBytes),
    receiptSha256: sha256(completionBytes),
    startSha256: sha256(startBytes),
    referenceSha256: referenceSha,
    browser,
    hardware,
    adapter,
    maxDelta,
    sample: {requestSha256: requestSha, inputTokens: 32768, outputTokens: 1, imageCount: 0, result: answer}
  };
}

export function assertCohortOperational(data: Record<string, unknown>) {
  assert.equal(data.cleanup, 'passed');
  assert.equal(data.recovery, 'passed');
  const offline = record(data.offline, 'Missing offline evidence');
  assert.equal(offline.cacheStatus, 'complete');
  assert.deepEqual(offline.blockedRequests, []);
  const image = record(record(data.cases, 'Missing five cases')['image-single'], 'Missing image case');
  assert.deepEqual(offline.result, image.result);
  const adapter = record(data.adapter, 'Cohort hardware adapter');
  assert.equal(adapter.isFallbackAdapter, false);
  const browser = record(data.browser, 'Cohort browser');
  assert.equal(browser.name, 'chromium');
  assert.ok(typeof browser.version === 'string' && browser.version);
  const hardware = record(data.hardware, 'Cohort browser hardware');
  assert.equal(hardware.status, 'Hardware accelerated');
  assert.ok(typeof hardware.renderer === 'string' && hardware.renderer.trim());
}

export function assertRawReceipt(
  receipt: Record<string, unknown>,
  expected: {sourceCommit: string; archive: unknown; producer: unknown; startedSha256: string; playwrightSha256: string}
) {
  assert.equal(receipt.status, 'completed-exact-raw-cohort');
  assert.equal(receipt.releaseApproved, false);
  for (const key of ['sourceCommit', 'archive', 'producer', 'startedSha256', 'playwrightSha256'] as const) {
    assert.deepEqual(receipt[key], expected[key], 'Raw capture receipt ' + key + ' differs.');
  }
}

export function assertCanonicalRawVectors(bytes: Uint8Array) {
  const lines = new TextDecoder('utf-8', {fatal: true, ignoreBOM: true}).decode(bytes).split('\n');
  assert.equal(lines.pop(), '', 'Expected newline-terminated raw JSONL.');
  for (const line of lines)
    assert.equal(JSON.stringify(JSON.parse(line)), line, 'Expected raw JSONL encoded by the exact producer.');
}

export function assertCohortMetrics(conformance: unknown, metrics: ReturnType<typeof inspectGemma4V2Conformance>) {
  assert.deepEqual(conformance, {probabilityEncoding: gemmaV2RawProbabilityEncoding, ...metrics});
  assert.equal(metrics.meetsUpstreamMetricThresholds, true);
}

function rawCohortTitle(mode: ReturnType<typeof persistentMode>) {
  if (mode === 'warm') return warmTitle;
  return mode ? persistentTitle : corpusTitle;
}

async function verifyPersistentCohort(data: Record<string, unknown>, installedRoot: string, rawReportPath: string) {
  const mode = persistentMode();
  if (mode !== 'warm') {
    assert.equal(data.warmSeed, undefined, 'A rehashed seed must use its separate warm evidence mode.');
    if (!mode) return;
    return assertPersistentTrace(data, mode, await checkedInventory(installedRoot));
  }
  const inventory = await checkedInventory(installedRoot);
  const seed = await completedWarmSeed();
  const clonePath = join(await realpath(dirname(rawReportPath)), 'profiles', warmProfileName);
  await assertClosedPrivateProfile(clonePath);
  const warmSeed = assertWarmTrace(data, seed, clonePath, inventory);
  return {...assertPersistentTrace(data, 'online', inventory, undefined, true), warmSeed};
}

export async function verifyRawCohort(rawReportPath: string) {
  const [{pins, archive, cpu, subset, golden}, producer, rawReport] = await Promise.all([
    preflight(),
    inspectSplitProducer(),
    readFile(rawReportPath)
  ]);
  const mode = persistentMode();
  const run = passingChromiumRun(rawReport, scratch, producer.producerPath, rawCohortTitle(mode));
  const data = parse(attachment(run, 'gemma-v2-public-run.json', 'application/json'), 'Raw cohort report');
  const vectors = attachment(run, 'gemma-v2-public-code-probabilities.jsonl', 'application/x-ndjson');
  assertCanonicalRawVectors(vectors);
  assert.equal(data.caseMode, 'five-cases-external-32k-required');
  assert.equal(data.producerSha256, producer.producerSha256);
  assert.deepEqual(data.archive, pins.archive);
  assert.equal(data.schemaVersion, 1);
  assert.equal(data.modelId, modelId);
  assert.equal(data.source, 'pinned-upstream');
  assert.equal(data.sourceCommit, pins.sourceCommit);
  assert.equal(data.sourceRevision, 'c3c845782095b31d438963ac2afcdfe41b49f757');
  assert.equal(data.checkpointRevision, 'a5f32ea1bffe2e91c25c0c641d444e8115e28212');
  assert.equal(data.catalogSha256, pins.catalogSha256);
  assert.equal(data.bundledManifestSha256, pins.manifestSha256);
  assert.equal(data.cpuFixtureSha256, pins.cpuFixtureSha256);
  const graph = pins.graphSha256;
  assert.equal(data.graphSha256, graph);
  const checkpointAssertion = mode === 'warm' ? 'independentPublicRangeObserved' : 'checkpointRangeObserved';
  assert.deepEqual(data.assertions, {[checkpointAssertion]: true, installedGraphResponseSha256: graph});
  const host = record(record(data.host, 'Raw cohort host').device, 'Raw cohort host device');
  assert.equal(host.isVirtual, false);
  assert.ok(typeof host.model === 'string' && host.model.trim());
  assertCohortOperational(data);
  const persistent = await verifyPersistentCohort(data, archive.installedRoot, rawReportPath);
  const cases = record(data.cases, 'Expected five raw cohort cases');
  assert.deepEqual(Object.keys(cases).sort(), ['image-multiple', 'image-single', 'noul', 'score', 'text-1024']);
  const metrics = inspectGemma4V2Conformance(golden, vectors, {bytes: subset, sha256: pins.subsetSha256}, 'raw-fp32');
  assert.equal(metrics.selectedRows, 360);
  assert.equal(metrics.questions, 1066);
  await assertGemmaV2RawCapture(data, archive, metrics.questions);
  assertCohortMetrics(data.conformance, metrics);

  return {pins, producer, archive, data, cases, cpu, metrics, persistent, playwrightSha256: sha256(rawReport)};
}

export function assertMatchingGpu(
  data: Record<string, unknown>,
  long: Pick<Awaited<ReturnType<typeof verifyCompleted32k>>, 'browser' | 'hardware' | 'adapter'>
) {
  assert.deepEqual(data.browser, long.browser);
  assert.equal(record(data.hardware, 'Raw cohort hardware').renderer, long.hardware.renderer);
  assert.deepEqual(data.adapter, long.adapter);
}

export function verifySplitNativeBuild(tvmBuildEvidencePath: string, installedRoot: string) {
  const sourcePackage = join(workspace, 'packages/web-decision');
  const bundled = Object.fromEntries(
    archiveLock.bundledRuntimeFiles.map(({manifestName, sourcePath}) => [manifestName, join(sourcePackage, sourcePath)])
  );
  return verifyTvmBuildEvidence(
    tvmBuildEvidencePath,
    [
      {
        modelId,
        manifestFiles: archiveLock.manifest.files,
        tvmVersion: '0.27.0.post1',
        tvmSourceRevision: '7b5acb32bd271342a24130f4ef439ff6c2e52c6c',
        sourceArtifacts: bundled,
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
    [sourcePackage, installedRoot]
  );
}

export async function verifySplit(rawReportPath: string, prefix32k: string, tvmBuildEvidencePath: string) {
  const [cohort, receiptBytes, startBytes] = await Promise.all([
    verifyRawCohort(rawReportPath),
    readFile(rawReportPath + '.completed.json'),
    readFile(join(dirname(rawReportPath), 'started.json'))
  ]);
  const {pins, producer, archive, data, cases, cpu, metrics, playwrightSha256} = cohort;
  const warmTrace = cohort.persistent && 'warmSeed' in cohort.persistent ? cohort.persistent.warmSeed : undefined;
  const warmStart = warmTrace ? warmStartReceipt(warmTrace) : {};
  assert.deepEqual(parse(startBytes, 'Raw capture start'), {
    status: 'started-raw-cohort',
    sourceCommit: pins.sourceCommit,
    archive: pins.archive,
    producer,
    ...warmStart
  });
  assertRawReceipt(parse(receiptBytes, 'Raw capture completion'), {
    startedSha256: sha256(startBytes),
    sourceCommit: pins.sourceCommit,
    archive: pins.archive,
    producer,
    playwrightSha256
  });
  const long = await verifyCompleted32k(prefix32k, cpu, {sourceCommit: pins.sourceCommit, archive: pins.archive});
  assertMatchingGpu(data, long);
  const replayPath = process.env.WEB_DECISION_SPLIT_VITE_AUDIT;
  const scriptPath = process.env.WEB_DECISION_SPLIT_VITE_AUDIT_SCRIPT;
  assert.ok(replayPath && scriptPath, 'The reviewed Vite replay report and script are required.');
  const viteReplay = await verifyViteReplay({
    reportPath: replayPath,
    scriptPath,
    workspace,
    sourceCommit: pins.sourceCommit,
    archive,
    startSha256: long.startSha256,
    requestSha256: long.sample.requestSha256,
    referenceSha256: long.referenceSha256
  });
  const cpuCases = validateGemmaV2CpuReferences(cpu);
  const numericDifferences = assertGemma4V2ReleaseRecord(
    {...data, cases: {...cases, 'text-32768': long.sample}},
    archive,
    {status: 'pinned-upstream-cpu', cases: cpuCases}
  );
  const build = await verifySplitNativeBuild(tvmBuildEvidencePath, archive.installedRoot);

  return {
    status: 'verified-gemma-v2-split-evidence',
    releaseApproved: false,
    sourceCommit: pins.sourceCommit,
    verifierCommit: pins.verifierCommit,
    source: pins.source,
    archive: pins.archive,
    producer,
    transport: {
      corpus: 'byte-preserving-installed-archive',
      longContext: 'installed-archive-vite-consumer',
      viteReplay,
      ...(cohort.persistent ? {persistentCohort: cohort.persistent} : {})
    },
    reports: {rawPlaywrightSha256: playwrightSha256, rawReceiptSha256: sha256(receiptBytes), independent32k: long},
    metrics,
    numericDifferences,
    build
  };
}

function warmStartReceipt(trace: Record<string, unknown>) {
  const {copy, rehash: _rehash, browserRemoteAttempts: _network, ...origin} = trace;
  return {warmSeed: {...origin, copiedProfilePathSha256: record(copy, 'Warm profile copy').copiedProfilePathSha256}};
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [raw, prefix, build, ...rest] = process.argv.slice(2);
  assert.ok(
    raw && prefix && build && rest.length === 0,
    'Usage: node --import tsx verify-split-evidence.mts raw.playwright.json 32768-run /tvm-build-evidence.json'
  );
  console.log(JSON.stringify(await verifySplit(raw, prefix, build), null, 2));
}
