import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {isAbsolute, join, resolve} from 'node:path';
import type {ArchiveInfo} from './verify-release-evidence.ts';
import {verifyTvmBuildEvidence} from './tvm-build-evidence.ts';

type AnyRecord = Record<string, any>;
type AssetMetadata = Readonly<{bytes: number; sha256: string}>;

export interface KananaEvidencePolicy {
  schemaVersion: 1;
  evidencePrefix: string;
  archive: {sha256: string; packageName: string; packageVersion: string};
  model: {
    modelId: string;
    repository: string;
    revision: string;
    checkpointSha256: string;
    manifestPath: string;
    manifestRawSha256: string;
    manifestSha256: string;
    artifactFiles: Record<string, AssetMetadata>;
  };
  referenceFixture: {path: string; sha256: string};
  textProbabilityTolerance: number;
  browser: {name: string; project: string};
  gpu: {api: 'webgpu'; requiredLimits: Record<string, number>};
  tests: Record<
    'text' | 'image' | 'offline',
    {
      file: string;
      title: string;
      kind: 'text-choice' | 'synthetic-image-choice' | 'cached-offline-text-choice';
      runtimeArtifacts: string[];
    }
  >;
}

export interface KananaManifestEvidence {
  path: string;
  rawSha256: string;
  manifestSha256: string;
  filesSha256: string;
  manifest: AnyRecord;
  files: Record<string, AssetMetadata>;
}

export interface KananaReleaseEvidenceInputs {
  archive: ArchiveInfo;
  policy: KananaEvidencePolicy;
  manifest: KananaManifestEvidence;
}

interface KananaReferenceFixture extends AnyRecord {
  modelId: string;
  revision: string;
  checkpointSha256: string;
  result: {answers: Record<string, {choice: string; confidence: number; probabilities: Record<string, number>}>};
}

export interface KananaVerificationContext {
  archive: ArchiveInfo;
  manifest: KananaManifestEvidence;
  policy: KananaEvidencePolicy;
  referenceFixture: KananaReferenceFixture;
  workspaceRoot: string;
}

export async function verifyKananaTvmBuildEvidence(
  evidencePath: string,
  inputs: KananaReleaseEvidenceInputs,
  sourcePackageRoot: string
) {
  const wasmArtifacts = Object.keys(inputs.manifest.files)
    .filter((name) => name.endsWith('.wasm'))
    .sort();
  const expectedGraphs = {
    'kanana-text-model.wasm': 'text',
    'kanana-vision-model.wasm': 'vision'
  } as const;
  assert.deepEqual(wasmArtifacts, Object.keys(expectedGraphs).sort(), 'Kanana TVM artifact set differs from policy.');

  const tvmVersion = inputs.manifest.manifest.runtime?.tvm;
  fail(typeof tvmVersion === 'string' && tvmVersion.length > 0, 'Kanana manifest must pin its TVM version.');
  const manifestDirectory = inputs.manifest.path.split('/').slice(0, -1);
  const sourceArtifacts = Object.fromEntries(
    wasmArtifacts.map((artifact) => [artifact, join(inputs.archive.installedRoot, ...manifestDirectory, artifact)])
  );
  const reportRequirements = Object.fromEntries(
    Object.entries(expectedGraphs).map(([artifact, graph]) => [
      artifact,
      {
        repository: inputs.policy.model.repository,
        modelRevision: inputs.policy.model.revision,
        checkpointSha256: inputs.policy.model.checkpointSha256,
        graph,
        buildBackend: 'native-tvm',
        weightSource: 'pinned-safetensors',
        onnxImportOrExportUsed: false
      }
    ])
  );
  const buildModel = {
    modelId: inputs.policy.model.modelId,
    manifestFiles: inputs.manifest.files,
    tvmVersion,
    sourceArtifacts,
    reportRequirements
  };
  return verifyTvmBuildEvidence(evidencePath, [buildModel], [sourcePackageRoot, inputs.archive.installedRoot]);
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/u;

function fail(condition: unknown, message: string): asserts condition {
  assert.ok(condition, message);
}

function normalizedPath(path: unknown, description: string): string {
  fail(typeof path === 'string' && path.length > 0, `${description} must be a non-empty path.`);
  fail(!path.includes('\\') && !isAbsolute(path), `${description} must be a safe relative path: ${path}`);
  const segments = path.split('/');
  fail(
    segments.every((segment) => segment && segment !== '.' && segment !== '..'),
    `${description} is unsafe: ${path}`
  );
  return segments.join('/');
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonical(item)])
    );
  }
  return value;
}

function canonicalSha256(value: unknown): string {
  return createHash('sha256')
    .update(`${JSON.stringify(canonical(value))}\n`)
    .digest('hex');
}

function artifactFiles(input: unknown): Record<string, AssetMetadata> {
  fail(input !== null && typeof input === 'object' && !Array.isArray(input), 'Kanana artifact files are required.');
  const files: Record<string, AssetMetadata> = {};
  for (const [path, value] of Object.entries(input)) {
    const item = value as AnyRecord;
    const safePath = normalizedPath(path, 'Kanana artifact file');
    fail(Number.isSafeInteger(item?.bytes) && item.bytes >= 0, `Invalid Kanana artifact byte length: ${path}.`);
    fail(
      typeof item?.sha256 === 'string' && SHA256_PATTERN.test(item.sha256),
      `Invalid Kanana artifact SHA-256: ${path}.`
    );
    files[safePath] = {bytes: item.bytes, sha256: item.sha256};
  }
  fail(Object.keys(files).length > 0, 'Kanana artifact manifest must not be empty.');
  return files;
}

export function validateKananaEvidencePolicy(input: unknown): KananaEvidencePolicy {
  fail(input !== null && typeof input === 'object' && !Array.isArray(input), 'Kanana evidence policy is required.');
  const policy = input as AnyRecord;
  fail(policy.schemaVersion === 1, 'Unsupported Kanana evidence policy schema.');
  fail(
    typeof policy.evidencePrefix === 'string' && policy.evidencePrefix.length > 0,
    'Kanana evidence prefix is required.'
  );

  const archive = policy.archive as AnyRecord;
  fail(archive && SHA256_PATTERN.test(archive.sha256), 'Kanana archive SHA-256 pin is invalid.');
  fail(archive.packageName === 'web-decision', 'Kanana package name must be web-decision.');
  fail(archive.packageVersion === '0.4.0', 'Kanana package version must be 0.4.0.');

  const model = policy.model as AnyRecord;
  fail(model?.modelId === 'kanana-2-v-decide-3b-2611', 'Kanana curated model ID pin is invalid.');
  for (const field of ['repository', 'revision'])
    fail(typeof model[field] === 'string' && model[field].length > 0, `Kanana ${field} pin is required.`);
  for (const field of ['checkpointSha256', 'manifestRawSha256', 'manifestSha256'])
    fail(typeof model[field] === 'string' && SHA256_PATTERN.test(model[field]), `Kanana ${field} pin is invalid.`);
  const manifestPath = normalizedPath(model.manifestPath, 'Kanana manifest path');
  const pinnedFiles = artifactFiles(model.artifactFiles);

  const fixture = policy.referenceFixture as AnyRecord;
  const referenceFixture = {
    path: normalizedPath(fixture?.path, 'Kanana reference fixture path'),
    sha256: fixture?.sha256
  };
  fail(SHA256_PATTERN.test(referenceFixture.sha256), 'Kanana reference fixture SHA-256 pin is invalid.');
  fail(
    typeof policy.textProbabilityTolerance === 'number' &&
      Number.isFinite(policy.textProbabilityTolerance) &&
      policy.textProbabilityTolerance >= 0,
    'Kanana text probability tolerance is invalid.'
  );

  const browser = policy.browser as AnyRecord;
  fail(browser?.name === 'chromium' && browser?.project === 'chromium', 'Kanana evidence requires Chromium.');
  const gpu = policy.gpu as AnyRecord;
  fail(gpu?.api === 'webgpu', 'Kanana evidence requires WebGPU.');
  fail(gpu.requiredLimits && typeof gpu.requiredLimits === 'object', 'Kanana GPU limit requirements are required.');
  const requiredLimits: Record<string, number> = {};
  for (const [name, value] of Object.entries(gpu.requiredLimits as AnyRecord)) {
    fail(typeof value === 'number' && Number.isSafeInteger(value) && value > 0, `Invalid Kanana GPU limit: ${name}.`);
    requiredLimits[name] = value;
  }

  const definitions = {
    text: {kind: 'text-choice', runtimeArtifacts: ['kanana-text-model.wasm']},
    image: {
      kind: 'synthetic-image-choice',
      runtimeArtifacts: ['kanana-text-model.wasm', 'kanana-vision-model.wasm']
    },
    offline: {kind: 'cached-offline-text-choice', runtimeArtifacts: ['kanana-text-model.wasm']}
  } as const;
  const tests = {} as KananaEvidencePolicy['tests'];
  for (const key of Object.keys(definitions) as Array<keyof typeof definitions>) {
    const item = (policy.tests as AnyRecord)?.[key] as AnyRecord;
    const file = normalizedPath(item?.file, `Kanana ${key} test file`);
    fail(file.startsWith('tests/browser/'), `Kanana ${key} test must be under tests/browser/.`);
    fail(typeof item?.title === 'string' && item.title.length > 0, `Kanana ${key} test title is required.`);
    assert.equal(item.kind, definitions[key].kind, `Kanana ${key} evidence kind is invalid.`);
    const runtimeArtifacts = item.runtimeArtifacts;
    fail(Array.isArray(runtimeArtifacts), `Kanana ${key} runtime artifacts are required.`);
    const paths = runtimeArtifacts.map((path: unknown) => normalizedPath(path, `Kanana ${key} runtime artifact`));
    assert.deepEqual(
      paths,
      definitions[key].runtimeArtifacts,
      `Kanana ${key} runtime artifact set differs from policy.`
    );
    fail(
      paths.every((path) => Object.hasOwn(pinnedFiles, path)),
      `Kanana ${key} test refers to an unpinned artifact.`
    );
    tests[key] = {file, title: item.title, kind: definitions[key].kind, runtimeArtifacts: paths};
  }

  return {
    schemaVersion: 1,
    evidencePrefix: policy.evidencePrefix,
    archive: {sha256: archive.sha256, packageName: archive.packageName, packageVersion: archive.packageVersion},
    model: {
      modelId: model.modelId,
      repository: model.repository,
      revision: model.revision,
      checkpointSha256: model.checkpointSha256,
      manifestPath,
      manifestRawSha256: model.manifestRawSha256,
      manifestSha256: model.manifestSha256,
      artifactFiles: pinnedFiles
    },
    referenceFixture,
    textProbabilityTolerance: policy.textProbabilityTolerance,
    browser: {name: browser.name, project: browser.project},
    gpu: {api: 'webgpu', requiredLimits},
    tests
  };
}

export function validateKananaManifestBytes(
  bytes: Buffer,
  policy: KananaEvidencePolicy
): Omit<KananaManifestEvidence, 'path'> {
  const rawSha256 = createHash('sha256').update(bytes).digest('hex');
  assert.equal(rawSha256, policy.model.manifestRawSha256, 'Kanana raw artifact manifest differs from its pin.');
  const manifest = JSON.parse(bytes.toString('utf8')) as AnyRecord;
  assert.equal(manifest.modelId, policy.model.modelId, 'Kanana artifact manifest model ID differs from its pin.');
  assert.equal(
    manifest.repository,
    policy.model.repository,
    'Kanana artifact manifest repository differs from its pin.'
  );
  assert.equal(manifest.revision, policy.model.revision, 'Kanana artifact manifest revision differs from its pin.');
  assert.equal(
    manifest.checkpointSha256,
    policy.model.checkpointSha256,
    'Kanana checkpoint digest differs from its pin.'
  );
  const manifestSha256 = canonicalSha256(manifest);
  assert.equal(manifestSha256, policy.model.manifestSha256, 'Kanana canonical artifact manifest differs from its pin.');
  const files = artifactFiles(manifest.files);
  assert.deepEqual(
    canonical(files),
    canonical(policy.model.artifactFiles),
    'Kanana artifact file hashes differ from policy.'
  );
  return {rawSha256, manifestSha256, filesSha256: canonicalSha256(files), manifest, files};
}

function requireText(value: unknown, description: string): void {
  fail(typeof value === 'string' && value.trim().length > 0, `${description} is required.`);
}

function validateLimits(actual: AnyRecord, required: Record<string, number>): void {
  fail(actual && typeof actual === 'object', 'Kanana GPU limits are missing.');
  for (const [name, minimum] of Object.entries(required)) {
    fail(Number.isFinite(actual[name]) && actual[name] >= minimum, `Kanana GPU does not meet ${name}.`);
  }
}

function collectPlaywrightTests(suites: AnyRecord[] = [], items: AnyRecord[] = []): AnyRecord[] {
  for (const suite of suites) {
    for (const spec of suite.specs ?? []) {
      for (const test of spec.tests ?? []) items.push({file: spec.file ?? suite.file, title: spec.title, test});
    }
    collectPlaywrightTests(suite.suites, items);
  }
  return items;
}

function recordFromTest(test: AnyRecord, prefix: string): AnyRecord {
  const results = test.results ?? [];
  assert.equal(results.length, 1, 'Each Kanana GPU test must have exactly one execution and no retry.');
  assert.equal(test.expectedStatus, 'passed', 'Every Kanana GPU test must expect a pass.');
  assert.equal(test.status, 'expected', 'Every Kanana GPU test must pass without skip or retry.');
  const result = results[0];
  assert.ok(result, 'Kanana GPU test has no execution result.');
  assert.equal(result.status, 'passed', 'Kanana GPU test did not pass.');
  assert.deepEqual(result.errors ?? [], [], 'Kanana GPU test contains errors.');
  const stdout = (result.stdout ?? [])
    .map((item: unknown) => (typeof item === 'string' ? item : ((item as AnyRecord)?.text ?? '')))
    .join('');
  const lines = stdout.split(/\r?\n/u).filter((line: string) => line.startsWith(prefix));
  assert.equal(lines.length, 1, 'Each passing Kanana GPU test must emit exactly one structured evidence record.');
  return JSON.parse(lines[0].slice(prefix.length));
}

function validateKananaArchiveRecord(
  evidence: AnyRecord,
  context: KananaVerificationContext,
  kind: keyof KananaEvidencePolicy['tests']
): void {
  const {archive, manifest, policy} = context;
  assert.equal(evidence.schemaVersion, 1, 'Unsupported Kanana browser evidence schema.');
  assert.equal(evidence.kind, policy.tests[kind].kind, `Unexpected Kanana evidence kind for ${kind}.`);
  const archiveRecord = evidence.archive;
  assert.equal(archiveRecord?.sha256, archive.archiveSha256, 'Kanana test used a different package archive.');
  assert.equal(
    archiveRecord?.packageTreeSha256,
    archive.packageTreeSha256,
    'Kanana test used a different package tree.'
  );
  assert.equal(archiveRecord?.bundleSha256, archive.bundleSha256, 'Kanana test used a different package bundle.');
  assert.equal(
    archiveRecord?.installedPackagePathSha256,
    archive.installedPackagePathSha256,
    'Kanana evidence points to a different installed package directory.'
  );
  assert.equal(archiveRecord?.entryPathSha256, archive.entryPathSha256, 'Kanana entrypoint path differs from archive.');
  assert.equal(archiveRecord?.packageName, archive.packageName, 'Kanana package name differs from archive.');
  assert.equal(archiveRecord?.packageVersion, archive.packageVersion, 'Kanana package version differs from archive.');

  const model = evidence.model;
  assert.equal(model?.requestedId, policy.model.modelId, 'Kanana requested model ID mismatch.');
  assert.equal(model?.loadedId, policy.model.modelId, 'Kanana loaded model ID mismatch.');
  assert.equal(model?.describedId, policy.model.modelId, 'Kanana descriptor model ID mismatch.');
  assert.equal(model?.checkpointRevision, policy.model.revision, 'Kanana checkpoint revision mismatch.');
  assert.equal(model?.checkpointSha256, policy.model.checkpointSha256, 'Kanana checkpoint digest mismatch.');

  const artifact = evidence.artifact;
  assert.equal(artifact?.modelId, policy.model.modelId, 'Kanana artifact model ID mismatch.');
  assert.equal(artifact?.manifestPath, manifest.path, 'Kanana evidence refers to a different static manifest path.');
  assert.equal(artifact?.manifestRawSha256, manifest.rawSha256, 'Kanana evidence refers to a different manifest file.');
  assert.equal(
    artifact?.manifestSha256,
    manifest.manifestSha256,
    'Kanana evidence refers to a different model manifest.'
  );
  assert.equal(artifact?.filesSha256, manifest.filesSha256, 'Kanana evidence refers to different model artifacts.');
  assert.deepEqual(artifact?.files, manifest.files, 'Kanana artifact hashes differ from the archive.');
  assert.deepEqual(
    artifact?.runtimeArtifacts?.map(({path}: AnyRecord) => path),
    policy.tests[kind].runtimeArtifacts,
    `Kanana ${kind} evidence omits a required TVM model artifact.`
  );
  for (const observed of artifact.runtimeArtifacts as AnyRecord[]) {
    assert.deepEqual(
      {bytes: observed.bytes, sha256: observed.sha256},
      manifest.files[observed.path],
      `Kanana runtime artifact differs from the pinned package: ${observed.path}.`
    );
  }

  assert.equal(evidence.browser?.name, policy.browser.name, 'Kanana browser name mismatch.');
  requireText(evidence.browser?.version, 'Kanana browser version');
  requireText(evidence.browser?.userAgent, 'Kanana browser user agent');
  const gpu = evidence.gpu;
  assert.equal(gpu?.api, policy.gpu.api, 'Kanana GPU API mismatch.');
  assert.equal(gpu?.isFallbackAdapter, false, 'Kanana inference used a fallback GPU adapter.');
  assert.equal(
    gpu?.adapterInfo?.source,
    'navigator.gpu.requestAdapter() and GPUAdapterInfo',
    'Kanana adapter information must come from the active WebGPU adapter.'
  );
  requireText(gpu?.adapterInfo?.value?.vendor, 'Kanana WebGPU adapter vendor');
  requireText(gpu?.adapterInfo?.value?.architecture, 'Kanana WebGPU adapter architecture');
  assert.equal(gpu.adapterInfo.value.isFallbackAdapter, false, 'Kanana adapter info reports a fallback adapter.');
  fail(Array.isArray(gpu.features), 'Kanana GPU features are missing.');
  validateLimits(gpu.limits, policy.gpu.requiredLimits);
}

function validateKananaResultIdentity(output: AnyRecord, context: KananaVerificationContext, mode: string): void {
  const {model} = context.policy;
  assert.equal(output?.modelId, model.modelId, `${mode} Kanana result model ID mismatch.`);
  assert.equal(output?.descriptionModelId, model.modelId, `${mode} Kanana descriptor model ID mismatch.`);
  assert.equal(output?.checkpointRevision, model.revision, `${mode} Kanana checkpoint revision mismatch.`);
  assert.equal(output?.checkpointSha256, model.checkpointSha256, `${mode} Kanana checkpoint digest mismatch.`);
  assert.equal(output?.backend, 'tvm-webgpu', `${mode} Kanana inference did not use TVM WebGPU.`);
  assert.equal(
    output?.assetManifestSha256,
    context.manifest.manifestSha256,
    `${mode} Kanana inference used a different model artifact manifest.`
  );
  fail(
    output?.usage && Number.isSafeInteger(output.usage.input_tokens) && output.usage.input_tokens > 0,
    `${mode} Kanana inference did not report consumed input tokens.`
  );
}

function validateKananaTextResult(output: AnyRecord, context: KananaVerificationContext, mode: string): void {
  validateKananaResultIdentity(output, context, mode);
  const expected = context.referenceFixture.result.answers.delivery_status;
  assert.ok(expected, 'Kanana reference fixture is missing delivery_status.');
  assert.equal(output.choice, expected.choice, `${mode} Kanana text choice differs from the pinned reference.`);
  const expectedKeys = Object.keys(expected.probabilities);
  assert.deepEqual(Object.keys(output.probabilities ?? {}), expectedKeys, `${mode} Kanana probability keys mismatch.`);
  fail(Array.isArray(output.probabilityEntries), `${mode} Kanana result has no ordered probability entries.`);
  assert.deepEqual(
    output.probabilityEntries.map((entry: unknown[]) => entry[0]),
    expectedKeys,
    `${mode} Kanana probability order differs from the pinned reference.`
  );
  let sum = 0;
  for (const [index, key] of expectedKeys.entries()) {
    const actual: number = output.probabilityEntries[index]?.[1];
    const expectedProbability = expected.probabilities[key];
    fail(expectedProbability !== undefined, `Kanana reference fixture has no probability for ${key}.`);
    fail(Number.isFinite(actual) && actual >= 0 && actual <= 1, `${mode} Kanana probability ${key} is invalid.`);
    assert.equal(output.probabilities[key], actual, `${mode} Kanana probability map and entries disagree for ${key}.`);
    fail(
      Math.abs(actual - expectedProbability) <= context.policy.textProbabilityTolerance,
      `${mode} Kanana probability ${key} differs from the pinned reference.`
    );
    sum += actual;
  }
  fail(Math.abs(sum - 1) <= 1e-6, `${mode} Kanana probabilities are not normalized.`);
  fail(Number.isFinite(output.confidence), `${mode} Kanana confidence is invalid.`);
  fail(
    Math.abs(output.confidence - expected.confidence) <= context.policy.textProbabilityTolerance,
    `${mode} Kanana confidence differs from the pinned reference.`
  );
}

function validateKananaImageResult(output: AnyRecord, context: KananaVerificationContext): void {
  validateKananaResultIdentity(output, context, 'Image');
  assert.equal(output.answer, 'rectangle', 'Kanana synthetic-image choice differs from the expected result.');
  const probabilities = output.probabilities as Record<string, number>;
  assert.deepEqual(Object.keys(probabilities ?? {}), ['rectangle', 'circle'], 'Kanana image option order mismatch.');
  const rectangle = probabilities.rectangle;
  const circle = probabilities.circle;
  fail(typeof rectangle === 'number' && Number.isFinite(rectangle), 'Kanana rectangle probability is invalid.');
  fail(typeof circle === 'number' && Number.isFinite(circle), 'Kanana circle probability is invalid.');
  fail(Math.abs(rectangle + circle - 1) <= 1e-6, 'Kanana image probabilities are not normalized.');
  fail(Array.isArray(output.probabilityEntries), 'Kanana image probabilities have no ordered entries.');
  assert.deepEqual(
    output.probabilityEntries.map((entry: unknown[]) => entry[0]),
    Object.keys(probabilities),
    'Kanana image probability order differs from its option order.'
  );
  fail(Number.isFinite(output.confidence), 'Kanana image confidence is invalid.');
  const selected = probabilities[output.answer];
  fail(typeof selected === 'number' && Number.isFinite(selected), 'Kanana selected probability is invalid.');
  const others: number[] = Object.entries(probabilities)
    .filter(([key]) => key !== output.answer)
    .map(([, probability]) => probability);
  const expectedConfidence = others.length
    ? Math.max(0, Math.min(1, selected - others.reduce((sum, probability) => sum + probability, 0) / others.length))
    : 1;
  fail(
    Math.abs(output.confidence - expectedConfidence) <= 1e-6,
    'Kanana image confidence differs from the selected-option margin.'
  );
}

function validateKananaRecord(
  record: AnyRecord,
  kind: keyof KananaEvidencePolicy['tests'],
  context: KananaVerificationContext
): void {
  validateKananaArchiveRecord(record, context, kind);
  if (kind === 'text') {
    assert.equal(
      record.referenceFixtureSha256,
      context.policy.referenceFixture.sha256,
      'Kanana text fixture pin mismatch.'
    );
    assert.equal(
      record.probabilityTolerance,
      context.policy.textProbabilityTolerance,
      'Kanana text probability tolerance mismatch.'
    );
    validateKananaTextResult(record.inference, context, 'Online');
    return;
  }
  if (kind === 'image') {
    assert.equal(record.input?.kind, 'synthetic-canvas-v1', 'Kanana image evidence must use the synthetic fixture.');
    assert.equal(record.input?.width, 256, 'Kanana synthetic image width mismatch.');
    assert.equal(record.input?.height, 256, 'Kanana synthetic image height mismatch.');
    fail(SHA256_PATTERN.test(record.input?.sha256), 'Kanana synthetic image digest is missing.');
    validateKananaImageResult(record.inference, context);
    return;
  }

  assert.equal(
    record.referenceFixtureSha256,
    context.policy.referenceFixture.sha256,
    'Kanana offline fixture pin mismatch.'
  );
  assert.equal(
    record.probabilityTolerance,
    context.policy.textProbabilityTolerance,
    'Kanana offline tolerance mismatch.'
  );
  assert.equal(record.cache?.primedOnline, true, 'Kanana offline test did not enable online caching.');
  assert.equal(record.cache?.requestedOffline, true, 'Kanana offline test did not request offline loading.');
  assert.equal(record.cache?.blockedNetwork, true, 'Kanana offline test did not block network requests.');
  assert(
    Number.isSafeInteger(record.cache?.fetchedRanges) && record.cache.fetchedRanges > 0,
    'Kanana cache priming fetched no verified ranges.'
  );
  assert(
    Number.isSafeInteger(record.cache?.fetchedBytes) && record.cache.fetchedBytes > 0,
    'Kanana cache priming fetched no bytes.'
  );
  assert.deepEqual(record.networkAttempts, [], 'Kanana offline inference attempted a network request.');
  validateKananaTextResult(record.inference?.online, context, 'Online cached');
  validateKananaTextResult(record.inference?.offline, context, 'Offline cached');
  for (const field of ['choice', 'confidence', 'probabilities', 'probabilityEntries']) {
    assert.deepEqual(
      record.inference.offline[field],
      record.inference.online[field],
      `Kanana offline ${field} differs from the cached online inference.`
    );
  }
}

export function validateKananaPlaywrightReport(
  report: AnyRecord,
  context: KananaVerificationContext
): {caseCount: number; browser: string; modelId: string} {
  const tests = collectPlaywrightTests(report.suites);
  const reportRoot = report.config?.rootDir;
  fail(typeof reportRoot === 'string' && isAbsolute(reportRoot), 'Kanana Playwright report has no absolute test root.');
  const expectedCount = Object.keys(context.policy.tests).length;
  assert.equal(report.stats?.expected, expectedCount, 'Kanana pass count differs from the required evidence matrix.');
  assert.equal(report.stats?.skipped, 0, 'Skipped Kanana tests block release evidence.');
  assert.equal(report.stats?.unexpected, 0, 'Failed Kanana tests block release evidence.');
  assert.equal(report.stats?.flaky, 0, 'Flaky Kanana tests block release evidence.');
  assert.deepEqual(report.errors ?? [], [], 'Kanana Playwright report contains global errors.');
  assert.equal(tests.length, expectedCount, 'Kanana report must contain all required text, image, and offline cases.');

  const observed = new Set<string>();
  type KananaTestKind = keyof KananaEvidencePolicy['tests'];
  type KananaTestRequirement = KananaEvidencePolicy['tests'][KananaTestKind];
  const requirements = Object.entries(context.policy.tests) as Array<[KananaTestKind, KananaTestRequirement]>;
  for (const item of tests) {
    const match: [KananaTestKind, KananaTestRequirement] | undefined = requirements.find(
      ([, requirement]) =>
        item.file && resolve(reportRoot, item.file) === resolve(context.workspaceRoot, requirement.file)
    );
    if (!match) throw new Error(`Unexpected Kanana release GPU test file: ${item.file}.`);
    const kind: KananaTestKind = match[0];
    const requirement: KananaTestRequirement = context.policy.tests[kind];
    assert.equal(item.title, requirement.title, `Kanana ${kind} test title differs from policy.`);
    assert.equal(item.test.projectName, context.policy.browser.project, `Kanana ${kind} browser project mismatch.`);
    assert.ok(!observed.has(kind), `Duplicate Kanana ${kind} evidence.`);
    validateKananaRecord(recordFromTest(item.test, context.policy.evidencePrefix), kind, context);
    observed.add(kind);
  }
  for (const kind of Object.keys(context.policy.tests) as Array<keyof KananaEvidencePolicy['tests']>) {
    assert.ok(observed.has(kind), `Missing Kanana ${kind} release evidence.`);
  }
  return {caseCount: observed.size, browser: context.policy.browser.name, modelId: context.policy.model.modelId};
}
