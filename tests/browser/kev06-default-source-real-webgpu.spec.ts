import assert from 'node:assert/strict';
import {readFileSync, realpathSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, test} from '@playwright/test';
import type {DecisionRequest, DecisionResult, ModelProgress} from 'web-decision';
import {
  BASELINE_EXPECTED_OUTPUT,
  BASELINE_REFERENCE,
  collectReleaseEvidenceInputs,
  sha256,
  UPSTREAM_EXPECTED_OUTPUT,
  UPSTREAM_REFERENCE
} from '../../scripts/release/verify-release-evidence.ts';
import {
  collectWindowsHardwareEvidence,
  createInferenceEvidence,
  type ExpectedInference,
  type ObservedInference
} from './kev06-release-evidence.ts';

type AssetMetadata = Readonly<{bytes: number; sha256: string}>;
type VerifiedFile = AssetMetadata & Readonly<{path: string}>;
type Answer = DecisionResult['answers'][number];
type Baseline = Readonly<{
  scope: string;
  packageSha256: string;
  request: DecisionRequest;
  selected: string;
  probabilities: readonly Readonly<{option: string; probability: number; logit: number}>[];
}>;
type UpstreamReferenceFixture = Readonly<{
  scope: string;
  modelId: string;
  graphSha256: string;
  weightsSha256: string;
  logits: readonly number[];
  probabilities: readonly number[];
  maxAbsoluteLogitDifference: number;
  maxAbsoluteProbabilityDifference: number;
}>;
type InferenceSnapshot = Omit<ObservedInference, 'probabilities'> & {probabilities: Answer['probabilities']};
type BrowserDescriptor = Readonly<{modelId: string | null; backend: string | null; manifestSha256: string | null}>;
type OnlineResult = InferenceSnapshot & {
  adapterInfo: GPUAdapterInfo;
  adapterIsFallback: false;
  adapterFeatures: string[];
  adapterLimits: Readonly<Record<string, number>>;
  userAgent: string;
  sourceProgress: ModelProgress[];
  disposed: boolean;
  sourceReachedFullLength: boolean;
  verifiedFiles: VerifiedFile[];
};
type OfflineResult = InferenceSnapshot;

const enabled = process.env.WEB_DECISION_KEV06_DEFAULT_GPU === '1';
const modelId = 'kev-0.6b-q4f16';
const upstreamSourceBytes = 335_357_952;
const evidencePrefix = 'WEB_DECISION_RELEASE_EVIDENCE_V1:';
const progressPrefix = 'KEV06_DEFAULT_PROGRESS ';
const baselineFixtureBytes = readFileSync(new URL('../fixtures/kev06-main-baseline.json', import.meta.url));
const upstreamFixtureBytes = readFileSync(new URL('../fixtures/kev06-upstream-onnx-reference.json', import.meta.url));
const baselineFixtureSha256 = sha256(baselineFixtureBytes);
const upstreamFixtureSha256 = sha256(upstreamFixtureBytes);
const baseline = JSON.parse(baselineFixtureBytes.toString('utf8')) as Baseline;
const upstreamReference = JSON.parse(upstreamFixtureBytes.toString('utf8')) as UpstreamReferenceFixture;
const baselineQuestion = baseline.request.questions[0];
if (!baselineQuestion) throw new Error('The baseline fixture has no question.');
const baselineOptionOrder = baselineQuestion.options;
const upstreamPath =
  '/onnx-community/kev-0.6b-ONNX/resolve/5a827091b3d58fe4fdfc02a5a5b2ce364047d678/onnx/model_q4f16.onnx_data';

test('stable Kev model ID loads pinned upstream weights and emits measured release evidence', async ({
  page,
  browser,
  browserName
}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KEV06_DEFAULT_GPU=1 to run the stable Kev upstream WebGPU check.');
  test.setTimeout(20 * 60 * 1000);

  expect(browserName).toBe('chromium');
  assert.equal(
    baselineFixtureSha256,
    BASELINE_REFERENCE.fixtureSha256,
    'Baseline fixture digest differs from its pin.'
  );
  assert.equal(
    upstreamFixtureSha256,
    UPSTREAM_REFERENCE.fixtureSha256,
    'Upstream fixture digest differs from its pin.'
  );
  assert.equal(baseline.scope, BASELINE_REFERENCE.scope);
  assert.equal(baseline.packageSha256, BASELINE_REFERENCE.packageSha256);
  assert.equal(upstreamReference.scope, UPSTREAM_REFERENCE.scope);
  assert.equal(upstreamReference.modelId, UPSTREAM_REFERENCE.modelId);
  assert.equal(upstreamReference.graphSha256, UPSTREAM_REFERENCE.graphSha256);
  assert.equal(upstreamReference.weightsSha256, UPSTREAM_REFERENCE.weightsSha256);
  assert.equal(baseline.selected, BASELINE_EXPECTED_OUTPUT.selected);
  assert.deepEqual(baselineOptionOrder, BASELINE_EXPECTED_OUTPUT.optionOrder);
  assert.deepEqual(
    baseline.probabilities.map(({logit}) => logit),
    BASELINE_EXPECTED_OUTPUT.logits
  );
  assert.deepEqual(
    baseline.probabilities.map(({probability}) => probability),
    BASELINE_EXPECTED_OUTPUT.probabilities
  );
  assert.deepEqual(upstreamReference.logits, UPSTREAM_EXPECTED_OUTPUT.logits);
  assert.deepEqual(upstreamReference.probabilities, UPSTREAM_EXPECTED_OUTPUT.probabilities);
  assert.equal(upstreamReference.maxAbsoluteLogitDifference, UPSTREAM_EXPECTED_OUTPUT.maxAbsoluteLogitDifference);
  assert.equal(
    upstreamReference.maxAbsoluteProbabilityDifference,
    UPSTREAM_EXPECTED_OUTPUT.maxAbsoluteProbabilityDifference
  );

  const archivePath = requiredEnvironment('WEB_DECISION_RELEASE_ARCHIVE');
  const installedPackagePath = requiredEnvironment('WEB_DECISION_RELEASE_PACKAGE_DIR');
  const installedBundlePath = requiredEnvironment('WEB_DECISION_INSTALLED_BUNDLE');
  const workspaceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
  const inputs = await collectReleaseEvidenceInputs({
    archivePath,
    installedPackagePath,
    sourcePackageRoot: join(workspaceRoot, 'packages/web-decision')
  });
  assert.equal(realpathSync(resolve(installedPackagePath)), inputs.archive.installedRoot);
  assert.equal(realpathSync(resolve(installedBundlePath)), inputs.archive.bundlePath);

  const model = inputs.models.find((item) => item.modelId === modelId);
  if (!model) throw new Error(`The candidate archive does not contain curated model ${modelId}.`);
  const manifest = inputs.manifests.get(modelId);
  if (!manifest) throw new Error(`The candidate archive has no pinned manifest for ${modelId}.`);
  const manifestFiles = manifest.manifest.files;
  if (!manifestFiles) throw new Error(`The candidate archive manifest has no file map for ${modelId}.`);
  expect(Object.keys(manifestFiles)).toHaveLength(14);
  const browserHardware = collectWindowsHardwareEvidence();

  page.on('console', (message) => {
    const text = message.text();
    if (text.startsWith(progressPrefix)) console.log(text);
  });
  const huggingFaceGets: string[] = [];
  let blockHuggingFaceGets = false;
  const offlineRequests: string[] = [];
  await page.route('https://huggingface.co/**', async (route) => {
    if (route.request().method() === 'GET') {
      huggingFaceGets.push(route.request().url());
      if (blockHuggingFaceGets) return route.abort();
    }
    return route.continue();
  });
  await page.route('**/*', async (route) => {
    if (!blockHuggingFaceGets) return route.fallback();
    offlineRequests.push(route.request().url());
    return route.abort();
  });
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');

  const browserDescriptor = await page.evaluate((): BrowserDescriptor => {
    const descriptor = window.webDecision.describeModel('kev-0.6b-q4f16');
    return descriptor
      ? {
          modelId: descriptor.modelId,
          backend: descriptor.backend.name,
          manifestSha256: descriptor.assets.manifestSha256
        }
      : {modelId: null, backend: null, manifestSha256: null};
  });
  expect(browserDescriptor).toEqual({
    modelId,
    backend: 'tvm-webgpu',
    manifestSha256: model.manifestSha256
  });

  const online = await page.evaluate(
    async ({expectedSourceBytes, request, expectedFiles}) => {
      const reportProgress = (stage: string, details: Record<string, unknown> = {}) =>
        console.info(`KEV06_DEFAULT_PROGRESS ${JSON.stringify({stage, ...details})}`);
      reportProgress('adapter.request.started');
      if (!navigator.gpu) throw new Error('WebGPU is unavailable in Chromium.');
      const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
      if (!adapter) throw new Error('Chromium did not provide a WebGPU adapter.');
      if (!adapter.features.has('shader-f16')) throw new Error('The adapter does not support shader-f16.');
      const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
      const adapterIsFallback = adapterWithFallback.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
      if (adapterIsFallback !== false) {
        throw new Error(`Expected a non-fallback WebGPU adapter; got ${String(adapterIsFallback)}.`);
      }
      reportProgress('adapter.request.completed', {
        adapterInfo: adapter.info ?? {},
        adapterIsFallback,
        features: [...adapter.features]
      });
      const sourceProgress: ModelProgress[] = [];
      reportProgress('load.started', {modelId: 'kev-0.6b-q4f16'});
      const model = await window.webDecision.loadDecisionModel({
        modelId: 'kev-0.6b-q4f16',
        onProgress(event) {
          if (event.phase === 'upstream-source') {
            const progress = {
              loadedBytes: event.loadedBytes,
              totalBytes: event.totalBytes,
              fraction: event.fraction
            };
            sourceProgress.push(progress);
            reportProgress('source-progress', progress);
          }
        }
      });
      reportProgress('load.completed', {
        model: model.description.modelId,
        backend: model.description.backend.name
      });
      let result: OnlineResult | undefined;
      try {
        reportProgress('inference.started');
        const decision = await model.decide(request);
        const answer = decision.answers[0];
        if (!answer) throw new Error('The model returned no answer.');
        reportProgress('inference.completed', {
          selected: answer.selected,
          optionOrder: answer.probabilities.map(({option}) => option)
        });
        result = {
          modelId: model.description.modelId,
          backend: model.description.backend.name,
          assetManifestSha256: model.description.assets.manifestSha256,
          selected: answer.selected,
          probabilities: answer.probabilities,
          adapterInfo: adapter.info ?? {},
          adapterIsFallback,
          adapterFeatures: [...adapter.features],
          adapterLimits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
            maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
            maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension
          },
          userAgent: navigator.userAgent,
          sourceProgress,
          disposed: false,
          sourceReachedFullLength: false,
          verifiedFiles: []
        };
      } finally {
        reportProgress('dispose.started');
        await model.dispose();
        reportProgress('dispose.completed', {disposed: true});
      }
      if (!result) throw new Error('The model returned no result.');
      result.disposed = true;
      const finalSourceProgress = sourceProgress[sourceProgress.length - 1];
      result.sourceReachedFullLength =
        finalSourceProgress?.loadedBytes === expectedSourceBytes &&
        finalSourceProgress.totalBytes === expectedSourceBytes &&
        finalSourceProgress.fraction === 1;
      const cache = await caches.open('web-decision-assets-v1');
      for (const [path, metadata] of Object.entries(expectedFiles)) {
        const key = new URL('https://kev-assets.web-decision.invalid/.web-decision-cache/sha256/');
        key.searchParams.set('digest', metadata.sha256);
        const response = await cache.match(key.href);
        if (!response) throw new Error(`The cached model file is missing: ${path}`);
        const bytes = await response.arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        const sha256 = [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
        result.verifiedFiles.push({path, bytes: bytes.byteLength, sha256});
      }
      return result;
    },
    {expectedSourceBytes: upstreamSourceBytes, request: baseline.request, expectedFiles: manifestFiles}
  );

  expect(huggingFaceGets.length).toBeGreaterThan(0);
  expect(huggingFaceGets.every((url) => new URL(url).pathname === upstreamPath)).toBe(true);
  expect(online.modelId).toBe(modelId);
  expect(online.backend).toBe('tvm-webgpu');
  expect(online.assetManifestSha256).toBe(model.manifestSha256);
  expect(online.adapterIsFallback).toBe(false);
  expect(online.sourceReachedFullLength).toBe(true);
  expect(online.sourceProgress.length).toBeGreaterThan(0);
  expect(online.disposed).toBe(true);
  expect(online.verifiedFiles).toHaveLength(14);
  const measuredFiles = [...online.verifiedFiles].sort((left, right) => left.path.localeCompare(right.path));
  const expectedFiles = Object.entries(manifestFiles)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([path, metadata]) => ({path, bytes: metadata.bytes, sha256: metadata.sha256}));
  expect(measuredFiles).toEqual(expectedFiles);
  expect(online.probabilities.map(({option}) => option)).toEqual(baselineOptionOrder);

  const selectedProbability = online.probabilities.find(({option}) => option === online.selected)?.probability;
  expect(selectedProbability).toBeGreaterThanOrEqual(
    Math.max(...online.probabilities.map(({probability}) => probability)) - 1e-6
  );

  const onlineHuggingFaceGets = [...huggingFaceGets];
  huggingFaceGets.length = 0;
  blockHuggingFaceGets = true;
  const offline = await page.evaluate(async (request): Promise<OfflineResult> => {
    const reportProgress = (stage: string, details: Record<string, unknown> = {}) =>
      console.info(`KEV06_DEFAULT_PROGRESS ${JSON.stringify({stage, ...details})}`);
    reportProgress('offline.reload.started', {modelId: 'kev-0.6b-q4f16'});
    const model = await window.webDecision.loadDecisionModel({modelId: 'kev-0.6b-q4f16', offline: true});
    reportProgress('offline.reload.loaded', {
      model: model.description.modelId,
      backend: model.description.backend.name
    });
    try {
      const decision = await model.decide(request);
      const answer = decision.answers[0];
      if (!answer) throw new Error('The model returned no offline answer.');
      return {
        modelId: model.description.modelId,
        backend: model.description.backend.name,
        assetManifestSha256: model.description.assets.manifestSha256,
        selected: answer.selected,
        probabilities: answer.probabilities
      };
    } finally {
      reportProgress('offline.dispose.started');
      await model.dispose();
      reportProgress('offline.reload.completed', {disposed: true});
    }
  }, baseline.request);

  const offlineNetworkAttempts = [...offlineRequests];
  expect(offlineNetworkAttempts).toEqual([]);
  expect(huggingFaceGets).toEqual([]);
  expect(offline.modelId).toBe(modelId);
  expect(offline.backend).toBe('tvm-webgpu');
  expect(offline.assetManifestSha256).toBe(online.assetManifestSha256);

  const onlineInference: ObservedInference = online;
  const offlineInference: ObservedInference = offline;
  const baselineExpected: ExpectedInference = {
    selected: baseline.selected,
    optionOrder: baselineOptionOrder,
    logits: baseline.probabilities.map(({logit}) => logit),
    probabilities: baseline.probabilities.map(({probability}) => probability),
    maxAbsoluteLogitDifference: BASELINE_EXPECTED_OUTPUT.maxAbsoluteLogitDifference,
    maxAbsoluteProbabilityDifference: BASELINE_EXPECTED_OUTPUT.maxAbsoluteProbabilityDifference
  };
  const upstreamExpected: ExpectedInference = {
    optionOrder: baselineOptionOrder,
    logits: upstreamReference.logits,
    probabilities: upstreamReference.probabilities,
    maxAbsoluteLogitDifference: upstreamReference.maxAbsoluteLogitDifference,
    maxAbsoluteProbabilityDifference: upstreamReference.maxAbsoluteProbabilityDifference
  };
  const inference = createInferenceEvidence(
    onlineInference,
    offlineInference,
    offlineNetworkAttempts,
    {
      scope: baseline.scope,
      packageSha256: baseline.packageSha256,
      fixtureSha256: baselineFixtureSha256
    },
    {
      scope: upstreamReference.scope,
      fixtureSha256: upstreamFixtureSha256,
      modelId: upstreamReference.modelId,
      graphSha256: upstreamReference.graphSha256,
      weightsSha256: upstreamReference.weightsSha256
    },
    baselineExpected,
    upstreamExpected
  );
  expect(inference.online.referenceMatched).toBe(true);
  expect(inference.upstreamReference.matched).toBe(true);
  expect(inference.offline.matchesOnline).toBe(true);
  expect(onlineHuggingFaceGets.length).toBeGreaterThan(0);

  const record = {
    schemaVersion: 1,
    testKind: 'real-model-webgpu',
    catalogSha256: inputs.catalogSha256,
    model: {
      requestedId: modelId,
      loadedId: online.modelId,
      describedId: browserDescriptor.modelId,
      assetManifestSha256: online.assetManifestSha256
    },
    artifact: {
      modelId,
      manifestPath: manifest.path,
      manifestSha256: manifest.manifestSha256,
      filesSha256: manifest.filesSha256,
      manifest: manifest.manifest,
      verifiedFiles: measuredFiles
    },
    archive: {
      sha256: inputs.archive.archiveSha256,
      packageTreeSha256: inputs.archive.packageTreeSha256,
      bundleSha256: inputs.archive.bundleSha256,
      installedPackagePathSha256: inputs.archive.installedPackagePathSha256,
      entryPathSha256: inputs.archive.entryPathSha256,
      packageName: inputs.archive.packageName,
      packageVersion: inputs.archive.packageVersion
    },
    browser: {name: browserName, version: browser.version(), userAgent: online.userAgent},
    device: browserHardware.device,
    gpu: {
      api: 'webgpu',
      isFallbackAdapter: online.adapterIsFallback,
      adapterInfo: {source: 'navigator.gpu.requestAdapter().info', value: online.adapterInfo},
      hardware: browserHardware.gpu.hardware,
      hardwareEvidence: browserHardware.gpu.hardwareEvidence,
      features: online.adapterFeatures,
      limits: online.adapterLimits
    },
    inference
  };

  for (const [feature, minimum] of Object.entries(model.backend.requiredLimits)) {
    expect(online.adapterLimits[feature], `WebGPU ${feature} limit`).toBeGreaterThanOrEqual(minimum);
  }
  for (const feature of model.backend.requiredFeatures) expect(online.adapterFeatures).toContain(feature);
  console.log(`${evidencePrefix}${JSON.stringify(record)}`);
});

function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Set ${name} to the exact release candidate input.`);
  return value;
}
