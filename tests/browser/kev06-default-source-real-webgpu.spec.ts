import {expect, test} from '@playwright/test';
import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import type {DecisionRequest, DecisionResult, ModelProgress} from 'web-decision';

type AssetMetadata = Readonly<{bytes: number; sha256: string}>;
type VerifiedFile = AssetMetadata & Readonly<{path: string}>;
type Answer = DecisionResult['answers'][number];
type Baseline = Readonly<{
  request: DecisionRequest;
  selected: string;
  probabilities: readonly Readonly<{option: string; probability: number; logit: number}>[];
}>;
type UpstreamReference = Readonly<{
  logits: readonly number[];
  probabilities: readonly number[];
  maxAbsoluteLogitDifference: number;
  maxAbsoluteProbabilityDifference: number;
}>;
type OnlineResult = {
  model: string;
  backend: string;
  adapterInfo: GPUAdapterInfo;
  adapterIsFallback: false;
  adapterFeatures: string[];
  adapterLimits: Readonly<Record<string, number>>;
  userAgent: string;
  selected: string;
  optionOrder: string[];
  probabilities: Answer['probabilities'];
  sourceProgress: ModelProgress[];
  disposed: boolean;
  sourceReachedFullLength: boolean;
  verifiedFiles: VerifiedFile[];
};

const enabled = process.env.WEB_DECISION_KEV06_DEFAULT_GPU === '1';
const modelId = 'kev-0.6b-q4f16';
const upstreamSourceBytes = 335_357_952;
const progressPrefix = 'KEV06_DEFAULT_PROGRESS ';
const baseline = JSON.parse(
  readFileSync(new URL('../fixtures/kev06-main-baseline.json', import.meta.url), 'utf8')
) as Baseline;
const upstreamReference = JSON.parse(
  readFileSync(new URL('../fixtures/kev06-upstream-onnx-reference.json', import.meta.url), 'utf8')
) as UpstreamReference;
const upstreamPath =
  '/onnx-community/kev-0.6b-ONNX/resolve/5a827091b3d58fe4fdfc02a5a5b2ce364047d678/onnx/model_q4f16.onnx_data';

test('stable Kev model ID loads pinned upstream weights and then reloads offline from the installed Chromium bundle', async ({
  page,
  browserName
}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KEV06_DEFAULT_GPU=1 to run the stable Kev upstream WebGPU check.');
  test.setTimeout(20 * 60 * 1000);

  expect(browserName).toBe('chromium');
  const installedBundle = process.env.WEB_DECISION_INSTALLED_BUNDLE;
  if (!installedBundle) throw new Error('Use a clean consumer installation of the candidate archive.');
  const installedPackage = dirname(dirname(installedBundle));
  const manifest = JSON.parse(readFileSync(join(installedPackage, 'assets/kev06-stable/manifest.json'), 'utf8')) as {
    files: Readonly<Record<string, AssetMetadata>>;
  };
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
      if (adapterIsFallback !== false)
        throw new Error(`Expected a non-fallback WebGPU adapter; got ${String(adapterIsFallback)}.`);
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
          model: model.description.modelId,
          backend: model.description.backend.name,
          adapterInfo: adapter.info ?? {},
          adapterIsFallback,
          adapterFeatures: [...adapter.features],
          adapterLimits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize
          },
          userAgent: navigator.userAgent,
          selected: answer.selected,
          optionOrder: answer.probabilities.map(({option}) => option),
          probabilities: answer.probabilities,
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
      result.verifiedFiles = [];
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
    {expectedSourceBytes: upstreamSourceBytes, request: baseline.request, expectedFiles: manifest.files}
  );

  expect(huggingFaceGets.length).toBeGreaterThan(0);
  expect(huggingFaceGets.every((url) => new URL(url).pathname === upstreamPath)).toBe(true);
  expect(online.model).toBe(modelId);
  expect(online.backend).toBe('tvm-webgpu');
  expect(online.adapterIsFallback).toBe(false);
  expect(online.sourceReachedFullLength).toBe(true);
  expect(online.sourceProgress.length).toBeGreaterThan(0);
  expect(online.disposed).toBe(true);
  expect(online.verifiedFiles).toEqual(
    Object.entries(manifest.files).map(([path, metadata]) => ({path, bytes: metadata.bytes, sha256: metadata.sha256}))
  );
  expect(online.optionOrder).toEqual(['billing', 'shipping']);
  expect(['billing', 'shipping']).toContain(online.selected);
  expect(online.selected).toBe(baseline.selected);
  for (const [index, option] of online.probabilities.entries()) {
    const expected = baseline.probabilities[index];
    const upstreamLogit = upstreamReference.logits[index];
    const upstreamProbability = upstreamReference.probabilities[index];
    if (!expected || upstreamLogit === undefined || upstreamProbability === undefined)
      throw new Error('The reference has fewer options than the model result.');
    expect(option.option).toBe(expected.option);
    expect(option.logit).toBeCloseTo(expected.logit, 1);
    expect(option.probability).toBeCloseTo(expected.probability, 3);
    expect(Math.abs(option.logit - upstreamLogit)).toBeLessThanOrEqual(upstreamReference.maxAbsoluteLogitDifference);
    expect(Math.abs(option.probability - upstreamProbability)).toBeLessThanOrEqual(
      upstreamReference.maxAbsoluteProbabilityDifference
    );
  }
  const selectedProbability = online.probabilities.find(({option}) => option === online.selected)?.probability;
  expect(selectedProbability).toBeGreaterThanOrEqual(
    Math.max(...online.probabilities.map(({probability}) => probability)) - 1e-6
  );

  const onlineHuggingFaceGets = [...huggingFaceGets];
  huggingFaceGets.length = 0;
  blockHuggingFaceGets = true;
  const offline = await page.evaluate(async (request) => {
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
        model: model.description.modelId,
        backend: model.description.backend.name,
        answer
      };
    } finally {
      reportProgress('offline.dispose.started');
      await model.dispose();
      reportProgress('offline.reload.completed', {disposed: true});
    }
  }, baseline.request);

  expect(huggingFaceGets).toEqual([]);
  expect(offlineRequests).toEqual([]);
  expect(offline.model).toBe(modelId);
  expect(offline.backend).toBe('tvm-webgpu');
  expect(offline.answer.selected).toBe(online.selected);
  expect(offline.answer.probabilities.map(({option}) => option)).toEqual(online.optionOrder);
  for (const [index, option] of offline.answer.probabilities.entries()) {
    const onlineOption = online.probabilities[index];
    if (!onlineOption) throw new Error('The offline result has more options than the online result.');
    expect(option.probability).toBeCloseTo(onlineOption.probability, 5);
  }
  console.log(
    JSON.stringify(
      {
        modelId,
        upstreamPath,
        upstreamSourceBytes,
        online,
        onlineHuggingFaceGets,
        offline,
        offlineHuggingFaceGets: huggingFaceGets,
        offlineRequests
      },
      null,
      2
    )
  );
});
