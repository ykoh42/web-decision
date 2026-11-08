import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KEV06_DEFAULT_GPU === '1';
const modelId = 'kev-0.6b-q4f16';
const upstreamSourceBytes = 335_357_952;
const progressPrefix = 'KEV06_DEFAULT_PROGRESS ';
const upstreamPath =
  '/onnx-community/kev-0.6b-ONNX/resolve/5a827091b3d58fe4fdfc02a5a5b2ce364047d678/onnx/model_q4f16.onnx_data';

test('stable Kev model ID loads pinned upstream weights and then reloads offline from the installed Chromium bundle', async ({
  page,
  browserName
}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KEV06_DEFAULT_GPU=1 to run the stable Kev upstream WebGPU check.');
  test.setTimeout(20 * 60 * 1000);

  expect(browserName).toBe('chromium');
  page.on('console', (message) => {
    const text = message.text();
    if (text.startsWith(progressPrefix)) console.log(text);
  });
  const huggingFaceGets = [];
  let blockHuggingFaceGets = false;
  await page.route('https://huggingface.co/**', async (route) => {
    if (route.request().method() === 'GET') {
      huggingFaceGets.push(route.request().url());
      if (blockHuggingFaceGets) return route.abort();
    }
    return route.continue();
  });
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');

  const online = await page.evaluate(
    async ({expectedSourceBytes}) => {
      const reportProgress = (stage, details = {}) =>
        console.info(`KEV06_DEFAULT_PROGRESS ${JSON.stringify({stage, ...details})}`);
      reportProgress('adapter.request.started');
      if (!navigator.gpu) throw new Error('WebGPU is unavailable in Chromium.');
      const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
      if (!adapter) throw new Error('Chromium did not provide a WebGPU adapter.');
      if (!adapter.features.has('shader-f16')) throw new Error('The adapter does not support shader-f16.');
      const adapterIsFallback = adapter.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
      if (adapterIsFallback !== false)
        throw new Error(`Expected a non-fallback WebGPU adapter; got ${String(adapterIsFallback)}.`);
      reportProgress('adapter.request.completed', {
        adapterInfo: adapter.info ?? {},
        adapterIsFallback,
        features: [...adapter.features]
      });
      const sourceProgress = [];
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
        model: model.description.model,
        backend: model.description.backend.name
      });
      let result;
      try {
        reportProgress('inference.started');
        const decision = await model.decide({
          state: 'A customer was charged twice.',
          questions: [
            {
              instructions: 'Which team should handle this request?',
              options: ['billing', 'shipping']
            }
          ]
        });
        const answer = decision.answers[0];
        reportProgress('inference.completed', {
          selected: answer.selected,
          optionOrder: answer.probabilities.map(({option}) => option)
        });
        result = {
          model: model.description.model,
          backend: model.description.backend.name,
          adapterInfo: adapter.info ?? {},
          adapterIsFallback,
          selected: answer.selected,
          optionOrder: answer.probabilities.map(({option}) => option),
          probabilities: answer.probabilities,
          sourceProgress
        };
      } finally {
        reportProgress('dispose.started');
        await model.dispose();
        reportProgress('dispose.completed', {disposed: model.closed});
      }
      result.disposed = model.closed;
      const finalSourceProgress = sourceProgress[sourceProgress.length - 1];
      result.sourceReachedFullLength =
        finalSourceProgress?.loadedBytes === expectedSourceBytes &&
        finalSourceProgress.totalBytes === expectedSourceBytes &&
        finalSourceProgress.fraction === 1;
      return result;
    },
    {expectedSourceBytes: upstreamSourceBytes}
  );

  expect(huggingFaceGets.length).toBeGreaterThan(0);
  expect(huggingFaceGets.every((url) => new URL(url).pathname === upstreamPath)).toBe(true);
  expect(online.model).toBe(modelId);
  expect(online.backend).toBe('tvm-webgpu');
  expect(online.adapterIsFallback).toBe(false);
  expect(online.sourceReachedFullLength).toBe(true);
  expect(online.sourceProgress.length).toBeGreaterThan(0);
  expect(online.disposed).toBe(true);
  expect(online.optionOrder).toEqual(['billing', 'shipping']);
  expect(['billing', 'shipping']).toContain(online.selected);
  const selectedProbability = online.probabilities.find(({option}) => option === online.selected)?.probability;
  expect(selectedProbability).toBeGreaterThanOrEqual(
    Math.max(...online.probabilities.map(({probability}) => probability)) - 1e-6
  );

  const onlineHuggingFaceGets = [...huggingFaceGets];
  huggingFaceGets.length = 0;
  blockHuggingFaceGets = true;
  const offline = await page.evaluate(async () => {
    const reportProgress = (stage, details = {}) =>
      console.info(`KEV06_DEFAULT_PROGRESS ${JSON.stringify({stage, ...details})}`);
    reportProgress('offline.reload.started', {modelId: 'kev-0.6b-q4f16'});
    const model = await window.webDecision.loadDecisionModel({modelId: 'kev-0.6b-q4f16', offline: true});
    reportProgress('offline.reload.loaded', {
      model: model.description.model,
      backend: model.description.backend.name
    });
    try {
      return {model: model.description.model, backend: model.description.backend.name};
    } finally {
      reportProgress('offline.dispose.started');
      await model.dispose();
      reportProgress('offline.reload.completed', {disposed: model.closed});
    }
  });

  expect(huggingFaceGets).toEqual([]);
  expect(offline.model).toBe(modelId);
  expect(offline.backend).toBe('tvm-webgpu');
  console.log(
    JSON.stringify(
      {
        modelId,
        upstreamPath,
        upstreamSourceBytes,
        online,
        onlineHuggingFaceGets,
        offline,
        offlineHuggingFaceGets: huggingFaceGets
      },
      null,
      2
    )
  );
});
