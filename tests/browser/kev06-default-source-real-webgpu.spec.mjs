import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KEV06_DEFAULT_GPU === '1';
const modelId = 'kev-0.6b-q4f16';
const upstreamSourceBytes = 335_357_952;
const progressPrefix = 'KEV06_DEFAULT_PROGRESS ';
const upstreamPath =
  '/onnx-community/kev-0.6b-ONNX/resolve/5a827091b3d58fe4fdfc02a5a5b2ce364047d678/onnx/model_q4f16.onnx_data';

test('stable Kev model ID loads pinned weights, checks aggregate resource limits and typed options, and infers offline', async ({
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
        const baseRequest = {
          state: 'A customer was charged twice.',
          questions: [
            {
              instructions: 'Which team should handle this request?',
              options: ['billing', 'shipping']
            }
          ]
        };
        reportProgress('inference.base.started');
        const decision = await model.decide(baseRequest);
        const answer = decision.answers[0];
        const base = {
          tokenCount: decision.tokenCount,
          selected: answer.selected,
          optionOrder: answer.probabilities.map(({option}) => option),
          probabilities: answer.probabilities
        };
        reportProgress('inference.base.completed', {
          tokenCount: base.tokenCount,
          selected: base.selected,
          optionOrder: base.optionOrder
        });

        const tokenLength = (text) => model.tokenizer.encode(text, {add_special_tokens: false}).ids.length;
        const exactTokenText = (target) => {
          const pieces = [' the', ' a', ' to', ' of', ' is', ' x', ' q', ' 0', ' yes', ' no', '. ', '! ', '\n'];
          for (const piece of pieces) {
            const candidate = piece.repeat(target);
            if (tokenLength(candidate) === target) return candidate;
          }
          let text = '';
          let count = tokenLength(text);
          while (count < target) {
            let nextText;
            for (const piece of pieces) {
              const candidate = text + piece;
              if (tokenLength(candidate) === count + 1) {
                nextText = candidate;
                break;
              }
            }
            if (nextText === undefined) throw new Error(`Could not construct exactly ${target} tokenizer tokens.`);
            text = nextText;
            count++;
          }
          return text;
        };
        const aggregateBaseline = await model.decide({
          state: '',
          questions: [{instructions: 'Choose.', options: ['no', 'yes']}]
        });
        const stateMarkerTokens = 1;
        const branchFixedTokens = aggregateBaseline.tokenCount - stateMarkerTokens - tokenLength('Choose.');

        const rowRequestFor = (rowTokenTarget) => ({
          state: '',
          questions: [
            {
              instructions: exactTokenText(rowTokenTarget - stateMarkerTokens - branchFixedTokens),
              options: ['no', 'yes']
            }
          ]
        });
        const exactRowTarget = 8192;
        const exactRowRequest = rowRequestFor(exactRowTarget);
        const exactRowQuestion = exactRowRequest.questions[0];
        const exactRowTokenCount = stateMarkerTokens + branchFixedTokens + tokenLength(exactRowQuestion.instructions);
        reportProgress('inference.question-row-8192.started', {rowTokenCount: exactRowTokenCount});
        const exactRowDecision = await model.decide(exactRowRequest);
        const exactRowAnswer = exactRowDecision.answers[0];
        const exactRow = {
          rowTokenCount: exactRowTokenCount,
          tokenCount: exactRowDecision.tokenCount,
          selected: exactRowAnswer.selected,
          optionOrder: exactRowAnswer.probabilities.map(({option}) => option)
        };
        reportProgress('inference.question-row-8192.completed', {
          rowTokenCount: exactRow.rowTokenCount,
          tokenCount: exactRow.tokenCount,
          selected: exactRow.selected
        });

        const overRowTarget = exactRowTarget + 1;
        const overRowRequest = rowRequestFor(overRowTarget);
        const overRowQuestion = overRowRequest.questions[0];
        const overRowTokenCount = stateMarkerTokens + branchFixedTokens + tokenLength(overRowQuestion.instructions);
        reportProgress('inference.question-row-8193.started', {rowTokenCount: overRowTokenCount});
        let overRow;
        try {
          await model.decide(overRowRequest);
          overRow = {code: 'UNEXPECTED_SUCCESS'};
        } catch (error) {
          overRow = {code: error.code ?? error.name, message: error.message};
        }
        overRow.rowTokenCount = overRowTokenCount;
        reportProgress('inference.question-row-8193.completed', overRow);

        // The empty state contributes one marker, so each 8,191-token branch stays within the 8,192 row budget.
        const branchTokenTargets = [8191, 8191, 8191, 4097, 4098];
        const aggregateRequest = {
          state: '',
          questions: branchTokenTargets.map((target) => ({
            instructions: exactTokenText(target - branchFixedTokens),
            options: ['no', 'yes']
          }))
        };
        const branchTokenCounts = aggregateRequest.questions.map(
          ({instructions}) => branchFixedTokens + tokenLength(instructions)
        );
        const packedTokenCount = 1 + branchTokenCounts.reduce((sum, count) => sum + count, 0);
        const attentionBufferBytes = packedTokenCount * packedTokenCount * 32;
        const deviceAttentionLimit = Math.min(
          model.gpu.limits.maxBufferSize,
          model.gpu.limits.maxStorageBufferBindingSize
        );
        if (attentionBufferBytes <= deviceAttentionLimit)
          throw new Error(
            'The loaded WebGPU device can fit this request; the M4 resource-preflight check is not applicable.'
          );
        reportProgress('inference.aggregate-resource-preflight.started', {
          packedTokenCount,
          questionCount: aggregateRequest.questions.length,
          branchTokenCounts,
          attentionBufferBytes,
          deviceAttentionLimit
        });
        let aggregateResource;
        try {
          await model.decide(aggregateRequest);
          aggregateResource = {code: 'UNEXPECTED_SUCCESS'};
        } catch (error) {
          aggregateResource = {code: error.code ?? error.name, message: error.message};
        }
        Object.assign(aggregateResource, {
          packedTokenCount,
          questionCount: aggregateRequest.questions.length,
          branchTokenCounts,
          attentionBufferBytes,
          deviceAttentionLimit
        });
        reportProgress('inference.aggregate-resource-preflight.completed', aggregateResource);

        const criteria = Object.fromEntries(Array.from({length: 255}, (_, index) => [`option-${index}`, null]));
        reportProgress('inference.typed-255-options.started', {optionCount: 255});
        const typedResult = await model.systemOne('A customer asks for a refund.', {
          route: {type: 'choice', instructions: 'Choose the appropriate action.', criteria},
          refund: {type: 'noul', instructions: 'Is the customer asking for a refund?'},
          priority: {type: 'score', instructions: 'Rate the urgency.', criteria: ['low', 'medium', 'high']}
        });
        const choice = typedResult.answers.route;
        const noul = typedResult.answers.refund;
        const score = typedResult.answers.priority;
        const typed = {
          model: typedResult.model,
          tokenCount: typedResult.tokenCount,
          choice: {
            type: choice.type,
            selected: choice.choice,
            optionOrder: choice.probabilityEntries.map(([option]) => option),
            probabilitiesFinite: choice.probabilityEntries.every(([, probability]) => Number.isFinite(probability)),
            probabilityTotal: choice.probabilityEntries.reduce((sum, [, probability]) => sum + probability, 0),
            confidence: choice.confidence
          },
          noul: {type: noul.type, probability: noul.noul},
          score: {
            type: score.type,
            value: score.score,
            legend: score.legend,
            probabilityCount: score.probabilityEntries.length,
            probabilitiesFinite: score.probabilityEntries.every(([, probability]) => Number.isFinite(probability)),
            probabilityTotal: score.probabilityEntries.reduce((sum, [, probability]) => sum + probability, 0)
          }
        };
        reportProgress('inference.typed-255-options.completed', {
          tokenCount: typed.tokenCount,
          choiceOptionCount: typed.choice.optionOrder.length,
          noulType: typed.noul.type,
          scoreType: typed.score.type
        });

        result = {
          model: model.description.model,
          backend: model.description.backend.name,
          adapterInfo: adapter.info ?? {},
          adapterIsFallback,
          selected: base.selected,
          optionOrder: base.optionOrder,
          probabilities: base.probabilities,
          base,
          rowBoundary: {exact: exactRow, over: overRow},
          aggregateResource,
          typed,
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
  expect(online.rowBoundary.exact.rowTokenCount).toBe(8192);
  expect(online.rowBoundary.exact.tokenCount).toBe(8192);
  expect(online.rowBoundary.exact.optionOrder).toEqual(['no', 'yes']);
  expect(['no', 'yes']).toContain(online.rowBoundary.exact.selected);
  expect(online.rowBoundary.over.rowTokenCount).toBe(8193);
  expect(online.rowBoundary.over.code).toBe('INVALID_INPUT');
  expect(online.aggregateResource.packedTokenCount).toBe(32769);
  expect(online.aggregateResource.questionCount).toBe(5);
  expect(online.aggregateResource.branchTokenCounts).toEqual([8191, 8191, 8191, 4097, 4098]);
  expect(online.aggregateResource.branchTokenCounts.every((count) => count <= 8191)).toBe(true);
  expect(online.aggregateResource.attentionBufferBytes).toBe(34_361_835_552);
  expect(online.aggregateResource.attentionBufferBytes).toBeGreaterThan(online.aggregateResource.deviceAttentionLimit);
  expect(online.aggregateResource.code).toBe('DEVICE_RESOURCE_LIMIT');
  expect(online.typed.model).toBe(modelId);
  expect(online.typed.tokenCount).toBeGreaterThan(0);
  expect(online.typed.choice.type).toBe('choice');
  expect(online.typed.choice.optionOrder).toEqual(Array.from({length: 255}, (_, index) => `option-${index}`));
  expect(online.typed.choice.probabilitiesFinite).toBe(true);
  expect(Math.abs(online.typed.choice.probabilityTotal - 1)).toBeLessThanOrEqual(255 * 0.00005 + 1e-8);
  expect(online.typed.choice.optionOrder).toContain(online.typed.choice.selected);
  expect(Number.isFinite(online.typed.choice.confidence)).toBe(true);
  expect(online.typed.choice.confidence).toBeGreaterThanOrEqual(0);
  expect(online.typed.choice.confidence).toBeLessThanOrEqual(1);
  expect(online.typed.noul.type).toBe('noul');
  expect(Number.isFinite(online.typed.noul.probability)).toBe(true);
  expect(online.typed.noul.probability).toBeGreaterThanOrEqual(0);
  expect(online.typed.noul.probability).toBeLessThanOrEqual(1);
  expect(online.typed.score.type).toBe('score');
  expect(Number.isFinite(online.typed.score.value)).toBe(true);
  expect(online.typed.score.legend).toEqual({0: 'low', 1: 'medium', 2: 'high'});
  expect(online.typed.score.probabilityCount).toBe(3);
  expect(online.typed.score.probabilitiesFinite).toBe(true);
  expect(online.typed.score.probabilityTotal).toBeCloseTo(1, 3);

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
      reportProgress('offline.inference.started');
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
      const inference = {
        tokenCount: decision.tokenCount,
        selected: answer.selected,
        optionOrder: answer.probabilities.map(({option}) => option),
        probabilities: answer.probabilities
      };
      reportProgress('offline.inference.completed', {
        tokenCount: inference.tokenCount,
        selected: inference.selected,
        optionOrder: inference.optionOrder
      });
      return {model: model.description.model, backend: model.description.backend.name, inference};
    } finally {
      reportProgress('offline.dispose.started');
      await model.dispose();
      reportProgress('offline.reload.completed', {disposed: model.closed});
    }
  });

  expect(huggingFaceGets).toEqual([]);
  expect(offline.model).toBe(modelId);
  expect(offline.backend).toBe('tvm-webgpu');
  expect(offline.inference.optionOrder).toEqual(online.optionOrder);
  expect(offline.inference.selected).toBe(online.selected);
  expect(offline.inference.probabilities.map(({probability}) => probability)).toEqual(
    online.probabilities.map(({probability}) => probability)
  );
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
