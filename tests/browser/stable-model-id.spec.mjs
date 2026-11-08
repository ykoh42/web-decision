import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_STABLE_MODEL_ID_REAL === '1';

test.skip(!enabled, 'Set WEB_DECISION_STABLE_MODEL_ID_REAL=1 to load the pinned TAMEV Micro model.');

test('stable model ID loads pinned assets without an assetsUrl', async ({page}) => {
  test.setTimeout(180_000);
  await page.goto('/');
  const report = await page.evaluate(async () => {
    const {describeStableModel, listStableModelIds, loadDecisionModel} = window.webDecision;
    const modelId = 'tamev-micro-minilm-int8';
    const model = await loadDecisionModel({modelId, cache: false});
    try {
      const result = await model.systemOne('A customer was charged twice.', {
        route: {
          type: 'choice',
          instructions: 'Which queue should handle this?',
          criteria: {billing: 'review duplicate charges', support: 'fix an app problem'}
        }
      });
      return {
        modelId,
        catalog: listStableModelIds(),
        descriptionId: model.description.model,
        describedId: describeStableModel(modelId).model,
        resultId: result.model,
        selected: result.answers.route.choice
      };
    } finally {
      await model.dispose();
    }
  });

  expect(report.catalog).toEqual(['tamev-micro-minilm-int8', 'tamev-nano-tinybert-int8', 'kev-0.6b-q4f16']);
  expect(report.descriptionId).toBe(report.modelId);
  expect(report.describedId).toBe(report.modelId);
  expect(report.resultId).toBe(report.modelId);
  expect(['billing', 'support']).toContain(report.selected);
});
