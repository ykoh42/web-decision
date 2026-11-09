import {expect, test} from '@playwright/test';
import type {
  LoadedStableTamevModel,
  StableTamevModelId,
  TamevModelName
} from '../../packages/web-decision/src/index.d.ts';

const enabled = process.env.WEB_DECISION_STABLE_MODEL_ID_REAL === '1';
const tamevModelIds = [
  'tamev-micro-minilm-int8',
  'tamev-nano-tinybert-int8',
  'tamev-nano-flash-tinybert-int8'
] as const;

for (const modelId of tamevModelIds) {
  test(`stable model ID ${modelId} loads pinned assets without an assetsUrl`, async ({page}) => {
    test.skip(!enabled, 'Set WEB_DECISION_STABLE_MODEL_ID_REAL=1 to load a pinned TAMEV model by stable ID.');
    if (!process.env.WEB_DECISION_INSTALLED_BUNDLE) {
      throw new Error(
        'The legacy TAMEV browser test requires WEB_DECISION_INSTALLED_BUNDLE to point to a compatible legacy bundle.'
      );
    }
    test.setTimeout(180_000);
    await page.goto('/');
    const report = await page.evaluate(async (modelId) => {
      const {describeStableModel, listStableModelIds, loadDecisionModel} =
        window.webDecision as unknown as typeof import('../../packages/web-decision/src/index.d.ts');
      const model = (await loadDecisionModel({modelId, cache: false})) as LoadedStableTamevModel<
        StableTamevModelId,
        TamevModelName
      >;
      try {
        const result = await model.systemOne('A customer was charged twice.', {
          route: {
            type: 'choice',
            instructions: 'Which queue should handle this?',
            criteria: {billing: 'review duplicate charges', support: 'fix an app problem'}
          },
          refund: {type: 'noul', instructions: 'Is the customer asking for a refund?'},
          priority: {
            type: 'score',
            instructions: 'How urgent is the issue?',
            criteria: ['low', 'medium', 'high']
          }
        });
        const route = result.answers.route;
        const refund = result.answers.refund;
        const priority = result.answers.priority;
        if (route?.type !== 'choice' || refund?.type !== 'noul' || priority?.type !== 'score') {
          throw new Error('Legacy TAMEV returned unexpected answer types.');
        }
        return {
          modelId,
          catalog: listStableModelIds(),
          descriptionId: model.description.model,
          describedId: describeStableModel(modelId).model,
          backend: model.description.backend.name,
          resultId: result.model,
          answerOrder: Object.keys(result.answers),
          choiceType: route.type,
          choice: route.choice,
          choiceOptionOrder: Object.keys(route.probabilities),
          choiceProbabilities: Object.entries(route.probabilities),
          choiceProbabilityTotal: Object.values(route.probabilities).reduce(
            (total, probability) => total + probability,
            0
          ),
          noulType: refund.type,
          noul: refund.noul,
          scoreType: priority.type,
          score: priority.score,
          scoreOptionOrder: Object.keys(priority.probabilities),
          scoreProbabilities: Object.entries(priority.probabilities),
          scoreProbabilityTotal: Object.values(priority.probabilities).reduce(
            (total, probability) => total + probability,
            0
          ),
          scoreLegend: Object.entries(priority.legend)
        };
      } finally {
        await model.dispose();
      }
    }, modelId);

    expect(report.catalog).toEqual([
      'tamev-micro-minilm-int8',
      'tamev-nano-tinybert-int8',
      'tamev-nano-flash-tinybert-int8',
      'kev-0.6b-q4f16'
    ]);
    expect(report.descriptionId).toBe(modelId);
    expect(report.describedId).toBe(modelId);
    expect(report.backend).toBe('onnxruntime-web-wasm');
    expect(report.resultId).toBe(modelId);
    expect(report.answerOrder).toEqual(['route', 'refund', 'priority']);
    expect(report.choiceType).toBe('choice');
    expect(report.noulType).toBe('noul');
    expect(report.scoreType).toBe('score');
    expect(report.choice).toMatch(/^(billing|support)$/);
    expect(report.choiceOptionOrder).toEqual(['billing', 'support']);
    expect(report.choiceProbabilityTotal).toBeCloseTo(1, 3);
    const choiceProbabilities = Object.fromEntries(report.choiceProbabilities);
    const selectedProbability = choiceProbabilities[report.choice];
    const highestChoiceProbability = Math.max(...Object.values(choiceProbabilities));
    expect(selectedProbability).toBeGreaterThanOrEqual(highestChoiceProbability - 0.0001);
    for (const [, probability] of report.choiceProbabilities) {
      expect(Number.isFinite(probability)).toBe(true);
      expect(probability).toBeGreaterThanOrEqual(0);
      expect(probability).toBeLessThanOrEqual(1);
    }
    expect(Number.isFinite(report.noul)).toBe(true);
    expect(report.noul).toBeGreaterThanOrEqual(0);
    expect(report.noul).toBeLessThanOrEqual(1);
    expect(Number.isFinite(report.score)).toBe(true);
    expect(report.score).toBeGreaterThanOrEqual(0);
    expect(report.score).toBeLessThanOrEqual(2);
    expect(report.scoreOptionOrder).toEqual(['0', '1', '2']);
    for (const [, probability] of report.scoreProbabilities) {
      expect(Number.isFinite(probability)).toBe(true);
      expect(probability).toBeGreaterThanOrEqual(0);
      expect(probability).toBeLessThanOrEqual(1);
    }
    expect(report.scoreProbabilityTotal).toBeCloseTo(1, 3);
    const expectedScore = report.scoreProbabilities.reduce(
      (total, [index, probability]) => total + Number(index) * probability,
      0
    );
    expect(report.score).toBeCloseTo(expectedScore, 3);
    expect(report.scoreLegend).toEqual([
      ['0', 'low'],
      ['1', 'medium'],
      ['2', 'high']
    ]);
  });
}
