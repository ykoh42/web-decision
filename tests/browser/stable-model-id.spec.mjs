import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_STABLE_MODEL_ID_REAL === '1';
const tamevModelIds = ['tamev-micro-minilm-int8', 'tamev-nano-tinybert-int8'];

for (const modelId of tamevModelIds) {
  test(`stable model ID ${modelId} loads pinned assets without an assetsUrl`, async ({page}) => {
    test.skip(!enabled, 'Set WEB_DECISION_STABLE_MODEL_ID_REAL=1 to load a pinned stable TAMEV model.');
    test.setTimeout(180_000);
    await page.goto('/');
    const report = await page.evaluate(async (modelId) => {
      const {describeStableModel, listStableModelIds, loadDecisionModel} = window.webDecision;
      const model = await loadDecisionModel({modelId, cache: false});
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
        return {
          modelId,
          catalog: listStableModelIds(),
          descriptionId: model.description.model,
          describedId: describeStableModel(modelId).model,
          backend: model.description.backend.name,
          resultId: result.model,
          answerOrder: Object.keys(result.answers),
          choiceType: result.answers.route.type,
          choice: result.answers.route.choice,
          choiceOptionOrder: Object.keys(result.answers.route.probabilities),
          choiceProbabilities: Object.entries(result.answers.route.probabilities),
          choiceProbabilityTotal: Object.values(result.answers.route.probabilities).reduce(
            (total, probability) => total + probability,
            0
          ),
          noulType: result.answers.refund.type,
          noul: result.answers.refund.noul,
          scoreType: result.answers.priority.type,
          score: result.answers.priority.score,
          scoreOptionOrder: Object.keys(result.answers.priority.probabilities),
          scoreProbabilities: Object.entries(result.answers.priority.probabilities),
          scoreProbabilityTotal: Object.values(result.answers.priority.probabilities).reduce(
            (total, probability) => total + probability,
            0
          ),
          scoreLegend: Object.entries(result.answers.priority.legend)
        };
      } finally {
        await model.dispose();
      }
    }, modelId);

    expect(report.catalog).toEqual(['tamev-micro-minilm-int8', 'tamev-nano-tinybert-int8', 'kev-0.6b-q4f16']);
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
