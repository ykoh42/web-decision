import {
  clearModelCache,
  describeModel,
  listModels,
  loadDecisionModel,
  WebDecisionError,
  type KananaDecisionRequest,
  type ModelId,
  type ModelProgress
} from 'web-decision';

async function checkConsumer(): Promise<void> {
  const id: ModelId = 'kanana-2-v-decide-3b-2611';
  const progress = (event: ModelProgress): void => {
    const fraction: number = event.fraction;
    void fraction;
  };
  const models = listModels();
  const description = describeModel(id);
  const model = await loadDecisionModel({modelId: id, onProgress: progress});
  const request: KananaDecisionRequest = {
    state: 'A customer requests a refund.',
    questions: {team: {type: 'choice', criteria: {billing: 'Billing', support: 'Support'}}}
  };
  const result = await model.decide(request);
  const answer = result.answers.team;
  const error = new WebDecisionError('INVALID_INPUT', 'Invalid input.');
  await model.dispose();
  await clearModelCache();
  void [models, description, answer, error];
}
void checkConsumer;

// @ts-expect-error The public API does not accept a legacy model selector.
void loadDecisionModel({model: 'kanana-2-v-decide-3b-2611'});
// @ts-expect-error Users cannot choose model sources.
void loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611', assetsUrl: 'https://example.org/'});
// @ts-expect-error Only catalog model IDs are accepted.
void loadDecisionModel({modelId: 'unknown'});
// @ts-expect-error This package candidate exposes Kanana only.
void loadDecisionModel({modelId: 'kev-0.6b-q4f16'});
