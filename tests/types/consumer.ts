import {
  clearModelCache,
  describeModel,
  listModels,
  loadDecisionModel,
  WebDecisionError,
  type KevQuestion,
  type ModelId,
  type ModelProgress
} from 'web-decision';

async function checkConsumer(): Promise<void> {
  const id: ModelId = 'kev-0.6b-q4f16';
  const progress = (event: ModelProgress): void => {
    const fraction: number = event.fraction;
    void fraction;
  };
  const models = listModels();
  const description = describeModel(id);
  const model = await loadDecisionModel({modelId: id, onProgress: progress});
  const answer = await model.decide({
    state: 'A customer requests a refund.',
    questions: [{instructions: 'Which team?', options: ['Billing', 'Support']}]
  });
  const selected: string | undefined = answer.answers[0]?.selected;
  const question: KevQuestion = {type: 'noul', instructions: 'Is this urgent?'};
  const typed = await model.systemOne('A request', {urgent: question});
  const returnedId: ModelId = typed.model;
  const error = new WebDecisionError('INVALID_INPUT', 'Invalid input.');
  await model.releaseCachedMemory();
  await model.dispose();
  await clearModelCache();
  void [models, description, selected, returnedId, error];
}
void checkConsumer;

// @ts-expect-error The public API does not accept a legacy model selector.
void loadDecisionModel({model: 'kev-0.6b-q4f16'});
// @ts-expect-error Users cannot choose model sources.
void loadDecisionModel({modelId: 'kev-0.6b-q4f16', assetsUrl: 'https://example.org/'});
// @ts-expect-error Only catalog model IDs are accepted.
void loadDecisionModel({modelId: 'unknown'});
