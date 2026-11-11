import {
  clearModelCache,
  describeModel,
  listModels,
  loadDecisionModel,
  WebDecisionError,
  type ModelId
} from 'web-decision';

async function checkPublicApi(): Promise<void> {
  for (const {modelId} of listModels()) {
    const id: ModelId = modelId;
    const description = describeModel(id);
    const model = await loadDecisionModel({modelId: id});
    await model.dispose();
    void description;
  }
  await clearModelCache();
  void new WebDecisionError('INVALID_INPUT', 'Invalid input.');
}
void checkPublicApi;

// @ts-expect-error The public API does not accept legacy model selectors.
void loadDecisionModel({model: 'kev-0.6b-q4f16'});
// @ts-expect-error Users cannot choose model sources.
void loadDecisionModel({modelId: 'kev-0.6b-q4f16', assetsUrl: 'https://example.org/'});
// @ts-expect-error Unknown IDs are not accepted.
void loadDecisionModel({modelId: 'unknown'});
