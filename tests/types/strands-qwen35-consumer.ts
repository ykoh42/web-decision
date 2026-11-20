import {
  kananaFloat,
  loadDecisionModel,
  type LoadedDecisionModel,
  type StrandsQwen35SystemOneRequest
} from 'web-decision';

const modelId = 'strands-decider-2b-qwen3.5-v1-2610';
const request: StrandsQwen35SystemOneRequest = {
  state: {threshold: kananaFloat(1)},
  questions: new Map([['intent', {type: 'choice', instructions: 'Decide.', criteria: {yes: null, no: null}}]]),
  images: ['base64-data']
};

async function checkQwen35() {
  const model = await loadDecisionModel({
    modelId,
    onProgress(progress) {
      const phase: 'model-assets' | 'vision-weights' | 'upstream-source' = progress.phase;
      void phase;
    }
  });
  const loadedId: typeof modelId = model.modelId;
  const result = await model.systemOne(request, {maxBatchTokens: null, maxLength: 4096});
  await model.releaseCachedMemory();
  await model.dispose();
  // @ts-expect-error Curated Qwen does not take a caller-selected asset URL.
  await loadDecisionModel({modelId, assetsUrl: '/mirror'});
  // @ts-expect-error A scalar is not source Content at the root.
  await model.systemOne({state: kananaFloat(1), questions: request.questions});
  return {loadedId, result};
}

async function disposeByModel(model: LoadedDecisionModel) {
  if (model.modelId === modelId) await model.releaseCachedMemory();
  await model.dispose();
}
void [checkQwen35, disposeByModel];
