import {loadDecisionModel, type StrandsGemma4V2DecisionModel, type StrandsGemma4V2SystemOneRequest} from 'web-decision';

async function evaluate(request: StrandsGemma4V2SystemOneRequest, signal: AbortSignal): Promise<void> {
  const model: StrandsGemma4V2DecisionModel = await loadDecisionModel({
    modelId: 'strands-decider-e2b-gemma4-v2-2611',
    maxLength: 32768
  });
  try {
    const result = await model.systemOne(request, {signal});
    const modelId: 'strands-decider-e2b-gemma4-v2-2611' = result.model;
    void modelId;
    await model.releaseCachedMemory();
    await model.prepareOffline({signal});
    await model.isOfflineReady(signal);
  } finally {
    await model.dispose();
  }
}
void evaluate;

// @ts-expect-error Gemma always uses its pinned assets.
void loadDecisionModel({modelId: 'strands-decider-e2b-gemma4-v2-2611', assetsUrl: 'https://example.org/'});
