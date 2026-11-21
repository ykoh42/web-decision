import {
  loadDecisionModel,
  type Mapika2bDecisionModel,
  type Mapika2bPrepareOfflineOptions,
  type Mapika2bSystemOneRequest
} from 'web-decision';

async function evaluate(request: Mapika2bSystemOneRequest, signal: AbortSignal): Promise<void> {
  const model: Mapika2bDecisionModel<'mapika-decider-2b-v12'> = await loadDecisionModel({
    modelId: 'mapika-decider-2b-v12'
  });
  try {
    const response = await model.systemOne(request);
    const tokens: number = response.usage.input_tokens;
    void tokens;
    const offlineOptions: Mapika2bPrepareOfflineOptions = {
      signal,
      onProgress(progress) {
        const fraction: number = progress.fraction;
        void fraction;
      }
    };
    await model.prepareOffline(offlineOptions);
  } finally {
    await model.dispose();
  }
}
void evaluate;

// @ts-expect-error Mapika loads the checkpoint pinned to its curated ID.
void loadDecisionModel({modelId: 'mapika-decider-2b-v12', assetsUrl: 'https://example.org/'});
