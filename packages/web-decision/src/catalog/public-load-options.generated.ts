export const APPROVED_MODEL_LOAD_OPTIONS: Readonly<Record<string, readonly string[]>> = Object.freeze({
  'kanana-2-v-decide-3b-2611': ['modelId', 'cache', 'offline', 'signal', 'onProgress'],
  'strands-decider-2b-qwen3.5-v1-2610': ['modelId', 'cache', 'offline', 'signal', 'onProgress'],
  'mapika-decider-2b-v12': [
    'modelId',
    'cache',
    'offline',
    'signal',
    'onProgress',
    'temperature',
    'temperatureByType',
    'limits'
  ],
  'strands-decider-e2b-gemma4-v2-2611': ['modelId', 'cache', 'offline', 'signal', 'onProgress', 'maxLength'],
  'strands-decider-e2b-gemma4-v1-2610': [
    'modelId',
    'cache',
    'offline',
    'signal',
    'onProgress',
    'maxLength',
    'maxBatch',
    'maxBatchTokens',
    'usePrefixCache'
  ]
});
