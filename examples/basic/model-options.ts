import type {LoadOptions, ModelDescription, ModelProgress} from 'web-decision';

/** Build load options only for a model offered by the installed package. */
export function loadOptions(
  models: readonly Pick<ModelDescription, 'modelId'>[],
  modelId: string,
  onProgress?: (event: ModelProgress) => void
): LoadOptions {
  const model = models.find((candidate) => candidate.modelId === modelId);
  if (!model) throw new Error('Select a supported model.');
  return onProgress ? {modelId: model.modelId, onProgress} : {modelId: model.modelId};
}
