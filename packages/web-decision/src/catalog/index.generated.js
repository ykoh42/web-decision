import modelSpec from './model-spec.json' with {type: 'json'};
import artifactLock from './artifacts.lock.json' with {type: 'json'};

function deepFreeze(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

export const MODEL_CATALOG = deepFreeze(
  modelSpec.models.map((model) => {
    const artifacts = artifactLock.byModelId[model.modelId];
    return {
      modelId: model.modelId,
      description: {
        ...model.description,
        assets: {
          ...model.description.assets,
          requiredFiles: ['manifest.json', ...Object.keys(artifacts.manifest.files)],
          manifestSha256: artifacts.manifest.canonicalSha256
        }
      },
      artifacts
    };
  })
);
export const MODEL_IDS = Object.freeze(MODEL_CATALOG.map((entry) => entry.modelId));
export const MODEL_CATALOG_BY_ID = Object.freeze(
  Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.modelId, entry]))
);
