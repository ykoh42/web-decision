import modelSpec from './model-spec.json' with {type: 'json'};
import artifactLock from './artifacts.lock.json' with {type: 'json'};

import type {
  CatalogModelDescription,
  ModelArtifactLock,
  ModelCatalogEntry,
  ModelId,
  ModelIdTuple
} from './types.generated.js';

type CuratedModelSpec = Readonly<{
  modelId: ModelId;
  description: Omit<CatalogModelDescription, 'assets'> & {
    assets: Omit<CatalogModelDescription['assets'], 'manifestSha256' | 'requiredFiles'>;
  };
}>;
const models = modelSpec.models as unknown as readonly CuratedModelSpec[];
const artifactsByModelId = artifactLock.byModelId as unknown as Readonly<Record<ModelId, ModelArtifactLock>>;

function deepFreeze<T>(value: T): Readonly<T> {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

const catalog: ModelCatalogEntry[] = models.map((model) => {
  const artifacts = artifactsByModelId[model.modelId];
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
});

export const MODEL_CATALOG = deepFreeze(catalog);
export const MODEL_IDS = Object.freeze(MODEL_CATALOG.map((entry) => entry.modelId)) as ModelIdTuple;
export const MODEL_CATALOG_BY_ID = Object.freeze(
  Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.modelId, entry])) as Record<ModelId, ModelCatalogEntry>
);
