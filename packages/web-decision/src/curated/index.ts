import {CACHE_NAME} from '../assets.js';
import {APPROVED_ADAPTER_LOADERS} from '../catalog/public-adapters.generated.ts';
import {MODEL_CATALOG, MODEL_CATALOG_BY_ID} from '../catalog/public-catalog.generated.ts';
import type {ModelId} from '../catalog/types.generated.js';
import {WebDecisionError} from './errors.ts';
import {readLoadOptions} from './load-options.ts';

export {WebDecisionError};
export {KananaFloat, kananaFloat} from '../kanana/json-float.ts';

const loaderDescriptions = Object.freeze(
  MODEL_CATALOG.map(({modelId, description}) => {
    const {adapter: _adapter, ...metadata} = description;
    return Object.freeze({modelId, ...metadata});
  })
);
const descriptions = Object.freeze(
  loaderDescriptions.map(({assets, ...metadata}) => {
    const {directArtifactRoot: _directArtifactRoot, ...publicAssets} = assets;
    return Object.freeze({...metadata, assets: Object.freeze(publicAssets)});
  })
);
const descriptionsById = Object.freeze(Object.fromEntries(descriptions.map((entry) => [entry.modelId, entry])));
const loaderDescriptionsById = Object.freeze(
  Object.fromEntries(loaderDescriptions.map((entry) => [entry.modelId, entry]))
);

function isModelId(value: unknown): value is ModelId {
  return typeof value === 'string' && Object.hasOwn(MODEL_CATALOG_BY_ID, value);
}

/** List only the models approved for public use in this package. */
export function listModels() {
  return descriptions;
}

/** Describe an approved model from this package without accessing the network. */
export function describeModel(modelId: ModelId) {
  if (!isModelId(modelId)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model ID.', undefined);
  const description = descriptionsById[modelId];
  if (!description) throw new WebDecisionError('INVALID_ASSET', 'The model description is unavailable.', undefined);
  return description;
}

/** Load a curated model from the source and artifact pinned by this package. */
export async function loadDecisionModel(input: unknown) {
  const {modelId, options} = readLoadOptions(input, isModelId);
  (options as {signal?: AbortSignal}).signal?.throwIfAborted();
  const description = describeModel(modelId);
  const loaderDescription = loaderDescriptionsById[modelId]!;
  const loadAdapter = APPROVED_ADAPTER_LOADERS[MODEL_CATALOG_BY_ID[modelId].description.adapter];
  if (!loadAdapter) throw new WebDecisionError('INVALID_ASSET', 'The model adapter is unavailable.', undefined);
  const {loadModel} = await loadAdapter();
  const model = await loadModel(options, loaderDescription);
  try {
    Object.defineProperty(model, 'description', {value: description});
    return model;
  } catch (error) {
    await Promise.allSettled([Promise.resolve().then(() => model.dispose?.())]);
    throw error;
  }
}

/** Delete this library's cached model assets for the current origin. */
export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}
