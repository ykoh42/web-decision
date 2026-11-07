import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {loadKev06DecisionModel} from './kev-loader.js';
import {describeModel, isDecisionModel, listDecisionModels} from './kev-preview-registry.js';

export {WebDecisionError, describeModel, isDecisionModel, listDecisionModels};

/** Load the only model in this preview build. */
export async function loadDecisionModel(options) {
  const modelName = options?.model === undefined ? 'kev-0.6b-q4f16' : options.model;
  const description = describeModel(modelName);
  if (options?.debertaReadback !== undefined)
    throw new WebDecisionError('INVALID_INPUT', 'debertaReadback is only supported by the DeBERTa model.');
  const model = await loadKev06DecisionModel(options, description);
  Object.defineProperty(model, 'description', {value: description});
  return model;
}

/** Delete only model assets cached by this library for the current origin. */
export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}
