import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {loadKev06DecisionModel} from './kev-loader.js';
import {TAMEV_MICRO_MODEL, TAMEV_NANO_MODEL} from './tamev/contract.mjs';
import {loadTamev} from './tamev/loader.mjs';
import {describeModel, isDecisionModel, listDecisionModels} from './kev-preview-registry.js';

export {WebDecisionError, describeModel, isDecisionModel, listDecisionModels};

/** Load a model in this release with its separately hosted pinned asset pack. */
export async function loadDecisionModel(options) {
  const modelName = options?.model;
  if (modelName === TAMEV_NANO_MODEL || modelName === TAMEV_MICRO_MODEL) {
    const description = describeModel(modelName);
    const model = await loadTamev(options);
    Object.defineProperty(model, 'description', {value: description});
    return model;
  }
  if (modelName !== undefined && modelName !== 'kev-0.6b-q4f16')
    throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  const selectedModel = modelName ?? 'kev-0.6b-q4f16';
  const description = describeModel(selectedModel);
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
