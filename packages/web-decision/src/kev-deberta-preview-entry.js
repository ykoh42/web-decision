import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {loadExperimentalDeberta} from './deberta/loader.mjs';
import {loadKev06DecisionModel} from './kev-loader.js';
import {describeModel, isDecisionModel, listDecisionModels} from './kev-deberta-preview-registry.js';

export {WebDecisionError, describeModel, isDecisionModel, listDecisionModels};

const DEFAULT_MODEL = 'kev-0.6b-q4f16';
const DEBERTA_MODEL = 'open-jev-deberta-v3-large-q4f16';

/** Load Kev 0.6B by default, or the explicitly selected DeBERTa route. */
export async function loadDecisionModel(options) {
  const modelName = options?.model ?? DEFAULT_MODEL;
  const description = describeModel(modelName);
  if (modelName === DEFAULT_MODEL) {
    if (options?.debertaReadback !== undefined)
      throw new WebDecisionError('INVALID_INPUT', 'debertaReadback is only supported by the DeBERTa model.');
    const model = await loadKev06DecisionModel(options, description);
    Object.defineProperty(model, 'description', {value: description});
    return model;
  }
  if (modelName === DEBERTA_MODEL) {
    const model = await loadExperimentalDeberta(options);
    Object.defineProperty(model, 'description', {value: description});
    return model;
  }
  throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
}

/** Delete only model assets cached by this library for the current origin. */
export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}
