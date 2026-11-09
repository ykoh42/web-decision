import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {loadExperimentalDeberta} from './deberta/loader.mjs';
import {describeModel, isDecisionModel, listDecisionModels} from './deberta-single-preview-registry.mjs';
import type {DebertaLoadOptions, LoadedDebertaDecisionModel, ModelDescription} from './deberta-single-preview.js';

const MODEL_ID = 'open-jev-deberta-v3-large-q4f16';

export {WebDecisionError, describeModel, isDecisionModel, listDecisionModels};

/** Load the explicitly selected pinned Open-Jev DeBERTa model. */
export async function loadDecisionModel(options: DebertaLoadOptions): Promise<LoadedDebertaDecisionModel> {
  if (!options || options.model !== MODEL_ID) {
    throw new WebDecisionError('INVALID_INPUT', 'Select the open-jev-deberta-v3-large-q4f16 model.');
  }
  const model = await loadExperimentalDeberta(options);
  Object.defineProperty(model, 'description', {value: describeModel(MODEL_ID)});
  return model as typeof model & {readonly description: ModelDescription};
}

/** Delete the decision-model assets cached by this library for the current origin. */
export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}
