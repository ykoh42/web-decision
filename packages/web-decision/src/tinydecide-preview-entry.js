import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {loadTinyDecide} from './tinydecide/loader.mjs';
import {describeModel, isDecisionModel, listDecisionModels} from './tinydecide-preview-registry.js';

export {WebDecisionError, describeModel, isDecisionModel, listDecisionModels};

export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}

export async function loadDecisionModel(options) {
  const description = describeModel(options?.model);
  const model = await loadTinyDecide(options);
  Object.defineProperty(model, 'description', {value: description});
  return model;
}
