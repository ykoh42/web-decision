import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {loadSkejja} from './skejja/loader.mjs';
import {describeSkejjaModel, isSkejjaModel, listSkejjaModels} from './skejja-preview-registry.mjs';

export {WebDecisionError, describeSkejjaModel as describeModel, isSkejjaModel as isDecisionModel};
export {listSkejjaModels as listDecisionModels};

export async function loadDecisionModel(options: unknown) {
  if (!options || !isSkejjaModel((options as {model?: unknown}).model)) {
    throw new WebDecisionError('INVALID_INPUT', 'Select one of the two Skejja edge32k-ft6 int8 models.');
  }
  const model = await loadSkejja(options);
  Object.defineProperty(model, 'description', {value: describeSkejjaModel((options as {model?: unknown}).model)});
  return model;
}

export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}
