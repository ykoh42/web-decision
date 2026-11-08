import {loadCuratedKevModel} from '../../kev-loader.js';
import type {ModelDescription} from '../types.js';

/** Adapt the curated descriptor to the model's pinned implementation. */
export function loadModel(options: object, description: ModelDescription) {
  Object.defineProperty(options, 'model', {value: description.modelId, enumerable: true});
  return loadCuratedKevModel(options, {...description, model: description.modelId});
}
