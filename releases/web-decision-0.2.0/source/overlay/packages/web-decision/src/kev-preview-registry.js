import modelDescription from './kev-preview-model.json' with {type: 'json'};
import tamevNanoDescription from './tamev-nano-model.json' with {type: 'json'};
import {WebDecisionError} from './kev.js';

const MODELS = Object.freeze([modelDescription.model, tamevNanoDescription.model]);

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

const descriptions = new Map([
  [modelDescription.model, deepFreeze(modelDescription)],
  [tamevNanoDescription.model, deepFreeze(tamevNanoDescription)]
]);

export function listDecisionModels() {
  return MODELS;
}

export function isDecisionModel(value) {
  return descriptions.has(value);
}

export function describeModel(model) {
  if (!isDecisionModel(model)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  return descriptions.get(model);
}
