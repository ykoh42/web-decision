import modelDescription from './tinydecide-preview-model.json' with {type: 'json'};
import {WebDecisionError} from './kev.js';

const MODEL = modelDescription.model;
const MODELS = Object.freeze([MODEL]);

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

const description = deepFreeze(modelDescription);

export function listDecisionModels() {
  return MODELS;
}

export function isDecisionModel(value) {
  return value === MODEL;
}

export function describeModel(model) {
  if (!isDecisionModel(model)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  return description;
}
