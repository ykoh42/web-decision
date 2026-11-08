import {getModelRegistration} from './model-registry.js';

/** Resolve once so routing and the loaded model's description use the same ID. */
export function selectModelRegistration(options) {
  return getModelRegistration(options?.model);
}

/** Route explicit model families while preserving the existing Kev default. */
export function selectModelFamily(options) {
  return selectModelRegistration(options).family;
}
