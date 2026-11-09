import {getModelRegistration} from './model-registry.js';

/** Resolve once so routing and the loaded model's description use the same ID. */
export function selectModelRegistration(options: {model?: unknown} | null | undefined) {
  return getModelRegistration(options?.model as string | undefined);
}

/** Route explicit model families while preserving the existing Kev default. */
export function selectModelFamily(options: {model?: unknown} | null | undefined) {
  return selectModelRegistration(options).family;
}
