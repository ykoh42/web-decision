import type {ModelId} from '../catalog/types.generated.js';
import {WebDecisionError} from './errors.ts';

const ALLOWED_OPTIONS = new Set([
  'modelId',
  'cache',
  'offline',
  'signal',
  'onProgress',
  'maxPrefetchBytes',
  'maxInputTokens',
  'maxInputCharacters'
]);

/** Validate the closed public load interface without evaluating caller getters. */
export function readLoadOptions(options: unknown, isModelId: (value: unknown) => value is ModelId) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw new WebDecisionError('INVALID_INPUT', 'Model load options are required.', undefined);

  const descriptors = Object.getOwnPropertyDescriptors(options);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== 'string' || !ALLOWED_OPTIONS.has(key))
      throw new WebDecisionError('INVALID_INPUT', 'Unsupported model load option.', undefined);
    const descriptor = descriptors[key];
    if (!descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new WebDecisionError('INVALID_INPUT', 'Unsupported model load option.', undefined);
  }
  const modelId = descriptors.modelId?.value;
  if (!isModelId(modelId)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model ID.', undefined);

  delete descriptors.modelId;
  return {modelId, options: Object.defineProperties(Object.create(null), descriptors)};
}
