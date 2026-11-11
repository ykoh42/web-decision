import type {ModelId} from '../catalog/types.generated.js';
import {APPROVED_MODEL_LOAD_OPTIONS} from '../catalog/public-load-options.generated.ts';
import {WebDecisionError} from './errors.ts';

const COMMON_OPTIONS = ['modelId', 'cache', 'offline', 'signal', 'onProgress'] as const;
type ModelLoadOptions = Readonly<Record<string, readonly string[]>>;

function optionsForModel(modelId: string, approvedLoadOptions: ModelLoadOptions): ReadonlySet<string> {
  return new Set(approvedLoadOptions[modelId] ?? []);
}

/** Validate the closed public load interface without evaluating caller getters. */
export function readLoadOptions(
  options: unknown,
  isModelId: (value: unknown) => value is ModelId,
  approvedLoadOptions: ModelLoadOptions = APPROVED_MODEL_LOAD_OPTIONS
) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw new WebDecisionError('INVALID_INPUT', 'Model load options are required.', undefined);

  const knownOptions = new Set([...COMMON_OPTIONS, ...Object.values(approvedLoadOptions).flat()]);
  const descriptors = Object.getOwnPropertyDescriptors(options);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== 'string' || !knownOptions.has(key))
      throw new WebDecisionError('INVALID_INPUT', 'Unsupported model load option.', undefined);
    const descriptor = descriptors[key];
    if (!descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new WebDecisionError('INVALID_INPUT', 'Unsupported model load option.', undefined);
  }
  const modelId = descriptors.modelId?.value;
  if (!isModelId(modelId)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model ID.', undefined);
  const allowedOptions = new Set([...COMMON_OPTIONS, ...optionsForModel(modelId, approvedLoadOptions)]);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== 'string' || !allowedOptions.has(key)) {
      const name = typeof key === 'string' ? ` "${key}"` : '';
      throw new WebDecisionError('INVALID_INPUT', `Unsupported load option${name} for model "${modelId}".`, undefined);
    }
  }

  delete descriptors.modelId;
  return {modelId, options: Object.defineProperties(Object.create(null), descriptors)};
}
