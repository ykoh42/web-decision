import {WebDecisionError} from './kev.js';
import {getModelRegistration} from './model-registry.js';

const routes = Object.freeze({
  'kev-0.6b-q4f16': 'kev-0.6b-q4f16'
} as const);

type StableModelId = keyof typeof routes;

const modelIds = Object.freeze(Object.keys(routes));
const descriptions = Object.freeze(
  Object.fromEntries(
    modelIds.map((modelId) => [
      modelId,
      Object.freeze({...getModelRegistration(routes[modelId as StableModelId]).description, model: modelId})
    ])
  )
);

export function listStableModelIds() {
  return modelIds;
}

export function isStableModelId(value: unknown): value is StableModelId {
  return typeof value === 'string' && Object.hasOwn(routes, value);
}

export function describeStableModel(modelId: unknown) {
  if (!isStableModelId(modelId)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported stable decision model.');
  return descriptions[modelId]!;
}

export function normalizeStableModelOptions(options: unknown) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw new WebDecisionError('INVALID_INPUT', 'Stable model load options are required.');

  const descriptors = Object.getOwnPropertyDescriptors(options) as PropertyDescriptorMap;
  const modelIdDescriptor = descriptors.modelId;
  if (!modelIdDescriptor || !Object.hasOwn(modelIdDescriptor, 'value') || !isStableModelId(modelIdDescriptor.value))
    throw new WebDecisionError('INVALID_INPUT', 'Unsupported stable decision model.');
  const modelDescriptor = descriptors.model;
  if (modelDescriptor && (!Object.hasOwn(modelDescriptor, 'value') || modelDescriptor.value !== undefined))
    throw new WebDecisionError('INVALID_INPUT', 'Use either modelId or model, not both.');

  const modelId = modelIdDescriptor.value as StableModelId;
  delete descriptors.modelId;
  delete descriptors.model;
  descriptors.model = {value: routes[modelId], enumerable: true};

  return {
    modelId,
    options: Object.defineProperties(Object.create(null), descriptors)
  };
}
