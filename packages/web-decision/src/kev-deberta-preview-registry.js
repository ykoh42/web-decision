import kevDescription from './kev-preview-model.json' with {type: 'json'};
import {WebDecisionError} from './kev.js';
import deberta from './deberta/contract.json' with {type: 'json'};
import {DEBERTA_REQUIRED_LIMITS} from './deberta/device.mjs';

const KEV_MODEL = kevDescription.model;
const DEBERTA_MODEL = deberta.model;
const MODELS = Object.freeze([KEV_MODEL, DEBERTA_MODEL]);

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

const debertaDescription = deepFreeze({
  model: DEBERTA_MODEL,
  precision: 'q4f16',
  checkpoint: deberta.checkpoint,
  assets: {
    integrity: 'pinned-files',
    location: 'separate',
    requiredFiles: [
      'manifest.json',
      'model.wasm',
      'tokenizer.json',
      'tokenizer_config.json',
      'model_config.json',
      'weights/tensor-cache.json',
      'mapping.json',
      'split.json'
    ],
    additionalFiles: 'Weight shards or ranges listed by the model manifest are also required.'
  },
  licenses: {
    project: 'Apache-2.0',
    modelTerms: 'separate-review-required',
    modelCard: `https://huggingface.co/${deberta.checkpoint.repository}/tree/${deberta.checkpoint.revision}`
  },
  backend: {
    name: 'tvm-webgpu',
    requiredFeatures: ['shader-f16'],
    requiredLimits: {...DEBERTA_REQUIRED_LIMITS},
    requiresNonFallbackAdapter: false
  },
  features: {
    textChoice: {
      status: 'implemented',
      abstention: 'not-implemented',
      limits: [
        {
          resource: 'tokens',
          scope: 'request',
          max: deberta.maxInputTokens,
          description: 'Complete packed request, including markers.'
        },
        {resource: 'tokens', scope: 'state', max: deberta.maxStateTokens, description: 'State content tokens.'},
        {resource: 'options', scope: 'question', min: 2, max: 255},
        {resource: 'questions', scope: 'request', min: 1, max: 512}
      ],
      handling: 'packed-request',
      tokenCount: 'packed-request',
      score: {
        basis: 'choices',
        temperature: 1.05,
        temperatureAppliedBy: 'decoder',
        calibration: 'not-guaranteed'
      }
    },
    imageChoice: {status: 'not-implemented'},
    rating: {status: 'not-implemented'}
  },
  verification: {
    currentBundleGpu: 'verified',
    currentBundleCpuWasm: 'not-applicable',
    currentBundleJavascript: 'not-applicable',
    priorEvidence:
      'Preview.36 passed five guarded real-WebGPU cases in Chromium, Firefox, and WebKit on one Apple M4 Pro.'
  }
});

const descriptions = new Map([
  [KEV_MODEL, deepFreeze(kevDescription)],
  [DEBERTA_MODEL, debertaDescription]
]);

export function listDecisionModels() {
  return MODELS;
}

export function isDecisionModel(value) {
  return descriptions.has(value);
}

export function describeModel(model) {
  const description = descriptions.get(model);
  if (!description) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  return description;
}
