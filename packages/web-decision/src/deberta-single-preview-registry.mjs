import {WebDecisionError} from './kev.js';
import deberta from './deberta/contract.json' with {type: 'json'};
import {DEBERTA_REQUIRED_LIMITS} from './deberta/device.mjs';

const MODEL_ID = deberta.model;
const MODEL_IDS = Object.freeze([MODEL_ID]);

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

const description = deepFreeze({
  model: MODEL_ID,
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
        {resource: 'questions', scope: 'request', min: 1}
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

export function listDecisionModels() {
  return MODEL_IDS;
}

export function isDecisionModel(value) {
  return value === MODEL_ID;
}

export function describeModel(model) {
  if (model !== MODEL_ID) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  return description;
}
