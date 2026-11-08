import kevDescription from './kev-preview-model.json' with {type: 'json'};
import {WebDecisionError} from './kev.js';
import deberta from './deberta/contract.json' with {type: 'json'};
import {DEBERTA_REQUIRED_LIMITS} from './deberta/device.mjs';
import reflexes from './kev-reflexes-contract.json' with {type: 'json'};

const KEV_MODEL = kevDescription.model;
const DEBERTA_MODEL = deberta.model;
const REFLEXES_MODEL = reflexes.model;
const REFLEXES_MANIFEST_SHA256 = '1cd07a9326e4d8273d8972fdfb4ba9ab23cf3ff1f5e5b99cbea263116eebee07';
const MODELS = Object.freeze([KEV_MODEL, DEBERTA_MODEL, REFLEXES_MODEL]);

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

const kevPreviewDescription = deepFreeze({
  ...kevDescription,
  verification: {
    ...kevDescription.verification,
    currentBundleGpu: 'not-revalidated',
    priorEvidence:
      'Preview.35 passed seven real-WebGPU cases in Chromium, Firefox, and WebKit on one Apple M4 Pro. Preview.39 has not completed GPU validation.'
  }
});

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
    currentBundleGpu: 'not-revalidated',
    currentBundleCpuWasm: 'not-applicable',
    currentBundleJavascript: 'not-applicable',
    priorEvidence:
      'Preview.36 passed five guarded real-WebGPU cases in Chromium, Firefox, and WebKit on one Apple M4 Pro. Preview.37 has not completed GPU validation.'
  }
});

const reflexesDescription = deepFreeze({
  model: REFLEXES_MODEL,
  precision: 'q4f16',
  checkpoint: reflexes.checkpoint,
  assets: {
    integrity: 'pinned-manifest',
    manifestSha256: REFLEXES_MANIFEST_SHA256,
    location: 'separate',
    requiredFiles: [
      'manifest.json',
      'model.wasm',
      'tokenizer.json',
      'tokenizer_config.json',
      'model_config.json',
      'weights/tensor-cache.json',
      'heads/param_394.bin',
      'heads/param_395.bin',
      'heads/param_396.bin',
      'heads/param_397.bin'
    ],
    additionalFiles: 'Weight shards listed by the model manifest are also required.'
  },
  licenses: {
    project: 'Apache-2.0',
    modelTerms: 'separate-review-required',
    modelCard: `https://huggingface.co/${reflexes.checkpoint.repository}/blob/${reflexes.checkpoint.revision}/README.md`
  },
  backend: {
    name: 'tvm-webgpu',
    requiredFeatures: ['shader-f16'],
    requiredLimits: {...kevDescription.backend.requiredLimits},
    requiresNonFallbackAdapter: false
  },
  features: {
    textChoice: {
      status: 'implemented',
      abstention: 'not-implemented',
      limits: [
        {
          resource: 'tokens',
          scope: 'state',
          max: reflexes.maxStateTokens - 1,
          description: 'Native Kev state limit; the model counts its state marker within the 8,192-token budget.'
        },
        {
          resource: 'tokens',
          scope: 'question',
          max: reflexes.maxBranchTokens,
          description:
            'Native Kev per-question row limit, including the shared state and question branch. There is no fixed aggregate request-token cap; available memory limits the packed request.'
        },
        {resource: 'options', scope: 'question', min: 1, max: 255},
        {resource: 'questions', scope: 'request', min: 1}
      ],
      handling: 'packed-request',
      tokenCount: 'packed-request',
      score: {
        basis: 'choices',
        temperature: 1,
        temperatureAppliedBy: 'decoder',
        calibration: 'not-guaranteed'
      }
    },
    typedTextQuestionTypes: ['choice', 'noul', 'score'],
    imageChoice: {status: 'not-implemented'},
    rating: {status: 'not-implemented'}
  },
  verification: {
    currentBundleGpu: 'not-revalidated',
    currentBundleCpuWasm: 'not-applicable',
    currentBundleJavascript: 'not-applicable',
    priorEvidence:
      'Preview.34 passed seven guarded real-WebGPU cases in Chromium, Firefox, and WebKit on one Apple M4 Pro. Preview.37 did not complete: Firefox failed to initialize a GPU device in one case, and WebKit was not run.'
  }
});

const descriptions = new Map([
  [KEV_MODEL, kevPreviewDescription],
  [DEBERTA_MODEL, debertaDescription],
  [REFLEXES_MODEL, reflexesDescription]
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
