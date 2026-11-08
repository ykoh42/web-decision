import {WebDecisionError} from './kev.js';
import {SKEJJA_INT8DYN_PROFILE, SKEJJA_INT8_PROFILE, SKEJJA_REPOSITORY, SKEJJA_REVISION} from './skejja/contract.mjs';

const MODEL_IDS = Object.freeze([SKEJJA_INT8_PROFILE.model, SKEJJA_INT8DYN_PROFILE.model]);
const modelCard = `https://huggingface.co/${SKEJJA_REPOSITORY}/tree/${SKEJJA_REVISION}`;

function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

function describe(profile) {
  return freeze({
    model: profile.model,
    precision: 'int8',
    checkpoint: {
      repository: SKEJJA_REPOSITORY,
      revision: SKEJJA_REVISION,
      graphSha256: profile.files[profile.onnxFile].sha256
    },
    assets: {
      integrity: 'pinned-files',
      location: 'separate',
      requiredFiles: Object.keys(profile.files),
      additionalFiles: 'Only the selected ONNX graph and its external-data sidecar are loaded.'
    },
    licenses: {project: 'Apache-2.0', modelTerms: 'separate-review-required', modelCard},
    backend: {
      name: 'onnxruntime-web-wasm',
      executionProvider: 'wasm',
      intraOpNumThreads: 1,
      requiredFeatures: [],
      requiredLimits: {},
      requiresNonFallbackAdapter: false
    },
    features: {
      textChoice: {
        status: 'implemented',
        abstention: 'not-implemented',
        limits: [
          {
            resource: 'tokens',
            scope: 'question',
            max: 512,
            description:
              'Complete prompt, including labels and model markers; long bodies are shortened from both ends.'
          },
          {
            resource: 'questions',
            scope: 'request',
            min: 1,
            description:
              'No package request-count cap. The upstream server defaults to 256 questions but allows a configured max_questions value.'
          },
          {resource: 'options', scope: 'question', min: 2, max: 255}
        ],
        handling: 'independent-questions',
        tokenCount: 'sum-question-rows',
        score: {
          basis: 'choices',
          temperature: profile.temperature,
          temperatureAppliedBy: 'decoder',
          calibration: 'not-guaranteed'
        }
      },
      typedTextQuestionTypes: ['choice', 'noul', 'score'],
      imageChoice: {status: 'not-implemented'},
      rating: {status: 'not-implemented'}
    },
    verification: {
      currentBundleGpu: 'not-applicable',
      currentBundleCpuWasm: 'verified',
      currentBundleJavascript: 'not-applicable',
      priorEvidence:
        'The isolated Skejja preview.42 archive passed real one-thread CPU/WASM inference in Chromium, Firefox, and WebKit on one Apple M4 Pro.'
    }
  });
}

const DESCRIPTIONS = new Map([
  [SKEJJA_INT8_PROFILE.model, describe(SKEJJA_INT8_PROFILE)],
  [SKEJJA_INT8DYN_PROFILE.model, describe(SKEJJA_INT8DYN_PROFILE)]
]);

export function listSkejjaModels() {
  return MODEL_IDS;
}

export function isSkejjaModel(value) {
  return DESCRIPTIONS.has(value);
}

export function describeSkejjaModel(model) {
  const description = DESCRIPTIONS.get(model);
  if (!description) throw new WebDecisionError('INVALID_INPUT', 'Unsupported Skejja decision model.');
  return description;
}
