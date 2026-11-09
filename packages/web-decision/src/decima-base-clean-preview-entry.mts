import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {DECIMA_BASE_CLEAN_MODEL, DECIMA_BASE_CLEAN_SPEC} from './decima/contract.mjs';
import {loadDecimaModel, validateDecimaLoadOptions} from './decima/loader.mjs';

type BaseCleanLimits = typeof DECIMA_BASE_CLEAN_SPEC & {readonly maxChoices?: number; readonly maxQuestions?: number};
const MODEL_IDS = Object.freeze([DECIMA_BASE_CLEAN_MODEL]);
const modelCard = `https://huggingface.co/${DECIMA_BASE_CLEAN_SPEC.repository}/tree/${DECIMA_BASE_CLEAN_SPEC.revision}`;

function freeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

function describeBaseClean() {
  return freeze({
    model: DECIMA_BASE_CLEAN_MODEL,
    precision: 'int8',
    checkpoint: {
      repository: DECIMA_BASE_CLEAN_SPEC.repository,
      revision: DECIMA_BASE_CLEAN_SPEC.revision
    },
    assets: {
      integrity: 'pinned-files',
      location: 'separate',
      requiredFiles: Object.keys(DECIMA_BASE_CLEAN_SPEC.files),
      additionalFiles: 'No additional files are required.'
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
            scope: 'state',
            max: DECIMA_BASE_CLEAN_SPEC.maxStateTokens,
            description: 'Complete question and state row; longer input is rejected before inference.'
          },
          {
            resource: 'tokens',
            scope: 'question',
            max: DECIMA_BASE_CLEAN_SPEC.maxChoiceTokens,
            description:
              'Uses the checkpoint choice_fit rendering and reports the adapted rows; each row includes question text and special tokens.'
          },
          {resource: 'options', scope: 'question', min: 2, max: (DECIMA_BASE_CLEAN_SPEC as BaseCleanLimits).maxChoices},
          {
            resource: 'questions',
            scope: 'request',
            min: 1,
            max: (DECIMA_BASE_CLEAN_SPEC as BaseCleanLimits).maxQuestions
          }
        ],
        handling: 'independent-questions',
        tokenCount: 'sum-question-rows',
        score: {
          basis: 'choices',
          temperature: DECIMA_BASE_CLEAN_SPEC.temperature,
          temperatureAppliedBy: 'model',
          calibration: 'not-guaranteed'
        }
      },
      typedTextQuestionTypes: [...DECIMA_BASE_CLEAN_SPEC.typedQuestionTypes],
      imageChoice: {status: 'not-implemented'},
      rating: {status: 'not-implemented'}
    },
    verification: {
      currentBundleGpu: 'not-applicable',
      currentBundleCpuWasm: DECIMA_BASE_CLEAN_SPEC.cpuWasmVerification,
      currentBundleJavascript: 'not-applicable',
      priorEvidence: 'See the package README for this model.'
    }
  });
}

const description = describeBaseClean();

export {WebDecisionError};

export function listDecisionModels() {
  return MODEL_IDS;
}

export function isDecisionModel(value: unknown): boolean {
  return value === DECIMA_BASE_CLEAN_MODEL;
}

export function describeModel(model: unknown) {
  if (!isDecisionModel(model)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  return description;
}

export async function loadDecisionModel(options: unknown) {
  const checked = validateDecimaLoadOptions(options);
  if (checked.model !== DECIMA_BASE_CLEAN_MODEL)
    throw new WebDecisionError('INVALID_INPUT', 'Select the Decima base-clean int8 model.');
  const model = await loadDecimaModel(options);
  Object.defineProperty(model, 'description', {value: description});
  return model;
}

export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}
