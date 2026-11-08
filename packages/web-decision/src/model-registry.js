import {KEV_MAX_BRANCH_TOKENS, KEV_MAX_OPTIONS, KEV_MAX_STATE_TOKENS, WebDecisionError} from './kev.js';
import {KEV_REQUIRED_LIMITS} from './kev-gpu.js';
import {KEV06_MANIFEST_SHA256} from './kev06/contract.js';
import {DEBERTA_REQUIRED_LIMITS} from './deberta/device.mjs';
import debertaContract from './deberta/contract.json' with {type: 'json'};
import reflexesContract from './kev-reflexes-contract.json' with {type: 'json'};
import {MAPIKA_MODEL, MAPIKA_MANIFEST_SHA256, MAPIKA_LARGEST_BUFFER} from './mapika/contract.js';
import {MAPIKA_Q4_MODEL, MAPIKA_Q4_MANIFEST_SHA256, MAPIKA_Q4_LARGEST_BUFFER} from './mapika-q4/contract.js';
import {KEV08_MODEL, KEV08_MANIFEST_SHA256, KEV08_LARGEST_BUFFER} from './kev08/contract.js';
import {STRANDS_FUSED_GRAPH_MAX_TOKENS} from './strands/contract.mjs';
import {LARGEST_BINDING, MANIFEST_SHA256 as STRANDS_MANIFEST_SHA256} from './strands/manifest.mjs';
import {
  STRANDS_GEMMA4_E2B_BASE_MODEL,
  STRANDS_GEMMA4_E2B_BASE_REVISION,
  STRANDS_GEMMA4_E2B_GRAPH_SHA256,
  STRANDS_GEMMA4_E2B_LIMITS,
  STRANDS_GEMMA4_E2B_MANIFEST_SHA256,
  STRANDS_GEMMA4_E2B_MODEL,
  STRANDS_GEMMA4_E2B_REPOSITORY,
  STRANDS_GEMMA4_E2B_REVISION
} from './strands/gemma4-e2b-contract.mjs';
import {NARI_FULL, NARI_MIXED} from './nari/variants.mjs';
import {IMAJEV_MODEL} from './imajev/contract.mjs';
import {IMAJEV_METADATA_SHA256} from './imajev/source.mjs';
import {MAX_H2O_OPTIONS} from './h2o/protocol.mjs';
import {REVISION as H2O_REVISION} from './h2o/plan.mjs';
import {FLASH_GRAPH_SHA256, FLASH_LIMITS, FLASH_MANIFEST_SHA256, FLASH_MODEL} from './pplx/contract.mjs';
import {CLEF_GRAPH_SHA256, CLEF_INVENTORY_SHA256, CLEF_LIMITS, CLEF_MODEL} from './clef/contract.mjs';
import {
  OPENDECIDER_NANO_MANIFEST_SHA256,
  OPENDECIDER_NANO_MODEL,
  OPENDECIDER_NANO_LIMITS,
  OPENDECIDER_NANO_MAX_TOKENS
} from './opendecider-nano/contract.mjs';
import {
  elephantVariant,
  ELEPHANT_512_MODEL,
  ELEPHANT_INVENTORY_SHA256,
  ELEPHANT_LIMITS,
  ELEPHANT_MODEL
} from './elephant/contract.mjs';
import {
  MAPIKA4_DLIGHT_ALL_GRAPH_SHA256,
  MAPIKA4_DLIGHT_ALL_MODEL,
  MAPIKA4_DLIGHT_GRAPH_SHA256,
  MAPIKA4_DLIGHT_MODEL,
  MAPIKA4_GRAPH_SHA256,
  MAPIKA4_INVENTORY_SHA256,
  MAPIKA4_LIMITS,
  MAPIKA4_MODEL
} from './mapika4b/contract.mjs';
import {DECIMA_AGENT_SPEC, DECIMA_BASE_CLEAN_SPEC, DECIMA_SMALL_SPEC} from './decima/contract.mjs';
import {
  TINYDECIDE_FILES,
  TINYDECIDE_MAX_CHOICES,
  TINYDECIDE_MAX_QUESTION_TOKENS,
  TINYDECIDE_MAX_STATE_TOKENS,
  TINYDECIDE_MODEL,
  TINYDECIDE_REPOSITORY,
  TINYDECIDE_REVISION
} from './tinydecide/contract.mjs';
import {TAMEV_MICRO_PROFILE, TAMEV_NANO_PROFILE, TAMEV_NANO_FLASH_PROFILE} from './tamev/contract.mjs';
import {SKEJJA_INT8_PROFILE, SKEJJA_INT8DYN_PROFILE} from './skejja/contract.mjs';
import {ETTIN_FILES, ETTIN_PROFILE} from './ettin/contract.mjs';
import {
  SAFEDECIDE_FILES,
  SAFEDECIDE_MAX_SEQUENCE_TOKENS,
  SAFEDECIDE_MODEL,
  SAFEDECIDE_REPOSITORY,
  SAFEDECIDE_REVISION,
  SAFEDECIDE_SOURCE_REVISION
} from './safedecide/contract.mjs';
import {
  CRIVO_FP32_SPEC,
  CRIVO_INT8_SPEC,
  CRIVO_MAX_SEQUENCE_TOKENS,
  CRIVO_REPOSITORY,
  CRIVO_REVISION,
  CRIVO_SOURCE_REPOSITORY,
  CRIVO_SOURCE_REVISION
} from './crivo/contract.mjs';
import {
  LAYA_FILES,
  LAYA_MAX_OPTIONS,
  LAYA_MAX_SEQUENCE_TOKENS,
  LAYA_MODEL,
  LAYA_REPOSITORY,
  LAYA_REVISION
} from './laya/contract.mjs';
import {
  LAYA_TYPED_FILES,
  LAYA_TYPED_MAX_OPTIONS,
  LAYA_TYPED_MAX_SEQUENCE_TOKENS,
  LAYA_TYPED_MODEL,
  LAYA_TYPED_REPOSITORY,
  LAYA_TYPED_REVISION
} from './laya-typed/contract.mjs';
import {
  XDECISION_FILES,
  XDECISION_MAX_OPTIONS,
  XDECISION_MAX_SEQUENCE_TOKENS,
  XDECISION_MODEL,
  XDECISION_REPOSITORY,
  XDECISION_REVISION
} from './xdecision/contract.mjs';
import {
  GLINER_MAX_OPTIONS,
  GLINER_MAX_TOKENS,
  GLINER_MOBILE_FILES,
  GLINER_MOBILE_MODEL,
  GLINER_MOBILE_REPOSITORY,
  GLINER_MOBILE_REVISION,
  GLINER_SOURCE_REPOSITORY,
  GLINER_SOURCE_REVISION
} from './gliner-mobile/contract.mjs';
import {
  D1_OMNI_FILES,
  D1_OMNI_MAX_TOKENS,
  D1_OMNI_MODEL,
  D1_OMNI_REPOSITORY,
  D1_OMNI_REVISION,
  D1_OMNI_SOURCE_REPOSITORY,
  D1_OMNI_SOURCE_REVISION,
  D1_OMNI_TEMPERATURES,
  D1_OMNI_TOTAL_BYTES
} from './d1-omni/contract.mjs';
import {
  VERDICT_FILES,
  VERDICT_GLOBAL_TEMPERATURE,
  VERDICT_MAX_CANDIDATES,
  VERDICT_MAX_TOKENS,
  VERDICT_MODEL,
  VERDICT_REPOSITORY,
  VERDICT_REVISION,
  VERDICT_SOURCE_REPOSITORY,
  VERDICT_SOURCE_REVISION,
  VERDICT_TEMPERATURES,
  VERDICT_TOTAL_BYTES
} from './verdict/contract.mjs';

const DEFAULT_MODEL = 'kev-0.6b-q4f16';
const STRANDS_MODEL = 'experimental-strands-qwen-2b-fp16';

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

function tokenLimit(scope, max, description) {
  return {resource: 'tokens', scope, max, description};
}

function choice({
  tokenLimits,
  maxQuestions,
  minOptions = 2,
  maxOptions = 255,
  maxOptionsDescription,
  maxQuestionsDescription,
  maxTotalOptions,
  maxTotalOptionsDescription,
  handling,
  tokenCount,
  abstention = 'not-implemented',
  scoreBasis = 'choices',
  temperature,
  temperatureAppliedBy
}) {
  return {
    status: 'implemented',
    abstention,
    limits: [
      ...tokenLimits,
      {
        resource: 'options',
        scope: 'question',
        min: minOptions,
        ...(maxOptions === null ? {} : {max: maxOptions}),
        ...(maxOptionsDescription ? {description: maxOptionsDescription} : {})
      },
      ...(maxTotalOptions === undefined
        ? []
        : [
            {
              resource: 'options',
              scope: 'request',
              min: 2,
              max: maxTotalOptions,
              ...(maxTotalOptionsDescription ? {description: maxTotalOptionsDescription} : {})
            }
          ]),
      {
        resource: 'questions',
        scope: 'request',
        min: 1,
        ...(maxQuestions === undefined ? {} : {max: maxQuestions}),
        ...(maxQuestionsDescription ? {description: maxQuestionsDescription} : {})
      }
    ],
    handling,
    tokenCount,
    score: {basis: scoreBasis, temperature, temperatureAppliedBy, calibration: 'not-guaranteed'}
  };
}

function bindingLimits(bytes) {
  return {maxBufferSize: bytes, maxStorageBufferBindingSize: bytes, maxStorageBuffersPerShaderStage: 8};
}

function registration({
  family,
  model,
  precision,
  checkpoint,
  assets,
  requiredFiles,
  limits,
  maxQuestions,
  minOptions,
  maxOptions,
  maxOptionsDescription,
  maxQuestionsDescription,
  maxTotalOptions,
  maxTotalOptionsDescription,
  handling,
  tokenCount,
  abstention,
  scoreBasis,
  temperature,
  temperatureAppliedBy = 'decoder',
  typedTextQuestionTypes,
  selectiveAction,
  requiredLimits,
  requiresNonFallbackAdapter = false,
  backend = 'tvm-webgpu',
  gpuVerification = backend === 'tvm-webgpu' ? 'not-revalidated' : 'not-applicable',
  cpuWasmVerification = 'verified',
  javascriptVerification = 'not-revalidated',
  additionalFiles = 'Weight shards or ranges listed by the model manifest are also required.',
  modelTerms = 'separate-review-required',
  modelLicenseUrl,
  sourceModelCard
}) {
  return deepFreeze({
    family,
    description: {
      model,
      precision,
      checkpoint,
      assets: {
        ...assets,
        location: 'separate',
        requiredFiles,
        additionalFiles
      },
      licenses: {
        project: 'Apache-2.0',
        modelTerms,
        modelCard: `https://huggingface.co/${checkpoint.repository}/tree/${checkpoint.revision}`,
        ...(modelLicenseUrl ? {modelLicenseUrl} : {}),
        ...(sourceModelCard ? {sourceModelCard} : {})
      },
      backend:
        backend === 'tvm-webgpu'
          ? {name: backend, requiredFeatures: ['shader-f16'], requiredLimits, requiresNonFallbackAdapter}
          : backend === 'onnxruntime-web-wasm'
            ? {
                name: backend,
                executionProvider: 'wasm',
                intraOpNumThreads: 1,
                requiredFeatures: [],
                requiredLimits: {},
                requiresNonFallbackAdapter: false
              }
            : backend === 'onnxruntime-web-webgpu'
              ? {
                  name: backend,
                  executionProvider: 'webgpu',
                  cpuFallback: 'runtime-managed',
                  requiredFeatures: [],
                  requiredLimits: {},
                  requiresNonFallbackAdapter: true
                }
              : {name: 'javascript'},
      features: {
        textChoice: choice({
          tokenLimits: limits,
          maxQuestions,
          minOptions,
          maxOptions,
          maxOptionsDescription,
          maxQuestionsDescription,
          maxTotalOptions,
          maxTotalOptionsDescription,
          handling,
          tokenCount,
          abstention,
          scoreBasis,
          temperature,
          temperatureAppliedBy
        }),
        ...(typedTextQuestionTypes ? {typedTextQuestionTypes} : {}),
        ...(selectiveAction ? {selectiveAction} : {}),
        imageChoice: {status: 'not-implemented'},
        rating: {status: 'not-implemented'}
      },
      verification: {
        currentBundleGpu: gpuVerification,
        currentBundleCpuWasm:
          backend === 'onnxruntime-web-wasm'
            ? cpuWasmVerification
            : backend === 'onnxruntime-web-webgpu'
              ? 'not-revalidated'
              : 'not-applicable',
        currentBundleJavascript: backend === 'javascript' ? javascriptVerification : 'not-applicable',
        priorEvidence: 'See the package README for each model.'
      }
    }
  });
}

const kevLimits = [
  tokenLimit(
    'state',
    KEV_MAX_STATE_TOKENS - 1,
    'Native Kev state limit; the model counts its state marker within the 8,192-token budget.'
  ),
  tokenLimit(
    'question',
    KEV_MAX_BRANCH_TOKENS,
    'Native Kev per-question row limit, including the shared state and question branch. There is no fixed aggregate request-token cap; available memory limits the packed request.'
  )
];
const packed = 'packed-request';
const independent = 'independent-questions';
const sumRows = 'sum-question-rows';
const mapikaCheckpoint = {repository: 'Mapika/decider-0.8b', revision: 'a0a01d6f8135298f400a8c856b355793012ae971'};
const kevFiles = [
  'manifest.json',
  'model.wasm',
  'tokenizer.json',
  'tokenizer_config.json',
  'model_config.json',
  'weights/tensor-cache.json'
];
const textGraphFiles = ['manifest.json', 'model.wasm', 'tokenizer.json', 'tokenizer_config.json', 'weights/index.json'];

function decimaRegistration(spec) {
  const stateLimit =
    spec.stateOverflow === 'reject'
      ? 'longer input is rejected before inference.'
      : 'longer rows are truncated and marked.';
  const choiceLimit = spec.choiceFit
    ? 'Uses the checkpoint choice_fit rendering and reports the adapted rows; each row includes question text and special tokens.'
    : 'Each option row includes question text and special tokens; shortened rows are marked.';
  return registration({
    family: 'decima',
    model: spec.model,
    precision: 'int8',
    checkpoint: {repository: spec.repository, revision: spec.revision},
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(spec.files),
    limits: [
      tokenLimit('state', spec.maxStateTokens, `Complete question and state row; ${stateLimit}`),
      tokenLimit('question', spec.maxChoiceTokens, choiceLimit)
    ],
    maxOptions: null,
    handling: independent,
    tokenCount: sumRows,
    temperature: spec.temperature,
    temperatureAppliedBy: 'model',
    backend: 'onnxruntime-web-wasm',
    cpuWasmVerification: spec.cpuWasmVerification,
    typedTextQuestionTypes: spec.typedQuestionTypes,
    additionalFiles: 'No additional files are required.'
  });
}

function tamevRegistration(profile, cpuWasmVerification) {
  return registration({
    family: 'tamev',
    model: profile.model,
    precision: 'int8',
    checkpoint: {repository: profile.repository, revision: profile.revision},
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(profile.files),
    limits: [
      tokenLimit(
        'state',
        profile.maxContextTokens,
        'Combined state and question context row, including special tokens; right truncation is reported.'
      ),
      tokenLimit(
        'question',
        profile.maxOptionTokens,
        'Each candidate option row, including special tokens; right truncation is reported.'
      )
    ],
    maxOptions: null,
    handling: independent,
    tokenCount: sumRows,
    temperature: profile.temperature,
    temperatureAppliedBy: 'model',
    backend: 'onnxruntime-web-wasm',
    cpuWasmVerification,
    typedTextQuestionTypes: ['choice', 'noul', 'score'],
    additionalFiles: 'No additional files are required.'
  });
}

function skejjaRegistration(profile, cpuWasmVerification) {
  return registration({
    family: 'skejja',
    model: profile.model,
    precision: 'int8',
    checkpoint: {
      repository: profile.repository,
      revision: profile.revision,
      graphSha256: profile.files[profile.onnxFile].sha256
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(profile.files),
    limits: [
      tokenLimit(
        'question',
        512,
        'Each question row is at most 512 tokens after adding its labels. Oversized state text is shortened from both ends.'
      ),
      {
        resource: 'questions',
        scope: 'request',
        min: 1,
        description:
          'No package request-count cap. The upstream server defaults to 256 questions but allows a configured max_questions value.'
      }
    ],
    maxOptions: profile.maxChoices,
    handling: independent,
    tokenCount: sumRows,
    temperature: profile.temperature,
    temperatureAppliedBy: 'decoder',
    backend: 'onnxruntime-web-wasm',
    cpuWasmVerification,
    typedTextQuestionTypes: ['choice', 'noul', 'score'],
    additionalFiles:
      'The pinned archive also includes its NOTICE and the alternate int8 ONNX graph with sidecar. Only the graph and sidecar listed for this variant are loaded.'
  });
}

function ettinRegistration(cpuWasmVerification) {
  return registration({
    family: 'ettin',
    model: ETTIN_PROFILE.model,
    precision: 'int8',
    checkpoint: {
      repository: ETTIN_PROFILE.repository,
      revision: ETTIN_PROFILE.revision,
      graphSha256: ETTIN_FILES['model_int8.onnx'].sha256
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(ETTIN_FILES),
    limits: [
      tokenLimit(
        'question',
        ETTIN_PROFILE.maxSequenceTokens,
        'Each question row includes model markers and options; long state and question text is proportionally truncated from the end.'
      )
    ],
    maxOptions: ETTIN_PROFILE.maxOptions,
    handling: independent,
    tokenCount: sumRows,
    temperature: 1,
    scoreBasis: 'ordered-levels',
    backend: 'onnxruntime-web-wasm',
    cpuWasmVerification,
    typedTextQuestionTypes: ['choice', 'noul', 'score'],
    additionalFiles: 'No additional files are required.'
  });
}

const models = [
  decimaRegistration(DECIMA_SMALL_SPEC),
  decimaRegistration(DECIMA_BASE_CLEAN_SPEC),
  decimaRegistration(DECIMA_AGENT_SPEC),
  tamevRegistration(TAMEV_MICRO_PROFILE, 'verified'),
  tamevRegistration(TAMEV_NANO_PROFILE, 'verified'),
  tamevRegistration(TAMEV_NANO_FLASH_PROFILE, 'verified'),
  skejjaRegistration(SKEJJA_INT8_PROFILE, 'verified'),
  skejjaRegistration(SKEJJA_INT8DYN_PROFILE, 'verified'),
  ettinRegistration('verified'),
  registration({
    family: 'laya',
    model: LAYA_MODEL,
    precision: 'f16',
    checkpoint: {
      repository: LAYA_REPOSITORY,
      revision: LAYA_REVISION,
      graphSha256: LAYA_FILES['onnx/model_fp16.onnx'].sha256
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(LAYA_FILES),
    limits: [
      tokenLimit(
        'question',
        LAYA_MAX_SEQUENCE_TOKENS,
        'One sequence per question; state, instructions and options are shortened and reported to fit the 1024-token context.'
      ),
      {resource: 'options', scope: 'question', min: 2, max: LAYA_MAX_OPTIONS},
      {resource: 'questions', scope: 'request', min: 1}
    ],
    maxOptions: LAYA_MAX_OPTIONS,
    handling: independent,
    tokenCount: sumRows,
    temperature: 1,
    scoreBasis: 'ordered-levels',
    backend: 'onnxruntime-web-webgpu',
    gpuVerification: 'verified',
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    additionalFiles: 'No additional files are required.'
  }),
  registration({
    family: 'laya-typed',
    model: LAYA_TYPED_MODEL,
    precision: 'f16',
    checkpoint: {
      repository: LAYA_TYPED_REPOSITORY,
      revision: LAYA_TYPED_REVISION,
      graphSha256: LAYA_TYPED_FILES['onnx/model_fp16.onnx'].sha256
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(LAYA_TYPED_FILES),
    limits: [
      tokenLimit(
        'question',
        LAYA_TYPED_MAX_SEQUENCE_TOKENS,
        'One sequence per question; state, instructions and options are shortened and reported to fit the 1024-token context.'
      ),
      {resource: 'options', scope: 'question', min: 2, max: LAYA_TYPED_MAX_OPTIONS},
      {resource: 'questions', scope: 'request', min: 1}
    ],
    maxOptions: LAYA_TYPED_MAX_OPTIONS,
    handling: independent,
    tokenCount: sumRows,
    temperature: 1,
    scoreBasis: 'ordered-levels',
    backend: 'onnxruntime-web-webgpu',
    gpuVerification: 'verified',
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    additionalFiles: 'No additional files are required.'
  }),
  registration({
    family: 'xdecision',
    model: XDECISION_MODEL,
    precision: 'f16',
    checkpoint: {
      repository: XDECISION_REPOSITORY,
      revision: XDECISION_REVISION,
      graphSha256: XDECISION_FILES['onnx/model_fp16.onnx'].sha256
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(XDECISION_FILES),
    limits: [
      tokenLimit(
        'question',
        XDECISION_MAX_SEQUENCE_TOKENS,
        'One sequence per question; state, instructions and options are shortened and reported to fit the 1024-token context.'
      ),
      {resource: 'options', scope: 'question', min: 2, max: XDECISION_MAX_OPTIONS},
      {resource: 'questions', scope: 'request', min: 1}
    ],
    maxOptions: XDECISION_MAX_OPTIONS,
    handling: independent,
    tokenCount: sumRows,
    temperature: {
      byType: {choice: 1.8352, score: 2.1167, noul: 1.6012},
      byQuestionTypeAndOptionCount: {
        'choice:2': 1.819,
        'choice:3-5': 1.8945,
        'choice:6-10': 1.774,
        'noul:2': 1.6012,
        'score:3-5': 2.1167
      }
    },
    temperatureAppliedBy: 'decoder',
    scoreBasis: 'ordered-levels',
    backend: 'onnxruntime-web-webgpu',
    gpuVerification: 'verified',
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    additionalFiles: 'No additional files are required.'
  }),
  registration({
    family: 'gliner-mobile',
    model: GLINER_MOBILE_MODEL,
    precision: 'q4f16',
    checkpoint: {
      repository: GLINER_MOBILE_REPOSITORY,
      revision: GLINER_MOBILE_REVISION,
      sourceRepository: GLINER_SOURCE_REPOSITORY,
      sourceRevision: GLINER_SOURCE_REVISION,
      graphSha256: GLINER_MOBILE_FILES['onnx/model_q4f16.onnx'].sha256
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(GLINER_MOBILE_FILES),
    limits: [
      tokenLimit(
        'request',
        GLINER_MAX_TOKENS,
        'All question schemas and state share one sequence. Long state words are shortened and truncation is reported.'
      ),
      tokenLimit('state', GLINER_MAX_TOKENS, 'State shares the 512-token context with every question schema.'),
      {resource: 'options', scope: 'question', min: 2, max: GLINER_MAX_OPTIONS},
      {resource: 'questions', scope: 'request', min: 1}
    ],
    maxOptions: GLINER_MAX_OPTIONS,
    handling: 'packed-request',
    tokenCount: 'packed-request',
    temperature: 1,
    scoreBasis: 'ordered-levels',
    backend: 'onnxruntime-web-webgpu',
    gpuVerification: 'verified',
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    additionalFiles: 'No additional files are required.'
  }),
  registration({
    family: 'tinydecide',
    model: TINYDECIDE_MODEL,
    precision: 'q4',
    checkpoint: {
      repository: TINYDECIDE_REPOSITORY,
      revision: TINYDECIDE_REVISION,
      weightSha256: TINYDECIDE_FILES['model.bin'].sha256
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(TINYDECIDE_FILES),
    limits: [
      tokenLimit(
        'state',
        TINYDECIDE_MAX_STATE_TOKENS,
        'Reads at most the first 127 state tokens and reports truncation.'
      ),
      tokenLimit('question', TINYDECIDE_MAX_QUESTION_TOKENS, 'Includes the question, options and special tokens.')
    ],
    maxOptions: TINYDECIDE_MAX_CHOICES,
    handling: packed,
    tokenCount: packed,
    temperature: 1.149999976158142,
    temperatureAppliedBy: 'model',
    backend: 'javascript',
    javascriptVerification: 'verified',
    typedTextQuestionTypes: ['choice', 'noul', 'score', 'span'],
    additionalFiles: 'No additional files are required.'
  }),
  registration({
    family: 'kev',
    model: DEFAULT_MODEL,
    precision: 'q4f16',
    checkpoint: {repository: 'onnx-community/kev-0.6b-ONNX', revision: '5a827091b3d58fe4fdfc02a5a5b2ce364047d678'},
    assets: {integrity: 'pinned-manifest', manifestSha256: KEV06_MANIFEST_SHA256},
    requiredFiles: kevFiles,
    limits: kevLimits,
    maxOptions: KEV_MAX_OPTIONS,
    maxOptionsDescription: 'Upstream Kev config declares a maximum of 255 options per question.',
    minOptions: 1,
    handling: packed,
    tokenCount: packed,
    temperature: 1,
    typedTextQuestionTypes: ['choice', 'noul', 'score'],
    requiredLimits: KEV_REQUIRED_LIMITS,
    gpuVerification: 'verified'
  }),
  registration({
    family: 'kev',
    model: reflexesContract.model,
    precision: 'q4f16',
    checkpoint: reflexesContract.checkpoint,
    assets: {integrity: 'pinned-files'},
    requiredFiles: [
      ...kevFiles,
      'heads/param_394.bin',
      'heads/param_395.bin',
      'heads/param_396.bin',
      'heads/param_397.bin'
    ],
    limits: kevLimits,
    maxOptions: KEV_MAX_OPTIONS,
    maxOptionsDescription: 'Upstream Kev config declares a maximum of 255 options per question.',
    minOptions: 1,
    handling: packed,
    tokenCount: packed,
    temperature: 1,
    typedTextQuestionTypes: ['choice', 'noul', 'score'],
    requiredLimits: KEV_REQUIRED_LIMITS,
    gpuVerification: 'not-revalidated'
  }),
  registration({
    family: 'deberta',
    model: debertaContract.model,
    precision: 'q4f16',
    checkpoint: debertaContract.checkpoint,
    assets: {integrity: 'pinned-files'},
    requiredFiles: [...kevFiles, 'mapping.json', 'split.json'],
    limits: [
      tokenLimit('request', debertaContract.maxInputTokens, 'Complete packed request, including markers.'),
      tokenLimit('state', debertaContract.maxStateTokens, 'State content tokens.')
    ],
    handling: packed,
    tokenCount: packed,
    temperature: 1.05,
    requiredLimits: DEBERTA_REQUIRED_LIMITS
  }),
  registration({
    family: 'mapika',
    model: MAPIKA_MODEL,
    precision: 'f16',
    checkpoint: mapikaCheckpoint,
    assets: {integrity: 'pinned-manifest', manifestSha256: MAPIKA_MANIFEST_SHA256},
    requiredFiles: textGraphFiles,
    limits: [tokenLimit('request', 512, 'Complete packed request, including prompt and answer slots.')],
    handling: packed,
    tokenCount: packed,
    temperature: 1.03,
    requiredLimits: bindingLimits(MAPIKA_LARGEST_BUFFER)
  }),
  registration({
    family: 'mapika-q4',
    model: MAPIKA_Q4_MODEL,
    precision: 'q4f16',
    checkpoint: mapikaCheckpoint,
    assets: {integrity: 'pinned-manifest', manifestSha256: MAPIKA_Q4_MANIFEST_SHA256},
    requiredFiles: textGraphFiles,
    limits: [tokenLimit('request', 128, 'Complete packed request, including prompt and answer slots.')],
    handling: packed,
    tokenCount: packed,
    temperature: 1.03,
    requiredLimits: bindingLimits(MAPIKA_Q4_LARGEST_BUFFER)
  }),
  registration({
    family: 'kev08',
    model: KEV08_MODEL,
    precision: 'f16',
    checkpoint: {repository: 'jaredpalmer/kev-0.8b', revision: 'bf75a6a8848ea6960ff2ed108d9ed44c2941174f'},
    assets: {integrity: 'pinned-manifest', manifestSha256: KEV08_MANIFEST_SHA256},
    requiredFiles: textGraphFiles,
    limits: [
      tokenLimit('question', 128, 'Each question row includes the shared state and model markers.'),
      tokenLimit('state', 127, 'State content tokens; the state marker uses one more token.')
    ],
    handling: independent,
    tokenCount: sumRows,
    temperature: 2.3510958125672174,
    temperatureAppliedBy: 'model',
    requiredLimits: bindingLimits(KEV08_LARGEST_BUFFER)
  }),
  registration({
    family: 'h2o',
    model: 'experimental-h2o-lightning-4b-q4f16',
    precision: 'q4f16',
    checkpoint: {
      repository: 'h2oai/h2o-lightning-4b',
      revision: H2O_REVISION,
      weightSha256: 'c9b6290d4c0c4338c37ce37425a978027473e4d45b84e038da2369b76ff8a377',
      graphSha256: '62fb8ddb2b98106cb24f925a585f930f87d9f6df25e32deb342aa3ae37850a50'
    },
    assets: {
      integrity: 'pinned-manifest',
      manifestSha256: '27f83988992c9d9fe2febe924848259890bcb4d39e1c57cb225dd8cb22610e9a'
    },
    requiredFiles: [
      'evidence/inventory.json',
      'h2o-public-metadata.json',
      'manifest.pending.json',
      'model.pending.wasm',
      'tokenizer/tokenizer.json',
      'tokenizer/tokenizer_config.json',
      'tokenizer_contract.json'
    ],
    limits: [tokenLimit('question', 512, 'Complete prompt for each separate question, including state and labels.')],
    maxOptions: MAX_H2O_OPTIONS,
    handling: independent,
    tokenCount: sumRows,
    temperature: 0.75,
    requiredLimits: bindingLimits(134216960)
  }),
  registration({
    family: 'nari',
    model: NARI_FULL.model,
    precision: 'q4f16',
    checkpoint: {
      repository: 'blackmallangcow/nari-decision-4b',
      revision: '029c4628a38859dce5b0db1742e9c97545aabe12',
      weightSha256: 'eb0ba6323ff262c5e66c90bece8d5d9519f428b2d04d07a4500a28e12770d77b',
      graphSha256: '20379bdae7e66ce31d241d8fe1b07e9ce15168dab11db898ab91fde6eb8afd92'
    },
    assets: {
      integrity: 'pinned-manifest',
      manifestSha256: NARI_FULL.metadataSha256
    },
    requiredFiles: ['nari-metadata.json', 'model.wasm', 'tokenizer.json', 'tokenizer_config.json', 'config.json'],
    limits: [
      tokenLimit(
        'request',
        361,
        'Current experimental package graph envelope, including prompt and answer slots; pinned model config allows max_len 2048.'
      )
    ],
    maxQuestions: 8,
    maxQuestionsDescription:
      'Current experimental package graph envelope; pinned model source has dynamic question rows and no question-count maximum.',
    maxOptions: 64,
    maxOptionsDescription:
      'Current experimental package graph envelope; pinned model source has dynamic option rows and no option-count maximum.',
    maxTotalOptions: 64,
    maxTotalOptionsDescription:
      'Current experimental package graph envelope, validated through 64 total options; not a pinned model capacity limit.',
    handling: packed,
    tokenCount: packed,
    temperature: {
      byOptionCount: [
        {min: 2, max: 2, value: 1},
        {min: 3, max: 5, value: 1.1189},
        {min: 6, max: 26, value: 0.9632},
        {min: 27, max: 64, value: 1.1189}
      ]
    },
    requiredLimits: {
      maxBufferSize: 134184960,
      maxStorageBufferBindingSize: 134184960,
      maxStorageBuffersPerShaderStage: 6
    }
  }),
  registration({
    family: 'strands',
    model: STRANDS_MODEL,
    precision: 'f16',
    checkpoint: {
      repository: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
      revision: '216d1ccb46a9e9b036eb9187b248db98480f77d4'
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: STRANDS_MANIFEST_SHA256},
    requiredFiles: [
      'fused-ordered-manifest-4096-candidate-v1.json',
      'full-fused-tvm-4096-candidate-v1/model.wasm',
      'checkpoint/tokenizer.json',
      'checkpoint/tokenizer_config.json'
    ],
    limits: [
      tokenLimit(
        'question',
        STRANDS_FUSED_GRAPH_MAX_TOKENS,
        'The compiled fused TVM candidate has a dynamic row shape from 2 through 4,096 tokens; requests above 4,096 are rejected without truncation.'
      )
    ],
    handling: independent,
    tokenCount: sumRows,
    temperature: 0.8629556148274228,
    requiredLimits: bindingLimits(LARGEST_BINDING),
    requiresNonFallbackAdapter: true
  }),
  registration({
    family: 'strands-gemma4',
    model: STRANDS_GEMMA4_E2B_MODEL,
    precision: 'f16',
    checkpoint: {
      repository: STRANDS_GEMMA4_E2B_REPOSITORY,
      revision: STRANDS_GEMMA4_E2B_REVISION,
      graphSha256: STRANDS_GEMMA4_E2B_GRAPH_SHA256,
      sourceRepository: STRANDS_GEMMA4_E2B_BASE_MODEL,
      sourceRevision: STRANDS_GEMMA4_E2B_BASE_REVISION
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: STRANDS_GEMMA4_E2B_MANIFEST_SHA256},
    requiredFiles: [
      'runtime-manifest-v26.json',
      'graph-v26/model.wasm',
      'graph-v26/input-specs.json',
      'checkpoint/tokenizer.json',
      'checkpoint/tokenizer_config.json',
      'tensor-cache-e2b-v26-fp16/tensor-cache-manifest.json',
      'tensor-cache-e2b-v26-fp16/tensor-cache.json',
      'tensor-cache-e2b-v26-fp16/tensor-source-map.json'
    ],
    limits: [
      tokenLimit(
        'question',
        4096,
        'Each independently encoded state and typed question row must fit the pinned 4,096-token context. Inputs are rejected when they exceed it; the adapter does not truncate.'
      )
    ],
    maxOptions: null,
    handling: independent,
    tokenCount: sumRows,
    scoreBasis: 'choices',
    temperature: {
      byType: {choice: 0.5608428296613732, noul: 1.0704394404346953, score: 1.4788446944896205},
      byQuestionTypeAndOptionCount: {}
    },
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    requiredLimits: STRANDS_GEMMA4_E2B_LIMITS,
    requiresNonFallbackAdapter: true,
    modelTerms:
      'Publisher-declared Apache-2.0; checkpoint provenance is incomplete and the prepared FP16 cache is a lossy conversion.',
    modelLicenseUrl: 'https://ai.google.dev/gemma/docs/gemma_4_license',
    additionalFiles:
      'Applications must separately host all 82 pinned FP16 cache shards (9,556,494,410 tensor bytes) and the remaining files listed by runtime-manifest-v26.json. The npm package contains no model weights. The cache was converted from BF16 and is lossy.'
  }),
  registration({
    family: 'opendecider-nano',
    model: OPENDECIDER_NANO_MODEL,
    precision: 'f16',
    checkpoint: {
      repository: 'manjunathshiva/opendecider-nano-ONNX',
      revision: '5137ae7b4f1cf8c55f1d05d3f5cdd3694d92da3e',
      graphSha256: '3c92c5e4dd01caae3afe4d3d9ffce7295343fe6fecceb39b030d895dd8fa4488'
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: OPENDECIDER_NANO_MANIFEST_SHA256},
    requiredFiles: [
      'manifest.json',
      'model.wasm',
      'tokenizer.json',
      'tokenizer_config.json',
      'weights/tensor-cache.json'
    ],
    limits: [
      tokenLimit('question', OPENDECIDER_NANO_MAX_TOKENS, 'Complete independent question row after state truncation.')
    ],
    maxOptions: null,
    handling: independent,
    tokenCount: 'max-question-rows',
    temperature: 1,
    requiredLimits: OPENDECIDER_NANO_LIMITS,
    requiresNonFallbackAdapter: true
  }),
  registration({
    family: 'safedecide',
    model: SAFEDECIDE_MODEL,
    precision: 'f32',
    checkpoint: {
      repository: SAFEDECIDE_REPOSITORY,
      revision: SAFEDECIDE_REVISION,
      graphSha256: SAFEDECIDE_FILES['onnx/model.onnx'].sha256,
      sourceRepository: 'yotambraun/safedecide',
      sourceRevision: SAFEDECIDE_SOURCE_REVISION
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(SAFEDECIDE_FILES),
    limits: [
      tokenLimit(
        'question',
        SAFEDECIDE_MAX_SEQUENCE_TOKENS,
        'Each state/question/option pair is right-truncated with longest-first to 640 tokens.'
      )
    ],
    maxOptions: null,
    handling: 'independent-options',
    tokenCount: 'sum-option-pairs',
    temperature: {
      byType: {choice: 1.1134611876261922, noul: 1.1006431532659389, score: 1.089708955367956},
      byQuestionTypeAndOptionCount: {}
    },
    scoreBasis: 'ordered-levels',
    backend: 'onnxruntime-web-webgpu',
    gpuVerification: 'not-revalidated',
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    selectiveAction: {
      status: 'implemented',
      actions: ['automate', 'escalate'],
      calibration: 'exchangeable-data',
      levels: [
        {alpha: 0.05, delta: 0.1, threshold: null},
        {alpha: 0.1, delta: 0.1, threshold: 0.7},
        {alpha: 0.2, delta: 0.1, threshold: 0.5}
      ]
    },
    modelTerms: 'Apache-2.0 as declared by the SafeDecide model card',
    sourceModelCard: `https://github.com/yotambraun/safedecide/tree/${SAFEDECIDE_SOURCE_REVISION}`,
    additionalFiles: 'No additional files are required.'
  }),
  ...[CRIVO_FP32_SPEC, CRIVO_INT8_SPEC].map((spec) =>
    registration({
      family: 'crivo',
      model: spec.model,
      precision: spec.precision,
      checkpoint: {
        repository: CRIVO_REPOSITORY,
        revision: CRIVO_REVISION,
        graphSha256: spec.files[spec.graphFile].sha256,
        sourceRepository: CRIVO_SOURCE_REPOSITORY,
        sourceRevision: CRIVO_SOURCE_REVISION
      },
      assets: {integrity: 'pinned-files'},
      requiredFiles: Object.keys(spec.files),
      additionalFiles:
        'The model graph and tokenizer are separate CC BY-NC 4.0 assets. The npm package does not include model files.',
      limits: [
        tokenLimit(
          'question',
          CRIVO_MAX_SEQUENCE_TOKENS,
          'One sequence per question. Inputs longer than the training context are rejected without truncation.'
        ),
        {resource: 'options', scope: 'question', min: 2},
        {resource: 'questions', scope: 'request', min: 1}
      ],
      maxOptions: null,
      handling: independent,
      tokenCount: sumRows,
      temperature: 1,
      scoreBasis: 'ordered-levels',
      backend: spec.backend,
      gpuVerification: spec.backend === 'onnxruntime-web-webgpu' ? 'not-revalidated' : 'not-applicable',
      cpuWasmVerification: spec.backend === 'onnxruntime-web-wasm' ? 'verified' : 'not-applicable',
      typedTextQuestionTypes: ['choice', 'score', 'noul'],
      modelTerms: 'CC BY-NC 4.0; non-commercial use only.',
      modelLicenseUrl: 'https://creativecommons.org/licenses/by-nc/4.0/',
      sourceModelCard: 'https://huggingface.co/' + CRIVO_SOURCE_REPOSITORY + '/tree/' + CRIVO_SOURCE_REVISION
    })
  )
];
const h2oDescription = models.find(
  (item) => item.description.model === 'experimental-h2o-lightning-4b-q4f16'
).description;
models.push(
  deepFreeze({
    family: 'h2o',
    description: {
      ...h2oDescription,
      model: 'experimental-h2o-lightning-4b-q4f16-dlight',
      checkpoint: {
        ...h2oDescription.checkpoint,
        graphSha256: '32da4bd915dec178f29828acd2a57a202df069f4c6c0f13b0d9e7c55ff782336'
      },
      assets: {
        ...h2oDescription.assets,
        requiredFiles: [
          ...h2oDescription.assets.requiredFiles.filter((path) => path !== 'model.pending.wasm'),
          'model.dlight.pending.wasm',
          'h2o-dlight-metadata.json'
        ]
      }
    }
  }),
  deepFreeze({
    family: 'h2o',
    description: {
      ...h2oDescription,
      model: 'experimental-h2o-lightning-4b-q4f16-dual',
      checkpoint: {
        ...h2oDescription.checkpoint,
        graphSha256: 'd4531ab4a4690b192ba205c42a4e9ca0117d09e9454ff90ee9958306f35ce9b8'
      },
      assets: {
        ...h2oDescription.assets,
        requiredFiles: [
          ...h2oDescription.assets.requiredFiles.filter((path) => path !== 'model.pending.wasm'),
          'model.dual.pending.wasm',
          'h2o-dual-metadata.json'
        ]
      }
    }
  })
);
const nariDescription = models.find((item) => item.description.model === NARI_FULL.model).description;
models.push(
  deepFreeze({
    family: 'nari',
    description: {
      ...nariDescription,
      model: NARI_MIXED.model,
      precision: 'mixed-f16-q4f16',
      checkpoint: {
        ...nariDescription.checkpoint,
        graphSha256: '07901194847cc394398f28b31006e5170ab4f45bb24480cf3c4ddcb471ba898f'
      },
      assets: {...nariDescription.assets, manifestSha256: NARI_MIXED.metadataSha256}
    }
  })
);
models.push(
  registration({
    family: 'imajev',
    model: IMAJEV_MODEL,
    precision: 'q4f16',
    checkpoint: {
      repository: 'mohit67890/imajev-4b',
      revision: '27d4e54fc11efeea2ea4d10e910643440d2333b5',
      graphSha256: '1166b2d125328550b3ab5b523de28902237dada6bd383aa293c0a5f5563161ba'
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: IMAJEV_METADATA_SHA256},
    requiredFiles: [
      'imajev-metadata.json',
      'model.wasm',
      'tokenizer.json',
      'tokenizer_config.json',
      'decision_readout.json',
      'calibration.json'
    ],
    limits: [tokenLimit('question', 128, 'Each question includes state, protocol text, choices and unknown.')],
    maxQuestions: 8,
    maxQuestionsDescription:
      'Pinned Jev public serving API limit; the package processes each question as an independent row.',
    maxOptions: 255,
    maxOptionsDescription:
      'Pinned Jev serving API permits 255 options with this route’s extended 256-code readout; the default 255-code readout permits 254.',
    handling: independent,
    tokenCount: 'max-question-rows',
    abstention: 'implemented',
    scoreBasis: 'conditional-known-choices',
    temperature: 1.3051569717552742,
    requiredLimits: {
      maxBufferSize: 134184960,
      maxStorageBufferBindingSize: 134184960,
      maxStorageBuffersPerShaderStage: 6
    }
  })
);
models.push(
  registration({
    family: 'pplx',
    model: FLASH_MODEL,
    precision: 'q4f16',
    checkpoint: {
      repository: 'perplexity-ai/pplx-decider-flash-v1.2',
      revision: '9ff7cf9838e59df461c276b7699d56b244d2af58',
      graphSha256: FLASH_GRAPH_SHA256
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: FLASH_MANIFEST_SHA256},
    requiredFiles: ['manifest.pending.json', 'model.pending.wasm', 'tokenizer.json', 'tokenizer_config.json'],
    limits: [tokenLimit('request', 512, 'Complete joint text request including protocol and question markers.')],
    maxQuestions: 255,
    maxQuestionsDescription:
      'The exported marker axis supports 1–255 readouts; the separate 512-token sequence budget may bind first.',
    minOptions: 1,
    maxOptionsDescription: 'The graph has 255 answer-code logits per question.',
    handling: packed,
    tokenCount: packed,
    temperature: 1.5539092221462472,
    requiredLimits: FLASH_LIMITS,
    requiresNonFallbackAdapter: true
  })
);
function registerMapika4(model, graphSha256) {
  return registration({
    family: 'mapika4b',
    model,
    precision: 'q4f16',
    checkpoint: {
      repository: 'Mapika/decider-4b',
      revision: '585473dec4fa93f0325c2bf02dda7273b61f44c1',
      graphSha256
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: MAPIKA4_INVENTORY_SHA256},
    requiredFiles: ['inventory.json', 'model.wasm', 'tokenizer.json', 'tokenizer_config.json'],
    limits: [tokenLimit('request', 128, 'Complete joint text request including choices and answer slots.')],
    maxQuestions: 255,
    maxQuestionsDescription:
      'The compiled slot axis supports 1–255 questions; the separate 128-token sequence budget may bind first.',
    maxOptionsDescription: 'The graph has 255 answer-code logits per question.',
    handling: packed,
    tokenCount: packed,
    temperature: 2.098,
    requiredLimits: MAPIKA4_LIMITS,
    requiresNonFallbackAdapter: true
  });
}
models.push(registerMapika4(MAPIKA4_MODEL, MAPIKA4_GRAPH_SHA256));
models.push(registerMapika4(MAPIKA4_DLIGHT_MODEL, MAPIKA4_DLIGHT_GRAPH_SHA256));
models.push(registerMapika4(MAPIKA4_DLIGHT_ALL_MODEL, MAPIKA4_DLIGHT_ALL_GRAPH_SHA256));
function registerElephant(model) {
  const {graphSha256, maxBranchTokens} = elephantVariant(model);
  return registration({
    family: 'elephant',
    model,
    precision: 'q4f16',
    checkpoint: {
      repository: 'baselabs/elephant-1',
      revision: 'b3f8a93462fb1051a9de031e94a7ff380e254072',
      graphSha256
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: ELEPHANT_INVENTORY_SHA256},
    requiredFiles: ['inventory.json', 'model.wasm', 'tokenizer.json', 'tokenizer_config.json'],
    limits: [
      tokenLimit(
        'question',
        maxBranchTokens,
        'One complete state and question branch including model markers. Packed-request memory scales with branch count and size.'
      )
    ],
    handling: independent,
    tokenCount: 'shared-prefix-and-branches',
    temperature: 1,
    requiredLimits: ELEPHANT_LIMITS,
    requiresNonFallbackAdapter: true
  });
}
models.push(registerElephant(ELEPHANT_MODEL));
models.push(registerElephant(ELEPHANT_512_MODEL));
models.push(
  registration({
    family: 'clef',
    model: CLEF_MODEL,
    precision: 'q4f16',
    checkpoint: {
      repository: 'Cloudflare/clef-flash',
      revision: 'fb4417d31c1d0f9e7e77f1141b40aaede7491e00',
      graphSha256: CLEF_GRAPH_SHA256
    },
    assets: {integrity: 'pinned-manifest', manifestSha256: CLEF_INVENTORY_SHA256},
    requiredFiles: ['inventory.json', 'model.wasm', 'tokenizer.json', 'tokenizer_config.json'],
    limits: [
      tokenLimit(
        'request',
        512,
        'Current package graph envelope for the complete packed request; symbolic axes have no larger runtime validation evidence.'
      )
    ],
    maxQuestions: 8,
    maxQuestionsDescription:
      'Current package graph-input envelope; pinned source and exported graph use dynamic question axes, with no larger runtime evidence.',
    maxOptions: 64,
    maxOptionsDescription:
      'Current package graph-input envelope; pinned source and exported graph use dynamic option axes, with no larger runtime evidence.',
    maxTotalOptions: 64,
    maxTotalOptionsDescription:
      'Current package graph-input envelope, validated through 64 options total; not a pinned model or serving API maximum.',
    handling: packed,
    tokenCount: packed,
    temperature: 1,
    requiredLimits: CLEF_LIMITS,
    requiresNonFallbackAdapter: true
  })
);
models.push(
  registration({
    family: 'd1-omni',
    model: D1_OMNI_MODEL,
    precision: 'q8',
    checkpoint: {
      repository: D1_OMNI_REPOSITORY,
      revision: D1_OMNI_REVISION,
      sourceRepository: D1_OMNI_SOURCE_REPOSITORY,
      sourceRevision: D1_OMNI_SOURCE_REVISION
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(D1_OMNI_FILES),
    additionalFiles: `Pinned text-only Q8 ONNX assets total ${D1_OMNI_TOTAL_BYTES} bytes. The image and audio encoders are not included.`,
    limits: [
      tokenLimit(
        'question',
        D1_OMNI_MAX_TOKENS,
        'Each text question is encoded independently; state and option text follow the upstream truncation rules.'
      )
    ],
    maxOptions: null,
    handling: independent,
    tokenCount: sumRows,
    temperature: {
      byType: {choice: D1_OMNI_TEMPERATURES.choice, score: D1_OMNI_TEMPERATURES.score, noul: D1_OMNI_TEMPERATURES.noul},
      byQuestionTypeAndOptionCount: Object.fromEntries(
        Object.entries(D1_OMNI_TEMPERATURES).filter(([key]) => key.includes(':'))
      )
    },
    temperatureAppliedBy: 'decoder',
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    scoreBasis: 'ordered-levels',
    backend: 'onnxruntime-web-webgpu',
    gpuVerification: 'not-revalidated',
    requiresNonFallbackAdapter: true,
    modelTerms: 'LFM Open License v1.0; review the revenue threshold and redistribution conditions before use.',
    modelLicenseUrl: `https://huggingface.co/${D1_OMNI_SOURCE_REPOSITORY}/blob/${D1_OMNI_SOURCE_REVISION}/LICENSE`,
    sourceModelCard: `https://huggingface.co/${D1_OMNI_SOURCE_REPOSITORY}/tree/${D1_OMNI_SOURCE_REVISION}`
  })
);
models.push(
  registration({
    family: 'verdict',
    model: VERDICT_MODEL,
    precision: 'f16',
    checkpoint: {
      repository: VERDICT_REPOSITORY,
      revision: VERDICT_REVISION,
      sourceRepository: VERDICT_SOURCE_REPOSITORY,
      sourceRevision: VERDICT_SOURCE_REVISION
    },
    assets: {integrity: 'pinned-files'},
    requiredFiles: Object.keys(VERDICT_FILES),
    additionalFiles: `Pinned FP16 ONNX and tokenizer assets total ${VERDICT_TOTAL_BYTES} bytes.`,
    limits: [
      tokenLimit(
        'question',
        VERDICT_MAX_TOKENS,
        'Each typed question is tokenized independently and right-truncated to the upstream Python engine limit.'
      )
    ],
    maxOptions: VERDICT_MAX_CANDIDATES,
    handling: independent,
    tokenCount: sumRows,
    abstention: 'implemented',
    scoreBasis: 'ordered-levels',
    temperature: {
      byCandidateCount: VERDICT_TEMPERATURES,
      fallback: VERDICT_GLOBAL_TEMPERATURE
    },
    temperatureAppliedBy: 'decoder',
    typedTextQuestionTypes: ['choice', 'score', 'noul'],
    backend: 'onnxruntime-web-webgpu',
    gpuVerification: 'not-revalidated',
    cpuWasmVerification: 'not-revalidated',
    javascriptVerification: 'not-applicable',
    requiresNonFallbackAdapter: true,
    modelTerms:
      'Apache-2.0 declared by the pinned model card; review the exact checkpoint and transformed assets separately.',
    modelLicenseUrl: `https://github.com/${VERDICT_SOURCE_REPOSITORY}/blob/${VERDICT_SOURCE_REVISION}/LICENSE`,
    sourceModelCard: `https://github.com/${VERDICT_SOURCE_REPOSITORY}/tree/${VERDICT_SOURCE_REVISION}`
  })
);
const byName = new Map(models.map((item) => [item.description.model, item]));
const names = Object.freeze(models.map((item) => item.description.model));

/** A registration is shared by routing and the public, immutable descriptor. */
export function getModelRegistration(model = DEFAULT_MODEL) {
  const result = typeof model === 'string' ? byName.get(model) : undefined;
  if (!result) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  return result;
}

/** Names identify implemented routes, not GPU compatibility or release status. */
export function listDecisionModels() {
  return names;
}

export function isDecisionModel(value) {
  return typeof value === 'string' && byName.has(value);
}

/** No network request or model load is needed. */
export function describeModel(model) {
  if (!isDecisionModel(model)) throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model.');
  return getModelRegistration(model).description;
}
