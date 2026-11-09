import assert from 'node:assert/strict';
import test from 'node:test';
import {describeModel as describeRegisteredModel, isDecisionModel, listDecisionModels} from '../src/model-registry.js';
import type {ModelDescription} from '../src/index.d.ts';

type TestLimit = {resource: string; scope: string; min?: number; max?: number; description?: string};
type TestDescription = Omit<ModelDescription, 'assets' | 'backend' | 'features'> & {
  features: Omit<ModelDescription['features'], 'textChoice'> & {
    textChoice: Omit<ModelDescription['features']['textChoice'], 'limits'> & {limits: readonly TestLimit[]};
  };
  assets: ModelDescription['assets'] & {
    manifestSha256?: string;
    sourceDataSha256?: string;
    candidateWasmSha256?: string;
  };
  backend: ModelDescription['backend'] & {
    requiredLimits: Readonly<Record<string, number>>;
    requiresNonFallbackAdapter?: boolean;
  };
};
const describeModel = describeRegisteredModel as unknown as (model: unknown) => TestDescription;
import {selectModelFamily} from '../src/model-family.js';
import {MAPIKA_MANIFEST_SHA256} from '../src/mapika/contract.js';
import {MAPIKA_Q4_MANIFEST_SHA256} from '../src/mapika-q4/contract.js';
import {KEV08_MANIFEST_SHA256} from '../src/kev08/contract.js';
import {KEV06_MANIFEST_SHA256} from '../src/kev06/contract.js';
import {
  EXPERIMENTAL_KEV_UPSTREAM_MANIFEST_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_WASM_SHA256
} from '../src/kev06/upstream-contract.js';
import {MANIFEST_SHA256} from '../src/strands/manifest.mjs';
import {MAX_TOKENS as STRANDS_MAX_TOKENS} from '../src/strands/choice.mjs';
import {KEV_MAX_BRANCH_TOKENS, KEV_MAX_OPTIONS, KEV_MAX_STATE_TOKENS} from '../src/kev.js';
import {REVISION as H2O_REVISION} from '../src/h2o/plan.mjs';
import deberta from '../src/deberta/contract.json' with {type: 'json'};

const currentCpuWasmVerification = new Map([
  ['experimental-decima-small-int8', 'verified'],
  ['experimental-decima-base-clean-int8', 'verified'],
  ['experimental-decima-agent-int8', 'verified'],
  ['experimental-tamev-micro-minilm-int8', 'verified'],
  ['experimental-tamev-nano-tinybert-int8', 'verified'],
  ['experimental-tamev-nano-flash-tinybert-int8', 'verified'],
  ['experimental-skejja-edge32k-ft6-int8', 'verified'],
  ['experimental-skejja-edge32k-ft6-int8dyn', 'verified'],
  ['experimental-ettin-68m-int8', 'verified'],
  ['experimental-xdecision-fp16', 'not-revalidated'],
  ['experimental-laya-multilingual-fp16', 'not-revalidated'],
  ['experimental-laya-typed-decisions-fp16', 'not-revalidated'],
  ['experimental-gliner2.5-decide-mobile-q4f16', 'not-revalidated'],
  ['experimental-d1-omni-600m-q8', 'not-revalidated'],
  ['experimental-openjev-verdict-151m-fp16', 'not-revalidated'],
  ['experimental-safedecide-140m-fp32', 'not-revalidated'],
  ['experimental-crivo-32m-v2-fp32', 'not-revalidated'],
  ['experimental-crivo-32m-v2-int8', 'verified']
]);

const currentGpuVerification = new Map([
  ['experimental-tinydecide-10m-q4', 'not-applicable'],
  ['experimental-decima-small-int8', 'not-applicable'],
  ['experimental-decima-base-clean-int8', 'not-applicable'],
  ['experimental-decima-agent-int8', 'not-applicable'],
  ['experimental-tamev-micro-minilm-int8', 'not-applicable'],
  ['experimental-tamev-nano-tinybert-int8', 'not-applicable'],
  ['experimental-tamev-nano-flash-tinybert-int8', 'not-applicable'],
  ['experimental-skejja-edge32k-ft6-int8', 'not-applicable'],
  ['experimental-skejja-edge32k-ft6-int8dyn', 'not-applicable'],
  ['experimental-ettin-68m-int8', 'not-applicable'],
  ['experimental-crivo-32m-v2-int8', 'not-applicable'],
  ['experimental-crivo-32m-v2-fp32', 'not-revalidated'],
  ['experimental-laya-multilingual-fp16', 'verified'],
  ['experimental-laya-typed-decisions-fp16', 'verified'],
  ['experimental-xdecision-fp16', 'verified'],
  ['experimental-gliner2.5-decide-mobile-q4f16', 'verified'],
  ['experimental-d1-omni-600m-q8', 'not-revalidated'],
  ['experimental-openjev-verdict-151m-fp16', 'not-revalidated'],
  ['kev-0.6b-q4f16', 'verified'],
  ['experimental-strands-qwen-2b-fp16', 'verified']
]);

const backendByFamily: Record<string, string> = {
  decima: 'onnxruntime-web-wasm',
  tinydecide: 'javascript',
  tamev: 'onnxruntime-web-wasm',
  skejja: 'onnxruntime-web-wasm',
  ettin: 'onnxruntime-web-wasm',
  laya: 'onnxruntime-web-webgpu',
  'laya-typed': 'onnxruntime-web-webgpu',
  xdecision: 'onnxruntime-web-webgpu',
  'gliner-mobile': 'onnxruntime-web-webgpu',
  'd1-omni': 'onnxruntime-web-webgpu',
  verdict: 'onnxruntime-web-webgpu',
  safedecide: 'onnxruntime-web-webgpu',
  default: 'tvm-webgpu'
};

test('all H2O descriptions identify the conversion revision and model card', () => {
  const revision = '052bb7d48c71abcd769d91994792bbc664b7681e';
  assert.equal(H2O_REVISION, revision);
  for (const model of [
    'experimental-h2o-lightning-4b-q4f16',
    'experimental-h2o-lightning-4b-q4f16-dlight',
    'experimental-h2o-lightning-4b-q4f16-dual'
  ]) {
    const description = describeModel(model);
    assert.equal(description.checkpoint.repository, 'h2oai/h2o-lightning-4b');
    assert.equal(description.checkpoint.revision, H2O_REVISION);
    assert.equal(description.licenses.modelCard, `https://huggingface.co/h2oai/h2o-lightning-4b/tree/${revision}`);
  }
});

test('the catalog and router describe the same implemented routes', () => {
  const expected = new Map([
    ['experimental-decima-small-int8', 'decima'],
    ['experimental-decima-base-clean-int8', 'decima'],
    ['experimental-decima-agent-int8', 'decima'],
    ['experimental-tamev-micro-minilm-int8', 'tamev'],
    ['experimental-tamev-nano-tinybert-int8', 'tamev'],
    ['experimental-tamev-nano-flash-tinybert-int8', 'tamev'],
    ['experimental-skejja-edge32k-ft6-int8', 'skejja'],
    ['experimental-skejja-edge32k-ft6-int8dyn', 'skejja'],
    ['experimental-ettin-68m-int8', 'ettin'],
    ['experimental-laya-multilingual-fp16', 'laya'],
    ['experimental-laya-typed-decisions-fp16', 'laya-typed'],
    ['experimental-xdecision-fp16', 'xdecision'],
    ['experimental-gliner2.5-decide-mobile-q4f16', 'gliner-mobile'],
    ['experimental-tinydecide-10m-q4', 'tinydecide'],
    ['kev-0.6b-q4f16', 'kev'],
    ['experimental-kev-0.6b-q4f16-upstream', 'kev'],
    ['kev-0.6b-reflexes-q4f16', 'kev'],
    ['open-jev-deberta-v3-large-q4f16', 'deberta'],
    ['mapika-0.8b-f16', 'mapika'],
    ['mapika-0.8b-q4f16', 'mapika-q4'],
    ['kev-0.8b-f16', 'kev08'],
    ['experimental-h2o-lightning-4b-q4f16', 'h2o'],
    ['experimental-nari-decision-4b-q4f16', 'nari'],
    ['experimental-strands-qwen-2b-fp16', 'strands'],
    ['experimental-strands-gemma4-e2b-fp16', 'strands-gemma4'],
    ['experimental-opendecider-nano-fp16', 'opendecider-nano'],
    ['experimental-safedecide-140m-fp32', 'safedecide'],
    ['experimental-crivo-32m-v2-fp32', 'crivo'],
    ['experimental-crivo-32m-v2-int8', 'crivo'],
    ['experimental-h2o-lightning-4b-q4f16-dlight', 'h2o'],
    ['experimental-h2o-lightning-4b-q4f16-dual', 'h2o'],
    ['experimental-nari-decision-4b-mixed16-q4f16', 'nari'],
    ['experimental-imajev-4b-candidate', 'imajev'],
    ['experimental-pplx-decider-flash-v1.2-q4f16-dlight', 'pplx'],
    ['experimental-mapika-4b-q4f16', 'mapika4b'],
    ['experimental-mapika-4b-q4f16-dlight', 'mapika4b'],
    ['experimental-mapika-4b-q4f16-dlight-all', 'mapika4b'],
    ['experimental-elephant-9b-q4f16-dlight', 'elephant'],
    ['experimental-elephant-9b-q4f16-dlight-512', 'elephant'],
    ['experimental-clef-9b-q4f16-selected', 'clef'],
    ['experimental-d1-omni-600m-q8', 'd1-omni'],
    ['experimental-openjev-verdict-151m-fp16', 'verdict']
  ]);
  assert.deepEqual(listDecisionModels(), [...expected.keys()]);
  assert.equal(selectModelFamily({}), 'kev');
  for (const [name, family] of expected) {
    assert.equal(isDecisionModel(name), true);
    assert.equal(selectModelFamily({model: name}), family);
    const description = describeModel(name);
    assert.equal(description.model, name);
    assert.equal(description.features.textChoice.status, 'implemented');
    assert.equal(
      description.features.textChoice.abstention,
      family === 'imajev' || family === 'verdict' ? 'implemented' : 'not-implemented'
    );
    assert.equal(description.features.imageChoice.status, 'not-implemented');
    assert.equal(description.features.rating.status, 'not-implemented');
    assert.equal(description.verification.currentBundleGpu, currentGpuVerification.get(name) ?? 'not-revalidated');
    assert.equal(
      description.verification.currentBundleCpuWasm,
      currentCpuWasmVerification.get(name) ?? 'not-applicable'
    );
    assert.equal(
      description.verification.currentBundleJavascript,
      family === 'tinydecide' ? 'verified' : 'not-applicable'
    );
    assert.equal(description.licenses.project, 'Apache-2.0');
    assert.match(description.licenses.modelCard, /^https:\/\/huggingface\.co\//);
    assert.equal(
      description.assets.location,
      ['kev-0.6b-q4f16', 'experimental-kev-0.6b-q4f16-upstream'].includes(name)
        ? 'package-and-pinned-upstream'
        : 'separate'
    );
    assert.ok(description.assets.requiredFiles.length >= 2);
    assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'questions')!.min, 1);
    const expectedBackend =
      family === 'crivo'
        ? name.endsWith('-int8')
          ? 'onnxruntime-web-wasm'
          : 'onnxruntime-web-webgpu'
        : (backendByFamily[family] ?? backendByFamily.default);
    assert.equal(description.backend.name, expectedBackend);
  }
});

test('experimental upstream Kev has separate pins and leaves the stable Kev route unchanged', () => {
  const stable = describeModel('kev-0.6b-q4f16');
  const experimental = describeModel('experimental-kev-0.6b-q4f16-upstream');

  assert.equal(stable.assets.integrity, 'pinned-manifest');
  assert.equal(stable.assets.manifestSha256, KEV06_MANIFEST_SHA256);
  assert.equal(experimental.assets.integrity, 'candidate-pinned-manifest-and-shards');
  assert.equal(experimental.assets.manifestSha256, EXPERIMENTAL_KEV_UPSTREAM_MANIFEST_SHA256);
  assert.equal(experimental.assets.sourceDataSha256, EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256);
  assert.equal(experimental.assets.candidateWasmSha256, EXPERIMENTAL_KEV_UPSTREAM_WASM_SHA256);
  assert.notEqual(experimental.assets.manifestSha256, stable.assets.manifestSha256);
  assert.equal(experimental.assets.location, 'package-and-pinned-upstream');
  assert.match(experimental.assets.additionalFiles, /lazily loaded packaged JavaScript chunk/u);
  assert.equal(experimental.checkpoint.weightSha256, EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256);
  assert.equal(experimental.backend.name, 'tvm-webgpu');
  assert.equal(experimental.verification.currentBundleGpu, 'not-revalidated');
  assert.equal(selectModelFamily({model: experimental.model}), 'kev');
});

test('Laya metadata identifies WebGPU execution and runtime-managed CPU fallback', () => {
  const description = describeModel('experimental-laya-multilingual-fp16');
  assert.deepEqual(description.backend, {
    name: 'onnxruntime-web-webgpu',
    executionProvider: 'webgpu',
    cpuFallback: 'runtime-managed',
    requiredFeatures: [],
    requiredLimits: {},
    requiresNonFallbackAdapter: true
  });
  assert.equal(description.features.textChoice.score.basis, 'ordered-levels');
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, 122);
  assert.deepEqual(
    description.features.textChoice.limits.find((item) => item.resource === 'questions'),
    {
      resource: 'questions',
      scope: 'request',
      min: 1
    }
  );
  assert.equal(
    description.features.textChoice.limits.some((item) => item.resource === 'tokens' && item.scope === 'request'),
    false
  );
  assert.deepEqual(description.features.typedTextQuestionTypes, ['choice', 'score', 'noul']);
  assert.equal(description.verification.currentBundleCpuWasm, 'not-revalidated');
});

test('Laya Typed-Decisions identifies its independent ModernBERT ONNX assets and limits', () => {
  const description = describeModel('experimental-laya-typed-decisions-fp16');
  assert.equal(description.checkpoint.repository, 'onnx-community/laya-typed-decisions-ONNX');
  assert.equal(description.checkpoint.revision, '79c441b22da79eb3727bb01e8452e29a32cfa548');
  assert.equal(description.checkpoint.graphSha256, '94c25ed9a1ebe3f13b85bdbd54cd20ed407fb1993e3993f3deda1bc5662b8ae1');
  assert.equal(description.precision, 'f16');
  assert.equal(description.verification.currentBundleGpu, 'verified');
  assert.equal(description.assets.integrity, 'pinned-files');
  assert.equal(description.assets.requiredFiles.length, 5);
  assert.equal(
    description.features.textChoice.limits.find((item) => item.resource === 'tokens' && item.scope === 'question')!.max,
    1024
  );
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, 122);
  assert.deepEqual(
    description.features.textChoice.limits.find((item) => item.resource === 'questions'),
    {
      resource: 'questions',
      scope: 'request',
      min: 1
    }
  );
  assert.equal(
    description.features.textChoice.limits.some((item) => item.resource === 'tokens' && item.scope === 'request'),
    false
  );
  assert.deepEqual(description.features.typedTextQuestionTypes, ['choice', 'score', 'noul']);
});

test('xDecision exposes per-row limits without a fixed question or aggregate token cap', () => {
  const description = describeModel('experimental-xdecision-fp16');
  assert.equal(
    description.features.textChoice.limits.find((item) => item.resource === 'tokens' && item.scope === 'question')!.max,
    1024
  );
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, 122);
  assert.deepEqual(
    description.features.textChoice.limits.find((item) => item.resource === 'questions'),
    {
      resource: 'questions',
      scope: 'request',
      min: 1
    }
  );
  assert.equal(
    description.features.textChoice.limits.some((item) => item.resource === 'tokens' && item.scope === 'request'),
    false
  );
});

test('Ettin reports native per-question and head limits without a request question cap', () => {
  const limits = describeModel('experimental-ettin-68m-int8').features.textChoice.limits;
  assert.equal(limits.find((item) => item.resource === 'tokens' && item.scope === 'question')!.max, 2048);
  assert.equal(limits.find((item) => item.resource === 'options' && item.scope === 'question')!.max, 10);
  assert.equal(limits.find((item) => item.resource === 'questions')!.max, undefined);
});

test('Crivo metadata describes row and chunk bounds without question or option ceilings', () => {
  for (const model of ['experimental-crivo-32m-v2-fp32', 'experimental-crivo-32m-v2-int8']) {
    const description = describeModel(model);
    const limits = description.features.textChoice.limits;
    assert.equal(limits.find((item) => item.resource === 'tokens' && item.scope === 'question')!.max, 2048);
    assert.deepEqual(
      limits.find((item) => item.resource === 'options'),
      {
        resource: 'options',
        scope: 'question',
        min: 2
      }
    );
    assert.deepEqual(
      limits.find((item) => item.resource === 'questions'),
      {
        resource: 'questions',
        scope: 'request',
        min: 1
      }
    );
    assert.equal(
      limits.some((item) => item.resource === 'tokens' && item.scope === 'request'),
      false
    );
  }
});

test('GLiNER Mobile describes its pinned Q4F16 WebGPU checkpoint and sequence limits', () => {
  const description = describeModel('experimental-gliner2.5-decide-mobile-q4f16');
  assert.equal(description.precision, 'q4f16');
  assert.equal(description.checkpoint.repository, 'onnx-community/GLiNER2.5-Decide-mobile-ONNX');
  assert.equal(description.checkpoint.revision, '880d1921d315b44648c9f6c6fb6ea0b87dea2b62');
  assert.equal(description.checkpoint.graphSha256, '99a4e4c57fb15095a7e0aada14303d3d4f4ae0c3ce09c7b456a444bc2aff0dc2');
  assert.deepEqual(description.assets.requiredFiles, [
    'config.json',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'onnx/model_q4f16.onnx',
    'onnx/model_q4f16.onnx_data'
  ]);
  assert.equal(description.backend.name, 'onnxruntime-web-webgpu');
  assert.equal(description.features.textChoice.handling, 'packed-request');
  assert.equal(description.features.textChoice.abstention, 'not-implemented');
  assert.deepEqual(description.features.typedTextQuestionTypes, ['choice', 'score', 'noul']);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'tokens')!.max, 512);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, 64);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'questions')!.max, undefined);
  assert.equal(description.verification.currentBundleGpu, 'verified');
});

test('d1-omni describes its pinned text-only Q8 route and separate model license', () => {
  const description = describeModel('experimental-d1-omni-600m-q8');
  assert.equal(description.precision, 'q8');
  assert.equal(description.checkpoint.repository, 'onnx-community/d1-omni-600M-ONNX');
  assert.equal(description.checkpoint.revision, '4ebc1b97bf1477485371c79d1cf8d5e4e8eebfa0');
  assert.equal(description.checkpoint.sourceRepository, 'LiquidAI/d1-omni-600M');
  assert.equal(description.checkpoint.sourceRevision, '02b55d7076f15129e59ab3f94783f32c4b088674');
  assert.equal(description.assets.integrity, 'pinned-files');
  assert.equal(description.backend.name, 'onnxruntime-web-webgpu');
  assert.equal(description.verification.currentBundleGpu, 'not-revalidated');
  const limits = description.features.textChoice.limits;
  assert.equal(limits.find((limit) => limit.resource === 'tokens' && limit.scope === 'question')?.max, 4096);
  assert.equal(limits.find((limit) => limit.resource === 'questions')?.max, undefined);
  assert.equal(limits.find((limit) => limit.resource === 'options')?.max, undefined);
  assert.match(description.licenses.modelTerms, /LFM Open License v1\.0/);
  assert.match(description.licenses.modelLicenseUrl!, /LiquidAI\/d1-omni-600M/);
  assert.match(description.licenses.sourceModelCard!, /LiquidAI\/d1-omni-600M/);
  assert.deepEqual(description.features.typedTextQuestionTypes, ['choice', 'score', 'noul']);
  assert.equal(description.features.textChoice.score.temperatureAppliedBy, 'decoder');
});

test('TinyDecide describes its pinned 4-bit files and token limits without a question-count cap', () => {
  const description = describeModel('experimental-tinydecide-10m-q4');
  assert.equal(description.precision, 'q4');
  assert.equal(description.checkpoint.repository, 'TheREZOR/TinyDecide');
  assert.equal(description.checkpoint.revision, 'aa745b73a2db9a43bce731f0e8e5c90a6899fe2c');
  assert.equal(description.assets.integrity, 'pinned-files');
  assert.equal(description.assets.requiredFiles.length, 2);
  assert.deepEqual(description.backend, {name: 'javascript'});
  assert.deepEqual(
    description.features.textChoice.limits.filter((item) => item.resource === 'tokens'),
    [
      {
        resource: 'tokens',
        scope: 'state',
        max: 127,
        description: 'Reads at most the first 127 state tokens and reports truncation.'
      },
      {
        resource: 'tokens',
        scope: 'question',
        max: 192,
        description: 'Includes the question, options and special tokens.'
      }
    ]
  );
  assert.equal(
    description.features.textChoice.limits.find((item) => item.resource === 'tokens' && item.scope === 'request'),
    undefined
  );
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, 32);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'questions')!.max, undefined);
  assert.equal(description.verification.currentBundleJavascript, 'verified');
});

test('Decima describes its pinned int8 ONNX files and CPU WASM limits', () => {
  const description = describeModel('experimental-decima-small-int8');
  assert.equal(description.precision, 'int8');
  assert.equal(description.checkpoint.repository, 'amyrmahdy/decima-small');
  assert.equal(description.checkpoint.revision, '480c4ea9e54f2946b947fb3ceb3c7cc7ece77522');
  assert.equal(description.assets.integrity, 'pinned-files');
  assert.deepEqual(description.assets.requiredFiles, [
    'encoder.onnx',
    'scorer.onnx',
    'decima.json',
    'tokenizer/tokenizer.json',
    'tokenizer/tokenizer_config.json'
  ]);
  assert.deepEqual(description.backend, {
    name: 'onnxruntime-web-wasm',
    executionProvider: 'wasm',
    intraOpNumThreads: 1,
    requiredFeatures: [],
    requiredLimits: {},
    requiresNonFallbackAdapter: false
  });
  assert.deepEqual(description.features.textChoice.limits, [
    {
      resource: 'tokens',
      scope: 'state',
      max: 512,
      description: 'Complete question and state row; longer rows are truncated and marked.'
    },
    {
      resource: 'tokens',
      scope: 'question',
      max: 64,
      description: 'Each option row includes question text and special tokens; shortened rows are marked.'
    },
    {resource: 'options', scope: 'question', min: 2},
    {resource: 'questions', scope: 'request', min: 1}
  ]);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, undefined);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'questions')!.max, undefined);
});

test('Decima base-clean describes its pinned CPU model and rejects overlong state', () => {
  const description = describeModel('experimental-decima-base-clean-int8');
  assert.equal(description.precision, 'int8');
  assert.equal(description.checkpoint.repository, 'amyrmahdy/decima-base-clean');
  assert.equal(description.checkpoint.revision, 'a37e006fd4015c3c96176c341f3d6620f8a101f3');
  assert.deepEqual(description.assets.requiredFiles, [
    'encoder.onnx',
    'scorer.onnx',
    'decima.json',
    'tokenizer/tokenizer.json',
    'tokenizer/tokenizer_config.json'
  ]);
  assert.equal(description.features.textChoice.score.temperature, 0.6929793512269696);
  assert.deepEqual(description.features.textChoice.limits.slice(0, 2), [
    {
      resource: 'tokens',
      scope: 'state',
      max: 512,
      description: 'Complete question and state row; longer input is rejected before inference.'
    },
    {
      resource: 'tokens',
      scope: 'question',
      max: 64,
      description:
        'Uses the checkpoint choice_fit rendering and reports the adapted rows; each row includes question text and special tokens.'
    }
  ]);
  assert.equal(description.verification.currentBundleCpuWasm, 'verified');
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, undefined);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'questions')!.max, undefined);
});

test('Decima Agent describes its pinned CPU route and native question types', () => {
  const description = describeModel('experimental-decima-agent-int8');
  assert.equal(description.checkpoint.repository, 'amyrmahdy/decima-agent');
  assert.equal(description.checkpoint.revision, '1acd2c4b1461c15ebc6b2b8f84dd9f8af18be245');
  assert.equal(description.verification.currentBundleCpuWasm, 'verified');
  assert.deepEqual(description.features.typedTextQuestionTypes, ['choice', 'score', 'noul']);
  assert.deepEqual(description.features.textChoice.limits.slice(0, 2), [
    {
      resource: 'tokens',
      scope: 'state',
      max: 4096,
      description: 'Complete question and state row; longer input is rejected before inference.'
    },
    {
      resource: 'tokens',
      scope: 'question',
      max: 64,
      description:
        'Uses the checkpoint choice_fit rendering and reports the adapted rows; each row includes question text and special tokens.'
    }
  ]);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'options')!.max, undefined);
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'questions')!.max, undefined);
});

test('OpenDecider-nano describes its pinned FP16 graph and independent token limit', () => {
  const description = describeModel('experimental-opendecider-nano-fp16');
  const choice = description.features.textChoice;
  assert.equal(choice.handling, 'independent-questions');
  assert.equal(choice.tokenCount, 'max-question-rows');
  assert.equal(choice.limits.find((item) => item.resource === 'tokens')!.max, 2048);
  assert.equal(
    Object.hasOwn(
      choice.limits.find((item) => item.resource === 'questions')!,
      'max'
    ),
    false
  );
  assert.deepEqual(
    choice.limits.filter((item) => item.resource === 'options').map(({scope, min, max}) => ({scope, min, max})),
    [{scope: 'question', min: 2, max: undefined}]
  );
  assert.equal(choice.abstention, 'not-implemented');
  assert.equal(description.checkpoint.revision, '5137ae7b4f1cf8c55f1d05d3f5cdd3694d92da3e');
  assert.equal(description.checkpoint.graphSha256, '3c92c5e4dd01caae3afe4d3d9ffce7295343fe6fecceb39b030d895dd8fa4488');
  assert.equal(description.assets.manifestSha256, '0eb73bbddd75ce7c93212c12436521e69769c37ae9affe5f9bd6d0b10a7d7c3e');
  assert.equal(description.backend.requiresNonFallbackAdapter, true);
});

test('Clef describes joint text-choice limits and pinned selected graph assets', () => {
  const description = describeModel('experimental-clef-9b-q4f16-selected');
  const choice = description.features.textChoice;
  assert.equal(choice.handling, 'packed-request');
  assert.equal(choice.tokenCount, 'packed-request');
  assert.equal(choice.abstention, 'not-implemented');
  const tokenLimit = choice.limits.find((item) => item.resource === 'tokens')!;
  const questionLimit = choice.limits.find((item) => item.resource === 'questions')!;
  assert.equal(tokenLimit.max, 512);
  assert.match(tokenLimit.description!, /Current package graph envelope/);
  assert.equal(questionLimit.max, 8);
  assert.match(questionLimit.description!, /dynamic question axes/);
  assert.deepEqual(
    choice.limits.filter((item) => item.resource === 'options').map(({scope, min, max}) => ({scope, min, max})),
    [
      {scope: 'question', min: 2, max: 64},
      {scope: 'request', min: 2, max: 64}
    ]
  );
  assert.match(
    choice.limits.find((item) => item.resource === 'options' && item.scope === 'question')!.description!,
    /dynamic option axes/
  );
  assert.match(
    choice.limits.find((item) => item.resource === 'options' && item.scope === 'request')!.description!,
    /not a pinned model/
  );
  assert.equal(description.checkpoint.repository, 'Cloudflare/clef-flash');
  assert.equal(description.checkpoint.revision, 'fb4417d31c1d0f9e7e77f1141b40aaede7491e00');
  assert.equal(description.checkpoint.graphSha256, 'b70b2b892be91a1d19559c043f0b79a1e257f60f4c738780a704b109c2404a0d');
  assert.equal(description.assets.manifestSha256, '1b7289e209668534c52759634d7d0655e0a66eac827c22d6cd02ec4d122adf22');
  assert.deepEqual(description.assets.requiredFiles, [
    'inventory.json',
    'model.wasm',
    'tokenizer.json',
    'tokenizer_config.json'
  ]);
  assert.equal(description.backend.requiresNonFallbackAdapter, true);
  assert.equal(description.backend.requiredLimits.maxStorageBufferBindingSize, 134217728);
});

test('Elephant describes independent branches and its shared-prefix token count', () => {
  const description = describeModel('experimental-elephant-9b-q4f16-dlight');
  const choice = description.features.textChoice;
  assert.equal(choice.handling, 'independent-questions');
  assert.equal(choice.tokenCount, 'shared-prefix-and-branches');
  const questionLimit = choice.limits.find((item) => item.resource === 'questions')!;
  assert.equal(questionLimit.min, 1);
  assert.equal(Object.hasOwn(questionLimit, 'max'), false);
  const branchLimit = choice.limits.find((item) => item.resource === 'tokens' && item.scope === 'question')!;
  assert.equal(branchLimit.max, 128);
  assert.match(branchLimit.description!, /Packed-request memory scales with branch count and size/);
  assert.equal(
    choice.limits.find((item) => item.resource === 'tokens' && item.scope === 'request'),
    undefined
  );
  assert.equal(choice.score.temperature, 1);
  assert.equal(description.backend.requiredLimits.maxStorageBufferBindingSize, 134217728);
});

test('Mapika 4B variants share capabilities but pin different graphs', () => {
  const baseline = describeModel('experimental-mapika-4b-q4f16');
  const candidate = describeModel('experimental-mapika-4b-q4f16-dlight');
  assert.deepEqual(candidate.features, baseline.features);
  assert.deepEqual(candidate.backend, baseline.backend);
  assert.equal(baseline.features.textChoice.limits.find((item) => item.resource === 'tokens')!.max, 128);
  assert.deepEqual(
    baseline.features.textChoice.limits.find((item) => item.resource === 'questions'),
    {
      resource: 'questions',
      scope: 'request',
      min: 1,
      max: 255,
      description:
        'The compiled slot axis supports 1–255 questions; the separate 128-token sequence budget may bind first.'
    }
  );
  assert.equal(
    baseline.features.textChoice.limits.find((item) => item.resource === 'options')!.description!,
    'The graph has 255 answer-code logits per question.'
  );
  assert.equal(candidate.checkpoint.graphSha256, '0fe97fa7314e66c15540ee870e0bcc30fe852ee8c0a231a94a7cd1eaad1b8375');
  assert.notEqual(candidate.checkpoint.graphSha256, baseline.checkpoint.graphSha256);
  const all = describeModel('experimental-mapika-4b-q4f16-dlight-all');
  assert.deepEqual(all.features, baseline.features);
  assert.equal(all.checkpoint.graphSha256, '1b4ecf6ec52f9c428bbe9c95fad4f95c5b91cf5f3655af81cbf1c83451b48723');
});

test('Imajev describes explicit abstention and independently processed question limits', () => {
  const description = describeModel('experimental-imajev-4b-candidate');
  const choice = description.features.textChoice;
  assert.equal(choice.abstention, 'implemented');
  assert.equal(choice.handling, 'independent-questions');
  assert.equal(choice.tokenCount, 'max-question-rows');
  assert.equal(choice.score.basis, 'conditional-known-choices');
  assert.equal(choice.limits.find((item) => item.resource === 'tokens')!.max, 128);
  const questionLimit = choice.limits.find((item) => item.resource === 'questions')!;
  assert.equal(questionLimit.max, 8);
  assert.match(questionLimit.description!, /Pinned Jev public serving API/);
  const optionLimit = choice.limits.find((item) => item.resource === 'options')!;
  assert.equal(optionLimit.max, 255);
  assert.match(optionLimit.description!, /extended 256-code readout/);
  assert.match(optionLimit.description!, /default 255-code readout permits 254/);
  assert.equal(description.assets.manifestSha256, '0741c89f225e04ef1d2b72c88d7b3913dcc006ca5975bfc64ee34386ceea73a1');
});

test('Perplexity describes a joint, online-only Q4 candidate', () => {
  const description = describeModel('experimental-pplx-decider-flash-v1.2-q4f16-dlight');
  assert.equal(description.features.textChoice.handling, 'packed-request');
  assert.equal(description.features.textChoice.tokenCount, 'packed-request');
  assert.equal(description.features.textChoice.limits.find((item) => item.resource === 'tokens')!.max, 512);
  assert.deepEqual(
    description.features.textChoice.limits.find((item) => item.resource === 'questions'),
    {
      resource: 'questions',
      scope: 'request',
      min: 1,
      max: 255,
      description:
        'The exported marker axis supports 1–255 readouts; the separate 512-token sequence budget may bind first.'
    }
  );
  assert.equal(description.features.textChoice.score.temperature, 1.5539092221462472);
  assert.deepEqual(
    description.features.textChoice.limits.find((item) => item.resource === 'options'),
    {
      resource: 'options',
      scope: 'question',
      min: 1,
      max: 255,
      description: 'The graph has 255 answer-code logits per question.'
    }
  );
  assert.equal(description.backend.requiresNonFallbackAdapter, true);
  assert.equal(description.checkpoint.graphSha256, '8040c84f36e254958b9eff3b2e7d75248857232d9c489f586881aebd7f43bfbc');
});

test('dual H2O shares its capabilities and pins a distinct graph and metadata', () => {
  const original = describeModel('experimental-h2o-lightning-4b-q4f16');
  const dual = describeModel('experimental-h2o-lightning-4b-q4f16-dual');
  assert.equal(dual.checkpoint.graphSha256, 'd4531ab4a4690b192ba205c42a4e9ca0117d09e9454ff90ee9958306f35ce9b8');
  assert.equal(dual.checkpoint.weightSha256, original.checkpoint.weightSha256);
  assert.deepEqual(dual.features, original.features);
  assert.deepEqual(dual.backend, original.backend);
  assert.ok(dual.assets.requiredFiles.includes('model.dual.pending.wasm'));
  assert.ok(dual.assets.requiredFiles.includes('h2o-dual-metadata.json'));
  assert.ok(!dual.assets.requiredFiles.includes('model.pending.wasm'));
});

test('Nari describes joint request limits and option-count temperatures', () => {
  const description = describeModel('experimental-nari-decision-4b-q4f16');
  const choice = description.features.textChoice;
  assert.equal(choice.handling, 'packed-request');
  const tokenLimit = choice.limits.find((limit) => limit.resource === 'tokens')!;
  assert.equal(tokenLimit.max, 361);
  assert.match(tokenLimit.description!, /Current experimental package graph envelope/);
  const questionLimit = choice.limits.find((limit) => limit.resource === 'questions')!;
  assert.equal(questionLimit.max, 8);
  assert.match(questionLimit.description!, /dynamic question rows/);
  assert.deepEqual(
    choice.limits.filter((limit) => limit.resource === 'options').map(({scope, max}) => ({scope, max})),
    [
      {scope: 'question', max: 64},
      {scope: 'request', max: 64}
    ]
  );
  assert.match(
    choice.limits.find((limit) => limit.resource === 'options' && limit.scope === 'question')!.description!,
    /dynamic option rows/
  );
  assert.match(
    choice.limits.find((limit) => limit.resource === 'options' && limit.scope === 'request')!.description!,
    /not a pinned model capacity/
  );
  assert.deepEqual(
    (
      choice.score.temperature as {byOptionCount: readonly {min: number; max: number; value: number}[]}
    ).byOptionCount.map(({min, max, value}) => [min, max, value]),
    [
      [2, 2, 1],
      [3, 5, 1.1189],
      [6, 26, 0.9632],
      [27, 64, 1.1189]
    ]
  );
});

test('mixed Nari has its own pinned graph and metadata with the same choice contract', () => {
  const full = describeModel('experimental-nari-decision-4b-q4f16');
  const mixed = describeModel('experimental-nari-decision-4b-mixed16-q4f16');
  assert.equal(mixed.precision, 'mixed-f16-q4f16');
  assert.notEqual(mixed.checkpoint.graphSha256, full.checkpoint.graphSha256);
  assert.notEqual(mixed.assets.manifestSha256, full.assets.manifestSha256);
  assert.deepEqual(mixed.features, full.features);
  assert.deepEqual(mixed.backend, full.backend);
});

test('DLight shares H2O capabilities while binding its own graph and assets', () => {
  const original = describeModel('experimental-h2o-lightning-4b-q4f16');
  const dlight = describeModel('experimental-h2o-lightning-4b-q4f16-dlight');
  assert.equal(original.checkpoint.graphSha256, '62fb8ddb2b98106cb24f925a585f930f87d9f6df25e32deb342aa3ae37850a50');
  assert.equal(dlight.checkpoint.graphSha256, '32da4bd915dec178f29828acd2a57a202df069f4c6c0f13b0d9e7c55ff782336');
  assert.equal(dlight.checkpoint.weightSha256, original.checkpoint.weightSha256);
  assert.equal(dlight.checkpoint.revision, original.checkpoint.revision);
  assert.equal(dlight.precision, original.precision);
  assert.deepEqual(dlight.features, original.features);
  assert.deepEqual(dlight.backend, original.backend);
  assert.equal(dlight.assets.manifestSha256, original.assets.manifestSha256);
  assert.ok(dlight.assets.requiredFiles.includes('model.dlight.pending.wasm'));
  assert.ok(dlight.assets.requiredFiles.includes('h2o-dlight-metadata.json'));
  assert.ok(!dlight.assets.requiredFiles.includes('model.pending.wasm'));
  assert.ok(Object.isFrozen(dlight.assets.requiredFiles));
  assert.throws(() => (dlight.assets.requiredFiles as string[]).push('extra'), TypeError);
});

test('unknown and unregistered candidates are rejected before loading', () => {
  for (const value of [undefined, null, 0, {}, '__proto__', 'toString', 'h2o-4b', 'nari-4b', 'imajev-4b']) {
    assert.equal(isDecisionModel(value), false);
    assert.throws(() => describeModel(value), {code: 'INVALID_INPUT'});
    if (value !== undefined) assert.throws(() => selectModelFamily({model: value}), {code: 'INVALID_INPUT'});
  }
});

test('limits distinguish a whole packed request from independently encoded questions', () => {
  const limit = (name: string, resource: string, scope: string) =>
    describeModel(name).features.textChoice.limits.find((item) => item.resource === resource && item.scope === scope);
  assert.equal(limit('kev-0.6b-q4f16', 'tokens', 'request'), undefined);
  assert.equal(limit('kev-0.6b-q4f16', 'tokens', 'state')!.max, KEV_MAX_STATE_TOKENS - 1);
  assert.equal(limit('kev-0.6b-q4f16', 'tokens', 'question')!.max, KEV_MAX_BRANCH_TOKENS);
  assert.match(limit('kev-0.6b-q4f16', 'tokens', 'question')!.description!, /native.*per-question/i);
  for (const model of ['kev-0.6b-q4f16', 'kev-0.6b-reflexes-q4f16']) {
    const description = describeModel(model);
    const questions = limit(model, 'questions', 'request')!;
    assert.equal(questions.min, 1);
    assert.equal(questions.max, undefined);
    assert.equal(limit(model, 'tokens', 'question')!.max, KEV_MAX_BRANCH_TOKENS);
    const options = limit(model, 'options', 'question')!;
    assert.equal(options.min, 1);
    assert.equal(options.max, KEV_MAX_OPTIONS);
    assert.match(options.description!, /upstream Kev/i);
    assert.deepEqual(description.features.typedTextQuestionTypes, ['choice', 'noul', 'score']);
  }
  for (const model of [
    'experimental-tamev-micro-minilm-int8',
    'experimental-tamev-nano-tinybert-int8',
    'experimental-tamev-nano-flash-tinybert-int8'
  ]) {
    const questions = limit(model, 'questions', 'request')!;
    assert.equal(questions.min, 1);
    assert.equal(questions.max, undefined);
    const options = limit(model, 'options', 'question')!;
    assert.equal(options.min, 2);
    assert.equal(options.max, undefined);
    assert.equal(limit(model, 'tokens', 'state')!.max, 128);
    assert.equal(limit(model, 'tokens', 'question')!.max, 48);
  }
  assert.equal(limit(deberta.model, 'tokens', 'request')!.max, deberta.maxInputTokens);
  assert.equal(limit(deberta.model, 'tokens', 'state')!.max, deberta.maxStateTokens);
  assert.equal(limit('mapika-0.8b-f16', 'tokens', 'request')!.max, 512);
  assert.equal(limit('mapika-0.8b-q4f16', 'tokens', 'request')!.max, 128);
  assert.equal(limit('kev-0.8b-f16', 'tokens', 'question')!.max, 128);
  assert.equal(limit('kev-0.8b-f16', 'tokens', 'state')!.max, 127);
  assert.equal(limit('experimental-strands-qwen-2b-fp16', 'tokens', 'question')!.max, STRANDS_MAX_TOKENS);
  assert.equal(limit('experimental-h2o-lightning-4b-q4f16', 'tokens', 'question')!.max, 512);
  assert.equal(limit('experimental-h2o-lightning-4b-q4f16', 'questions', 'request')!.max, undefined);
  assert.equal(limit('experimental-h2o-lightning-4b-q4f16', 'options', 'question')!.max, 255);
  assert.equal(describeModel('experimental-h2o-lightning-4b-q4f16').features.textChoice.score.temperature, 0.75);
  assert.equal(
    describeModel('experimental-h2o-lightning-4b-q4f16').features.textChoice.handling,
    'independent-questions'
  );
  assert.equal(
    describeModel('experimental-h2o-lightning-4b-q4f16').features.textChoice.tokenCount,
    'sum-question-rows'
  );
  assert.equal(limit(deberta.model, 'questions', 'request')!.max, undefined);
  assert.equal(limit('mapika-0.8b-q4f16', 'questions', 'request')!.max, undefined);
  assert.equal(describeModel('mapika-0.8b-q4f16').features.textChoice.tokenCount, 'packed-request');
  assert.equal(describeModel('kev-0.8b-f16').features.textChoice.tokenCount, 'sum-question-rows');
});

test('Skejja advertises its upstream option limit and row-scoped token budget without a package question cap', () => {
  for (const model of ['experimental-skejja-edge32k-ft6-int8', 'experimental-skejja-edge32k-ft6-int8dyn']) {
    const choice = describeModel(model).features.textChoice;
    const tokens = choice.limits.find((limit) => limit.resource === 'tokens')!;
    const questions = choice.limits.find((limit) => limit.resource === 'questions')!;
    const options = choice.limits.find((limit) => limit.resource === 'options')!;

    assert.deepEqual([tokens.scope, tokens.max], ['question', 512]);
    assert.equal(questions.min, 1);
    assert.equal(questions.max, undefined);
    assert.match(questions.description!, /upstream server defaults to 256/i);
    assert.deepEqual([options.scope, options.min, options.max], ['question', 2, 255]);
  }
});

test('Verdict exposes per-row and class limits without a request question cap', () => {
  const choice = describeModel('experimental-openjev-verdict-151m-fp16').features.textChoice;
  assert.equal(choice.handling, 'independent-questions');
  assert.equal(choice.tokenCount, 'sum-question-rows');
  assert.equal(choice.limits.find((item) => item.resource === 'tokens' && item.scope === 'question')!.max, 512);
  assert.equal(choice.limits.find((item) => item.resource === 'questions')!.max, undefined);
  assert.equal(choice.limits.find((item) => item.resource === 'options' && item.scope === 'question')!.max, 24);
});

test('SafeDecide describes its pinned FP32 option-pair route and selective action rules', () => {
  const model = 'experimental-safedecide-140m-fp32';
  const description = describeModel(model);
  const choice = description.features.textChoice;

  assert.equal(isDecisionModel(model), true);
  assert.equal(selectModelFamily({model}), 'safedecide');
  assert.equal(description.precision, 'f32');
  assert.equal(description.checkpoint.repository, 'yotambr/safedecide-140m');
  assert.equal(description.checkpoint.revision, '1934a00e6ba5242668c3f1b06fa39fb438741207');
  assert.equal(description.checkpoint.graphSha256, 'e987d2e4c8da29b8366d2c914b93483ffe661f30593b966202051ee86811eacc');
  assert.equal(description.backend.name, 'onnxruntime-web-webgpu');
  assert.equal(choice.handling, 'independent-options');
  assert.equal(choice.tokenCount, 'sum-option-pairs');
  assert.deepEqual(
    choice.limits.filter((limit) => limit.resource === 'options').map(({scope, min, max}) => [scope, min, max]),
    [['question', 2, undefined]]
  );
  assert.equal(choice.limits.find((limit) => limit.resource === 'questions')!.max, undefined);
  assert.equal(choice.limits.find((limit) => limit.resource === 'tokens')!.max, 640);
  assert.deepEqual(description.features.selectiveAction!.levels, [
    {alpha: 0.05, delta: 0.1, threshold: null},
    {alpha: 0.1, delta: 0.1, threshold: 0.7},
    {alpha: 0.2, delta: 0.1, threshold: 0.5}
  ]);
  assert.equal(description.verification.currentBundleGpu, 'not-revalidated');
});

test('asset descriptions pin the Kev 0.6B manifest and distinguish other contracts', () => {
  assert.equal(describeModel('kev-0.6b-q4f16').assets.integrity, 'pinned-manifest');
  assert.equal(describeModel('kev-0.6b-q4f16').assets.manifestSha256, KEV06_MANIFEST_SHA256);
  assert.equal(describeModel('kev-0.6b-reflexes-q4f16').assets.integrity, 'pinned-files');
  assert.equal(describeModel('open-jev-deberta-v3-large-q4f16').assets.integrity, 'pinned-files');
  for (const [name, expected] of [
    ['mapika-0.8b-f16', MAPIKA_MANIFEST_SHA256],
    ['mapika-0.8b-q4f16', MAPIKA_Q4_MANIFEST_SHA256],
    ['kev-0.8b-f16', KEV08_MANIFEST_SHA256],
    ['experimental-strands-qwen-2b-fp16', MANIFEST_SHA256],
    ['experimental-strands-gemma4-e2b-fp16', 'a1c6e40dcb52720b75fe093db1e99e4376ca18b0785e4da4054b962a67b974da'],
    ['experimental-h2o-lightning-4b-q4f16', '27f83988992c9d9fe2febe924848259890bcb4d39e1c57cb225dd8cb22610e9a']
  ]) {
    assert.equal(describeModel(name).assets.integrity, 'pinned-manifest');
    assert.equal(describeModel(name).assets.manifestSha256, expected);
  }
});

test('Strands Qwen 2B records its verified WebGPU route and declared model terms', () => {
  const description = describeModel('experimental-strands-qwen-2b-fp16');
  assert.equal(description.checkpoint.repository, 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610');
  assert.equal(description.checkpoint.revision, '216d1ccb46a9e9b036eb9187b248db98480f77d4');
  assert.equal(description.backend.name, 'tvm-webgpu');
  assert.equal(description.verification.currentBundleGpu, 'verified');
  assert.equal(description.features.textChoice.limits.find((limit) => limit.resource === 'tokens')!.max, 4096);
  const optionLimit = description.features.textChoice.limits.find((limit) => limit.resource === 'options')!;
  assert.deepEqual([optionLimit.min, optionLimit.max], [2, 255]);
  assert.match(optionLimit.description!, /pinned fused TVM graph.*2–255.*rejected before inference/u);
  assert.equal(description.licenses.modelTerms, 'Apache-2.0 (publisher-declared; training-data source terms vary)');
  assert.match(
    description.licenses.modelLicenseUrl!,
    /\/blob\/216d1ccb46a9e9b036eb9187b248db98480f77d4\/LICENSE\.md$/u
  );
});

test('Gemma 4 registration reports the pinned E2B runtime contract and source limits', () => {
  const description = describeModel('experimental-strands-gemma4-e2b-fp16');
  const limits = description.features.textChoice.limits;
  assert.equal(description.checkpoint.repository, 'StrandsAgents/strands-decider-E2B-gemma4-v1-2610');
  assert.equal(description.checkpoint.revision, 'd383ecf8534b13d49b3b567a1b66f88fca6d50ba');
  assert.equal(description.assets.integrity, 'pinned-manifest');
  assert.deepEqual(description.features.typedTextQuestionTypes, ['choice', 'score', 'noul']);
  assert.equal(limits.find((limit) => limit.resource === 'tokens' && limit.scope === 'question')!.max, 4096);
  assert.equal(limits.find((limit) => limit.resource === 'options' && limit.scope === 'question')!.max, undefined);
  assert.equal(limits.find((limit) => limit.resource === 'questions' && limit.scope === 'request')!.max, undefined);
  assert.equal(description.verification.currentBundleGpu, 'not-revalidated');
  assert.match(description.assets.additionalFiles, /lossy/u);
});

test('catalog data and nested metadata cannot be mutated by callers', () => {
  const models = listDecisionModels();
  const description = describeModel('mapika-0.8b-q4f16');
  assert.ok(Object.isFrozen(models));
  assert.ok(Object.isFrozen(description));
  assert.ok(Object.isFrozen(description.features.textChoice.limits));
  assert.ok(Object.isFrozen(description.features.textChoice.limits[0]));
  assert.ok(Object.isFrozen(description.backend.requiredLimits));
  assert.throws(() => (models as string[]).push('imajev-4b'), TypeError);
  assert.throws(() => ((description.features.textChoice.limits[0] as {max?: number}).max = 999), TypeError);
  assert.equal(describeModel('mapika-0.8b-q4f16').features.textChoice.limits[0]!.max, 128);
});
