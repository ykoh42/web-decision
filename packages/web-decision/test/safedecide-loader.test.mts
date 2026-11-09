import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SAFEDECIDE_FILES,
  SAFEDECIDE_MODEL,
  SAFEDECIDE_SPEC,
  SAFEDECIDE_TOTAL_BYTES
} from '../src/safedecide/contract.mjs';
import {
  loadFakeSafeDecide,
  validateSafeDecideLoadOptions,
  validateSafeDecideModelConfig
} from '../src/safedecide/loader.mjs';

const decisionConfig = {
  base_model: 'jhu-clsp/mmBERT-small',
  profile: 'clean',
  max_length: 640,
  temperatures: {choice: 1.1134611876261922, noul: 1.1006431532659389, score: 1.089708955367956},
  selective_rule: {
    alpha: 0.1,
    delta: 0.1,
    threshold: 0.7,
    statement: 'With probability >= 90%, risk among automated decisions is <= 10%.'
  },
  selective_rules: [
    {
      alpha: 0.05,
      delta: 0.1,
      threshold: null,
      statement: 'With probability >= 90%, risk among automated decisions is <= 5%.'
    },
    {
      alpha: 0.1,
      delta: 0.1,
      threshold: 0.7,
      statement: 'With probability >= 90%, risk among automated decisions is <= 10%.'
    },
    {
      alpha: 0.2,
      delta: 0.1,
      threshold: 0.5,
      statement: 'With probability >= 90%, risk among automated decisions is <= 20%.'
    }
  ]
};
const modelConfig = {
  model_type: 'modernbert',
  vocab_size: 256000,
  hidden_size: 384,
  num_hidden_layers: 22,
  max_position_embeddings: 8192,
  bos_token_id: 2,
  eos_token_id: 1,
  pad_token_id: 0,
  mask_token_id: 4
};
const tokenizerConfig = {max_length: 384, truncation_side: 'right', padding_side: 'right'};

function jsonBytes(value: unknown) {
  return new TextEncoder().encode(JSON.stringify(value));
}

test('SafeDecide pins the complete asset set and permits the declared large graph', () => {
  assert.equal(SAFEDECIDE_MODEL, 'experimental-safedecide-140m-fp32');
  assert.equal(SAFEDECIDE_FILES['onnx/model.onnx'].bytes, 563_099_019);
  assert.equal(
    SAFEDECIDE_TOTAL_BYTES,
    Object.values(SAFEDECIDE_FILES).reduce((sum, file) => sum + file.bytes, 0)
  );
  assert.equal(SAFEDECIDE_SPEC.maxAssetBytes, SAFEDECIDE_FILES['onnx/model.onnx'].bytes);
  assert.equal(SAFEDECIDE_SPEC.maxSequenceTokens, 640);
  assert.equal(Object.hasOwn(SAFEDECIDE_SPEC, 'maxOptionPairs'), false);
  assert.equal(Object.hasOwn(SAFEDECIDE_SPEC, 'maxOptions'), false);
  assert.equal(Object.hasOwn(SAFEDECIDE_SPEC, 'maxQuestions'), false);
});

test('SafeDecide load options use opt-in browser caching for the 563 MB graph', () => {
  const defaults = validateSafeDecideLoadOptions({model: SAFEDECIDE_MODEL});
  assert.equal(
    defaults.base.href,
    `https://huggingface.co/${SAFEDECIDE_SPEC.repository}/resolve/${SAFEDECIDE_SPEC.revision}/`
  );
  assert.equal(defaults.cacheEnabled, false);
  assert.equal(defaults.offline, false);

  const checked = validateSafeDecideLoadOptions({
    model: SAFEDECIDE_MODEL,
    assetsUrl: 'https://localhost/assets/safedecide/'
  });
  assert.equal(checked.cacheEnabled, false);
  assert.equal(checked.base.pathname, '/assets/safedecide/');
  assert.throws(
    () =>
      validateSafeDecideLoadOptions({model: SAFEDECIDE_MODEL, assetsUrl: 'https://localhost/assets/', offline: true}),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      validateSafeDecideLoadOptions({model: SAFEDECIDE_MODEL, assetsUrl: 'https://localhost/assets/', unknown: true}),
    {code: 'INVALID_INPUT'}
  );
});

test('SafeDecide config binds the 640-token tokenizer, temperatures, and certified thresholds', () => {
  assert.equal(validateSafeDecideModelConfig(modelConfig, decisionConfig, tokenizerConfig), decisionConfig);
  assert.throws(
    () => validateSafeDecideModelConfig(modelConfig, {...decisionConfig, max_length: 384}, tokenizerConfig),
    {code: 'INVALID_ASSET'}
  );
  assert.throws(
    () =>
      validateSafeDecideModelConfig(modelConfig, {...decisionConfig, question_bias: {custom: [1]}}, tokenizerConfig),
    {code: 'INVALID_ASSET'}
  );
});

test('SafeDecide loader creates a WebGPU session only after checking metadata and acquiring a hardware adapter', async () => {
  const readNames: string[] = [];
  const released: string[] = [];
  const created: {bytes: number[]; options: unknown}[] = [];
  const files = new Map<string, Uint8Array | ArrayBuffer>([
    ['config.json', jsonBytes(modelConfig)],
    ['safedecide_config.json', jsonBytes(decisionConfig)],
    ['tokenizer_config.json', jsonBytes(tokenizerConfig)],
    ['tokenizer.json', jsonBytes({minimal: true})],
    ['onnx/model.onnx', Uint8Array.from([1, 2, 3, 4]).buffer]
  ]);
  const fakeSession = {
    inputNames: ['input_ids', 'attention_mask'],
    outputNames: ['logits'],
    async release() {
      released.push('session');
    }
  };
  const loaded = {fakeLoaded: true};
  let sourceBase;
  const result = await loadFakeSafeDecide(
    {model: SAFEDECIDE_MODEL},
    {
      fake: true,
      openSource: async ({spec, base}: {spec: unknown; base: URL}) => {
        assert.equal(spec, SAFEDECIDE_SPEC);
        sourceBase = base.href;
        return {
          async read(name: string) {
            readNames.push(name);
            return files.get(name);
          }
        };
      },
      createTokenizer(
        data: unknown,
        tokenizerFile: unknown,
        checkedModelConfig: {modelType: string},
        maxLength: number
      ) {
        assert.deepEqual(data, {minimal: true});
        assert.equal(checkedModelConfig.modelType, 'modernbert');
        assert.equal(maxLength, 640);
        return {special: {padId: 0}};
      },
      gpuApi: {
        async requestAdapter() {
          return {info: {isFallbackAdapter: false}};
        }
      },
      ort: {
        env: {webgpu: {}},
        Tensor: class {},
        InferenceSession: {
          async create(bytes: Uint8Array, options: unknown) {
            created.push({bytes: Array.from(bytes), options});
            return fakeSession;
          }
        }
      },
      createRunner(ort: unknown, session: unknown, padId: number) {
        assert.equal(session, fakeSession);
        assert.equal(padId, 0);
        return {
          async run() {},
          async dispose() {
            released.push('runner');
          }
        };
      },
      createLoadedSession(components: {config: unknown}) {
        assert.deepEqual(components.config, decisionConfig);
        return loaded;
      }
    }
  );

  assert.equal(result, loaded);
  assert.equal(sourceBase, `https://huggingface.co/${SAFEDECIDE_SPEC.repository}/resolve/${SAFEDECIDE_SPEC.revision}/`);
  assert.deepEqual(readNames, [
    'config.json',
    'safedecide_config.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'onnx/model.onnx'
  ]);
  assert.deepEqual(created, [
    {
      bytes: [1, 2, 3, 4],
      options: {executionProviders: ['webgpu'], graphOptimizationLevel: 'all'}
    }
  ]);
  assert.deepEqual(released, []);
});

test('SafeDecide releases an ONNX session when the graph interface is mismatched', async () => {
  const badSession = {
    inputNames: ['input_ids'],
    outputNames: ['logits'],
    async release() {}
  };
  const files = new Map<string, Uint8Array | ArrayBuffer>([
    ['config.json', jsonBytes(modelConfig)],
    ['safedecide_config.json', jsonBytes(decisionConfig)],
    ['tokenizer_config.json', jsonBytes(tokenizerConfig)],
    ['tokenizer.json', jsonBytes({})],
    ['onnx/model.onnx', new ArrayBuffer(4)]
  ]);
  let released = false;
  badSession.release = async () => {
    released = true;
  };
  await assert.rejects(
    loadFakeSafeDecide(
      {model: SAFEDECIDE_MODEL, assetsUrl: 'https://localhost/assets/'},
      {
        fake: true,
        openSource: async () => ({read: async (name: string) => files.get(name)}),
        createTokenizer: () => ({special: {padId: 0}}),
        gpuApi: {requestAdapter: async () => ({info: {isFallbackAdapter: false}})},
        ort: {
          env: {webgpu: {}},
          Tensor: class {},
          InferenceSession: {create: async () => badSession}
        }
      }
    ),
    {code: 'INVALID_ASSET'}
  );
  assert.equal(released, true);
});
