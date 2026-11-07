import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createTamevTokenizer
} from '../source/overlay/packages/web-decision/src/tamev/tokenizer.mjs';
import {
  TAMEV_MICRO_MODEL,
  TAMEV_MICRO_PROFILE,
  TAMEV_MICRO_TOTAL_BYTES,
  TAMEV_NANO_MODEL,
  TAMEV_NANO_PROFILE,
  TAMEV_NANO_TOTAL_BYTES
} from '../source/overlay/packages/web-decision/src/tamev/contract.mjs';
import {
  calibrateTamevProbabilities,
  finishTamevDecision,
  prepareTamevDecision
} from '../source/overlay/packages/web-decision/src/tamev/protocol.mjs';
import {createTamevRunner} from '../source/overlay/packages/web-decision/src/tamev/runner.mjs';
import {
  finishTamevSystemOne,
  prepareTamevSystemOne
} from '../source/overlay/packages/web-decision/src/tamev/typed.mjs';
import {
  describeModel,
  isDecisionModel,
  listDecisionModels
} from '../source/overlay/packages/web-decision/src/kev-preview-registry.js';

const tokenizerData = {model: {type: 'WordPiece'}};
const tokenizerConfig = {
  tokenizer_class: 'BertTokenizer',
  cls_token: '[CLS]',
  sep_token: '[SEP]',
  pad_token: '[PAD]',
  max_length: 128,
  padding_side: 'right',
  truncation_side: 'right'
};

class FakeTokenizer {
  token_to_id(token) {
    return {'[CLS]': 101, '[SEP]': 102, '[PAD]': 0}[token];
  }

  encode(text) {
    const tokens = text.trim() ? text.trim().split(/\s+/u) : [];
    return {ids: [101, ...tokens.map((_, index) => 103 + index), 102]};
  }
}

function makeTokenizer() {
  return createTamevTokenizer(FakeTokenizer, tokenizerData, tokenizerConfig, TAMEV_NANO_PROFILE);
}

test('release registry exposes Kev and exactly two TAMEV routes', () => {
  assert.deepEqual(listDecisionModels(), ['kev-0.6b-q4f16', TAMEV_NANO_MODEL, TAMEV_MICRO_MODEL]);
  assert.equal(isDecisionModel(TAMEV_NANO_MODEL), true);
  assert.equal(isDecisionModel('experimental-tamev-nano-flash-tinybert-int8'), false);
  assert.equal(isDecisionModel(TAMEV_MICRO_MODEL), true);
  const description = describeModel(TAMEV_NANO_MODEL);
  assert.equal(description.backend.name, 'onnxruntime-web-wasm');
  assert.equal(description.verification.currentBundleCpuWasm, 'verified');
  assert.equal(description.assets.totalBytes, TAMEV_NANO_TOTAL_BYTES);
  assert.equal(description.checkpoint.revision, TAMEV_NANO_PROFILE.revision);
  assert.equal(Object.isFrozen(description), true);
  assert.equal(Object.isFrozen(description.assets.requiredFiles), true);
  const microDescription = describeModel(TAMEV_MICRO_MODEL);
  assert.equal(microDescription.backend.name, 'onnxruntime-web-wasm');
  assert.equal(microDescription.assets.totalBytes, TAMEV_MICRO_TOTAL_BYTES);
  assert.equal(microDescription.checkpoint.revision, TAMEV_MICRO_PROFILE.revision);
  assert.equal(microDescription.verification.currentBundleGpu, 'not-applicable');
  assert.equal(microDescription.verification.currentBundleCpuWasm, 'verified');
  assert.equal(describeModel('kev-0.6b-q4f16').verification.currentBundleGpu, 'verified');
});

test('TAMEV Micro pins its own protocol, weights and calibration', () => {
  assert.equal(TAMEV_MICRO_PROFILE.name, 'TAMEV Micro MiniLM');
  assert.equal(TAMEV_MICRO_PROFILE.expectedConfig.backbone, 'sentence-transformers/all-MiniLM-L6-v2');
  assert.equal(TAMEV_MICRO_PROFILE.expectedConfig.pooling, 'mean');
  assert.equal(TAMEV_MICRO_PROFILE.expectedConfig.temperature, 1.66);
  assert.equal(TAMEV_MICRO_PROFILE.maxContextTokens, 128);
  assert.equal(TAMEV_MICRO_PROFILE.maxOptionTokens, 48);
  assert.equal(TAMEV_MICRO_TOTAL_BYTES, 24161003);
  assert.deepEqual(Object.keys(TAMEV_MICRO_PROFILE.files), [
    'model_int8.onnx',
    'tokenizer.json',
    'tokenizer_config.json',
    'config.json'
  ]);
  assert.equal(TAMEV_MICRO_PROFILE.calibrationByK[3], 1.1722);
});

test('TAMEV Nano tokenizes model rows and reports both exact and over-limit boundaries', () => {
  const tokenizer = makeTokenizer();
  const exactContext = tokenizer.encodeContext('x '.repeat(126).trim());
  const overContext = tokenizer.encodeContext('x '.repeat(127).trim());
  const exactOption = tokenizer.encodeOption('x '.repeat(46).trim());
  const overOption = tokenizer.encodeOption('x '.repeat(47).trim());

  assert.equal(exactContext.ids.length, 128);
  assert.equal(exactContext.truncated, false);
  assert.equal(overContext.ids.length, 128);
  assert.equal(overContext.truncated, true);
  assert.equal(overContext.ids.at(-1), tokenizer.special.sepId);
  assert.equal(exactOption.ids.length, 48);
  assert.equal(exactOption.truncated, false);
  assert.equal(overOption.ids.length, 48);
  assert.equal(overOption.truncated, true);
});

test('generic decision preserves option order, calibration and first-winner behavior', () => {
  const prepared = prepareTamevDecision(
    makeTokenizer(),
    {state: 'payment issue', questions: [{instructions: 'choose route', options: ['billing', 'fraud', 'support']}]},
    TAMEV_NANO_PROFILE
  );
  const calibrated = calibrateTamevProbabilities([0.2, 0.5, 0.3], 3, {
    calibration_by_k: TAMEV_NANO_PROFILE.calibrationByK
  });
  const result = finishTamevDecision(prepared, [{logits: [1, 2, 3], probabilities: [0.2, 0.5, 0.3]}]);

  assert.deepEqual(prepared.questions[0].options, ['billing', 'fraud', 'support']);
  assert.deepEqual(result.answers[0].probabilities.map(({option}) => option), ['billing', 'fraud', 'support']);
  assert.equal(result.answers[0].selected, 'fraud');
  assert.ok(Math.abs(calibrated.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.equal(calibrated.indexOf(Math.max(...calibrated)), 1);
  assert.equal(result.truncated, undefined);
});

test('typed protocol preserves Choice, Noul, Score and question insertion order', () => {
  const prepared = prepareTamevSystemOne(
    {payment: 'duplicate charge'},
    {
      route: {
        type: 'choice',
        instructions: 'Select a route',
        criteria: {billing: 'review duplicate charge', fraud: 'investigate unauthorized use'}
      },
      urgent: {type: 'noul', instructions: 'Is immediate help needed?'},
      severity: {type: 'score', instructions: 'Rate impact', criteria: ['low', 'medium', 'high']}
    },
    makeTokenizer()
  );
  const result = finishTamevSystemOne(
    prepared,
    [
      {probabilities: [0.8, 0.2]},
      {probabilities: [0.2, 0.8]},
      {probabilities: [0.2, 0.3, 0.5]}
    ],
    TAMEV_NANO_MODEL
  );

  assert.deepEqual(Object.keys(result.answers), ['route', 'urgent', 'severity']);
  assert.equal(result.answers.route.type, 'choice');
  assert.equal(result.answers.route.choice, 'billing');
  assert.equal(result.answers.urgent.noul, 0.8);
  assert.equal(result.answers.severity.score, 1.3);
  assert.deepEqual(result.answers.severity.legend, {0: 'low', 1: 'medium', 2: 'high'});
  assert.equal(result.model, TAMEV_NANO_MODEL);
});

test('runner sends padded int64 tensors with a dynamic option axis and releases its session', async () => {
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, {type, data, dims});
    }
  }
  let inputsSeen;
  let released = false;
  const session = {
    async run(inputs) {
      inputsSeen = inputs;
      return {
        logits: {type: 'float32', dims: [1, 3], data: new Float32Array([1, 2, 3])},
        probs: {type: 'float32', dims: [1, 3], data: new Float32Array([0.2, 0.3, 0.5])}
      };
    },
    async release() {
      released = true;
    }
  };
  const tokenizer = makeTokenizer();
  const runner = createTamevRunner({Tensor}, session, tokenizer);
  const result = await runner.run([101, 102], [[101, 102], [101, 103, 102], [101, 104, 105, 102]]);

  assert.deepEqual(inputsSeen.ctx_input_ids.dims, [1, 2]);
  assert.deepEqual(inputsSeen.opt_input_ids.dims, [1, 3, 4]);
  assert.equal(inputsSeen.opt_input_ids.type, 'int64');
  assert.ok(result.probabilities.every((value, index) => Math.abs(value - [0.2, 0.3, 0.5][index]) < 1e-7));
  await runner.dispose();
  assert.equal(released, true);
  await assert.rejects(() => runner.run([101, 102], [[101, 102], [101, 102]]), {code: 'SESSION_CLOSED'});
});
