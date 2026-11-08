import assert from 'node:assert/strict';
import test from 'node:test';
import {
  TAMEV_MICRO_CALIBRATION_BY_K,
  TAMEV_MICRO_FILES,
  TAMEV_MICRO_MODEL,
  TAMEV_MICRO_PROFILE,
  TAMEV_MICRO_TOTAL_BYTES,
  TAMEV_NANO_FILES,
  TAMEV_NANO_MODEL,
  TAMEV_NANO_PROFILE,
  TAMEV_NANO_TOTAL_BYTES,
  TAMEV_NANO_FLASH_FILES,
  TAMEV_NANO_FLASH_MODEL,
  TAMEV_NANO_FLASH_PROFILE,
  TAMEV_NANO_FLASH_TOTAL_BYTES
} from '../src/tamev/contract.mjs';
import {loadTamev, loadTamevMicro, loadTamevNano, validateTamevLoadOptions} from '../src/tamev/loader.mjs';
import {openTamevSource} from '../src/tamev/source.mjs';
import {calibrateTamevProbabilities, finishTamevDecision, prepareTamevDecision} from '../src/tamev/protocol.mjs';
import {createTamevRunner} from '../src/tamev/runner.mjs';
import {createTamevSession} from '../src/tamev/session.mjs';
import {createTamevTokenizer} from '../src/tamev/tokenizer.mjs';
import {finishTamevSystemOne, prepareTamevSystemOne, renderTamevValue} from '../src/tamev/typed.mjs';

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
  constructor() {}

  token_to_id(token) {
    return {'[CLS]': 101, '[SEP]': 102, '[PAD]': 0}[token];
  }

  encode(text) {
    const content = text.trim()
      ? text
          .trim()
          .split(/\s+/u)
          .map((word) => word.length + 200)
      : [];
    return {ids: [101, ...content, 102]};
  }
}

function tokenizer() {
  return createTamevTokenizer(FakeTokenizer, tokenizerData, tokenizerConfig);
}

test('TAMEV pins the current Micro checkpoint and its complete local file set', () => {
  assert.equal(TAMEV_MICRO_MODEL, 'experimental-tamev-micro-minilm-int8');
  assert.deepEqual(Object.keys(TAMEV_MICRO_FILES), [
    'model_int8.onnx',
    'tokenizer.json',
    'tokenizer_config.json',
    'config.json'
  ]);
  assert.equal(TAMEV_MICRO_TOTAL_BYTES, 24161003);
  assert.equal(
    Object.values(TAMEV_MICRO_FILES).every(({sha256}) => /^[0-9a-f]{64}$/u.test(sha256)),
    true
  );
});

test('TAMEV Nano pins its latest TinyBERT checkpoint and 15.6 MB ONNX asset set', () => {
  assert.equal(TAMEV_NANO_MODEL, 'experimental-tamev-nano-tinybert-int8');
  assert.equal(TAMEV_NANO_PROFILE.revision, 'ad15b45f35526070d48fb91f6597d2805a2d80af');
  assert.equal(TAMEV_NANO_TOTAL_BYTES, 15585045);
  assert.deepEqual(Object.keys(TAMEV_NANO_FILES), [
    'model_int8.onnx',
    'tokenizer.json',
    'tokenizer_config.json',
    'config.json'
  ]);
  assert.equal(TAMEV_NANO_PROFILE.expectedConfig.option_fusion, 'cross');
  assert.equal(
    Object.values(TAMEV_NANO_FILES).every(({sha256}) => /^[0-9a-f]{64}$/u.test(sha256)),
    true
  );
});

test('TAMEV Nano Flash pins its bilinear TinyBERT checkpoint and complete ONNX asset set', () => {
  assert.equal(TAMEV_NANO_FLASH_MODEL, 'experimental-tamev-nano-flash-tinybert-int8');
  assert.equal(TAMEV_NANO_FLASH_PROFILE.revision, '8049ddb646d4c4393df65cb08d0ad2380ef0934a');
  assert.equal(TAMEV_NANO_FLASH_TOTAL_BYTES, 15579274);
  assert.deepEqual(Object.keys(TAMEV_NANO_FLASH_FILES), [
    'model_int8.onnx',
    'tokenizer.json',
    'tokenizer_config.json',
    'config.json'
  ]);
  assert.equal(
    TAMEV_NANO_FLASH_PROFILE.expectedConfig.released_artifact,
    'Fused model (backbone + bilinear pointer head)'
  );
  assert.equal(Object.hasOwn(TAMEV_NANO_FLASH_PROFILE.expectedConfig, 'option_fusion'), false);
  assert.equal(
    Object.values(TAMEV_NANO_FLASH_FILES).every(({sha256}) => /^[0-9a-f]{64}$/u.test(sha256)),
    true
  );
});

test('TAMEV tokenizer preserves BERT markers and reports right truncation', () => {
  const encoder = tokenizer();
  assert.deepEqual(encoder.encodeContext('hello').ids, [101, 205, 102]);
  const exact = encoder.encodeContext('x '.repeat(126));
  assert.equal(exact.ids.length, 128);
  assert.equal(exact.truncated, false);
  assert.equal(exact.ids.at(-1), 102);
  const over = encoder.encodeContext('x '.repeat(127));
  assert.equal(over.ids.length, 128);
  assert.equal(over.truncated, true);
  assert.equal(over.ids.at(-1), 102);
  assert.equal(encoder.encodeOption('x '.repeat(47)).ids.length, 48);
  assert.equal(encoder.encodeOption('x '.repeat(48)).truncated, true);
});

test('TAMEV Nano accepts its BERT tokenizer defaults and applies model row limits', () => {
  const encoder = createTamevTokenizer(
    FakeTokenizer,
    tokenizerData,
    {tokenizer_class: 'BertTokenizer', cls_token: '[CLS]', sep_token: '[SEP]', pad_token: '[PAD]'},
    TAMEV_NANO_PROFILE
  );
  assert.equal(encoder.encodeContext('x '.repeat(126)).ids.length, 128);
  assert.equal(encoder.encodeContext('x '.repeat(127)).truncated, true);
  assert.equal(encoder.encodeOption('x '.repeat(46)).ids.length, 48);
  assert.equal(encoder.encodeOption('x '.repeat(47)).truncated, true);
});

test('TAMEV generic decisions preserve option order and count unpadded encoded tokens', () => {
  const prepared = prepareTamevDecision(tokenizer(), {
    state: 'one two',
    questions: [{instructions: 'pick one', options: ['alpha', 'a longer option']}]
  });
  assert.equal(prepared.questions.length, 1);
  assert.deepEqual(prepared.questions[0].options, ['alpha', 'a longer option']);
  assert.equal(prepared.questions[0].contextIds.length, 6);
  assert.deepEqual(
    prepared.questions[0].optionRows.map((row) => row.length),
    [3, 5]
  );
  assert.equal(prepared.tokenCount, 14);
  assert.equal(prepared.truncated, false);
  const result = finishTamevDecision(prepared, [{logits: [1, 2], probabilities: [0.2, 0.8]}]);
  assert.equal(result.answers[0].selected, 'a longer option');
  assert.equal(result.answers[0].selectedIndex, 1);
  assert.deepEqual(
    result.answers[0].probabilities.map(({option}) => option),
    ['alpha', 'a longer option']
  );
});

test('TAMEV generic choices preserve the first winner above the spread argument limit', () => {
  const optionCount = 200_000;
  const options = Array.from({length: optionCount}, (_, index) => `option-${index}`);
  const probabilities = new Array(optionCount).fill(0);
  probabilities[0] = 0.5;
  probabilities[optionCount - 1] = 0.5;
  const logits = new Array(optionCount).fill(0);

  const result = finishTamevDecision({questions: [{options}], tokenCount: optionCount}, [{logits, probabilities}]);

  assert.equal(result.answers[0].selected, 'option-0');
  assert.equal(result.answers[0].selectedIndex, 0);
  assert.equal(result.answers[0].probabilities.length, optionCount);
  assert.equal(result.answers[0].probabilities[0].option, 'option-0');
  assert.equal(result.answers[0].probabilities.at(-1).option, 'option-199999');
});

test('TAMEV generic and typed choices preserve 78-option order', async () => {
  const options = Array.from({length: 78}, (_, index) => `candidate-${index}`);
  const probabilities = options.map((_, index) => (index === options.length - 1 ? 0.5 : 0.5 / (options.length - 1)));
  const calls = [];
  const runner = {
    async run(contextIds, optionRows) {
      calls.push(optionRows);
      return {
        logits: optionRows.map((_, index) => index),
        probabilities
      };
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const model = createTamevSession({runner, tokenizer: tokenizer(), config: {calibration_by_k: {}}});

  const generic = await model.decide({
    state: 'state',
    questions: [{instructions: 'pick', options}]
  });
  assert.equal(calls[0].length, 78);
  assert.equal(generic.answers[0].selected, options.at(-1));
  assert.equal(generic.answers[0].selectedIndex, 77);
  assert.deepEqual(
    generic.answers[0].probabilities.map(({option}) => option),
    options
  );

  const criteria = Object.fromEntries(options.map((option) => [option, null]));
  const typed = await model.systemOne('state', {
    choice: {type: 'choice', criteria}
  });
  assert.equal(calls[1].length, 78);
  assert.equal(typed.answers.choice.choice, options.at(-1));
  assert.deepEqual(Object.keys(typed.answers.choice.probabilities), options);

  await model.dispose();
});

test('TAMEV generic and typed requests preserve order above 256 questions', async () => {
  const calls = [];
  const runner = {
    async run(contextIds, optionRows) {
      calls.push(optionRows.length);
      return {logits: [1, 0], probabilities: [0.75, 0.25]};
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const model = createTamevSession({runner, tokenizer: tokenizer(), config: {calibration_by_k: {}}});
  const names = Array.from({length: 257}, (_, index) => `question-${index}`);
  const generic = await model.decide({
    state: 'state',
    questions: names.map((instructions) => ({instructions, options: ['first', 'second']}))
  });
  const questions = Object.fromEntries(
    names.map((name) => [name, {type: 'choice', instructions: name, criteria: {first: null, second: null}}])
  );
  const typed = await model.systemOne('state', questions);

  assert.equal(generic.answers.length, 257);
  assert.equal(generic.answers.at(-1).selected, 'first');
  assert.deepEqual(Object.keys(typed.answers), names);
  assert.equal(typed.answers[names.at(-1)].choice, 'first');
  assert.equal(calls.length, 514);
  assert.ok(calls.every((count) => count === 2));
  await model.dispose();
});

test('TAMEV snapshots a Proxy data descriptor once without invoking a later accessor', async () => {
  const counts = {descriptorReads: 0, getterReads: 0, tokenizer: 0, runner: 0};
  const options = new Proxy(
    Object.defineProperties([], {
      0: {
        configurable: true,
        enumerable: true,
        get() {
          counts.getterReads++;
          throw Error('option getter invoked');
        }
      },
      1: {configurable: true, enumerable: true, writable: true, value: 'B'}
    }),
    {
      getOwnPropertyDescriptor(target, key) {
        if (key === '0' && counts.descriptorReads++ === 0)
          return {configurable: true, enumerable: true, writable: true, value: 'A'};
        return Reflect.getOwnPropertyDescriptor(target, key);
      }
    }
  );
  const encodedOptions = [];
  const model = createTamevSession({
    runner: {
      async run() {
        counts.runner++;
        return {logits: [1, 0], probabilities: [0.75, 0.25]};
      },
      async releaseCachedMemory() {},
      async dispose() {}
    },
    tokenizer: {
      encodeContext() {
        counts.tokenizer++;
        return {ids: [1], truncated: false};
      },
      encodeOption(text) {
        counts.tokenizer++;
        encodedOptions.push(text);
        return {ids: [2], truncated: false};
      }
    },
    config: {calibration_by_k: {}}
  });

  const result = await model.decide({state: 'state', questions: [{instructions: 'Choose', options}]});
  assert.equal(result.answers[0].selected, 'A');
  assert.deepEqual(encodedOptions, ['A', 'B']);
  assert.equal(counts.descriptorReads, 1);
  assert.equal(counts.getterReads, 0);
  assert.equal(counts.tokenizer, 3);
  assert.equal(counts.runner, 1);
  await model.dispose();
});

test('TAMEV preserves inherited data fields while rejecting inherited accessors', async () => {
  const counts = {getter: 0, tokenizer: 0};
  const runner = {
    async run(_context, options) {
      return {logits: options.map(() => 0), probabilities: options.map(() => 1 / options.length)};
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const encoder = {
    encodeContext() {
      counts.tokenizer++;
      return {ids: [1], truncated: false};
    },
    encodeOption() {
      counts.tokenizer++;
      return {ids: [2], truncated: false};
    }
  };
  const model = createTamevSession({runner, tokenizer: encoder, config: {calibration_by_k: {}}});
  const inheritedQuestion = Object.create({instructions: 'Choose', options: ['A', 'B']});
  const inheritedRequest = Object.create({state: 'state', questions: [inheritedQuestion]});
  const result = await model.decide(inheritedRequest);
  assert.equal(result.answers[0].selected, 'A');

  let getterReads = 0;
  const questionPrototype = Object.create({options: ['A', 'B']});
  Object.defineProperty(questionPrototype, 'instructions', {
    configurable: true,
    get() {
      getterReads++;
      throw Error('instruction getter invoked');
    }
  });
  await assert.rejects(model.decide({state: 'state', questions: [Object.create(questionPrototype)]}), {
    code: 'INVALID_INPUT'
  });
  assert.equal(getterReads, 0);
  assert.equal(counts.tokenizer, 3);
  await model.dispose();
});

test('TAMEV per-option-count calibration uses only checkpoint entries and preserves the winner', () => {
  for (const [count, delta] of Object.entries(TAMEV_MICRO_CALIBRATION_BY_K)) {
    const source = Array.from({length: Number(count)}, (_, index) => (index === 0 ? 0.7 : 0.3 / (Number(count) - 1)));
    const adjusted = calibrateTamevProbabilities(source, Number(count), {
      calibration_by_k: TAMEV_MICRO_CALIBRATION_BY_K
    });
    assert.equal(adjusted.indexOf(Math.max(...adjusted)), 0);
    assert.ok(Math.abs(adjusted.reduce((total, value) => total + value, 0) - 1) < 1e-12);
    assert.ok(delta > 0);
  }
  const uncalibrated = [0.7, 0.3];
  assert.deepEqual(
    calibrateTamevProbabilities(uncalibrated, 6, {calibration_by_k: TAMEV_MICRO_CALIBRATION_BY_K}),
    uncalibrated
  );
});

test('TAMEV typed inputs keep upstream rendering, option labels, and question kinds', () => {
  const contextTexts = [];
  const encoder = {
    encodeContext(text) {
      contextTexts.push(text);
      return {ids: [101, 202, 102], truncated: false};
    },
    encodeOption(text) {
      return {ids: [101, text.length + 200, 102], truncated: false};
    }
  };
  const prepared = prepareTamevSystemOne(
    {customer: {issue: 'charged twice'}, urgent: false},
    {
      route: {type: 'choice', instructions: 'Select a queue', criteria: {billing: 'duplicate charge', fraud: null}},
      emergency: {type: 'noul', instructions: 'Is the customer stranded?', criteria: {yes: 'no access to cash'}},
      severity: {type: 'score', instructions: 'Rate urgency', criteria: ['routine', 'elevated', 'critical']}
    },
    encoder
  );
  assert.equal(contextTexts[0], 'customer:\n  issue: charged twice\nurgent: False [SEP] Select a queue');
  assert.equal(prepared.questions[0].options[0], 'billing: duplicate charge');
  assert.equal(prepared.questions[0].options[1], 'fraud');
  assert.deepEqual(prepared.questions[1].options, ['No / False', 'Yes / True: no access to cash']);
  assert.deepEqual(prepared.questions[2].labels, ['0', '1', '2']);
  assert.equal(renderTamevValue(['one', {two: 'three'}]), '- one\n- two: three');
});

test('TAMEV typed results retain native confidence, yes probability and expected score semantics', () => {
  const prepared = {
    questions: [
      {id: 'route', type: 'choice', labels: ['billing', 'fraud'], tokenCount: 7, truncated: false},
      {id: 'urgent', type: 'noul', labels: ['false', 'true'], tokenCount: 6, truncated: true},
      {
        id: 'risk',
        type: 'score',
        labels: ['0', '1', '2'],
        legend: ['low', 'medium', 'high'],
        tokenCount: 8,
        truncated: false
      },
      {
        id: 'riskTie',
        type: 'score',
        labels: ['0', '1', '2', '3'],
        legend: ['low', 'medium', 'high', 'critical'],
        tokenCount: 0,
        truncated: false
      }
    ]
  };
  const result = finishTamevSystemOne(
    prepared,
    [
      {probabilities: [0.2, 0.8]},
      {probabilities: [0.25, 0.75]},
      {probabilities: [0.1, 0.6, 0.3]},
      {probabilities: [0.3, 0.2, 0.3, 0.2]}
    ],
    TAMEV_MICRO_MODEL
  );
  assert.equal(result.model, TAMEV_MICRO_MODEL);
  assert.equal(result.tokenCount, 21);
  assert.equal(result.truncated, true);
  assert.deepEqual(result.answers.route, {
    type: 'choice',
    choice: 'fraud',
    confidence: 0.6,
    probabilities: {billing: 0.2, fraud: 0.8}
  });
  assert.deepEqual(result.answers.urgent, {type: 'noul', noul: 0.75});
  assert.deepEqual(result.answers.risk, {
    type: 'score',
    score: 1.2,
    confidence: 0.8,
    legend: {0: 'low', 1: 'medium', 2: 'high'},
    probabilities: {0: 0.1, 1: 0.6, 2: 0.3}
  });
  assert.equal(result.answers.riskTie.confidence, 0.5333);
});

test('TAMEV typed choices preserve the first winner above the spread argument limit', () => {
  const optionCount = 200_000;
  const labels = Array.from({length: optionCount}, (_, index) => `choice-${index}`);
  const probabilities = new Array(optionCount).fill(0);
  probabilities[0] = 0.5;
  probabilities[optionCount - 1] = 0.5;

  const result = finishTamevSystemOne(
    {
      questions: [{id: 'choice', type: 'choice', labels, tokenCount: optionCount, truncated: false}]
    },
    [{probabilities}],
    TAMEV_NANO_MODEL
  );

  assert.equal(result.answers.choice.type, 'choice');
  assert.equal(result.answers.choice.choice, 'choice-0');
  assert.equal(result.answers.choice.confidence, 0.5);
  assert.equal(Object.keys(result.answers.choice.probabilities).length, optionCount);
  assert.equal(Object.keys(result.answers.choice.probabilities)[0], 'choice-0');
  assert.equal(Object.keys(result.answers.choice.probabilities).at(-1), 'choice-199999');
});

test('TAMEV rejects unsupported typed questions, invalid lengths and cyclic JSON', () => {
  const encoder = tokenizer();
  assert.throws(() => prepareTamevSystemOne('state', {rank: {type: 'rank', criteria: ['a', 'b']}}, encoder), {
    code: 'INVALID_INPUT'
  });
  assert.throws(
    () => prepareTamevDecision(encoder, {state: 'state', questions: [{instructions: '', options: ['only one']}]}),
    {code: 'INVALID_INPUT'}
  );
  const cyclic = {};
  cyclic.self = cyclic;
  assert.throws(() => renderTamevValue(cyclic), {code: 'INVALID_INPUT'});
});

test('TAMEV runner sends dynamic int64 context and option tensors to the pinned graph', async () => {
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, {type, data, dims});
    }
  }
  let inputs;
  let releases = 0;
  const runner = createTamevRunner(
    {Tensor},
    {
      async run(value) {
        inputs = value;
        return {
          logits: new Tensor('float32', Float32Array.of(0.5, 1.5), [1, 2]),
          probs: new Tensor('float32', Float32Array.of(0.25, 0.75), [1, 2])
        };
      },
      async release() {
        releases++;
      }
    },
    {special: {padId: 0}, encodeContext() {}}
  );
  const result = await runner.run(
    [101, 2, 102],
    [
      [101, 3, 102],
      [101, 4, 5, 102]
    ]
  );
  assert.deepEqual(inputs.ctx_input_ids.dims, [1, 3]);
  assert.deepEqual(inputs.opt_input_ids.dims, [1, 2, 4]);
  assert.deepEqual(Array.from(inputs.opt_input_ids.data), [101n, 3n, 102n, 0n, 101n, 4n, 5n, 102n]);
  assert.deepEqual(Array.from(inputs.opt_attention_mask.data), [1n, 1n, 1n, 0n, 1n, 1n, 1n, 1n]);
  assert.deepEqual(result, {logits: [0.5, 1.5], probabilities: [0.25, 0.75]});
  await runner.releaseCachedMemory();
  await Promise.all([runner.dispose(), runner.dispose()]);
  assert.equal(releases, 1);
});

test('TAMEV runner preserves a 78-row option axis and output order', async () => {
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, {type, data, dims});
    }
  }
  const optionRows = Array.from({length: 78}, (_, index) => [101, index + 200, 102]);
  let inputs;
  const runner = createTamevRunner(
    {Tensor},
    {
      async run(value) {
        inputs = value;
        return {
          logits: new Tensor(
            'float32',
            Float32Array.from({length: optionRows.length}, (_, index) => index),
            [1, 78]
          ),
          probs: new Tensor(
            'float32',
            Float32Array.from({length: optionRows.length}, (_, index) => (index === 77 ? 0.5 : 0.5 / 77)),
            [1, 78]
          )
        };
      },
      async release() {}
    },
    {special: {padId: 0}, encodeContext() {}}
  );

  const result = await runner.run([101, 2, 102], optionRows);
  assert.deepEqual(inputs.opt_input_ids.dims, [1, 78, 3]);
  assert.deepEqual(Array.from(inputs.opt_input_ids.data).slice(0, 9), [
    101n,
    200n,
    102n,
    101n,
    201n,
    102n,
    101n,
    202n,
    102n
  ]);
  assert.deepEqual(Array.from(inputs.opt_input_ids.data).slice(-3), [101n, 277n, 102n]);
  assert.equal(result.logits.length, 78);
  assert.deepEqual(result.logits.slice(0, 3), [0, 1, 2]);
  assert.equal(result.logits.at(-1), 77);
  assert.equal(result.probabilities.at(-1), 0.5);
});

test('TAMEV runner batches 200k option rows without an argument-count ceiling', async () => {
  const optionCount = 200_000;
  const optionRows = Array(optionCount).fill([101, 202, 102]);
  const logits = new Float32Array(optionCount);
  const probabilities = new Float32Array(optionCount);
  probabilities[0] = 1;
  let inputs;

  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, {type, data, dims});
    }
  }

  const runner = createTamevRunner(
    {Tensor},
    {
      async run(value) {
        inputs = value;
        return {
          logits: new Tensor('float32', logits, [1, optionCount]),
          probs: new Tensor('float32', probabilities, [1, optionCount])
        };
      },
      async release() {}
    },
    {special: {padId: 0}, encodeContext() {}}
  );

  const result = await runner.run([101, 2, 102], optionRows);
  assert.deepEqual(inputs.opt_input_ids.dims, [1, optionCount, 3]);
  assert.equal(inputs.opt_input_ids.data.length, optionCount * 3);
  assert.equal(result.logits.length, optionCount);
  assert.equal(result.probabilities[0], 1);
  assert.equal(result.probabilities.at(-1), 0);
  await runner.dispose();
});

test('TAMEV session calibrates generic choices and preserves the raw typed protocol', async () => {
  const calls = [];
  let disposed = 0;
  const runner = {
    async run(contextIds, optionRows) {
      calls.push({contextIds, optionRows});
      return {logits: [1, 2], probabilities: [0.7, 0.3]};
    },
    async releaseCachedMemory() {},
    async dispose() {
      disposed++;
    }
  };
  const model = createTamevSession({
    runner,
    tokenizer: tokenizer(),
    config: {calibration_by_k: TAMEV_MICRO_CALIBRATION_BY_K}
  });
  const generic = await model.decide({
    state: 'state',
    questions: [{instructions: 'pick', options: ['first', 'second']}]
  });
  const typed = await model.systemOne('state', {
    pick: {type: 'choice', instructions: 'pick', criteria: {first: null, second: null}}
  });
  assert.notEqual(generic.answers[0].probabilities[0].probability, 0.7);
  assert.ok(generic.answers[0].probabilities[0].probability < 0.7);
  assert.equal(typed.answers.pick.probabilities.first, 0.7);
  assert.equal(calls.length, 2);
  await Promise.all([model.dispose(), model.dispose()]);
  assert.equal(disposed, 1);
  await assert.rejects(model.decide({state: 'state', questions: [{instructions: '', options: ['a', 'b']}]}), {
    code: 'SESSION_CLOSED'
  });
});

test('TAMEV loader defaults to pinned revisions, allows mirrors, and rejects bad options', async () => {
  for (const profile of [TAMEV_MICRO_PROFILE, TAMEV_NANO_PROFILE, TAMEV_NANO_FLASH_PROFILE]) {
    const checked = validateTamevLoadOptions({model: profile.model}, profile);
    assert.equal(checked.base.href, `https://huggingface.co/${profile.repository}/resolve/${profile.revision}/`);
    assert.equal(checked.cacheEnabled, true);
    assert.equal(checked.offline, false);
    assert.equal(
      validateTamevLoadOptions({model: profile.model, assetsUrl: 'https://example.test/model'}, profile).base.href,
      'https://example.test/model/'
    );
  }
  assert.equal(
    validateTamevLoadOptions({
      model: TAMEV_MICRO_MODEL,
      assetsUrl: 'https://example.test/model/',
      maxPrefetchBytes: 0,
      debertaReadback: undefined
    }).base.href,
    'https://example.test/model/'
  );
  assert.throws(() => validateTamevLoadOptions({model: 'other', assetsUrl: '/model/'}), {code: 'INVALID_INPUT'});
  assert.throws(() => validateTamevLoadOptions({model: TAMEV_MICRO_MODEL, assetsUrl: '/model/', maxPrefetchBytes: 1}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateTamevLoadOptions({model: TAMEV_MICRO_MODEL, assetsUrl: 'file:///model/'}), {
    code: 'INVALID_INPUT'
  });
  assert.equal(validateTamevLoadOptions({model: TAMEV_NANO_MODEL, offline: true}, TAMEV_NANO_PROFILE).offline, true);
  assert.throws(
    () => validateTamevLoadOptions({model: TAMEV_NANO_MODEL, offline: true, cache: false}, TAMEV_NANO_PROFILE),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      validateTamevLoadOptions({model: TAMEV_NANO_FLASH_MODEL, offline: true, cache: false}, TAMEV_NANO_FLASH_PROFILE),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(() => validateTamevLoadOptions({model: TAMEV_MICRO_MODEL, assetsUrl: '/model/', other: true}), {
    code: 'INVALID_INPUT'
  });
});

test('TAMEV Nano Flash loads by ID, keeps typed behavior and WASM, and accepts a trusted mirror', async () => {
  const config = {
    ...TAMEV_NANO_FLASH_PROFILE.expectedConfig,
    calibration_by_k: TAMEV_NANO_FLASH_PROFILE.calibrationByK
  };
  const source = {
    async read(name) {
      if (name === 'tokenizer.json') return new TextEncoder().encode(JSON.stringify(tokenizerData)).buffer;
      if (name === 'tokenizer_config.json') return new TextEncoder().encode(JSON.stringify(tokenizerConfig)).buffer;
      if (name === 'config.json') return new TextEncoder().encode(JSON.stringify(config)).buffer;
      if (name === 'model_int8.onnx') return Uint8Array.of(1, 2, 3).buffer;
      throw new Error(`unexpected asset ${name}`);
    }
  };

  for (const assetsUrl of [undefined, 'https://mirror.example.test/tamev/nano-flash']) {
    let sourceBase;
    let sessionOptions;
    const ort = {
      Tensor: class {
        constructor(type, data, dims) {
          Object.assign(this, {type, data, dims});
        }
      },
      InferenceSession: {
        async create(bytes, options) {
          assert.equal(bytes.byteLength, 3);
          sessionOptions = options;
          return {
            inputNames: ['ctx_input_ids', 'ctx_attention_mask', 'opt_input_ids', 'opt_attention_mask'],
            outputNames: ['logits', 'probs'],
            async run() {
              return {
                logits: {type: 'float32', dims: [1, 2], data: Float32Array.of(0.2, 0.8)},
                probs: {type: 'float32', dims: [1, 2], data: Float32Array.of(0.4, 0.6)}
              };
            },
            async release() {}
          };
        }
      }
    };
    const options = {model: TAMEV_NANO_FLASH_MODEL};
    if (assetsUrl !== undefined) options.assetsUrl = assetsUrl;
    const loaded = await loadTamev(options, {
      ort,
      Tokenizer: FakeTokenizer,
      openSource: async (checked) => {
        sourceBase = checked.base.href;
        return source;
      }
    });

    const expectedBase =
      assetsUrl ??
      `https://huggingface.co/${TAMEV_NANO_FLASH_PROFILE.repository}/resolve/${TAMEV_NANO_FLASH_PROFILE.revision}/`;
    assert.equal(sourceBase, expectedBase.endsWith('/') ? expectedBase : `${expectedBase}/`);
    assert.deepEqual(sessionOptions, {
      executionProviders: ['wasm'],
      intraOpNumThreads: 1,
      interOpNumThreads: 1,
      graphOptimizationLevel: 'all'
    });
    const result = await loaded.systemOne('state', {
      route: {type: 'choice', criteria: {billing: 'billing', fraud: 'fraud'}}
    });
    assert.equal(result.model, TAMEV_NANO_FLASH_MODEL);
    assert.equal(result.answers.route.choice, 'fraud');
    await loaded.dispose();
  }
});

test('TAMEV stable load failures report the public ID', async () => {
  await assert.rejects(
    loadTamev(
      {model: TAMEV_NANO_FLASH_MODEL},
      {
        publicModelId: 'tamev-nano-flash-tinybert-int8',
        openSource: async () => {
          throw new Error('asset request failed');
        }
      }
    ),
    (error) => {
      assert.equal(error.code, 'MODEL_LOAD_FAILED');
      assert.equal(error.message, 'Could not load tamev-nano-flash-tinybert-int8.');
      assert.equal(error.message.includes('experimental-'), false);
      return true;
    }
  );
});

test('TAMEV Nano Flash verifies default and mirror bytes and stays offline on cache misses', async (t) => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    requests.push(String(url));
    return new Response(new Uint8Array(TAMEV_NANO_FLASH_FILES['config.json'].bytes));
  });

  const sources = [
    {
      options: {model: TAMEV_NANO_FLASH_MODEL},
      base: `https://huggingface.co/${TAMEV_NANO_FLASH_PROFILE.repository}/resolve/${TAMEV_NANO_FLASH_PROFILE.revision}/`
    },
    {
      options: {model: TAMEV_NANO_FLASH_MODEL, assetsUrl: 'https://mirror.example.test/tamev/nano-flash/'},
      base: 'https://mirror.example.test/tamev/nano-flash/'
    }
  ];
  for (const {options} of sources) {
    const checked = validateTamevLoadOptions(options, TAMEV_NANO_FLASH_PROFILE);
    const source = await openTamevSource(checked, TAMEV_NANO_FLASH_PROFILE);
    await assert.rejects(source.read('config.json'), {code: 'INVALID_ASSET', message: /Integrity check failed/u});
  }
  assert.deepEqual(
    requests,
    sources.map(({base}) => new URL('config.json', base).href)
  );

  const previousCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: {
      async open() {
        return {
          async match() {
            return null;
          }
        };
      }
    }
  });
  t.after(() => {
    if (previousCaches) Object.defineProperty(globalThis, 'caches', previousCaches);
    else delete globalThis.caches;
  });
  const offline = validateTamevLoadOptions({model: TAMEV_NANO_FLASH_MODEL, offline: true}, TAMEV_NANO_FLASH_PROFILE);
  const offlineSource = await openTamevSource(offline, TAMEV_NANO_FLASH_PROFILE);
  await assert.rejects(offlineSource.read('config.json'), {
    code: 'ASSET_LOAD_FAILED',
    message: /not available offline/u
  });
  assert.equal(requests.length, 2);
});

test('TAMEV loader checks pinned configuration and ONNX interface before returning a session', async () => {
  const config = {
    ...TAMEV_MICRO_PROFILE.expectedConfig,
    calibration_by_k: TAMEV_MICRO_CALIBRATION_BY_K
  };
  const source = {
    async read(name) {
      if (name === 'tokenizer.json') return new TextEncoder().encode(JSON.stringify(tokenizerData)).buffer;
      if (name === 'tokenizer_config.json') return new TextEncoder().encode(JSON.stringify(tokenizerConfig)).buffer;
      if (name === 'config.json') return new TextEncoder().encode(JSON.stringify(config)).buffer;
      if (name === 'model_int8.onnx') return Uint8Array.of(1, 2, 3).buffer;
      throw new Error(`unexpected asset ${name}`);
    }
  };
  let sessionOptions;
  let released = 0;
  const ort = {
    Tensor: class {
      constructor(type, data, dims) {
        Object.assign(this, {type, data, dims});
      }
    },
    InferenceSession: {
      async create(bytes, options) {
        assert.equal(bytes.byteLength, 3);
        sessionOptions = options;
        return {
          inputNames: ['ctx_input_ids', 'ctx_attention_mask', 'opt_input_ids', 'opt_attention_mask'],
          outputNames: ['logits', 'probs'],
          async run() {
            return {
              logits: {type: 'float32', dims: [1, 2], data: Float32Array.of(0.2, 0.8)},
              probs: {type: 'float32', dims: [1, 2], data: Float32Array.of(0.4, 0.6)}
            };
          },
          async release() {
            released++;
          }
        };
      }
    }
  };
  let sourceBase;
  const loaded = await loadTamevMicro(
    {model: TAMEV_MICRO_MODEL, cache: false},
    {
      ort,
      Tokenizer: FakeTokenizer,
      openSource: async (checked) => {
        sourceBase = checked.base.href;
        return source;
      }
    }
  );
  assert.equal(
    sourceBase,
    `https://huggingface.co/${TAMEV_MICRO_PROFILE.repository}/resolve/${TAMEV_MICRO_PROFILE.revision}/`
  );
  assert.deepEqual(sessionOptions, {
    executionProviders: ['wasm'],
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
    graphOptimizationLevel: 'all'
  });
  const result = await loaded.decide({state: 'state', questions: [{instructions: 'pick', options: ['a', 'b']}]});
  assert.equal(result.answers[0].selected, 'b');
  await loaded.dispose();
  assert.equal(released, 1);
});

test('TAMEV Nano loader accepts the pinned cross-fusion ONNX graph and typed API', async () => {
  const config = {
    ...TAMEV_NANO_PROFILE.expectedConfig,
    calibration_by_k: TAMEV_NANO_PROFILE.calibrationByK
  };
  const source = {
    async read(name) {
      if (name === 'tokenizer.json') return new TextEncoder().encode(JSON.stringify(tokenizerData)).buffer;
      if (name === 'tokenizer_config.json') {
        return new TextEncoder().encode(
          JSON.stringify({tokenizer_class: 'BertTokenizer', cls_token: '[CLS]', sep_token: '[SEP]', pad_token: '[PAD]'})
        ).buffer;
      }
      if (name === 'config.json') return new TextEncoder().encode(JSON.stringify(config)).buffer;
      if (name === 'model_int8.onnx') return Uint8Array.of(1, 2, 3).buffer;
      throw new Error(`unexpected asset ${name}`);
    }
  };
  let released = 0;
  const ort = {
    Tensor: class {
      constructor(type, data, dims) {
        Object.assign(this, {type, data, dims});
      }
    },
    InferenceSession: {
      async create() {
        return {
          inputNames: ['ctx_input_ids', 'ctx_attention_mask', 'opt_input_ids', 'opt_attention_mask'],
          outputNames: ['logits', 'probs'],
          async run() {
            return {
              logits: {type: 'float32', dims: [1, 2], data: Float32Array.of(0.2, 0.8)},
              probs: {type: 'float32', dims: [1, 2], data: Float32Array.of(0.4, 0.6)}
            };
          },
          async release() {
            released++;
          }
        };
      }
    }
  };
  let sourceBase;
  const loaded = await loadTamevNano(
    {model: TAMEV_NANO_MODEL},
    {
      ort,
      Tokenizer: FakeTokenizer,
      openSource: async (checked) => {
        sourceBase = checked.base.href;
        return source;
      }
    }
  );
  assert.equal(
    sourceBase,
    `https://huggingface.co/${TAMEV_NANO_PROFILE.repository}/resolve/${TAMEV_NANO_PROFILE.revision}/`
  );
  const typed = await loaded.systemOne('state', {
    route: {type: 'choice', criteria: {billing: 'billing', fraud: 'fraud'}}
  });
  assert.equal(typed.model, TAMEV_NANO_MODEL);
  assert.equal(typed.answers.route.choice, 'fraud');
  await loaded.dispose();
  assert.equal(released, 1);
});
