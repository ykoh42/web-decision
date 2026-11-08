import assert from 'node:assert/strict';
import test from 'node:test';
import {
  CRIVO_BATCH_SIZE,
  CRIVO_BATCH_TOKEN_BUDGET,
  CRIVO_FP32_SPEC,
  CRIVO_INT8_SPEC,
  CRIVO_MAX_SEQUENCE_TOKENS,
  getCrivoSpec
} from '../src/crivo/contract.mjs';
import {loadCrivoModel, validateCrivoLoadOptions} from '../src/crivo/loader.mjs';
import {
  finishCrivoDecision,
  finishCrivoSystemOne,
  prepareCrivoDecision,
  prepareCrivoSystemOne,
  validateCrivoCallOptions
} from '../src/crivo/protocol.mjs';
import {createCrivoRunner} from '../src/crivo/runner.mjs';
import {createCrivoSession} from '../src/crivo/session.mjs';
import {createCrivoTokenizer, validateCrivoConfig} from '../src/crivo/tokenizer.mjs';

const special = Object.freeze({cls: 50281, sep: 50282, pad: 50283, opt: 50368});

function fakeTokenizer() {
  const encoded = [];
  return {
    encoded,
    special,
    encode(text) {
      encoded.push(text);
      return Array.from(text, (character) => character.codePointAt(0) + 10);
    }
  };
}

test('Crivo pins both graph variants and their external file manifests', () => {
  assert.equal(CRIVO_FP32_SPEC.model, 'experimental-crivo-32m-v2-fp32');
  assert.equal(CRIVO_INT8_SPEC.model, 'experimental-crivo-32m-v2-int8');
  assert.equal(CRIVO_FP32_SPEC.files['onnx/model.onnx'].bytes, 128385184);
  assert.equal(CRIVO_INT8_SPEC.files['onnx/model_quantized.onnx'].bytes, 32729616);
  assert.equal(getCrivoSpec(CRIVO_FP32_SPEC.model), CRIVO_FP32_SPEC);
  assert.equal(getCrivoSpec('unknown'), undefined);
});

test('Crivo validates the pinned model and tokenizer configuration', () => {
  const config = {
    model_type: 'custom',
    vocab_size: 50369,
    max_position_embeddings: 7999,
    cls_token_id: 50281,
    sep_token_id: 50282,
    pad_token_id: 50283,
    decision: {opt_token: '[OPT]', opt_token_id: 50368, question_types: {noul: 0, choice: 1, score: 2}}
  };
  const tokenizerConfig = {
    backend: 'tokenizers',
    cls_token: '[CLS]',
    sep_token: '[SEP]',
    pad_token: '[PAD]',
    max_length: 7999,
    model_max_length: 7999
  };
  assert.equal(validateCrivoConfig(config, tokenizerConfig), undefined);
  assert.throws(
    () => validateCrivoConfig({...config, decision: {...config.decision, opt_token_id: 2}}, tokenizerConfig),
    {code: 'INVALID_ASSET'}
  );
});

test('Crivo tokenizer checks special token IDs and does not add automatic tokens', () => {
  class Tokenizer {
    constructor() {}
    token_to_id(token) {
      return {'[CLS]': 50281, '[SEP]': 50282, '[PAD]': 50283, '[OPT]': 50368}[token];
    }
    encode(text, options) {
      assert.deepEqual(options, {add_special_tokens: false});
      return {ids: Array.from(text, (character) => character.codePointAt(0) + 10)};
    }
  }
  const tokenizer = createCrivoTokenizer(Tokenizer, {model: {type: 'BPE'}}, {});
  assert.deepEqual(tokenizer.encode('yes'), [131, 111, 125]);
  assert.throws(
    () =>
      createCrivoTokenizer(
        class extends Tokenizer {
          token_to_id(token) {
            return token === '[OPT]' ? 1 : super.token_to_id(token);
          }
        },
        {model: {type: 'BPE'}},
        {}
      ),
    {code: 'INVALID_ASSET'}
  );
});

test('Crivo renders sorted JSON and preserves typed option order and task semantics', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareCrivoSystemOne(
    {z: 1, a: 'ok'},
    {
      intent: {type: 'choice', instructions: ' Pick intent ', criteria: {refund: 'Money back', other: 'other'}},
      claim: {type: 'noul', instructions: 'Duplicate charge?', criteria: {true: 'charged twice', false: 'one charge'}},
      urgency: {type: 'score', instructions: 'Rate urgency', criteria: ['Low', 'Medium', 'High']}
    },
    tokenizer
  );
  assert.equal(tokenizer.encoded[0], 'Pick intent');
  assert.equal(tokenizer.encoded[1], '{"a":"ok","z":1}');
  assert.ok(tokenizer.encoded.includes('refund: Money back'));
  assert.ok(tokenizer.encoded.includes('other'));
  assert.ok(tokenizer.encoded.includes('false: one charge'));
  assert.ok(tokenizer.encoded.includes('true: charged twice'));
  assert.ok(tokenizer.encoded.includes('0: Low'));
  assert.deepEqual(
    prepared.rows.map((row) => [row.type, row.qtype]),
    [
      ['choice', 1],
      ['noul', 0],
      ['score', 2]
    ]
  );

  const result = finishCrivoSystemOne(
    prepared,
    [Float32Array.of(3, 0), Float32Array.of(0, 2), Float32Array.of(0, 0, 0)],
    CRIVO_FP32_SPEC.model
  );
  assert.equal(result.model, CRIVO_FP32_SPEC.model);
  assert.equal(result.answers.intent.choice, 'refund');
  assert.ok(result.answers.claim.noul > 0.8);
  assert.ok(Math.abs(result.answers.urgency.score - 1) < 1e-6);
  assert.equal(Object.hasOwn(result.answers.intent, 'confidence'), false);
});

test('Crivo validates typed criteria and structured JSON edge cases before tokenization', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareCrivoSystemOne(
    {aa: 2, a: 1},
    {
      score: {type: 'score', instructions: 'Rate', criteria: {low: 'Low', high: 'High'}},
      noul: {type: 'noul', instructions: 'Valid?', criteria: null}
    },
    tokenizer
  );
  assert.equal(tokenizer.encoded[1], '{"a":1,"aa":2}');
  assert.deepEqual(
    prepared.rows.map((row) => row.optionTexts),
    [
      ['0: Low', '1: High'],
      ['No.', 'Yes.']
    ]
  );

  const cyclic = {};
  cyclic.self = cyclic;
  const sparseScore = new Array(2);
  const symbolCriteria = {one: null, two: null, [Symbol('extra')]: 'ignored'};
  const invalidQuestions = [
    {q: {type: 'choice', instructions: 'Pick', criteria: {only: null}}},
    {q: {type: 'score', instructions: 'Rate', criteria: ['only']}},
    {q: {type: 'score', instructions: 'Rate', criteria: sparseScore}},
    {q: {type: 'noul', instructions: 'Valid?', criteria: {maybe: 'perhaps'}}},
    {q: {type: 'choice', instructions: 'Pick', criteria: symbolCriteria}},
    {q: {type: 'rank', instructions: 'Pick', criteria: {a: null, b: null}}}
  ];
  for (const questions of invalidQuestions)
    assert.throws(() => prepareCrivoSystemOne('', questions, tokenizer), {code: 'INVALID_INPUT'});
  assert.throws(() => prepareCrivoSystemOne(cyclic, {q: {type: 'noul', instructions: 'Valid?'}}, tokenizer), {
    code: 'INVALID_INPUT'
  });
  assert.throws(
    () => prepareCrivoSystemOne({notFinite: Number.NaN}, {q: {type: 'noul', instructions: 'Valid?'}}, tokenizer),
    {code: 'INVALID_INPUT'}
  );
});

test('Crivo rejects mismatched question and option output axes', () => {
  const prepared = prepareCrivoSystemOne(
    '',
    {
      choice: {type: 'choice', instructions: 'Pick', criteria: {a: null, b: null}}
    },
    fakeTokenizer()
  );
  assert.throws(() => finishCrivoSystemOne(prepared, [], CRIVO_FP32_SPEC.model), {code: 'INFERENCE_ERROR'});
  assert.throws(() => finishCrivoSystemOne(prepared, [Float32Array.of(1)], CRIVO_FP32_SPEC.model), {
    code: 'INFERENCE_ERROR'
  });
  assert.throws(() => finishCrivoDecision(prepared, [[Number.POSITIVE_INFINITY]]), {code: 'INFERENCE_ERROR'});
});

test('Crivo rejects malformed typed inputs and call options before encoding', () => {
  const tokenizer = fakeTokenizer();
  const question = {type: 'choice', instructions: 'Pick', criteria: {a: null, b: null}};
  const extraArray = ['a', 'b'];
  extraArray.extra = true;
  const sparseArray = new Array(2);
  const inheritedCriteria = Object.create({prototypeField: true});
  inheritedCriteria.a = null;
  inheritedCriteria.b = null;
  const accessor = {};
  Object.defineProperty(accessor, 'signal', {
    enumerable: true,
    get() {
      throw Error('signal getter invoked');
    }
  });

  for (const questions of [
    {},
    {'': question},
    {q: {...question, extra: true}},
    {q: {...question, type: 'unknown'}},
    {q: {...question, type: 'choice', criteria: []}},
    {q: {...question, type: 'choice', criteria: inheritedCriteria}},
    {q: {...question, type: 'score', criteria: 'not an array or object'}},
    {q: {...question, type: 'score', criteria: sparseArray}},
    {q: {...question, type: 'score', criteria: extraArray}},
    {q: {...question, type: 'noul', criteria: []}},
    {q: {...question, instructions: () => 'not JSON'}}
  ])
    assert.throws(() => prepareCrivoSystemOne('', questions, tokenizer), {code: 'INVALID_INPUT'});

  const extraState = ['valid'];
  extraState.extra = 'not JSON';
  const genericCases = [
    {state: 1, questions: [{instructions: 'Pick', options: ['a', 'b']}]},
    {state: '', questions: []},
    {state: '', questions: [{instructions: null, options: ['a', 'b']}]},
    {state: '', questions: [{instructions: 'Pick', options: 'not an array'}]},
    {state: extraState, questions: [{instructions: 'Pick', options: ['a', 'b']}]}
  ];
  for (const request of genericCases)
    assert.throws(() => prepareCrivoDecision(request, tokenizer), {code: 'INVALID_INPUT'});
  assert.throws(() => prepareCrivoSystemOne('', {q: question}, {}), TypeError);
  assert.throws(() => prepareCrivoSystemOne(() => 'not JSON', {q: {type: 'noul', instructions: 'Valid?'}}, tokenizer), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => prepareCrivoSystemOne('', {}, tokenizer), {code: 'INVALID_INPUT'});
  assert.throws(() => prepareCrivoDecision({state: '', questions: []}, {}), TypeError);

  assert.equal(validateCrivoCallOptions(null), undefined);
  assert.equal(validateCrivoCallOptions({}), undefined);
  assert.throws(() => validateCrivoCallOptions(accessor), {code: 'INVALID_INPUT'});
  assert.throws(() => validateCrivoCallOptions({signal: {}}), {code: 'INVALID_INPUT'});
  const controller = new AbortController();
  assert.equal(validateCrivoCallOptions({signal: controller.signal}), controller.signal);
});

test('Crivo rejects overlong input without truncating and accepts exact 2,048-token rows', () => {
  const tokenizer = fakeTokenizer();
  const question = {type: 'choice', criteria: {a: null, b: null}};
  const exactState = 'x'.repeat(CRIVO_MAX_SEQUENCE_TOKENS - 7);
  const exact = prepareCrivoSystemOne(exactState, {one: question}, tokenizer);
  assert.equal(exact.rows[0].inputIds.length, CRIVO_MAX_SEQUENCE_TOKENS);
  assert.throws(() => prepareCrivoSystemOne(exactState + 'x', {one: question}, tokenizer), {code: 'INVALID_INPUT'});
});

test('Crivo accepts over-200K text when its encoded row fits the native token limit', async () => {
  const longText = 'x'.repeat(200_001);
  const encoded = [];
  const tokenizer = {
    special,
    encode(text) {
      encoded.push(text);
      return text.length > 200_000 ? [17] : Array.from(text, (character) => character.codePointAt(0) + 10);
    }
  };
  let runnerCalls = 0;
  const model = createCrivoSession({
    model: CRIVO_FP32_SPEC.model,
    tokenizer,
    runner: {
      async run(rows) {
        runnerCalls++;
        assert.ok(rows[0].inputIds.length <= CRIVO_MAX_SEQUENCE_TOKENS);
        return [Float32Array.of(0, 1)];
      },
      async dispose() {}
    }
  });

  const result = await model.systemOne(longText, {
    one: {type: 'choice', instructions: longText, criteria: {a: longText, b: null}}
  });
  assert.ok(encoded.filter((text) => text.length > 200_000).length >= 3);
  assert.ok(result.tokenCount <= CRIVO_MAX_SEQUENCE_TOKENS);
  assert.equal(runnerCalls, 1);
  await model.dispose();
});

test('Crivo rejects a row beyond 2,048 tokens before inference', async () => {
  let runnerCalls = 0;
  const model = createCrivoSession({
    model: CRIVO_FP32_SPEC.model,
    tokenizer: fakeTokenizer(),
    runner: {
      async run() {
        runnerCalls++;
        return [];
      },
      async dispose() {}
    }
  });
  const overBudgetState = 'x'.repeat(CRIVO_MAX_SEQUENCE_TOKENS - 6);

  await assert.rejects(
    model.systemOne(overBudgetState, {one: {type: 'choice', criteria: {a: null, b: null}}}),
    (error) => error.code === 'INVALID_INPUT' && error.message.includes('2048')
  );
  assert.equal(runnerCalls, 0);
  await model.dispose();
});

test('Crivo accepts options up to the row-token limit without a fixed option-count cap', () => {
  const maxOptions = Math.floor((CRIVO_MAX_SEQUENCE_TOKENS - 4) / 2);
  const tokenizer = {
    special,
    encode(text) {
      return text ? [17] : [];
    }
  };
  const labels = Array.from({length: maxOptions}, (_, index) => `option-${index}`);
  const criteria = Object.fromEntries(labels.map((label) => [label, null]));
  const typed = prepareCrivoSystemOne('', {one: {type: 'choice', instructions: 'pick', criteria}}, tokenizer);
  const generic = prepareCrivoDecision({state: '', questions: [{instructions: 'pick', options: labels}]}, tokenizer);

  assert.equal(maxOptions, 1022);
  assert.equal(typed.rows[0].optionPositions.length, maxOptions);
  assert.equal(typed.rows[0].inputIds.length, CRIVO_MAX_SEQUENCE_TOKENS);
  assert.equal(generic.rows[0].optionPositions.length, maxOptions);
  assert.equal(generic.rows[0].inputIds.length, CRIVO_MAX_SEQUENCE_TOKENS);

  const tooManyLabels = [...labels, 'overflow'];
  const tooManyCriteria = Object.fromEntries(tooManyLabels.map((label) => [label, null]));
  assert.throws(
    () =>
      prepareCrivoSystemOne('', {one: {type: 'choice', instructions: 'pick', criteria: tooManyCriteria}}, tokenizer),
    (error) => error.code === 'INVALID_INPUT' && error.message.includes('2048')
  );
  assert.throws(
    () => prepareCrivoDecision({state: '', questions: [{instructions: 'pick', options: tooManyLabels}]}, tokenizer),
    (error) => error.code === 'INVALID_INPUT' && error.message.includes('2048')
  );
});

test('generic Crivo decisions keep their supplied option order', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareCrivoDecision(
    {state: 'ticket', questions: [{instructions: 'intent', options: ['later', 'first', 'middle']}]},
    tokenizer
  );
  assert.deepEqual(prepared.rows[0].labels, ['later', 'first', 'middle']);
  const result = finishCrivoDecision(prepared, [Float32Array.of(0, 3, 1)]);
  assert.equal(result.answers[0].selected, 'first');
});

test('Crivo rejects accessor option values without reading large option arrays', () => {
  const tokenizer = fakeTokenizer();
  const options = new Array(300).fill('unused');
  let reads = 0;
  Object.defineProperty(options, '0', {
    get() {
      reads++;
      throw Error('option getter invoked');
    }
  });
  assert.throws(
    () => prepareCrivoDecision({state: 'ticket', questions: [{instructions: 'intent', options}]}, tokenizer),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(reads, 0);
  assert.equal(tokenizer.encoded.length, 0);
});

test('Crivo snapshots a Proxy data descriptor once without invoking a later accessor', () => {
  const tokenizer = fakeTokenizer();
  let descriptorReads = 0,
    getterReads = 0;
  const options = new Proxy(
    Object.defineProperties([], {
      0: {
        configurable: true,
        enumerable: true,
        get() {
          getterReads++;
          throw Error('option getter invoked');
        }
      },
      1: {configurable: true, enumerable: true, writable: true, value: 'B'}
    }),
    {
      getOwnPropertyDescriptor(target, key) {
        if (key === '0' && descriptorReads++ === 0)
          return {configurable: true, enumerable: true, writable: true, value: 'A'};
        return Reflect.getOwnPropertyDescriptor(target, key);
      }
    }
  );

  const prepared = prepareCrivoDecision({state: 'ticket', questions: [{instructions: 'intent', options}]}, tokenizer);
  assert.deepEqual(prepared.rows[0].labels, ['A', 'B']);
  assert.equal(descriptorReads, 1);
  assert.equal(getterReads, 0);
  assert.ok(tokenizer.encoded.includes('A'));
});

test('Crivo generic decisions preserve inherited data fields without reading inherited getters', () => {
  const tokenizer = fakeTokenizer();
  const question = Object.create({instructions: 'intent', options: ['A', 'B']});
  const request = Object.create({state: 'ticket', questions: [question]});
  const prepared = prepareCrivoDecision(request, tokenizer);
  assert.deepEqual(prepared.rows[0].labels, ['A', 'B']);

  let getterReads = 0;
  const accessorQuestion = Object.create({options: ['A', 'B']});
  Object.defineProperty(Object.getPrototypeOf(accessorQuestion), 'instructions', {
    configurable: true,
    get() {
      getterReads++;
      throw Error('instruction getter invoked');
    }
  });
  assert.throws(
    () => prepareCrivoDecision(Object.create({state: 'ticket', questions: [accessorQuestion]}), tokenizer),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(getterReads, 0);
});

test('Crivo handles more than 256 questions and chunks requests above the aggregate batch budget', async () => {
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, {type, data, dims});
    }

    dispose() {}
  }

  const calls = [];
  const runner = createCrivoRunner(
    {Tensor},
    {
      async run(feeds) {
        calls.push(feeds);
        const [batchRows, optionCount] = feeds.opt_mask.dims;
        return {
          logits: new Tensor('float32', new Float32Array(batchRows * optionCount), [batchRows, optionCount])
        };
      },
      async release() {}
    }
  );
  const model = createCrivoSession({model: CRIVO_FP32_SPEC.model, tokenizer: fakeTokenizer(), runner});
  const createTypedQuestions = (count) =>
    Object.fromEntries(
      Array.from({length: count}, (_, index) => [
        `question-${index}`,
        {type: 'choice', instructions: 'pick one', criteria: {a: null, b: null}}
      ])
    );
  const createGenericQuestions = (count) =>
    Array.from({length: count}, () => ({instructions: 'pick one', options: ['a', 'b']}));
  const assertBatches = (start, expectedRows) => {
    const batches = calls.slice(start);
    assert.equal(
      batches.reduce((total, feeds) => total + feeds.input_ids.dims[0], 0),
      expectedRows
    );
    assert.equal(batches.length, Math.ceil(expectedRows / CRIVO_BATCH_SIZE));
    for (const feeds of batches) {
      const [batchRows, paddedLength] = feeds.input_ids.dims;
      assert.ok(batchRows > 0 && batchRows <= CRIVO_BATCH_SIZE);
      assert.ok(paddedLength <= CRIVO_MAX_SEQUENCE_TOKENS);
      assert.ok(batchRows * paddedLength <= CRIVO_BATCH_TOKEN_BUDGET);
    }
    return batches;
  };

  for (const count of [256, 257]) {
    const start = calls.length;
    const result = await model.systemOne('', createTypedQuestions(count));
    assert.equal(Object.keys(result.answers).length, count);
    assertBatches(start, count);
  }

  const genericStart = calls.length;
  const generic = await model.decide({state: '', questions: createGenericQuestions(257)});
  assert.equal(generic.answers.length, 257);
  assertBatches(genericStart, 257);

  const largeStart = calls.length;
  const large = await model.systemOne('', createTypedQuestions(5000));
  assert.equal(Object.keys(large.answers).length, 5000);
  assert.ok(large.tokenCount > CRIVO_BATCH_TOKEN_BUDGET);
  assertBatches(largeStart, 5000);
  await model.dispose();
});

test('Crivo batches questions by the upstream row and token budgets', async () => {
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
      this.disposed = false;
    }
    dispose() {
      this.disposed = true;
    }
  }
  const calls = [];
  const session = {
    async run(feeds) {
      calls.push(feeds);
      const [batch, options] = feeds.opt_mask.dims;
      return {logits: new Tensor('float32', new Float32Array(batch * options), [batch, options])};
    },
    async release() {}
  };
  const runner = createCrivoRunner({Tensor}, session);
  const rows = Array.from({length: CRIVO_BATCH_SIZE + 1}, () => ({
    inputIds: Array.from({length: CRIVO_MAX_SEQUENCE_TOKENS}, () => 1),
    optionPositions: [1, 2],
    qtype: 1
  }));
  const output = await runner.run(rows);
  assert.equal(output.length, CRIVO_BATCH_SIZE + 1);
  assert.deepEqual(
    calls.map((feeds) => feeds.input_ids.dims[0]),
    [CRIVO_BATCH_SIZE, 1]
  );
  assert.deepEqual(calls[0].input_ids.dims, [CRIVO_BATCH_SIZE, CRIVO_MAX_SEQUENCE_TOKENS]);
  assert.equal(calls[0].input_ids.dims[0] * calls[0].input_ids.dims[1], CRIVO_BATCH_TOKEN_BUDGET);
  assert.deepEqual(calls[0].opt_mask.dims, [CRIVO_BATCH_SIZE, 2]);
  assert.equal(calls[0].opt_mask.type, 'bool');
  await runner.dispose();
});

test('Crivo loader requires an explicit model route and a directory URL', () => {
  assert.throws(() => validateCrivoLoadOptions({assetsUrl: 'https://example.test/crivo/'}), {code: 'INVALID_INPUT'});
  assert.throws(() => validateCrivoLoadOptions({model: CRIVO_FP32_SPEC.model, assetsUrl: 'file:///tmp/crivo/'}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(
    () =>
      validateCrivoLoadOptions({
        model: CRIVO_INT8_SPEC.model,
        assetsUrl: 'https://example.test/crivo/',
        offline: true
      }),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(
    validateCrivoLoadOptions({model: CRIVO_INT8_SPEC.model, assetsUrl: 'https://example.test/crivo'}).base.pathname,
    '/crivo/'
  );
});

test('Crivo loader selects WebGPU or single-thread WASM and loads only the pinned graph', async () => {
  const config = {
    model_type: 'custom',
    vocab_size: 50369,
    max_position_embeddings: 7999,
    cls_token_id: 50281,
    sep_token_id: 50282,
    pad_token_id: 50283,
    decision: {opt_token: '[OPT]', opt_token_id: 50368, question_types: {noul: 0, choice: 1, score: 2}}
  };
  const tokenizerConfig = {
    backend: 'tokenizers',
    cls_token: '[CLS]',
    sep_token: '[SEP]',
    pad_token: '[PAD]',
    max_length: 7999,
    model_max_length: 7999
  };
  const tokenizerJson = {model: {type: 'BPE'}};
  const jsonBytes = (value) => new TextEncoder().encode(JSON.stringify(value)).buffer;

  for (const spec of [CRIVO_FP32_SPEC, CRIVO_INT8_SPEC]) {
    const graphBytes = new Uint8Array([1, 2, 3]).buffer;
    const reads = [];
    const calls = [];
    let releaseCount = 0;
    const adapter = {info: {isFallbackAdapter: false}};
    const ort = {
      Tensor: class Tensor {},
      env: spec.backend === 'onnxruntime-web-webgpu' ? {webgpu: {}} : {wasm: {}},
      InferenceSession: {
        async create(bytes, options) {
          calls.push({bytes, options});
          return {
            inputNames: ['input_ids', 'attention_mask', 'opt_positions', 'opt_mask', 'qtype'],
            outputNames: ['logits'],
            async run() {},
            async release() {
              releaseCount++;
            }
          };
        }
      }
    };
    const loaded = await loadCrivoModel(
      {model: spec.model, assetsUrl: 'https://example.test/crivo/'},
      {
        ort,
        gpuApi: {
          async requestAdapter(options) {
            assert.equal(options.forceFallbackAdapter, false);
            return adapter;
          }
        },
        openSource: async () => ({
          async read(name) {
            reads.push(name);
            if (name === 'config.json') return jsonBytes(config);
            if (name === 'tokenizer_config.json') return jsonBytes(tokenizerConfig);
            if (name === 'tokenizer.json') return jsonBytes(tokenizerJson);
            assert.equal(name, spec.graphFile);
            return graphBytes;
          }
        }),
        createTokenizer: () => fakeTokenizer()
      }
    );
    assert.deepEqual(reads.sort(), ['config.json', 'tokenizer.json', 'tokenizer_config.json', spec.graphFile].sort());
    assert.deepEqual(calls[0].options.executionProviders, [
      spec.backend === 'onnxruntime-web-webgpu' ? 'webgpu' : 'wasm'
    ]);
    assert.equal(calls[0].options.intraOpNumThreads, spec.backend === 'onnxruntime-web-wasm' ? 1 : undefined);
    if (spec.backend === 'onnxruntime-web-webgpu') assert.equal(ort.env.webgpu.adapter, adapter);
    else assert.equal(ort.env.wasm.numThreads, 1);
    await loaded.dispose();
    assert.equal(releaseCount, 1);
  }
});

test('Crivo session queues calls and closes after disposal', async () => {
  let releaseCount = 0;
  const tokenizer = fakeTokenizer();
  const session = createCrivoSession({
    model: CRIVO_INT8_SPEC.model,
    tokenizer,
    runner: {
      async run(rows) {
        return rows.map((row) => (row.type === 'noul' ? Float32Array.of(0, 2) : Float32Array.of(2, 0)));
      },
      async dispose() {
        releaseCount++;
      }
    }
  });
  const result = await session.systemOne('state', {
    claim: {type: 'noul', instructions: 'true?'},
    intent: {type: 'choice', instructions: 'choose', criteria: {first: null, second: null}}
  });
  assert.ok(result.answers.claim.noul > 0.8);
  assert.equal(result.answers.intent.choice, 'first');
  await session.dispose();
  assert.equal(releaseCount, 1);
  await assert.rejects(() => session.systemOne('state', {x: {type: 'noul'}}), {code: 'SESSION_CLOSED'});
});
