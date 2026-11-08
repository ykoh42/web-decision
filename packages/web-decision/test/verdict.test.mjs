import assert from 'node:assert/strict';
import test from 'node:test';
import {
  VERDICT_FILES,
  VERDICT_GLOBAL_TEMPERATURE,
  VERDICT_MAX_CANDIDATES,
  VERDICT_MAX_LABELS,
  VERDICT_MAX_TOKENS,
  VERDICT_MODEL,
  VERDICT_TEMPERATURES,
  VERDICT_TOTAL_BYTES
} from '../src/verdict/contract.mjs';
import {loadVerdictModel, validateVerdictLoadOptions} from '../src/verdict/loader.mjs';
import {createVerdictTokenizer} from '../src/verdict/tokenizer.mjs';
import {finishVerdictSystemOne, prepareVerdictSystemOne} from '../src/verdict/protocol.mjs';
import {createVerdictRunner, VERDICT_INPUT_NAMES} from '../src/verdict/runner.mjs';
import {createVerdictSession} from '../src/verdict/session.mjs';

function fakeTokenizer() {
  const encoded = [];
  return {
    encoded,
    special: {cls: 50281, sep: 50282, pad: 50283, label: 50368, query: 50369},
    encode(text) {
      encoded.push(text);
      const ids = [50281, ...Array.from(text, (character) => 100 + character.codePointAt(0)), 50282];
      return ids.length <= VERDICT_MAX_TOKENS
        ? {ids, truncated: false}
        : {ids: [ids[0], ...ids.slice(1, VERDICT_MAX_TOKENS - 1), ids.at(-1)], truncated: true};
    }
  };
}

function validQuestions() {
  return {
    route: {
      type: 'choice',
      instructions: 'Where should this request go?',
      criteria: {billing: 'billing questions', technical: 'technical support'}
    },
    severity: {
      type: 'score',
      instructions: 'How urgent is the request?',
      criteria: [
        {id: 'low', description: 'low urgency', value: 0},
        {id: 'high', description: 'high urgency', value: 10}
      ]
    },
    supported: {
      type: 'noul',
      proposition: 'The evidence supports an automated response.',
      semantics: 'conditional_on_sufficient_evidence_v2'
    }
  };
}

test('Verdict pins the six-file 151M FP16 bundle and its input limits', () => {
  assert.equal(VERDICT_MODEL, 'experimental-openjev-verdict-151m-fp16');
  assert.equal(VERDICT_MAX_CANDIDATES, 24);
  assert.equal(VERDICT_MAX_LABELS, 25);
  assert.equal(VERDICT_MAX_TOKENS, 512);
  assert.equal(VERDICT_GLOBAL_TEMPERATURE, 2.8039);
  assert.deepEqual(Object.keys(VERDICT_FILES), [
    'bundle_manifest.json',
    'calibrator.json',
    'config.json',
    'model_fp16.onnx',
    'tokenizer.json',
    'tokenizer_config.json'
  ]);
  assert.equal(VERDICT_TOTAL_BYTES, 307_374_582);
  assert.equal(VERDICT_FILES['model_fp16.onnx'].bytes, 303_785_047);
});

test('Verdict tokenizer verifies special IDs and right-truncates while retaining boundary tokens', () => {
  class Tokenizer {
    token_to_id(token) {
      return {'[CLS]': 50281, '[SEP]': 50282, '[PAD]': 50283, '<<LABEL>>': 50368, '<<SEP>>': 50369}[token];
    }
    encode(text) {
      return {ids: [50281, ...Array.from({length: text.length}, (_, index) => index + 1), 50282]};
    }
  }
  const data = {model: {type: 'BPE', vocab: {example: 1}}};
  const config = {
    cls_token: '[CLS]',
    sep_token: '[SEP]',
    pad_token: '[PAD]',
    model_input_names: ['input_ids', 'attention_mask']
  };
  const tokenizer = createVerdictTokenizer(Tokenizer, data, config);
  const encoded = tokenizer.encode('x'.repeat(600));
  assert.equal(encoded.ids.length, 512);
  assert.equal(encoded.ids[0], 50281);
  assert.equal(encoded.ids.at(-1), 50282);
  assert.equal(encoded.truncated, true);
  assert.deepEqual(tokenizer.special, {cls: 50281, sep: 50282, pad: 50283, label: 50368, query: 50369});
  assert.throws(() => createVerdictTokenizer(Tokenizer, data, {...config, pad_token: '[WRONG]'}), {
    code: 'INVALID_ASSET'
  });
});

test('Verdict formats Choice, Score, and abstaining Noul as the pinned model expects', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareVerdictSystemOne(
    'A customer was charged twice and requests a refund.',
    validQuestions(),
    tokenizer
  );
  assert.equal(prepared.rows.length, 3);
  assert.equal(prepared.rows[0].candidateIds.join(','), 'billing,technical,__insufficient_evidence__');
  assert.equal(
    prepared.rows[0].labels.join(','),
    'It is billing questions,It is technical support,insufficient evidence'
  );
  assert.equal(
    prepared.rows[1].labels.join(','),
    'low urgency (Value: 0),high urgency (Value: 10),insufficient evidence'
  );
  assert.equal(
    prepared.rows[2].labels.join(','),
    'true: The evidence supports an automated response.,false: not The evidence supports an automated response.,insufficient evidence'
  );
  assert.equal(
    tokenizer.encoded[0],
    '<<LABEL>>It is billing questions<<LABEL>>It is technical support<<LABEL>>insufficient evidence<<SEP>>Question: Where should this request go?\n\nContext:\nA customer was charged twice and requests a refund.'
  );
  assert.equal(
    tokenizer.encoded[2],
    '<<LABEL>>true: The evidence supports an automated response.<<LABEL>>false: not The evidence supports an automated response.<<LABEL>>insufficient evidence<<SEP>>Context:\nA customer was charged twice and requests a refund.\n\nEvaluate proposition: The evidence supports an automated response.'
  );
});

test('Verdict decodes calibrated probabilities, conditional scores, and explicit abstention', () => {
  const prepared = prepareVerdictSystemOne('context', validQuestions(), fakeTokenizer());
  const result = finishVerdictSystemOne(
    prepared,
    [
      [5, 0, 1],
      [0, 2, 1],
      [1, 0, 4]
    ],
    {temperature: 2.8039, per_k: {3: 1}}
  );
  assert.equal(result.model, VERDICT_MODEL);
  assert.equal(result.outputTokenCount, 0);
  assert.equal(result.forwardPassCount, 1);
  assert.equal(result.answers.route.type, 'choice');
  assert.equal(result.answers.route.choice, 'billing');
  assert.equal(result.answers.route.isAbstention, false);
  assert.equal(result.answers.route.calibrationSource, 'per-cardinality');
  assert.ok(
    Math.abs(Object.values(result.answers.route.probabilities).reduce((sum, value) => sum + value, 0) - 1) < 1e-12
  );
  assert.equal(result.answers.severity.selectedLevelId, 'high');
  assert.equal(result.answers.severity.selectedValue, 10);
  assert.ok(Math.abs(result.answers.severity.expectedScore - 8.807970779778824) < 1e-12);
  assert.equal(result.answers.supported.selectedOutcome, '__insufficient_evidence__');
  assert.equal(result.answers.supported.pTrueGivenSufficientEvidence, null);
  assert.ok(result.answers.supported.pInsufficientEvidence > 0.9);
});

test('Verdict uses per-cardinality calibration and the pinned global fallback', () => {
  const questions = {
    two: {
      type: 'choice',
      instructions: 'Pick one.',
      criteria: {first: 'first', second: 'second'}
    },
    three: {
      type: 'choice',
      instructions: 'Pick one.',
      criteria: {first: 'first', second: 'second', third: 'third'}
    }
  };
  const prepared = prepareVerdictSystemOne('context', questions, fakeTokenizer());
  const result = finishVerdictSystemOne(
    prepared,
    [
      [1, 0, -1],
      [1, 0, -1, -2]
    ],
    {
      temperature: 2.8039,
      per_k: {3: 5.0069}
    }
  );
  assert.equal(result.answers.two.calibrationSource, 'per-cardinality');
  assert.equal(result.answers.two.calibrationTemperature, 5.0069);
  assert.equal(result.answers.three.calibrationSource, 'global');
  assert.equal(result.answers.three.calibrationTemperature, 2.8039);
});

test('Verdict accepts the model capacity and rejects unsupported cardinality or invalid score semantics', () => {
  const tokenizer = fakeTokenizer();
  const atLimit = Object.fromEntries(
    Array.from({length: VERDICT_MAX_CANDIDATES}, (_, index) => [`option-${index}`, `description ${index}`])
  );
  const prepared = prepareVerdictSystemOne(
    'context',
    {choice: {type: 'choice', instructions: 'Pick one.', criteria: atLimit}},
    tokenizer
  );
  assert.equal(prepared.rows[0].candidateIds.length, VERDICT_MAX_LABELS);
  assert.throws(
    () =>
      prepareVerdictSystemOne(
        'context',
        {choice: {type: 'choice', instructions: 'Pick one.', criteria: {...atLimit, overflow: 'extra'}}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareVerdictSystemOne(
        'context',
        {
          score: {
            type: 'score',
            instructions: 'Rate it.',
            criteria: [
              {id: 'low', description: 'low', value: 1},
              {id: 'high', description: 'high', value: 1}
            ]
          }
        },
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareVerdictSystemOne(
        'context',
        {noul: {type: 'noul', proposition: 'It is true.', semantics: 'conditional_on_sufficient_evidence_v1'}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareVerdictSystemOne(
        'context',
        {choice: {type: 'choice', instructions: 'Pick one.', criteria: {__insufficient_evidence__: 'reserved'}}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
});

test('Verdict accepts large question sets and validates string context and question records', () => {
  const tokenizer = fakeTokenizer();
  const manyQuestions = Object.fromEntries(
    Array.from({length: 257}, (_, index) => [
      `q${index}`,
      {type: 'choice', instructions: 'Pick one.', criteria: {a: 'A', b: 'B'}}
    ])
  );
  assert.equal(prepareVerdictSystemOne('context', manyQuestions, tokenizer).rows.length, 257);
  assert.throws(() => prepareVerdictSystemOne({context: 'object'}, validQuestions(), tokenizer), {
    code: 'INVALID_INPUT'
  });
  const accessor = {};
  Object.defineProperty(accessor, 'q', {
    get() {
      return validQuestions().route;
    }
  });
  assert.throws(() => prepareVerdictSystemOne('context', accessor, tokenizer), {code: 'INVALID_INPUT'});
  assert.throws(() => prepareVerdictSystemOne('context', {}, tokenizer), {code: 'INVALID_INPUT'});
});

test('Verdict generic decisions accept 257 questions without a request-wide count cap', async () => {
  const runner = {
    async run(rows) {
      return rows.map(() => [2, 1, 0]);
    },
    async dispose() {}
  };
  const model = createVerdictSession({runner, tokenizer: fakeTokenizer(), calibrator: {temperature: 1}});
  const questions = Array.from({length: 257}, (_, index) => ({
    instructions: `Route request ${index}.`,
    options: ['billing', 'support']
  }));
  const result = await model.decide({state: 'context', questions});
  assert.equal(result.answers.length, 257);
  assert.equal(result.answers.at(-1).selected, 'billing');
  await model.dispose();
});

test('Verdict runner pads dynamic rows and extracts each row only through its active candidate count', async () => {
  const calls = [];
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
    dispose() {
      this.disposed = true;
    }
  }
  const session = {
    inputNames: [...VERDICT_INPUT_NAMES],
    outputNames: ['logits'],
    async run(feeds, outputNames) {
      calls.push({feeds, outputNames});
      const batch = feeds.input_ids.dims[0];
      const data = new Float32Array(batch * 25);
      for (let row = 0; row < batch; row++) data[row * 25] = row + 1;
      return {
        logits: {
          type: 'float32',
          dims: [batch, 25],
          data,
          dispose() {
            this.disposed = true;
          }
        }
      };
    },
    async release() {
      this.released = true;
    }
  };
  const runner = createVerdictRunner({Tensor}, session);
  const rows = await runner.run([
    {ids: [101, 102, 103], candidateIds: ['a', 'b', 'abstain']},
    {ids: [201, 202], candidateIds: ['c', 'd', 'abstain']}
  ]);
  assert.deepEqual(rows, [
    [1, 0, 0],
    [2, 0, 0]
  ]);
  assert.deepEqual(Array.from(calls[0].feeds.input_ids.data), [101n, 102n, 103n, 201n, 202n, 50283n]);
  assert.deepEqual(Array.from(calls[0].feeds.attention_mask.data), [1n, 1n, 1n, 1n, 1n, 0n]);
  assert.deepEqual(calls[0].outputNames, ['logits']);
  assert.equal(calls[0].feeds.input_ids.disposed, true);
  await runner.dispose();
  assert.equal(session.released, true);
});

test('Verdict runner chunks 257 rows under the per-dispatch limits', async () => {
  const batches = [];
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
    dispose() {}
  }
  const session = {
    inputNames: [...VERDICT_INPUT_NAMES],
    outputNames: ['logits'],
    async run(feeds) {
      const [batch, sequence] = feeds.input_ids.dims;
      batches.push([batch, sequence]);
      const data = new Float32Array(batch * 25);
      for (let index = 0; index < batch; index++) data[index * 25] = 1;
      return {logits: {type: 'float32', dims: [batch, 25], data, dispose() {}}};
    },
    async release() {}
  };
  const runner = createVerdictRunner({Tensor}, session),
    rows = Array.from({length: 257}, (_, index) => ({
      ids: [50281, ...Array(510).fill((index % 100) + 100), 50282],
      candidateIds: ['choice', 'other', '__insufficient_evidence__']
    }));

  const result = await runner.run(rows);
  assert.equal(result.length, 257);
  assert.equal(batches.length, 33);
  assert.deepEqual(batches.slice(0, 32), Array(32).fill([8, 512]));
  assert.deepEqual(batches.at(-1), [1, 512]);
  assert.ok(batches.every(([batch, sequence]) => batch * sequence <= 4096));
  await runner.dispose();
});

test('Verdict session preserves explicit abstention through the generic decision method', async () => {
  const runner = {
    async run(rows) {
      return rows.map(() => [0, 0, 10]);
    },
    async dispose() {}
  };
  const model = createVerdictSession({runner, tokenizer: fakeTokenizer(), calibrator: {temperature: 2.8039}});
  const result = await model.decide({
    state: 'The context does not identify a matching route.',
    questions: [{instructions: 'Where should this go?', options: ['billing', 'technical']}]
  });
  assert.equal(result.answers[0].selected, '__insufficient_evidence__');
  assert.equal(result.answers[0].selectedIndex, -1);
  assert.equal(result.answers[0].abstained, true);
  assert.ok(result.answers[0].abstentionProbability > 0.9);
  assert.deepEqual(
    result.answers[0].probabilities.map(({option}) => option),
    ['billing', 'technical', '__insufficient_evidence__']
  );
  await model.dispose();
  assert.throws(() => model.systemOne('context', validQuestions()), {code: 'SESSION_CLOSED'});
});

test('Verdict load options reject unknown fields, malformed URLs, and online prefetch', () => {
  const valid = {model: VERDICT_MODEL, assetsUrl: 'https://example.test/verdict/'};
  assert.equal(validateVerdictLoadOptions(valid).base.href, valid.assetsUrl);
  assert.throws(() => validateVerdictLoadOptions({...valid, maxPrefetchBytes: 1}), {code: 'INVALID_INPUT'});
  assert.throws(() => validateVerdictLoadOptions({...valid, offline: true, cache: false}), {code: 'INVALID_INPUT'});
  assert.throws(() => validateVerdictLoadOptions({...valid, unknown: true}), {code: 'INVALID_INPUT'});
  assert.throws(() => validateVerdictLoadOptions({...valid, assetsUrl: 'file:///tmp/verdict/'}), {
    code: 'INVALID_INPUT'
  });
});

test('Verdict loader pins its assets, strict WebGPU provider, and non-fallback adapter', async () => {
  const bundle = {
    bundle_version: '1.0.0',
    model_architecture: 'knowledgator/gliclass-modern-base-v2.0',
    opset_version: 17,
    max_capacity_logits: 25
  };
  const config = {
    model_type: 'GLiClass',
    max_num_classes: 25,
    class_token_index: 50368,
    text_token_index: 50369,
    vocab_size: 50370,
    encoder_config: {model_type: 'modernbert', max_position_embeddings: 8192}
  };
  const calibrator = {
    model_id: 'openjev-modernbert-151.4m',
    format_version: 'rlcd-calibrator-v1',
    temperature: 2.8039,
    per_k: VERDICT_TEMPERATURES
  };
  const tokenizerConfig = {
    cls_token: '[CLS]',
    sep_token: '[SEP]',
    pad_token: '[PAD]',
    model_input_names: ['input_ids', 'attention_mask']
  };
  const tokenizerData = {model: {type: 'BPE', vocab: {token: 1}}};
  const jsonBytes = (value) => new TextEncoder().encode(JSON.stringify(value));
  const assets = {
    'bundle_manifest.json': jsonBytes(bundle),
    'calibrator.json': jsonBytes(calibrator),
    'config.json': jsonBytes(config),
    'model_fp16.onnx': new Uint8Array([1, 2, 3]),
    'tokenizer.json': jsonBytes(tokenizerData),
    'tokenizer_config.json': jsonBytes(tokenizerConfig)
  };
  const reads = [];
  class Tokenizer {
    token_to_id(token) {
      return {'[CLS]': 50281, '[SEP]': 50282, '[PAD]': 50283, '<<LABEL>>': 50368, '<<SEP>>': 50369}[token];
    }
    encode() {
      return {ids: [50281, 123, 50282]};
    }
  }
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
    dispose() {}
  }
  const session = {
    inputNames: ['input_ids', 'attention_mask'],
    outputNames: ['logits'],
    async run(feeds) {
      const batch = feeds.input_ids.dims[0];
      const data = new Float32Array(batch * 25);
      for (let row = 0; row < batch; row++) data[row * 25] = 2;
      return {logits: {type: 'float32', dims: [batch, 25], data, dispose() {}}};
    },
    async release() {
      this.released = true;
    }
  };
  const adapter = {info: {isFallbackAdapter: false}};
  const ort = {
    Tensor,
    env: {webgpu: {}},
    InferenceSession: {
      async create(bytes, options) {
        assert.deepEqual(Array.from(bytes), [1, 2, 3]);
        assert.deepEqual(options.executionProviders, ['webgpu']);
        assert.equal(options.graphOptimizationLevel, 'all');
        return session;
      }
    }
  };
  const model = await loadVerdictModel(
    {model: VERDICT_MODEL, assetsUrl: 'https://example.test/verdict/', cache: false},
    {
      ort,
      Tokenizer,
      openSource: async () => ({
        async read(name) {
          reads.push(name);
          return assets[name];
        }
      }),
      gpuApi: {
        async requestAdapter(options) {
          assert.deepEqual(options, {powerPreference: 'high-performance', forceFallbackAdapter: false});
          return adapter;
        }
      }
    }
  );
  assert.deepEqual(reads.sort(), Object.keys(VERDICT_FILES).sort());
  assert.equal(ort.env.webgpu.adapter, adapter);
  assert.equal(ort.env.webgpu.forceFallbackAdapter, false);
  const result = await model.systemOne('context', {
    route: {type: 'choice', instructions: 'Where?', criteria: {billing: 'billing', technical: 'technical'}}
  });
  assert.equal(result.answers.route.choice, 'billing');
  await model.dispose();
  assert.equal(session.released, true);
});
