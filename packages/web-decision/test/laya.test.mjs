import assert from 'node:assert/strict';
import test from 'node:test';
import {
  LAYA_FILES,
  LAYA_MAX_HEAD_TOKENS,
  LAYA_MAX_OPTIONS,
  LAYA_MAX_SEQUENCE_TOKENS,
  LAYA_MODEL,
  LAYA_SPEC
} from '../src/laya/contract.mjs';
import {loadLayaModel, validateLayaLoadOptions} from '../src/laya/loader.mjs';
import {createLayaSession} from '../src/laya/session.mjs';
import {loadLayaTypedDecisionsModel} from '../src/laya-typed/loader.mjs';
import {LAYA_TYPED_FILES, LAYA_TYPED_MODEL, LAYA_TYPED_SPEC} from '../src/laya-typed/contract.mjs';
import {loadXDecisionModel} from '../src/xdecision/loader.mjs';
import {finishLayaSystemOne, prepareLayaSystemOne, validateLayaDecisionRequest} from '../src/laya/protocol.mjs';
import {createLayaRunner} from '../src/laya/runner.mjs';
import {createLayaTokenizer} from '../src/laya/tokenizer.mjs';

class CharacterTokenizer {
  constructor() {}

  token_to_id(token) {
    return {'<bos>': 2, '<eos>': 1, '<mask>': 4, '<pad>': 0}[token];
  }

  encode(text) {
    return {
      ids: Array.from(text)
        .filter((character) => character !== ' ')
        .map((character) => character.codePointAt(0) + 10)
    };
  }
}

class TypedCharacterTokenizer extends CharacterTokenizer {
  token_to_id(token) {
    return {'[CLS]': 50281, '[SEP]': 50282, '[PAD]': 50283, '[MASK]': 50284}[token];
  }
}

function tokenizer() {
  return createLayaTokenizer(
    CharacterTokenizer,
    {model: {vocab: {'<bos>': 2}}},
    {cls_token: '<bos>', sep_token: '<eos>', mask_token: '<mask>', pad_token: '<pad>'},
    {sep_token_id: 1, mask_token_id: 4, pad_token_id: 0}
  );
}

function bytes(value) {
  return new TextEncoder().encode(JSON.stringify(value)).buffer;
}

function config() {
  return {
    model_type: 'modernbert',
    vocab_size: 256000,
    mask_token_id: 4,
    sep_token_id: 1,
    pad_token_id: 0,
    laya: {
      max_len: 1024,
      head_max_len: 256,
      max_option_tokens: 48,
      qtypes: {choice: 0, score: 1, noul: 2},
      temperature: [1, 1, 1],
      temperature_by_options: {},
      temperature_clamp: [0.5, 5]
    }
  };
}

function loadDependencies(overrides = {}) {
  const values = {
    'config.json': config(),
    'tokenizer_config.json': {cls_token: '<bos>', sep_token: '<eos>', mask_token: '<mask>', pad_token: '<pad>'},
    'tokenizer.json': {model: {vocab: {'<bos>': 2}}},
    'onnx/model_fp16.onnx': new Uint8Array([1, 2, 3]).buffer,
    'onnx/model_fp16.onnx_data': new Uint8Array([4, 5, 6]).buffer,
    ...overrides
  };
  const released = {count: 0};
  const calls = [];
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, {type, data, dims});
    }

    dispose() {
      this.disposed = true;
    }
  }
  const session = {
    inputNames: ['input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'],
    outputNames: ['logits', 'act_logits'],
    async run(feeds) {
      calls.push(feeds);
      const [batch, markers] = feeds.marker_pos.dims;
      return {
        logits: new Tensor(
          'float32',
          new Float32Array(batch * markers).map((_, index) => index % markers),
          [batch, markers]
        ),
        act_logits: new Tensor('float32', new Float32Array(batch * 2).fill(0), [batch, 2])
      };
    },
    async release() {
      released.count++;
    }
  };
  const ort = {
    Tensor,
    env: {webgpu: {}},
    InferenceSession: {
      async create(graph, options) {
        assert.deepEqual(Array.from(graph), [1, 2, 3]);
        assert.deepEqual(options.executionProviders, ['webgpu']);
        assert.equal(options.graphOptimizationLevel, 'all');
        assert.equal(options.externalData[0].path, 'model_fp16.onnx_data');
        assert.deepEqual(Array.from(options.externalData[0].data), [4, 5, 6]);
        return session;
      }
    }
  };
  return {
    calls,
    released,
    ort,
    adapter: {features: new Set(), info: {isFallbackAdapter: false}, limits: {}, requestDevice() {}},
    dependencies: {
      ort,
      Tokenizer: CharacterTokenizer,
      async openSource() {
        return {
          async read(name) {
            const value = values[name];
            if (value === undefined) throw new Error(`Unexpected asset ${name}`);
            return value instanceof ArrayBuffer ? value : bytes(value);
          }
        };
      },
      gpuApi: {
        async requestAdapter(options) {
          assert.deepEqual(options, {powerPreference: 'high-performance', forceFallbackAdapter: false});
          return this.adapter;
        },
        adapter: null
      }
    }
  };
}

function genericQuestions(count) {
  return Array.from({length: count}, () => ({instructions: 'pick', options: ['a', 'b']}));
}

function typedQuestions(count) {
  return Object.fromEntries(
    Array.from({length: count}, (_, index) => [
      `question-${index}`,
      {type: 'choice', instructions: 'pick', criteria: ['a', 'b']}
    ])
  );
}

function assertChunkedRows(calls, start, expectedRows) {
  const batches = calls.slice(start);
  assert.equal(
    batches.reduce((sum, feeds) => sum + feeds.input_ids.dims[0], 0),
    expectedRows
  );
  assert.equal(batches.length, Math.ceil(expectedRows / 8));
  for (const feeds of batches) {
    const [batchRows, sequenceLength] = feeds.input_ids.dims;
    assert.ok(batchRows > 0 && batchRows <= 8);
    assert.ok(sequenceLength <= LAYA_MAX_SEQUENCE_TOKENS);
    assert.ok(batchRows * sequenceLength <= LAYA_MAX_SEQUENCE_TOKENS);
    assert.equal(feeds.marker_pos.dims[0], batchRows);
  }
}

test('Laya pins the ONNX graph, external weights, tokenizer, and config', () => {
  assert.equal(LAYA_MODEL, 'experimental-laya-multilingual-fp16');
  assert.deepEqual(Object.keys(LAYA_FILES), [
    'config.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'onnx/model_fp16.onnx',
    'onnx/model_fp16.onnx_data'
  ]);
  assert.equal(LAYA_FILES['onnx/model_fp16.onnx_data'].bytes, 643_817_472);
  assert.ok(Object.values(LAYA_FILES).every(({sha256}) => /^[0-9a-f]{64}$/u.test(sha256)));
});

test('Laya tokenizer follows tokenizer special IDs despite the conflicting model CLS config field', () => {
  const created = tokenizer();
  assert.deepEqual(created.special, {clsId: 2, sepId: 1, maskId: 4, padId: 0, maskToken: '<mask>'});
  assert.deepEqual(created.encode('abc'), [107, 108, 109]);
  assert.throws(
    () =>
      createLayaTokenizer(
        CharacterTokenizer,
        {model: {vocab: {'<bos>': 2}}},
        {cls_token: '<bos>', sep_token: '<eos>', mask_token: '<mask>', pad_token: '<pad>'},
        {sep_token_id: 99, mask_token_id: 4, pad_token_id: 0}
      ),
    {code: 'INVALID_ASSET'}
  );
});

test('Laya protocol preserves named Choice, ordered Score and Noul outputs', () => {
  const prepared = prepareLayaSystemOne(
    'case state',
    {
      route: {type: 'choice', instructions: 'choose', criteria: {billing: 'refunds', support: 'help'}},
      urgency: {type: 'score', instructions: 'rate', criteria: ['low', 'medium', 'high']},
      true_case: {type: 'noul', instructions: 'true?'}
    },
    tokenizer()
  );
  assert.equal(prepared.rows.length, 3);
  assert.deepEqual(
    prepared.rows.map(({qtype}) => qtype),
    [0, 1, 2]
  );
  const result = finishLayaSystemOne(
    prepared,
    {rows: prepared.rows.map(({options}) => ({logits: options.map((_, index) => index), actLogits: [0, 0]}))},
    {byType: [1, 1, 1], byOptions: {}},
    LAYA_MODEL
  );
  assert.equal(result.model, LAYA_MODEL);
  assert.equal(result.answers.route.type, 'choice');
  assert.deepEqual(Object.keys(result.answers.route.probabilities), ['billing', 'support']);
  const expected = result.answers.urgency.probabilities['1'] + 2 * result.answers.urgency.probabilities['2'];
  assert.ok(Math.abs(result.answers.urgency.score - expected) < 0.0002);
  assert.equal(result.answers.urgency.score, result.answers.urgency.expected);
  assert.equal(result.answers.urgency.legend['2'], 'high');
  assert.equal(result.answers.true_case.type, 'noul');
  assert.ok(result.answers.true_case.noul > result.answers.true_case.probabilities.false);
  assert.equal(result.answers.route.actionProbability, 0.5);
  assert.equal(result.answers.route.actionLogits.length, 2);
});

test('Laya reports context and head truncation at the 1024 and 256 token boundaries', () => {
  const encoder = tokenizer();
  const context = prepareLayaSystemOne(
    'x'.repeat(2000),
    {id: {type: 'choice', instructions: 'q', criteria: ['a', 'b']}},
    encoder
  );
  assert.equal(context.rows[0].tokenCount, LAYA_MAX_SEQUENCE_TOKENS);
  assert.equal(context.rows[0].stateTruncated, true);

  const longState = 'x'.repeat(200_001);
  const longText = 'y'.repeat(200_001);
  const longContext = prepareLayaSystemOne(
    longState,
    {id: {type: 'choice', instructions: longText, criteria: {first: longText, second: 'b'}}},
    encoder
  );
  assert.equal(longState.length > 200_000, true);
  assert.equal(longContext.rows[0].tokenCount, LAYA_MAX_SEQUENCE_TOKENS);
  assert.equal(longContext.rows[0].stateTruncated, true);
  assert.equal(longContext.rows[0].instructionTruncated, true);
  assert.equal(longContext.rows[0].optionTruncated, true);

  const options = Array.from({length: LAYA_MAX_OPTIONS}, (_, index) => `o${String(index).padStart(3, '0')}`);
  const head = prepareLayaSystemOne(
    '',
    {id: {type: 'choice', instructions: 'q'.repeat(100), criteria: options}},
    encoder
  );
  assert.ok(head.rows[0].markerPositions.at(-1) < LAYA_MAX_HEAD_TOKENS);
  assert.equal(head.rows[0].tokenCount, LAYA_MAX_HEAD_TOKENS + 1);
  assert.equal(head.rows[0].markerPositions.length, LAYA_MAX_OPTIONS);
  assert.equal(head.rows[0].optionTruncated, true);
  assert.throws(
    () =>
      prepareLayaSystemOne(
        '',
        {
          id: {type: 'choice', instructions: 'q', criteria: [...options, 'overflow']}
        },
        encoder
      ),
    {code: 'INVALID_INPUT'}
  );
});

test('Laya routes reject an option-head overflow before inference', async () => {
  const runnerCalls = [];
  const criteria = Array.from({length: LAYA_MAX_OPTIONS + 1}, (_, index) => `option-${index}`);
  for (const model of [LAYA_MODEL, LAYA_TYPED_MODEL]) {
    const session = createLayaSession({
      model,
      runner: {
        async run(rows) {
          runnerCalls.push(rows);
          return {rows: []};
        },
        async dispose() {}
      },
      tokenizer: tokenizer(),
      temperatures: {byType: [1, 1, 1], byOptions: {}}
    });

    await assert.rejects(session.systemOne('', {pick: {type: 'choice', instructions: 'Choose.', criteria}}), {
      code: 'INVALID_INPUT'
    });
    assert.equal(runnerCalls.length, 0);
    await session.dispose();
  }
});

test('Laya JSON accepts null-prototype objects but rejects cycles and malformed generic requests', () => {
  const state = Object.assign(Object.create(null), {result: {ok: true}});
  assert.doesNotThrow(() => prepareLayaSystemOne(state, {id: {type: 'noul', instructions: 'valid?'}}, tokenizer()));
  state.loop = state;
  assert.throws(() => prepareLayaSystemOne(state, {id: {type: 'noul', instructions: 'valid?'}}, tokenizer()), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateLayaDecisionRequest({state: {}, questions: []}), {code: 'INVALID_INPUT'});
});

test('Laya validates model selection, local assets URL, and bounded load options', () => {
  assert.throws(() => validateLayaLoadOptions({model: LAYA_MODEL, assetsUrl: '/laya/'}), {code: 'INVALID_INPUT'});
  assert.throws(() => validateLayaLoadOptions({model: LAYA_MODEL, assetsUrl: 'file:///laya/'}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(
    () => validateLayaLoadOptions({model: LAYA_MODEL, assetsUrl: 'https://example.test/laya/', maxPrefetchBytes: 1}),
    {code: 'INVALID_INPUT'}
  );
  assert.doesNotThrow(() => validateLayaLoadOptions({model: LAYA_MODEL, assetsUrl: 'https://example.test/laya/'}));
});

test('Laya routes default to distinct pinned Hugging Face revisions and preserve mirror directories', () => {
  for (const [model, spec] of [
    [LAYA_MODEL, LAYA_SPEC],
    [LAYA_TYPED_MODEL, LAYA_TYPED_SPEC]
  ]) {
    const expectedBase = `https://huggingface.co/${spec.repository}/resolve/${spec.revision}/`;
    const defaults = validateLayaLoadOptions({model}, model, spec);
    assert.equal(defaults.base.href, expectedBase);
    assert.equal(defaults.cacheEnabled, true);
    assert.equal(defaults.offline, false);

    const mirror = validateLayaLoadOptions({model, assetsUrl: 'https://mirror.example.test/laya'}, model, spec);
    assert.equal(mirror.base.href, 'https://mirror.example.test/laya/');
    assert.equal(mirror.cacheEnabled, true);
  }
});

test('Laya loader defaults to its pinned source, preserves mirrors, and keeps WebGPU without fallback', async () => {
  for (const assetsUrl of [undefined, 'https://example.test/laya/']) {
    const fixture = loadDependencies();
    const adapter = (fixture.dependencies.gpuApi.adapter = fixture.adapter);
    let sourceBase;
    const openSource = fixture.dependencies.openSource;
    fixture.dependencies.openSource = async (options) => {
      sourceBase = options.base.href;
      return openSource(options);
    };
    const options = {model: LAYA_MODEL, cache: false};
    if (assetsUrl !== undefined) options.assetsUrl = assetsUrl;
    const model = await loadLayaModel(options, fixture.dependencies);
    assert.equal(
      sourceBase,
      assetsUrl ?? `https://huggingface.co/${LAYA_SPEC.repository}/resolve/${LAYA_SPEC.revision}/`
    );
    assert.equal(fixture.ort.env.webgpu.adapter, adapter);
    assert.equal(fixture.ort.env.webgpu.forceFallbackAdapter, false);
    const result = await model.systemOne('state', {
      route: {type: 'choice', instructions: 'choose', criteria: ['a', 'b']}
    });
    assert.equal(result.answers.route.choice, 'b');
    assert.equal(fixture.calls.length, 1);
    assert.ok(Object.values(fixture.calls[0]).every((tensor) => tensor.disposed));
    await model.dispose();
    assert.equal(fixture.released.count, 1);
  }
});

test('Laya and xDecision accept larger requests and keep ONNX chunks within the row and token bounds', async () => {
  const routes = [
    {
      model: LAYA_MODEL,
      load: loadLayaModel,
      assetsUrl: 'https://example.test/laya/'
    },
    {
      model: 'experimental-xdecision-fp16',
      load: loadXDecisionModel,
      assetsUrl: 'https://example.test/xdecision/'
    }
  ];

  for (const route of routes) {
    const fixture = loadDependencies();
    fixture.dependencies.gpuApi.adapter = fixture.adapter;
    const model = await route.load(
      {model: route.model, assetsUrl: route.assetsUrl, cache: false},
      fixture.dependencies
    );

    for (const count of [256, 257, 1025]) {
      const start = fixture.calls.length;
      const result = await model.decide({state: 's', questions: genericQuestions(count)});
      assert.equal(result.answers.length, count);
      assertChunkedRows(fixture.calls, start, count);
    }

    const beforeInvalidBatch = fixture.calls.length;
    const invalidBatch = genericQuestions(257);
    invalidBatch[256] = {instructions: 'pick', options: ['only one']};
    await assert.rejects(model.decide({state: 's', questions: invalidBatch}), {code: 'INVALID_INPUT'});
    assert.equal(fixture.calls.length, beforeInvalidBatch);

    const start = fixture.calls.length;
    const typedResult = await model.systemOne('s', typedQuestions(257));
    assert.equal(Object.keys(typedResult.answers).length, 257);
    assertChunkedRows(fixture.calls, start, 257);

    await model.dispose();
    assert.equal(fixture.released.count, 1);
  }
});

test('Laya Typed-Decisions preserves its tokenizer IDs and typed outputs through the shared WebGPU route', async () => {
  const typedConfig = config();
  typedConfig.vocab_size = 50368;
  typedConfig.sep_token_id = 50282;
  typedConfig.pad_token_id = 50283;
  delete typedConfig.mask_token_id;
  typedConfig.laya.temperature = [1.0148024559020996, 1.0374259948730469, 1.0575125217437744];
  typedConfig.laya.temperature_by_options = {
    'choice:3-5': 1.7601518630981445,
    'choice:6-10': 1.0000158548355103,
    'score:3-5': 1.2514300346374512,
    'noul:2': 1.983399510383606,
    'choice:11+': 0.10058280825614929,
    'choice:2': 1.9063563346862793
  };
  const fixture = loadDependencies({
    'config.json': typedConfig,
    'tokenizer_config.json': {
      cls_token: '[CLS]',
      sep_token: '[SEP]',
      mask_token: '[MASK]',
      pad_token: '[PAD]'
    }
  });
  fixture.dependencies.Tokenizer = TypedCharacterTokenizer;
  const openSource = fixture.dependencies.openSource;
  let sourceBase;
  fixture.dependencies.openSource = async (options) => {
    assert.equal(options.spec, LAYA_TYPED_SPEC);
    sourceBase = options.base.href;
    return openSource(options);
  };
  fixture.dependencies.gpuApi.adapter = fixture.adapter;

  const model = await loadLayaTypedDecisionsModel({model: LAYA_TYPED_MODEL, cache: false}, fixture.dependencies);
  assert.equal(sourceBase, `https://huggingface.co/${LAYA_TYPED_SPEC.repository}/resolve/${LAYA_TYPED_SPEC.revision}/`);
  const result = await model.systemOne('The package arrived.', {
    delivery: {type: 'choice', instructions: 'Choose the status.', criteria: ['arrived', 'pending']}
  });

  assert.equal(result.model, LAYA_TYPED_MODEL);
  assert.equal(result.answers.delivery.choice, 'pending');
  assert.deepEqual(Object.keys(result.answers.delivery.probabilities), ['arrived', 'pending']);
  assert.equal(fixture.calls[0].input_ids.data[0], 50281n);
  const markerPosition = Number(fixture.calls[0].marker_pos.data[0]);
  assert.equal(fixture.calls[0].input_ids.data[markerPosition], 50284n);

  const typedStart = fixture.calls.length;
  const manyTypedQuestions = await model.systemOne('s', typedQuestions(257));
  assert.equal(Object.keys(manyTypedQuestions.answers).length, 257);
  assertChunkedRows(fixture.calls, typedStart, 257);

  const longResult = await model.systemOne('x'.repeat(200_001), {
    delivery: {type: 'choice', instructions: 'Choose the status.', criteria: ['arrived', 'pending']}
  });
  assert.equal(longResult.tokenCount, LAYA_MAX_SEQUENCE_TOKENS);
  assert.equal(longResult.truncated, true);
  assert.equal(fixture.calls.length, 35);

  await assert.rejects(
    model.systemOne('', {
      delivery: {
        type: 'choice',
        instructions: 'Choose the status.',
        criteria: Array.from({length: LAYA_MAX_OPTIONS + 1}, (_, index) => `option-${index}`)
      }
    }),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(fixture.calls.length, 35);
  assert.equal(LAYA_TYPED_FILES['onnx/model_fp16.onnx_data'].bytes, 842_587_136);
  await model.dispose();
  assert.equal(fixture.released.count, 1);
});

test('Laya runner rejects non-finite output and releases its ONNX session once', async () => {
  const tokenizerValue = tokenizer();
  let releaseCount = 0;
  const runner = createLayaRunner(
    {Tensor: class Tensor {}},
    {
      async run() {
        return {logits: {type: 'float32', dims: [1, 2], data: new Float32Array([NaN, 0])}, act_logits: {}};
      },
      async release() {
        releaseCount++;
      }
    },
    tokenizerValue
  );
  const rows = prepareLayaSystemOne(
    's',
    {id: {type: 'choice', instructions: 'q', criteria: ['a', 'b']}},
    tokenizerValue
  ).rows;
  await assert.rejects(runner.run(rows), {code: 'INFERENCE_ERROR'});
  await Promise.all([runner.dispose(), runner.dispose()]);
  assert.equal(releaseCount, 1);
});
