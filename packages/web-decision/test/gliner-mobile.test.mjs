import assert from 'node:assert/strict';
import test from 'node:test';
import {
  GLINER_MOBILE_FILES,
  GLINER_MOBILE_MODEL,
  GLINER_MOBILE_SPEC,
  GLINER_MOBILE_TOTAL_BYTES,
  GLINER_MAX_OPTIONS,
  GLINER_MAX_TOKENS
} from '../src/gliner-mobile/contract.mjs';
import {loadGlinerMobileModel, validateGlinerMobileLoadOptions} from '../src/gliner-mobile/loader.mjs';
import {finishGlinerMobileSystemOne, prepareGlinerMobileSystemOne} from '../src/gliner-mobile/protocol.mjs';
import {createGlinerMobileRunner} from '../src/gliner-mobile/runner.mjs';
import {createGlinerMobileSession} from '../src/gliner-mobile/session.mjs';
import {createGlinerMobileTokenizer} from '../src/gliner-mobile/tokenizer.mjs';

const special = Object.freeze({
  '[SEP_STRUCT]': 128001,
  '[SEP_TEXT]': 128002,
  '[P]': 128003,
  '[L]': 128007,
  '[DESCRIPTION]': 128010,
  '[PAD]': 0
});

function protocolTokenizer() {
  return {
    special,
    encode(text) {
      return Array.from(text, (character) => character.codePointAt(0) + 100);
    },
    encodeState(text) {
      const normalized = /[.!?]$/u.test(text.trim()) ? text.trim() : `${text.trim()}.`;
      return (normalized.match(/\S+/gu) ?? []).map((_, index) => 200 + index);
    }
  };
}

function modelConfig() {
  return {
    model_type: 'deberta-v2',
    vocab_size: 128011,
    max_position_embeddings: 512,
    gliner2: {
      max_len: 512,
      temperature: 1,
      source: 'fastino/GLiNER2.5-Decide',
      inputs: 'input_ids, attention_mask, marker_positions (index of each [L] token)',
      output: "logits (batch, markers); softmax within each question's markers",
      special_tokens: [
        '[SEP_STRUCT]',
        '[SEP_TEXT]',
        '[P]',
        '[C]',
        '[E]',
        '[R]',
        '[L]',
        '[EXAMPLE]',
        '[OUTPUT]',
        '[DESCRIPTION]'
      ]
    },
    'transformers.js_config': {
      dtype: 'q4f16',
      use_external_data_format: {'model_q4f16.onnx': 1}
    }
  };
}

class Tensor {
  constructor(type, data, dims) {
    Object.assign(this, {type, data, dims, disposed: false});
  }

  dispose() {
    this.disposed = true;
  }
}

class TestTokenizer {
  constructor() {}

  token_to_id(token) {
    return special[token];
  }

  encode() {
    return {ids: [17]};
  }
}

function loaderDependencies() {
  const reads = [];
  const released = {count: 0};
  const runCalls = [];
  const tensors = [];
  const files = {
    'config.json': modelConfig(),
    'special_tokens_map.json': {pad_token: {content: '[PAD]'}},
    'tokenizer_config.json': {
      tokenizer_class: 'DebertaV2TokenizerFast',
      backend: 'tokenizers',
      pad_token: '[PAD]'
    },
    'tokenizer.json': {model: {type: 'Unigram'}}
  };
  const ort = {
    Tensor: class extends Tensor {
      constructor(...args) {
        super(...args);
        tensors.push(this);
      }
    },
    env: {webgpu: {}},
    InferenceSession: {
      async create(graph, options) {
        assert.deepEqual(Array.from(graph), [1, 2, 3]);
        assert.deepEqual(options.executionProviders, ['webgpu']);
        assert.equal(options.graphOptimizationLevel, 'all');
        assert.equal(options.externalData[0].path, 'model_q4f16.onnx_data');
        assert.deepEqual(Array.from(options.externalData[0].data), [4, 5, 6]);
        return {
          inputNames: ['input_ids', 'attention_mask', 'marker_positions'],
          outputNames: ['logits'],
          async run(feeds, outputNames) {
            runCalls.push({feeds, outputNames});
            const markers = feeds.marker_positions.dims[1];
            return {
              logits: new ort.Tensor(
                'float32',
                Float32Array.from({length: markers}, (_, i) => i),
                [1, markers]
              )
            };
          },
          async release() {
            released.count++;
          }
        };
      }
    }
  };

  return {
    reads,
    released,
    runCalls,
    tensors,
    ort,
    dependencies: {
      ort,
      Tokenizer: TestTokenizer,
      async openSource() {
        return {
          async read(name) {
            reads.push(name);
            if (name === 'onnx/model_q4f16.onnx') return Uint8Array.of(1, 2, 3).buffer;
            if (name === 'onnx/model_q4f16.onnx_data') return Uint8Array.of(4, 5, 6).buffer;
            const value = files[name];
            if (!value) throw new Error(`Unexpected asset ${name}`);
            return new TextEncoder().encode(JSON.stringify(value)).buffer;
          }
        };
      },
      gpuApi: {
        async requestAdapter(options) {
          assert.deepEqual(options, {powerPreference: 'high-performance', forceFallbackAdapter: false});
          return {info: {isFallbackAdapter: false}, isFallbackAdapter: false};
        }
      }
    }
  };
}

test('GLiNER Mobile pins the complete external-data asset set', () => {
  assert.equal(GLINER_MOBILE_MODEL, 'experimental-gliner2.5-decide-mobile-q4f16');
  assert.equal(GLINER_MOBILE_TOTAL_BYTES, 353_619_944);
  assert.equal(
    Object.values(GLINER_MOBILE_FILES).reduce((sum, file) => sum + file.bytes, 0),
    GLINER_MOBILE_TOTAL_BYTES
  );
  assert.ok(Object.values(GLINER_MOBILE_FILES).every(({sha256}) => /^[0-9a-f]{64}$/u.test(sha256)));
  assert.equal(GLINER_MAX_OPTIONS, 64);
  assert.equal(GLINER_MAX_TOKENS, 512);
});

test('GLiNER Mobile builds one packed schema and preserves typed Choice, Score, and Noul semantics', () => {
  const prepared = prepareGlinerMobileSystemOne(
    'A customer case.',
    {
      route: {
        type: 'choice',
        instructions: 'Choose a queue.',
        criteria: {billing: 'review charges', support: 'fix apps'}
      },
      urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'medium', 'high']},
      repeated: {type: 'noul', instructions: 'Was the charge repeated?', criteria: {true: 'yes', false: 'no'}}
    },
    protocolTokenizer()
  );
  assert.equal(prepared.questions.length, 3);
  assert.equal(prepared.questions[0].markerOffset, 0);
  assert.equal(prepared.questions[1].markerOffset, 2);
  assert.equal(prepared.questions[2].markerOffset, 5);
  assert.equal(prepared.markerPositions.length, 7);
  assert.ok(prepared.inputIds.includes(special['[SEP_STRUCT]']));
  assert.equal(prepared.inputIds.at(-1) > 200, true);

  const result = finishGlinerMobileSystemOne(prepared, [0, 2, 0, Math.log(2), 0, Math.log(3), 0]);
  assert.equal(result.model, GLINER_MOBILE_MODEL);
  assert.equal(result.answers.route.choice, 'support');
  assert.ok(
    Math.abs(result.answers.route.probabilities.billing + result.answers.route.probabilities.support - 1) < 1e-12
  );
  assert.equal(result.answers.urgency.score, 1);
  assert.equal(result.answers.urgency.expected, 1);
  assert.equal(result.answers.urgency.legend['2'], 'high');
  assert.ok(Math.abs(result.answers.repeated.noul - 0.75) < 1e-12);
  assert.ok(Math.abs(result.answers.repeated.probabilities.true - 0.75) < 1e-12);
  assert.ok(Math.abs(result.answers.repeated.probabilities.false - 0.25) < 1e-12);
  assert.equal(result.answers.repeated.confidence, result.answers.repeated.noul);
});

test('GLiNER Mobile truncates state to the shared context and rejects schemas that exceed it', () => {
  const tokenizer = protocolTokenizer();
  const question = {pick: {type: 'choice', instructions: 'Choose.', criteria: ['one', 'two']}};
  const emptyStateTokens = tokenizer.encodeState('').length;
  const schemaLength = prepareGlinerMobileSystemOne('', question, tokenizer).inputIds.length - emptyStateTokens;
  const stateCapacity = GLINER_MAX_TOKENS - schemaLength;
  const exactState = prepareGlinerMobileSystemOne('case '.repeat(stateCapacity), question, tokenizer);
  assert.equal(exactState.tokenCount, GLINER_MAX_TOKENS);
  assert.equal(exactState.truncated, false);
  const oneOverState = prepareGlinerMobileSystemOne('case '.repeat(stateCapacity + 1), question, tokenizer);
  assert.equal(oneOverState.tokenCount, GLINER_MAX_TOKENS);
  assert.equal(oneOverState.truncated, true);

  const longState = 'case '.repeat(40_001);
  assert.ok(longState.length > 200_000);
  const truncatedLongState = prepareGlinerMobileSystemOne(longState, question, tokenizer);
  assert.equal(truncatedLongState.tokenCount, GLINER_MAX_TOKENS);
  assert.equal(truncatedLongState.truncated, true);

  const oneTokenSchemaLength =
    prepareGlinerMobileSystemOne('', {pick: {type: 'choice', instructions: 'x', criteria: ['one', 'two']}}, tokenizer)
      .inputIds.length - emptyStateTokens;
  const exactInstructions = 'x'.repeat(GLINER_MAX_TOKENS - (oneTokenSchemaLength - 1));
  const exactSchema = prepareGlinerMobileSystemOne(
    '',
    {pick: {type: 'choice', instructions: exactInstructions, criteria: ['one', 'two']}},
    tokenizer
  );
  assert.equal(exactSchema.tokenCount, GLINER_MAX_TOKENS);
  assert.throws(
    () =>
      prepareGlinerMobileSystemOne(
        '',
        {pick: {type: 'choice', instructions: `${exactInstructions}x`, criteria: ['one', 'two']}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
});

test('GLiNER Mobile applies the 512-token schema limit to long Choice and Noul descriptions', async () => {
  const description = 'x'.repeat(200_001);
  const baseTokenizer = protocolTokenizer();
  const tokenizer = {
    ...baseTokenizer,
    encode(text) {
      if (text.includes(description)) return new Array(GLINER_MAX_TOKENS + 1).fill(17);
      return baseTokenizer.encode(text);
    }
  };

  for (const [type, criteria] of [
    ['choice', {first: description, second: null}],
    ['noul', {true: description, false: null}]
  ]) {
    let inferenceCalls = 0;
    const model = createGlinerMobileSession({
      tokenizer,
      runner: {
        async run() {
          inferenceCalls++;
          return [0, 1];
        },
        async dispose() {}
      }
    });

    await assert.rejects(
      model.systemOne('', {pick: {type, instructions: 'Choose.', criteria}}),
      (error) => error.code === 'INVALID_INPUT' && error.message.includes('512-token context')
    );
    assert.equal(inferenceCalls, 0);
    await model.dispose();
  }
});

test('GLiNER Mobile rejects invalid array shapes, over-limit labels, and reserved protocol tokens', () => {
  const tokenizer = protocolTokenizer();
  const sparse = new Array(2);
  sparse[1] = 'two';
  assert.throws(
    () =>
      prepareGlinerMobileSystemOne('', {pick: {type: 'choice', instructions: 'Choose.', criteria: sparse}}, tokenizer),
    {code: 'INVALID_INPUT'}
  );
  const accessor = ['one', 'two'];
  Object.defineProperty(accessor, '0', {
    get() {
      throw new Error('must not invoke');
    }
  });
  assert.throws(
    () =>
      prepareGlinerMobileSystemOne(
        '',
        {pick: {type: 'choice', instructions: 'Choose.', criteria: accessor}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareGlinerMobileSystemOne(
        '',
        {
          pick: {
            type: 'choice',
            instructions: 'Choose.',
            criteria: Array.from({length: GLINER_MAX_OPTIONS + 1}, (_, i) => `o${i}`)
          }
        },
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  const tooManyQuestions = Object.fromEntries(
    Array.from({length: 257}, (_, index) => [
      String(index),
      {type: 'choice', instructions: 'Pick.', criteria: ['a', 'b']}
    ])
  );
  assert.throws(
    () => prepareGlinerMobileSystemOne('', tooManyQuestions, tokenizer),
    (error) => error.code === 'INVALID_INPUT' && error.message.includes('512-token context')
  );
  for (const [state, question] of [
    ['literal [L] marker', {instructions: 'Choose.', criteria: ['one', 'two']}],
    ['', {instructions: 'Choose [SEP_TEXT].', criteria: ['one', 'two']}],
    ['', {instructions: 'Choose.', criteria: ['[L]', 'two']}]
  ]) {
    assert.throws(() => prepareGlinerMobileSystemOne(state, {pick: {type: 'choice', ...question}}, tokenizer), {
      code: 'INVALID_INPUT'
    });
  }
});

test('GLiNER Mobile generic decide preserves option order and returns per-option logits', async () => {
  const runCalls = [];
  const model = createGlinerMobileSession({
    tokenizer: protocolTokenizer(),
    runner: {
      async run(inputIds, markerPositions) {
        runCalls.push({inputIds, markerPositions});
        return [0, 2];
      },
      async dispose() {}
    }
  });
  const result = await model.decide({
    state: 'Customer requests a refund.',
    questions: [{instructions: 'Choose the queue.', options: ['billing', 'support']}]
  });
  assert.equal(result.answers[0].selected, 'support');
  assert.equal(result.answers[0].selectedIndex, 1);
  assert.deepEqual(
    result.answers[0].probabilities.map(({option}) => option),
    ['billing', 'support']
  );
  assert.deepEqual(
    result.answers[0].probabilities.map(({logit}) => logit),
    [0, 2]
  );
  assert.equal(runCalls.length, 1);
  await model.dispose();
});

test('GLiNER Mobile loads by ID or mirror with hardware WebGPU and typed inference intact', async () => {
  for (const assetsUrl of [undefined, 'https://example.test/gliner/']) {
    const fixture = loaderDependencies();
    let sourceBase;
    const openSource = fixture.dependencies.openSource;
    fixture.dependencies.openSource = async (options) => {
      sourceBase = options.base.href;
      return openSource(options);
    };
    const options = {model: GLINER_MOBILE_MODEL, cache: false};
    if (assetsUrl !== undefined) options.assetsUrl = assetsUrl;
    const model = await loadGlinerMobileModel(options, fixture.dependencies);
    assert.equal(
      sourceBase,
      assetsUrl ?? `https://huggingface.co/${GLINER_MOBILE_SPEC.repository}/resolve/${GLINER_MOBILE_SPEC.revision}/`
    );
    const result = await model.systemOne('A case.', {
      pick: {type: 'choice', instructions: 'Choose.', criteria: ['one', 'two']}
    });
    assert.equal(result.answers.pick.choice, 'two');
    assert.deepEqual(fixture.reads.sort(), Object.keys(GLINER_MOBILE_FILES).sort());
    assert.equal(fixture.runCalls[0].outputNames[0], 'logits');
    assert.equal(fixture.runCalls[0].feeds.input_ids.dims[0], 1);
    assert.equal(fixture.runCalls[0].feeds.marker_positions.dims[1], 2);
    assert.ok(fixture.tensors.every((tensor) => tensor.disposed));
    await model.dispose();
    assert.equal(fixture.released.count, 1);
  }
});

test('GLiNER Mobile rejects unsupported loading options before reading assets', async () => {
  const defaults = validateGlinerMobileLoadOptions({model: GLINER_MOBILE_MODEL});
  assert.equal(
    defaults.base.href,
    `https://huggingface.co/${GLINER_MOBILE_SPEC.repository}/resolve/${GLINER_MOBILE_SPEC.revision}/`
  );
  assert.equal(defaults.cacheEnabled, true);
  assert.equal(defaults.offline, false);
  assert.throws(() => validateGlinerMobileLoadOptions({model: GLINER_MOBILE_MODEL, assetsUrl: 'file:///model/'}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(
    () =>
      validateGlinerMobileLoadOptions({
        model: GLINER_MOBILE_MODEL,
        assetsUrl: 'https://example.test/',
        maxPrefetchBytes: 1
      }),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      validateGlinerMobileLoadOptions({
        model: GLINER_MOBILE_MODEL,
        assetsUrl: 'https://example.test/',
        cache: false,
        offline: true
      }),
    {code: 'INVALID_INPUT'}
  );
});

test('GLiNER Mobile tokenizer verifies pinned special tokens and vocabulary bounds', () => {
  const config = modelConfig();
  const tokenizer = createGlinerMobileTokenizer(
    TestTokenizer,
    {model: {type: 'Unigram'}},
    {tokenizer_class: 'DebertaV2TokenizerFast', backend: 'tokenizers', pad_token: '[PAD]'},
    {pad_token: {content: '[PAD]'}},
    config
  );
  assert.deepEqual(tokenizer.special, special);
  assert.deepEqual(tokenizer.encode('sample'), [17]);
  assert.deepEqual(tokenizer.encodeState('Hello world'), [17, 17, 17]);
  assert.throws(
    () =>
      createGlinerMobileTokenizer(
        class extends TestTokenizer {
          token_to_id(token) {
            return token === '[L]' ? 128011 : special[token];
          }
        },
        {model: {type: 'Unigram'}},
        {tokenizer_class: 'DebertaV2TokenizerFast', backend: 'tokenizers', pad_token: '[PAD]'},
        {pad_token: {content: '[PAD]'}},
        config
      ),
    {code: 'INVALID_ASSET'}
  );
});

test('GLiNER Mobile runner validates output dimensions and disposes feeds and outputs', async () => {
  const tensors = [];
  class TrackedTensor extends Tensor {
    constructor(...args) {
      super(...args);
      tensors.push(this);
    }
  }
  const session = {
    async run(feeds) {
      assert.equal(feeds.input_ids.type, 'int64');
      assert.equal(feeds.input_ids.dims[0], 1);
      assert.equal(feeds.marker_positions.dims[1], 2);
      return {logits: new TrackedTensor('float32', Float32Array.of(1, 2), [1, 2])};
    },
    async release() {}
  };
  const runner = createGlinerMobileRunner({Tensor: TrackedTensor}, session);
  assert.deepEqual(await runner.run([1, 2, 3], [0, 2]), [1, 2]);
  assert.ok(tensors.every((tensor) => tensor.disposed));
  await runner.dispose();
});
