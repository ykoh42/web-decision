import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {
  ETTIN_FILES,
  ETTIN_MARKER_IDS,
  ETTIN_MODEL,
  ETTIN_PROFILE,
  ETTIN_REVISION,
  ETTIN_TOTAL_BYTES
} from '../src/ettin/contract.mjs';
import {loadEttin, validateEttinLoadOptions} from '../src/ettin/loader.mjs';
import {
  finishEttinDecision,
  finishEttinSystemOne,
  prepareEttinDecision,
  prepareEttinSystemOne
} from '../src/ettin/protocol.mjs';
import {createEttinRunner} from '../src/ettin/runner.mjs';
import {createEttinSession} from '../src/ettin/session.mjs';
import {openEttinSource} from '../src/ettin/source.mjs';
import {createEttinTokenizer} from '../src/ettin/tokenizer.mjs';

const markerMap = new Map([
  ['[DECISION]', ETTIN_MARKER_IDS.decision],
  ['[STATE]', ETTIN_MARKER_IDS.state],
  ['[QUESTION]', ETTIN_MARKER_IDS.question],
  ...ETTIN_MARKER_IDS.options.map((id, index) => [`[OPT_${index}]`, id])
]);

class FakeTokenizer {
  constructor() {}

  token_to_id(token) {
    return markerMap.get(token);
  }

  encode(text) {
    return {ids: Array.from(text, (character) => character.codePointAt(0) + 100)};
  }
}

function tokenizer() {
  return createEttinTokenizer(FakeTokenizer, {model: {type: 'BPE'}}, {backend: 'tokenizers'});
}

function row({choice = [1, 2], noul = [0, 1], score = [0, 1, 2, 3, 4, 5, 6]} = {}) {
  return {
    choiceLogits: [...choice, ...Array(10 - choice.length).fill(-3.4028234663852886e38)],
    noulLogits: noul,
    scoreLogits: score
  };
}

function makeLoaderDependencies({config, inputNames, outputNames} = {}) {
  const configValue = config ?? {
    base_model: 'jhu-clsp/ettin-encoder-68m',
    max_len: 2048,
    max_options: 10,
    max_score_levels: 7,
    onnx_opset: 18,
    onnx_batch_size: 1,
    onnx_sequence_dynamic: true,
    onnx_inputs: ['input_ids', 'attention_mask', 'decision_pos', 'option_pos', 'option_mask', 'score_n'],
    onnx_outputs: ['choice_logits', 'noul_logits', 'score_logits']
  };
  const values = {
    'config.json': configValue,
    'manifest.json': {
      version: 'epoch2-int8',
      modelFile: 'model_int8.onnx',
      sizeBytes: ETTIN_FILES['model_int8.onnx'].bytes,
      sha256: ETTIN_FILES['model_int8.onnx'].sha256,
      maxOptions: 10,
      maxScoreLevels: 7,
      maxSequenceLength: 2048
    },
    'tokenizer.json': {model: {type: 'BPE'}},
    'tokenizer_config.json': {backend: 'tokenizers'},
    'model_int8.onnx': new Uint8Array([1, 2, 3]).buffer
  };
  const reads = [];
  const released = {count: 0};
  const ort = {
    Tensor: class Tensor {
      constructor(type, data, dims) {
        Object.assign(this, {type, data, dims});
      }
    },
    InferenceSession: {
      async create(graph, options) {
        assert.deepEqual(Array.from(graph), [1, 2, 3]);
        assert.deepEqual(options.executionProviders, ['wasm']);
        assert.equal(options.intraOpNumThreads, 1);
        assert.equal(options.interOpNumThreads, 1);
        return {
          inputNames: inputNames ?? [
            'input_ids',
            'attention_mask',
            'decision_pos',
            'option_pos',
            'option_mask',
            'score_n'
          ],
          outputNames: outputNames ?? ['choice_logits', 'noul_logits', 'score_logits'],
          async run() {
            return {
              choice_logits: {type: 'float32', dims: [1, 10], data: Float32Array.from([0, 2, ...Array(8).fill(-10)])},
              noul_logits: {type: 'float32', dims: [1, 2], data: Float32Array.from([0, 1])},
              score_logits: {type: 'float32', dims: [1, 7], data: Float32Array.from([0, 1, 2, 3, 4, 5, 6])}
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
    dependencies: {
      ort,
      Tokenizer: FakeTokenizer,
      async openSource() {
        return {
          async read(name) {
            reads.push(name);
            const value = values[name];
            if (name === 'model_int8.onnx') return value;
            return new TextEncoder().encode(JSON.stringify(value)).buffer;
          }
        };
      }
    }
  };
}

test('Decider Ettin pins its checkpoint, graph, and complete 74.2 MB file set', () => {
  assert.equal(ETTIN_MODEL, 'experimental-ettin-68m-int8');
  assert.equal(ETTIN_REVISION, 'df8600e1f6e220b302cdd49897c2b1ff76e021ae');
  assert.equal(ETTIN_TOTAL_BYTES, 74_200_147);
  assert.deepEqual(Object.keys(ETTIN_FILES), [
    'config.json',
    'manifest.json',
    'model_int8.onnx',
    'tokenizer.json',
    'tokenizer_config.json'
  ]);
  assert.equal(
    Object.values(ETTIN_FILES).every(({sha256}) => /^[0-9a-f]{64}$/u.test(sha256)),
    true
  );
  assert.equal(Object.hasOwn(ETTIN_PROFILE, 'maxTextChars'), false);
  assert.equal(Object.hasOwn(ETTIN_PROFILE, 'maxQuestions'), false);
  assert.deepEqual(
    ETTIN_PROFILE.markerIds.options,
    Array.from({length: 10}, (_, index) => 50371 + index)
  );
});

test('Decider Ettin tokenizer requires the trained BPE markers at their pinned IDs', () => {
  const encoder = tokenizer();
  assert.deepEqual(
    encoder.encode('hello'),
    Array.from('hello', (character) => character.codePointAt(0) + 100)
  );
  assert.throws(() => createEttinTokenizer(FakeTokenizer, {model: {type: 'WordPiece'}}, {backend: 'tokenizers'}), {
    code: 'INVALID_ASSET'
  });
  class WrongMarkerTokenizer extends FakeTokenizer {
    token_to_id(token) {
      return markerMap.get(token) + (token === '[OPT_2]' ? 1 : 0);
    }
  }
  assert.throws(() => createEttinTokenizer(WrongMarkerTokenizer, {model: {type: 'BPE'}}, {backend: 'tokenizers'}), {
    code: 'INVALID_ASSET'
  });
});

test('Decider Ettin generic choices preserve marker positions, option order, and labels', () => {
  const prepared = prepareEttinDecision(tokenizer(), {
    state: 'A',
    questions: [{instructions: 'Q', options: ['one', 'two']}]
  });
  const question = prepared.questions[0];
  assert.deepEqual(question.inputIds, [50368, 50369, 165, 50370, 181, 50371, 211, 210, 201, 50372, 216, 219, 211]);
  assert.deepEqual(question.optionPositions, [5, 9, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(question.optionMask, [true, true, false, false, false, false, false, false, false, false]);
  assert.deepEqual(question.options, ['one', 'two']);
  assert.equal(question.tokenCount, 13);

  const result = finishEttinDecision(prepared, [row({choice: [0, 2]})]);
  assert.equal(result.answers[0].selected, 'two');
  assert.equal(result.answers[0].selectedIndex, 1);
  assert.deepEqual(
    result.answers[0].probabilities.map(({option}) => option),
    ['one', 'two']
  );
  assert.ok(Math.abs(result.answers[0].probabilities.reduce((sum, item) => sum + item.probability, 0) - 1) < 1e-4);
});

test('Decider Ettin truncates only state and question tokens to the exact sequence cap', () => {
  const prepared = prepareEttinDecision(tokenizer(), {
    state: 's'.repeat(3000),
    questions: [{instructions: 'q', options: ['a', 'b']}]
  });
  const question = prepared.questions[0];
  assert.equal(question.tokenCount, 2048);
  assert.equal(question.truncated, true);
  assert.equal(question.inputIds.length, 2048);
  assert.equal(question.inputIds.filter((id) => id === ETTIN_MARKER_IDS.options[0]).length, 1);
  assert.equal(question.inputIds.filter((id) => id === ETTIN_MARKER_IDS.options[1]).length, 1);
  assert.deepEqual(
    prepareEttinDecision(tokenizer(), {
      state: 's',
      questions: [{instructions: '', options: ['a', 'b']}]
    }).truncated,
    false
  );
});

test('Decider Ettin accepts text above 200,000 characters and applies the native row budget', () => {
  const prepared = prepareEttinDecision(tokenizer(), {
    state: 's'.repeat(200_001),
    questions: [{instructions: 'q', options: ['a', 'b']}]
  });

  assert.equal(prepared.questions[0].tokenCount, 2048);
  assert.equal(prepared.questions[0].inputIds.length, 2048);
  assert.equal(prepared.questions[0].truncated, true);
});

test('Decider Ettin accepts more than 256 questions and runs one batch-one row per question', async () => {
  const questionCount = 257;
  const makeQuestions = () =>
    Array.from({length: questionCount}, (_, index) => ({
      instructions: `Choose for question ${index}.`,
      options: ['first', 'second']
    }));
  const makeTypedQuestions = () =>
    Object.fromEntries(
      Array.from({length: questionCount}, (_, index) => [
        `q${index}`,
        {type: 'choice', instructions: `Choose for question ${index}.`, criteria: {first: null, second: null}}
      ])
    );

  const generic = prepareEttinDecision(tokenizer(), {state: 'state', questions: makeQuestions()});
  const typed = prepareEttinSystemOne('state', makeTypedQuestions(), tokenizer());
  assert.equal(generic.questions.length, questionCount);
  assert.equal(typed.questions.length, questionCount);

  let calls = 0;
  let active = 0;
  let maximumActive = 0;
  const model = createEttinSession({
    tokenizer: tokenizer(),
    runner: {
      async run() {
        active++;
        maximumActive = Math.max(maximumActive, active);
        calls++;
        await Promise.resolve();
        active--;
        return row();
      },
      async dispose() {}
    }
  });

  const genericResult = await model.decide({state: 'state', questions: makeQuestions()});
  assert.equal(genericResult.answers.length, questionCount);
  assert.equal(calls, questionCount);

  const typedResult = await model.systemOne('state', makeTypedQuestions());
  assert.equal(Object.keys(typedResult.answers).length, questionCount);
  assert.equal(calls, questionCount * 2);
  assert.equal(maximumActive, 1);
  await model.dispose();
});

test('Decider Ettin enforces the exact option-token row boundary before inference', async () => {
  const calls = [];
  const model = createEttinSession({
    runner: {
      async run(question) {
        calls.push(question.inputIds.length);
        return row();
      },
      async dispose() {}
    },
    tokenizer: tokenizer()
  });

  const exact = await model.decide({
    state: 's',
    questions: [{instructions: 'q', options: ['a'.repeat(1021), 'b'.repeat(1021)]}]
  });
  assert.equal(exact.answers.length, 1);
  assert.deepEqual(calls, [2048]);

  await assert.rejects(
    model.decide({
      state: 's',
      questions: [{instructions: 'q', options: ['a'.repeat(1022), 'b'.repeat(1022)]}]
    }),
    {code: 'INVALID_INPUT'}
  );
  assert.deepEqual(calls, [2048]);
  await model.dispose();
});

test('Decider Ettin rejects invalid generic choice bounds and overlong option content', () => {
  const encoder = tokenizer();
  const request = (options) => ({state: 'state', questions: [{instructions: 'question', options}]});
  assert.throws(() => prepareEttinDecision(encoder, request(['one'])), {code: 'INVALID_INPUT'});
  assert.throws(() => prepareEttinDecision(encoder, request(Array.from({length: 11}, (_, index) => `o${index}`))), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => prepareEttinDecision(encoder, request(['x'.repeat(2048), 'y'.repeat(2048)])), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => prepareEttinDecision(encoder, request(['', 'valid'])), {code: 'INVALID_INPUT'});
});

test('Decider Ettin typed inputs preserve Choice, Noul, and ordered Score semantics', () => {
  const prepared = prepareEttinSystemOne(
    'A duplicate charge.',
    {
      route: {type: 'choice', instructions: 'Pick one.', criteria: {billing: 'review charges', support: null}},
      truth: {type: 'noul', instructions: 'Was the charge duplicated?'},
      impact: {type: 'score', instructions: 'Rate impact.', criteria: ['low', 'medium', 'high']}
    },
    tokenizer()
  );
  assert.deepEqual(
    prepared.questions.map(({type}) => type),
    ['choice', 'noul', 'score']
  );
  assert.deepEqual(prepared.questions[0].labels, ['billing', 'support']);
  assert.equal(prepared.questions[0].optionCount, 2);
  assert.deepEqual(prepared.questions[1].optionMask, Array(10).fill(false));
  assert.equal(prepared.questions[1].scoreCount, 0);
  assert.equal(prepared.questions[2].scoreCount, 3);
  assert.equal(prepared.questions[2].inputIds.includes(ETTIN_MARKER_IDS.options[0]), false);

  const result = finishEttinSystemOne(
    prepared,
    [row({choice: [0, 2]}), row({noul: [-2, 2]}), row({score: [-3, -2, 3]})],
    ETTIN_MODEL
  );
  assert.equal(result.model, ETTIN_MODEL);
  assert.equal(result.answers.route.choice, 'support');
  assert.ok(result.answers.truth.noul > 0.95);
  assert.equal(result.answers.impact.score, 2);
  assert.ok(result.answers.impact.expected > 1.5);
  assert.deepEqual(result.answers.impact.legend, {0: 'low', 1: 'medium', 2: 'high'});
});

test('Decider Ettin typed API rejects unsupported Noul criteria and invalid choice or score limits', () => {
  const encoder = tokenizer();
  assert.throws(
    () =>
      prepareEttinSystemOne(
        'state',
        {
          truth: {type: 'noul', instructions: 'Is it true?', criteria: {true: 'custom meaning'}}
        },
        encoder
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareEttinSystemOne(
        'state',
        {
          route: {type: 'choice', criteria: Array.from({length: 11}, (_, index) => `option ${index}`)}
        },
        encoder
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareEttinSystemOne(
        'state',
        {
          score: {
            type: 'score',
            instructions: 'Rate.',
            criteria: Array.from({length: 8}, (_, index) => `level ${index}`)
          }
        },
        encoder
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(() => prepareEttinSystemOne('state', {noul: {type: 'noul'}}, encoder), {code: 'INVALID_INPUT'});
});

test('Decider Ettin runner builds the graph feeds and validates its three heads', async () => {
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, {type, data, dims});
    }
  }
  let feed;
  let releases = 0;
  const session = {
    async run(value) {
      feed = value;
      return {
        choice_logits: {type: 'float32', dims: [1, 10], data: Float32Array.from([0, 1, ...Array(8).fill(-1e30)])},
        noul_logits: {type: 'float32', dims: [1, 2], data: Float32Array.from([0, 1])},
        score_logits: {type: 'float32', dims: [1, 7], data: Float32Array.from([0, 1, 2, 3, 4, 5, 6])}
      };
    },
    async release() {
      releases++;
    }
  };
  const runner = createEttinRunner({Tensor}, session);
  const question = prepareEttinDecision(tokenizer(), {
    state: 'A',
    questions: [{instructions: 'Q', options: ['one', 'two']}]
  }).questions[0];
  const result = await runner.run(question);
  assert.equal(result.choiceLogits[1], 1);
  assert.deepEqual(feed.input_ids.dims, [1, 13]);
  assert.deepEqual(feed.option_pos.dims, [1, 10]);
  assert.equal(feed.option_mask.type, 'bool');
  assert.deepEqual(Array.from(feed.option_mask.data), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
  await runner.dispose();
  await runner.dispose();
  assert.equal(releases, 1);
  await assert.rejects(runner.run(question), {code: 'SESSION_CLOSED'});
});

test('Decider Ettin session queues calls, exposes the typed heads, and closes cleanly', async () => {
  const calls = [];
  let disposed = 0;
  const runner = {
    async run(question) {
      calls.push(question.type);
      return row({
        choice: question.optionCount === 2 ? [0, 1] : [0, 1],
        noul: [0, 2],
        score: [0, 1, 2, 0, 0, 0, 0]
      });
    },
    async dispose() {
      disposed++;
    }
  };
  const model = createEttinSession({runner, tokenizer: tokenizer()});
  const generic = await model.decide({state: 'state', questions: [{instructions: 'Pick.', options: ['a', 'b']}]});
  const typed = await model.systemOne('state', {
    choice: {type: 'choice', instructions: 'Pick.', criteria: ['a', 'b']},
    truth: {type: 'noul', instructions: 'Is it true?'},
    score: {type: 'score', instructions: 'Rate.', criteria: ['low', 'high']}
  });
  assert.equal(generic.answers[0].selected, 'b');
  assert.equal(typed.answers.choice.choice, 'b');
  assert.ok(typed.answers.truth.noul > 0.8);
  assert.equal(typed.answers.score.type, 'score');
  assert.deepEqual(calls, ['choice', 'choice', 'noul', 'score']);
  await model.releaseCachedMemory();
  await model.dispose();
  await model.dispose();
  assert.equal(disposed, 1);
  await assert.rejects(model.decide({state: 'state', questions: [{instructions: 'Pick.', options: ['a', 'b']}]}), {
    code: 'SESSION_CLOSED'
  });
});

test('Decider Ettin load options default to the pinned source and support a trusted mirror', () => {
  assert.equal(
    validateEttinLoadOptions({model: ETTIN_MODEL}).base.href,
    'https://huggingface.co/aravind1706/decider-ettin/resolve/df8600e1f6e220b302cdd49897c2b1ff76e021ae/'
  );
  assert.deepEqual(validateEttinLoadOptions({model: ETTIN_MODEL, assetsUrl: 'https://example.test/ettin'}), {
    base: new URL('https://example.test/ettin/'),
    signal: undefined,
    cacheEnabled: true,
    offline: false,
    onProgress: undefined
  });
  assert.throws(() => validateEttinLoadOptions({model: ETTIN_MODEL, assetsUrl: 'file:///model/'}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateEttinLoadOptions({model: ETTIN_MODEL, assetsUrl: null}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateEttinLoadOptions({model: ETTIN_MODEL, assetsUrl: '/model/', maxPrefetchBytes: 1}), {
    code: 'INVALID_INPUT'
  });
});

test('Decider Ettin asset source verifies a file and reports loaded bytes', async (context) => {
  const bytes = new Uint8Array([1, 2, 3]);
  const previousFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url) => {
    requests.push(String(url));
    return new Response(bytes);
  };
  context.after(() => {
    globalThis.fetch = previousFetch;
  });

  const source = await openEttinSource(
    {base: new URL('https://assets.example.test/ettin/'), cacheEnabled: false, offline: false},
    {
      files: {'tiny.bin': {bytes: bytes.byteLength, sha256: createHash('sha256').update(bytes).digest('hex')}},
      totalBytes: bytes.byteLength
    }
  );
  const result = await source.read('tiny.bin');
  assert.deepEqual(Array.from(new Uint8Array(result)), [1, 2, 3]);
  assert.deepEqual(requests, ['https://assets.example.test/ettin/tiny.bin']);
  await assert.rejects(source.read('../escape.bin'), {code: 'INVALID_ASSET'});
});

test('Decider Ettin source reports progress for a verified read', async (context) => {
  const bytes = new Uint8Array([4, 5]);
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(bytes);
  context.after(() => {
    globalThis.fetch = previousFetch;
  });
  const progress = [];
  const source = await openEttinSource(
    {
      base: new URL('https://assets.example.test/ettin/'),
      cacheEnabled: false,
      offline: false,
      onProgress(value) {
        progress.push(value);
      }
    },
    {
      files: {'tiny.bin': {bytes: bytes.byteLength, sha256: createHash('sha256').update(bytes).digest('hex')}},
      totalBytes: bytes.byteLength
    }
  );
  await source.read('tiny.bin');
  assert.deepEqual(progress, [{loadedBytes: 2, totalBytes: 2, fraction: 1}]);
});

test('Decider Ettin loader verifies model metadata and returns a working WASM session', async () => {
  const harness = makeLoaderDependencies();
  const model = await loadEttin(
    {model: ETTIN_MODEL, assetsUrl: 'https://assets.example.test/ettin/'},
    harness.dependencies
  );
  const result = await model.decide({state: 'state', questions: [{instructions: 'Pick one.', options: ['a', 'b']}]});
  assert.equal(result.answers[0].selected, 'b');
  assert.deepEqual(harness.reads.sort(), [
    'config.json',
    'manifest.json',
    'model_int8.onnx',
    'tokenizer.json',
    'tokenizer_config.json'
  ]);
  await model.dispose();
  assert.equal(harness.released.count, 1);
});

test('Decider Ettin loader rejects mismatched graph metadata and releases a graph with an invalid interface', async () => {
  const badConfig = makeLoaderDependencies({config: {base_model: 'wrong'}});
  await assert.rejects(
    loadEttin({model: ETTIN_MODEL, assetsUrl: 'https://assets.example.test/ettin/'}, badConfig.dependencies),
    {code: 'INVALID_ASSET'}
  );
  assert.equal(badConfig.released.count, 0);

  const badGraph = makeLoaderDependencies({inputNames: ['unexpected'], outputNames: ['also_unexpected']});
  await assert.rejects(
    loadEttin({model: ETTIN_MODEL, assetsUrl: 'https://assets.example.test/ettin/'}, badGraph.dependencies),
    {code: 'INVALID_ASSET'}
  );
  assert.equal(badGraph.released.count, 1);
});
