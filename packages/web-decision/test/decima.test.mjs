import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DECIMA_AGENT_MODEL,
  DECIMA_AGENT_TEMPERATURE,
  DECIMA_BASE_CLEAN_MODEL,
  DECIMA_SMALL_MODEL,
  DECIMA_SMALL_TEMPERATURE
} from '../src/decima/contract.mjs';
import {loadDecimaModel, validateDecimaLoadOptions} from '../src/decima/loader.mjs';
import {normalizeDecimaText} from '../src/decima/normalize.mjs';
import {decodeDecimaScores, finishDecimaChoice} from '../src/decima/protocol.mjs';
import {createDecimaRunner} from '../src/decima/runner.mjs';
import {createDecimaTokenizer, prepareDecimaQuestion} from '../src/decima/tokenizer.mjs';
import {finishDecimaSystemOne, prepareDecimaSystemOne} from '../src/decima/typed.mjs';

const vocab = {'<s>': 0, '<pad>': 1, '</s>': 2};
const tokenizerConfig = {cls_token: '<s>', pad_token: '<pad>', sep_token: '</s>'};
const tokenizerData = {model: {vocab}};
const config = {
  state_prefix: 'query: ',
  choice_prefix: 'passage: ',
  max_state_tokens: 512,
  max_choice_tokens: 64,
  temperature: DECIMA_SMALL_TEMPERATURE,
  question_in_state: true,
  question_in_choices: true,
  ordinal: {ord_g_w: [0.2, -0.1], ord_g_b: 0.1, ord_gap_w: [0.05, -0.04], ord_gap_b: -0.2}
};

class FakeTokenizer {
  constructor() {}
  encode(text) {
    return {ids: Array.from(text, (character) => character.codePointAt(0) + 3)};
  }
}

function makeTokenizer() {
  return createDecimaTokenizer(FakeTokenizer, tokenizerData, tokenizerConfig);
}

class FakeTensor {
  constructor(type, data, dims) {
    Object.assign(this, {type, data, dims});
  }
}

function fakeOrt() {
  const released = [];
  const options = [];
  return {
    released,
    options,
    Tensor: FakeTensor,
    InferenceSession: {
      async create(bytes, sessionOptions) {
        options.push(sessionOptions);
        const encoder = new Uint8Array(bytes)[0] === 1;
        return {
          async run(feeds) {
            if (encoder) {
              const [batch, width] = feeds.input_ids.dims;
              return {token_states: new FakeTensor('float32', new Float32Array(batch * width * 2), [batch, width, 2])};
            }
            const count = feeds.choice_h.dims[0];
            return {
              scores: new FakeTensor(
                'float32',
                Float32Array.from({length: count}, (_, index) => index),
                [count]
              ),
              z: new FakeTensor('float32', new Float32Array(count * 2).fill(0.25), [count, 2])
            };
          },
          async release() {
            released.push(encoder ? 'encoder' : 'scorer');
          }
        };
      }
    }
  };
}

test('Decima normalization matches NFC and Persian or Arabic character rules', () => {
  assert.equal(normalizeDecimaText('  Cafe\u0301  '), 'Café');
  assert.equal(normalizeDecimaText('ي ك ة أ ١۲', 'fa'), 'ی ک ه ا 12');
  assert.equal(normalizeDecimaText('ی ک ۱2', 'ar'), 'ي ك 12');
  assert.equal(normalizeDecimaText('  a\t b  ', 'en'), 'a\t b');
});

test('Decima tokenizer resolves Unigram vocabulary arrays and added special-token IDs', () => {
  const tokenizer = createDecimaTokenizer(
    FakeTokenizer,
    {
      model: {
        vocab: [
          ['<s>', 0],
          ['<pad>', 0],
          ['</s>', 0],
          ['<unk>', 0]
        ]
      },
      added_tokens: [
        {id: 0, content: '<s>'},
        {id: 1, content: '<pad>'},
        {id: 2, content: '</s>'}
      ]
    },
    tokenizerConfig
  );
  assert.deepEqual(tokenizer.special, {clsId: 0, sepId: 2, padId: 1});
});

test('Decima tokenizer keeps XLM-R special-token layout and marks the exact state boundary', () => {
  const tokenizer = makeTokenizer();
  const state = 'x'.repeat(501);
  const prepared = prepareDecimaQuestion(state, 'i', ['a', 'b'], config, tokenizer);
  assert.equal(prepared.stateIds.length, 512);
  assert.deepEqual(prepared.stateIds.slice(0, 2), [0, 'q'.codePointAt(0) + 3]);
  assert.equal(prepared.stateIds.at(-1), 2);
  const truncated = prepareDecimaQuestion(`${state}x`, 'i', ['a', 'b'], config, tokenizer);
  assert.equal(truncated.stateIds.length, 512);
  assert.equal(truncated.truncated, true);
});

test('Decima base-clean rejects state above 512 encoded tokens and accepts the boundary', () => {
  const baseCleanConfig = {
    ...config,
    state_prefix: '',
    choice_prefix: '',
    temperature: 0.6929793512269696,
    choice_fit: true,
    state_overflow: 'reject'
  };
  const exact = prepareDecimaQuestion('x'.repeat(508), 'Q', ['a', 'b'], baseCleanConfig, makeTokenizer());
  assert.equal(exact.stateIds.length, 512);
  assert.throws(() => prepareDecimaQuestion('x'.repeat(509), 'Q', ['a', 'b'], baseCleanConfig, makeTokenizer()), {
    code: 'INVALID_INPUT'
  });
});

test('Decima choice rows respect the 64-token cap and fit colliding truncated choices', () => {
  const tokenizer = makeTokenizer();
  const exact = prepareDecimaQuestion('', 'q', ['x'.repeat(51), 'b'], config, tokenizer);
  assert.equal(exact.choiceIds[0].length, 64);
  assert.equal(exact.truncated, false);
  const truncated = prepareDecimaQuestion('', 'q', ['x'.repeat(52), 'b'], config, tokenizer);
  assert.equal(truncated.choiceIds[0].length, 64);
  assert.equal(truncated.truncated, true);
  const collision = prepareDecimaQuestion('', 'q'.repeat(90), ['first', 'second'], config, tokenizer);
  assert.notDeepEqual(collision.choiceIds[0], collision.choiceIds[1]);
  assert.equal(collision.truncated, true);
  assert.equal(
    prepareDecimaQuestion(
      '',
      'pick',
      Array.from({length: 257}, (_, index) => `option-${index}`),
      config,
      tokenizer
    ).choices.length,
    257
  );
});

test('Decima choice-fit keeps a short question intact and only marks adapted rows truncated', () => {
  const baseCleanConfig = {...config, state_prefix: '', choice_prefix: '', choice_fit: true};
  const prepared = prepareDecimaQuestion('state', 'Which action?', ['allow', 'deny'], baseCleanConfig, makeTokenizer());
  assert.equal(prepared.truncated, false);
  assert.deepEqual(prepared.choiceIds[0], [0, ...makeTokenizer().encode('Which action? allow'), 2]);
});

test('Decima language normalization passes unknown codes through and normalizes Arabic scripts', () => {
  const tokenizer = makeTokenizer();
  const persian = prepareDecimaQuestion('كيك', 'راهنما', ['الف', 'ب'], config, tokenizer, 'fa');
  const arabic = prepareDecimaQuestion('كيك', 'راهنما', ['الف', 'ب'], config, tokenizer, 'ar');
  assert.notDeepEqual(persian.stateIds, arabic.stateIds);
  assert.doesNotThrow(() => prepareDecimaQuestion('', 'question', ['a', 'b'], config, tokenizer, 'xx'));
  assert.throws(() => prepareDecimaQuestion('', 'question', ['a', 'b'], config, tokenizer, ' '), {
    code: 'INVALID_INPUT'
  });
});

test('Decima decoders preserve softmax, ordinal and independent-rank probabilities', () => {
  const raw = {scores: [1, 0, -1], features: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]};
  const rank = decodeDecimaScores(raw, config, 'rank');
  assert.ok(rank.every((value) => value > 0 && value < 1));
  assert.notEqual(
    rank.reduce((sum, value) => sum + value, 0),
    1
  );
  for (const kind of ['choice', 'score']) {
    const probabilities = decodeDecimaScores(raw, config, kind);
    assert.ok(probabilities.every((value) => value >= 0 && Number.isFinite(value)));
    assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  }
  const answer = finishDecimaChoice({choices: ['first', 'second', 'third']}, [0.1, 0.7, 0.2], [0, 1, 0.5]);
  assert.equal(answer.selected, 'second');
  assert.equal(answer.selectedIndex, 1);
  assert.equal(answer.probabilities[1].logit, 1);
  assert.throws(() => decodeDecimaScores({scores: [NaN]}, config, 'choice'), {code: 'INFERENCE_ERROR'});
  assert.throws(() => decodeDecimaScores({scores: [1, 0], features: []}, config, 'score'), {
    code: 'INFERENCE_ERROR'
  });
  assert.throws(() => finishDecimaChoice({choices: ['x']}, [1], []), {code: 'INFERENCE_ERROR'});
});

test('Decima System One prepares choice, ordinal choice, score, noul and rank without collapsing rank scores', () => {
  const prepared = prepareDecimaSystemOne(
    {ticket: 'charged twice'},
    {
      route: {type: 'choice', instructions: 'Which team?', criteria: {billing: 'payment', support: 'technical issue'}},
      priority: {type: 'choice', ordinal: true, instructions: 'Choose priority.', criteria: ['low', 'high']},
      urgency: {type: 'score', instructions: 'How urgent?', criteria: ['low', 'medium', 'high']},
      valid: {type: 'noul', instructions: 'Is it valid?', criteria: {true: 'matches policy'}},
      steps: {type: 'rank', instructions: 'Rank these steps.', criteria: ['refund', 'investigate', 'escalate']}
    }
  );
  assert.equal(prepared.state, '{"ticket":"charged twice"}');
  assert.equal(prepared.questions[1].type, 'score');
  assert.equal(prepared.questions[1].outputType, 'choice');
  assert.match(prepared.questions[3].instructions, /False if: —/);
  assert.equal(prepared.questions[4].outputType, 'rank');
  const answers = prepared.questions.map((question, index) => ({
    probabilities:
      question.type === 'noul'
        ? [0.8, 0.2]
        : question.type === 'rank'
          ? [0.8, 0.6, 0.2]
          : [0.2, 0.5, 0.3].slice(0, question.labels.length),
    tokenCount: 20,
    truncated: index === 0
  }));
  const result = finishDecimaSystemOne(prepared, answers);
  assert.equal(result.model, DECIMA_SMALL_MODEL);
  assert.equal(result.tokenCount, 100);
  assert.equal(result.truncated, true);
  assert.equal(result.answers.route.type, 'choice');
  assert.equal(result.answers.urgency.type, 'score');
  assert.equal(result.answers.valid.type, 'noul');
  assert.equal(result.answers.steps.type, 'rank');
  assert.deepEqual(result.answers.steps.order, ['refund', 'investigate', 'escalate']);
  assert.equal(finishDecimaSystemOne(prepared, answers, DECIMA_BASE_CLEAN_MODEL).model, DECIMA_BASE_CLEAN_MODEL);
  assert.ok(
    Math.abs(Object.values(result.answers.steps.probabilities).reduce((sum, value) => sum + value, 0) - 1.6) < 1e-12
  );
});

test('Decima System One keeps per-question limits without a request question or option cap', () => {
  const question = {type: 'choice', criteria: ['a', 'b']};
  const questions = Object.fromEntries(Array.from({length: 257}, (_, index) => [`q${index}`, question]));
  assert.equal(prepareDecimaSystemOne('state', questions).questions.length, 257);
  assert.equal(
    prepareDecimaSystemOne('state', {many: {type: 'choice', criteria: Array.from({length: 257}, (_, i) => `c${i}`)}})
      .questions[0].choices.length,
    257
  );
  assert.throws(() => prepareDecimaSystemOne('state', {q: {type: 'score', criteria: ['a']}}), {code: 'INVALID_INPUT'});
  assert.throws(() => prepareDecimaSystemOne('state', {q: {type: 'score', criteria: Array(11).fill('level')}}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => prepareDecimaSystemOne('state', {q: {type: 'choice', criteria: ['same', 'same']}}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => prepareDecimaSystemOne('state', {q: {type: 'other', criteria: ['a', 'b']}}), {
    code: 'INVALID_INPUT'
  });
  const accessor = {};
  Object.defineProperty(accessor, 'type', {
    get() {
      throw new Error('must not evaluate');
    }
  });
  assert.throws(() => prepareDecimaSystemOne('state', {q: accessor}), {code: 'INVALID_INPUT'});
});

test('Decima runner pads int64 token inputs, checks graph outputs and releases both sessions', async () => {
  const calls = [];
  const encoder = {
    async run(feeds) {
      calls.push(feeds);
      const [batch, width] = feeds.input_ids.dims;
      return {token_states: new FakeTensor('float32', new Float32Array(batch * width * 2), [batch, width, 2])};
    },
    async release() {
      calls.push('release-encoder');
    }
  };
  const scorer = {
    async run(feeds) {
      calls.push(feeds);
      const count = feeds.choice_h.dims[0];
      return {
        scores: new FakeTensor(
          'float32',
          Float32Array.from({length: count}, (_, index) => index),
          [count]
        ),
        z: new FakeTensor('float32', new Float32Array(count * 2), [count, 2])
      };
    },
    async release() {
      calls.push('release-scorer');
    }
  };
  const runner = createDecimaRunner({Tensor: FakeTensor}, encoder, scorer, makeTokenizer());
  const result = await runner.score([[1, 2]], [[3], [4, 5]]);
  assert.deepEqual(result.scores, [0, 1]);
  const input = calls.find((call) => call.input_ids?.dims[0] === 2);
  assert.equal(input.input_ids.type, 'int64');
  assert.deepEqual(input.input_ids.dims, [2, 2]);
  assert.equal(input.input_ids.data[1], 1n);
  assert.equal(input.attention_mask.data[1], 0n);
  await runner.dispose();
  assert.ok(calls.includes('release-encoder'));
  assert.ok(calls.includes('release-scorer'));
  await assert.rejects(runner.encode([[1]]), {code: 'SESSION_CLOSED'});
});

test('Decima load options accept HTTP(S) assets and reject unsupported fields', () => {
  assert.equal(
    validateDecimaLoadOptions({model: DECIMA_SMALL_MODEL, assetsUrl: 'https://example.test/model'}).base.href,
    'https://example.test/model/'
  );
  const baseClean = validateDecimaLoadOptions({
    model: DECIMA_BASE_CLEAN_MODEL,
    assetsUrl: 'https://example.test/base-clean/'
  });
  assert.equal(baseClean.model, DECIMA_BASE_CLEAN_MODEL);
  assert.equal(baseClean.spec.stateOverflow, 'reject');
  assert.throws(() => validateDecimaLoadOptions(null), {code: 'INVALID_INPUT'});
  assert.throws(() => validateDecimaLoadOptions({model: DECIMA_SMALL_MODEL, assetsUrl: 'file:///model/'}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(
    () =>
      validateDecimaLoadOptions({model: DECIMA_SMALL_MODEL, assetsUrl: 'https://example.test/', maxPrefetchBytes: 1}),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      validateDecimaLoadOptions({
        model: DECIMA_SMALL_MODEL,
        assetsUrl: 'https://example.test/',
        offline: true,
        cache: false
      }),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () => validateDecimaLoadOptions({model: DECIMA_SMALL_MODEL, assetsUrl: 'https://example.test/', gpu: true}),
    {code: 'INVALID_INPUT'}
  );
});

test('Decima Agent pins its own int8 checkpoint and 8,192-token CPU profile', () => {
  const checked = validateDecimaLoadOptions({
    model: DECIMA_AGENT_MODEL,
    assetsUrl: 'https://example.test/decima-agent/'
  });
  assert.equal(checked.spec.repository, 'amyrmahdy/decima-agent');
  assert.equal(checked.spec.revision, '1acd2c4b1461c15ebc6b2b8f84dd9f8af18be245');
  assert.equal(checked.spec.maxStateTokens, 4096);
  assert.equal(checked.spec.sourceMaxStateTokens, 8192);
  assert.equal(checked.spec.maxChoiceTokens, 64);
  assert.equal(checked.spec.temperature, DECIMA_AGENT_TEMPERATURE);
  assert.equal(checked.spec.stateOverflow, 'reject');
  assert.equal(checked.spec.choiceFit, true);
  assert.deepEqual(checked.spec.typedQuestionTypes, ['choice', 'score', 'noul']);
});

test('Decima loader uses pinned WASM sessions and returns a usable decision API', async () => {
  const ort = fakeOrt();
  const files = new Map([
    ['tokenizer/tokenizer.json', new TextEncoder().encode(JSON.stringify(tokenizerData))],
    ['tokenizer/tokenizer_config.json', new TextEncoder().encode(JSON.stringify(tokenizerConfig))],
    ['decima.json', new TextEncoder().encode(JSON.stringify(config))],
    ['encoder.onnx', new Uint8Array([1]).buffer],
    ['scorer.onnx', new Uint8Array([2]).buffer]
  ]);
  const reads = [];
  const model = await loadDecimaModel(
    {model: DECIMA_SMALL_MODEL, assetsUrl: 'https://example.test/assets/'},
    {
      ort,
      Tokenizer: FakeTokenizer,
      async openSource() {
        return {
          read: async (name) => {
            reads.push(name);
            return files.get(name);
          }
        };
      }
    }
  );
  assert.deepEqual(reads.sort(), [...files.keys()].sort());
  assert.equal(ort.options.length, 2);
  assert.ok(
    ort.options.every((options) => options.executionProviders[0] === 'wasm' && options.intraOpNumThreads === 1)
  );
  const decision = await model.decide({
    state: 'ticket',
    questions: [{instructions: 'Choose team.', options: ['billing', 'security']}]
  });
  assert.equal(decision.answers[0].selected, 'security');
  assert.equal(decision.answers[0].probabilities.length, 2);
  const manyQuestions = Array.from({length: 257}, () => ({instructions: 'Choose.', options: ['yes', 'no']}));
  const manyAnswers = await model.decide({state: 'ticket', questions: manyQuestions});
  assert.equal(manyAnswers.answers.length, 257);
  const manyOptions = Array.from({length: 257}, (_, index) => `option-${index}`);
  const manyOptionAnswer = await model.decide({
    state: 'ticket',
    questions: [{instructions: 'Choose.', options: manyOptions}]
  });
  assert.equal(manyOptionAnswer.answers[0].probabilities.length, 257);
  const typedQuestions = Object.fromEntries(
    Array.from({length: 257}, (_, index) => [`q${index}`, {type: 'choice', criteria: ['yes', 'no']}])
  );
  const manyTypedAnswers = await model.systemOne('ticket', typedQuestions);
  assert.equal(Object.keys(manyTypedAnswers.answers).length, 257);
  const manyTypedOptions = await model.systemOne('ticket', {
    many: {type: 'choice', criteria: manyOptions}
  });
  assert.equal(Object.keys(manyTypedOptions.answers.many.probabilities).length, 257);
  const languageDecision = await model.decide(
    {state: 'account activity', questions: [{instructions: 'Choose team.', options: ['billing', 'security']}]},
    {language: 'xx'}
  );
  assert.equal(languageDecision.answers[0].selected, 'security');
  const typed = await model.systemOne(
    'ticket',
    {
      route: {type: 'choice', instructions: 'Choose team.', criteria: {billing: 'money', security: 'account access'}},
      truth: {type: 'noul', instructions: 'Is this accurate?'},
      score: {type: 'score', instructions: 'How urgent?', criteria: ['low', 'medium', 'high']},
      rank: {type: 'rank', instructions: 'Rank next steps.', criteria: ['refund', 'investigate', 'escalate']}
    },
    {language: 'xx'}
  );
  assert.equal(typed.answers.route.type, 'choice');
  assert.equal(typed.answers.truth.type, 'noul');
  assert.equal(typed.answers.score.type, 'score');
  assert.equal(typed.answers.rank.type, 'rank');
  let languageGetterRan = false;
  const invalidOptions = {};
  Object.defineProperty(invalidOptions, 'language', {
    get() {
      languageGetterRan = true;
      throw new Error('must not evaluate');
    }
  });
  await assert.rejects(
    model.decide({state: 'ticket', questions: [{instructions: 'Choose.', options: ['a', 'b']}]}, invalidOptions),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(languageGetterRan, false);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    model.decide(
      {state: 'ticket', questions: [{instructions: 'Choose.', options: ['a', 'b']}]},
      {signal: controller.signal}
    )
  );
  await model.releaseCachedMemory();
  await model.dispose();
  assert.deepEqual(ort.released.sort(), ['encoder', 'scorer']);
  await assert.rejects(model.decide({state: 'ticket', questions: [{instructions: 'Choose.', options: ['a', 'b']}]}), {
    code: 'SESSION_CLOSED'
  });
});

test('Decima base-clean loader keeps its model ID and rejects long state', async () => {
  const ort = fakeOrt();
  const baseCleanConfig = {
    ...config,
    state_prefix: '',
    choice_prefix: '',
    temperature: 0.6929793512269696,
    choice_fit: true
  };
  const files = new Map([
    ['tokenizer/tokenizer.json', new TextEncoder().encode(JSON.stringify(tokenizerData))],
    ['tokenizer/tokenizer_config.json', new TextEncoder().encode(JSON.stringify(tokenizerConfig))],
    ['decima.json', new TextEncoder().encode(JSON.stringify(baseCleanConfig))],
    ['encoder.onnx', new Uint8Array([1]).buffer],
    ['scorer.onnx', new Uint8Array([2]).buffer]
  ]);
  const model = await loadDecimaModel(
    {model: DECIMA_BASE_CLEAN_MODEL, assetsUrl: 'https://example.test/base-clean/'},
    {
      ort,
      Tokenizer: FakeTokenizer,
      async openSource() {
        return {read: async (name) => files.get(name)};
      }
    }
  );
  const typed = await model.systemOne('ticket', {route: {type: 'choice', criteria: ['billing', 'security']}});
  assert.equal(typed.model, DECIMA_BASE_CLEAN_MODEL);
  await assert.rejects(
    model.decide({state: 'x'.repeat(520), questions: [{instructions: 'Choose.', options: ['a', 'b']}]}),
    {code: 'INVALID_INPUT'}
  );
  await model.dispose();
});

test('Decima Agent loader preserves model ID, typed choices and rejects unsupported rank', async () => {
  const ort = fakeOrt();
  const agentConfig = {
    ...config,
    state_prefix: '',
    choice_prefix: '',
    max_state_tokens: 8192,
    temperature: DECIMA_AGENT_TEMPERATURE,
    choice_fit: true
  };
  const files = new Map([
    ['tokenizer/tokenizer.json', new TextEncoder().encode(JSON.stringify(tokenizerData))],
    ['tokenizer/tokenizer_config.json', new TextEncoder().encode(JSON.stringify(tokenizerConfig))],
    ['decima.json', new TextEncoder().encode(JSON.stringify(agentConfig))],
    ['encoder.onnx', new Uint8Array([1]).buffer],
    ['scorer.onnx', new Uint8Array([2]).buffer]
  ]);
  const model = await loadDecimaModel(
    {model: DECIMA_AGENT_MODEL, assetsUrl: 'https://example.test/decima-agent/'},
    {
      ort,
      Tokenizer: FakeTokenizer,
      async openSource() {
        return {read: async (name) => files.get(name)};
      }
    }
  );
  const typed = await model.systemOne('state', {gate: {type: 'choice', criteria: ['allow', 'ask', 'deny']}});
  assert.equal(typed.model, DECIMA_AGENT_MODEL);
  assert.equal(typed.answers.gate.choice, 'deny');
  await assert.rejects(model.systemOne('state', {gate: {type: 'rank', criteria: ['allow', 'deny']}}), {
    code: 'INVALID_INPUT'
  });
  await model.dispose();
});
