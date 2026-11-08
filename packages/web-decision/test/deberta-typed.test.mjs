import test from 'node:test';
import assert from 'node:assert/strict';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {WebDecisionError} from '../src/kev.js';

function typedSession() {
  const model = new ExperimentalDebertaSession({});
  let request;
  model.decide = async (value) => {
    request = value;
    return {
      tokenCount: 41,
      answers: [
        {
          selected: 'refund',
          selectedIndex: 1,
          probabilities: [
            {option: 'fees', probability: 0.25, logit: 0},
            {option: 'refund', probability: 0.75, logit: 1}
          ]
        },
        {
          selected: 'high',
          selectedIndex: 2,
          probabilities: [
            {option: 'low', probability: 0.1, logit: 0},
            {option: 'medium', probability: 0.2, logit: 1},
            {option: 'high', probability: 0.7, logit: 2}
          ]
        },
        {
          selected: 'yes',
          selectedIndex: 1,
          probabilities: [
            {option: 'no', probability: 0.2, logit: 0},
            {option: 'yes', probability: 0.8, logit: 1}
          ]
        }
      ]
    };
  };
  return {
    model,
    get request() {
      return request;
    }
  };
}

test('DeBERTa systemOne preserves Open-Jev questions and decodes typed distributions', async () => {
  const context = typedSession();
  const {model} = context;
  const state = 'The customer was charged twice and wants a refund.';
  const result = await model.systemOne(state, {
    area: {type: 'choice', instructions: 'Which area?', options: ['fees', 'refund']},
    tone: {type: 'score', instructions: 'How negative?', options: ['low', 'medium', 'high']},
    wantsRefund: {type: 'noul', instructions: 'The customer wants money back.'}
  });

  assert.deepEqual(context.request, {
    state,
    questions: [
      {instructions: 'Which area?', options: ['fees', 'refund']},
      {instructions: 'How negative?', options: ['low', 'medium', 'high']},
      {instructions: 'The customer wants money back.', options: ['no', 'yes']}
    ]
  });
  assert.equal(result.model, 'open-jev-deberta-v3-large-q4f16');
  assert.equal(result.tokenCount, 41);
  assert.deepEqual(result.answers.area, {
    type: 'choice',
    choice: 'refund',
    confidence: 0.75,
    probabilities: {fees: 0.25, refund: 0.75}
  });
  assert.equal(result.answers.tone.type, 'score');
  assert.ok(Math.abs(result.answers.tone.score - 1.6) < 1e-12);
  assert.deepEqual(result.answers.tone.legend, {0: 'low', 1: 'medium', 2: 'high'});
  assert.deepEqual(result.answers.tone.probabilities, {0: 0.1, 1: 0.2, 2: 0.7});
  assert.equal(result.answers.tone.confidence, 0.7);
  assert.deepEqual(result.answers.wantsRefund, {
    type: 'noul',
    noul: 0.8,
    probabilities: {false: 0.2, true: 0.8},
    confidence: 0.8
  });
});

test('DeBERTa systemOne does not impose a fixed named-question count', async () => {
  const model = new ExperimentalDebertaSession({});
  model.decide = async (request) => ({
    tokenCount: 1,
    answers: request.questions.map(() => ({
      selectedIndex: 1,
      probabilities: [{probability: 0.25}, {probability: 0.75}]
    }))
  });
  const questions = Object.fromEntries(
    Array.from({length: 513}, (_, index) => [
      `question${index}`,
      {type: 'noul', instructions: `Is statement ${index} true?`}
    ])
  );

  const result = await model.systemOne('', questions);
  assert.equal(Object.keys(result.answers).length, 513);
  assert.equal(result.answers.question512.noul, 0.75);
});

test('DeBERTa systemOne enforces its native question schema and option limits before inference', async () => {
  const {model} = typedSession();
  const invalid = [
    [42, {q: {type: 'choice', instructions: 'Choose.', options: ['a', 'b']}}],
    ['state', {}],
    ['state', {q: {type: 'choice', instructions: 'Choose.', options: ['only']}}],
    ['state', {q: {type: 'choice', instructions: 'Choose.', options: Array.from({length: 256}, (_, i) => `o${i}`)}}],
    ['state', {q: {type: 'score', instructions: 'Rate.', options: ['one']}}],
    ['state', {q: {type: 'score', instructions: 'Rate.', options: Array.from({length: 11}, (_, i) => `l${i}`)}}],
    ['state', {q: {type: 'score', instructions: 'Rate.', options: ['same', 'same']}}],
    ['state', {q: {type: 'noul', instructions: 'Is it true?', options: ['no', 'yes']}}],
    ['state', {q: {type: 'unknown', instructions: 'Choose.', options: ['a', 'b']}}],
    ['state', {q: {type: 'choice', instructions: 'Choose.', criteria: ['a', 'b']}}]
  ];

  for (const [state, questions] of invalid)
    await assert.rejects(model.systemOne(state, questions), (error) => error.code === 'INVALID_INPUT');
});

test('DeBERTa systemOne rejects getters and sparse option arrays without reading them', async () => {
  const {model} = typedSession();
  let getterCalls = 0;
  const questions = Object.defineProperty({}, 'q', {
    enumerable: true,
    get() {
      getterCalls++;
      return {type: 'choice', instructions: 'Choose.', options: ['a', 'b']};
    }
  });
  await assert.rejects(model.systemOne('state', questions), {code: 'INVALID_INPUT'});
  assert.equal(getterCalls, 0);

  const sparse = new Array(2);
  sparse[1] = 'b';
  await assert.rejects(model.systemOne('state', {q: {type: 'choice', instructions: 'Choose.', options: sparse}}), {
    code: 'INVALID_INPUT'
  });
});

test('DeBERTa systemOne checks disposal before parsing new questions', async () => {
  const {model} = typedSession();
  model.closed = true;
  await assert.rejects(
    model.systemOne('state', null),
    (error) => error instanceof WebDecisionError && error.code === 'SESSION_CLOSED'
  );
});
