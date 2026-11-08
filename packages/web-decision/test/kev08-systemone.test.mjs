import test from 'node:test';
import assert from 'node:assert/strict';
import {packKev08Request} from '../src/kev08.js';
import {prepareKev08SystemOne, finishKev08SystemOne} from '../src/kev08/system-one.mjs';
import {ExperimentalKev08Session, validateKev08Row} from '../src/kev08/session.js';

function createTokenizer() {
  return {
    encode(text) {
      return {ids: Array.from(text, (character) => character.codePointAt(0) + 200)};
    }
  };
}

const limits = {maxInputTokens: 8192, maxStateTokens: 4096};

function prepare(state, questions) {
  return prepareKev08SystemOne(state, questions);
}

function sessionFixture() {
  const events = [];
  const lifecycle = {loss: null, disposed: false};
  const session = new ExperimentalKev08Session({
    runtime: {
      endScope() {
        events.push('end');
      },
      dispose() {
        events.push('dispose');
        lifecycle.disposed = true;
      }
    },
    device: {},
    gpu: {},
    fn: null,
    weights: [],
    tokenizer: createTokenizer(),
    limits,
    uncaptured: [],
    lifecycle
  });
  return {session, events};
}

test('Kev 0.8B systemOne renders JSON state and typed questions with the upstream protocol', () => {
  const prepared = prepare(
    {
      ticket: {subject: 'Refund', messages: ['first', {text: 'second'}]},
      open: true,
      empty: null
    },
    {
      department: {type: 'choice', instructions: {task: 'Pick one.'}, criteria: {billing: 'charges', support: null}},
      score: {type: 'score', criteria: ['low', 'high', 'high']},
      refund: {type: 'noul', criteria: {true: false, ignored: 'extra'}}
    }
  );

  assert.equal(
    prepared.request.state,
    'ticket:\n  subject: Refund\n  messages:\n    - first\n    - text: second\nopen: True\nempty: '
  );
  assert.deepEqual(prepared.request.questions, [
    {instructions: 'task: Pick one.', options: ['billing: charges', 'support']},
    {instructions: '', options: ['low', 'high', 'high']},
    {instructions: '', options: ['no', 'yes: False']}
  ]);
  assert.deepEqual(prepared.metadata, [
    {name: 'department', type: 'choice', keys: ['billing', 'support']},
    {name: 'score', type: 'score', legend: ['low', 'high', 'high']},
    {name: 'refund', type: 'noul'}
  ]);
  assert.deepEqual([...prepared.packing.allowDuplicateOptionsFor], [1]);
});

test('typed results preserve Kev Choice, Score, and Noul outputs and upstream rounding', () => {
  const prepared = prepare('state', {
    choice: {type: 'choice', criteria: {red: null, blue: 'cool'}},
    score: {type: 'score', instructions: 'Rate.', criteria: ['low', 'mid', 'high']},
    noul: {type: 'noul', instructions: null}
  });
  const result = finishKev08SystemOne(prepared, {
    tokenCount: 123,
    answers: [
      {
        selectedIndex: 1,
        probabilities: [{probability: 0.1}, {probability: 0.9}]
      },
      {
        selectedIndex: 2,
        probabilities: [{probability: 0.2}, {probability: 0.3}, {probability: 0.5}]
      },
      {
        selectedIndex: 1,
        probabilities: [{probability: 0.012344}, {probability: 0.987656}]
      }
    ]
  });

  assert.equal(result.model, 'kev-0.8b-f16');
  assert.equal(result.tokenCount, 123);
  assert.deepEqual(result.answers.choice, {
    type: 'choice',
    choice: 'blue',
    confidence: 0.8,
    probabilities: {red: 0.1, blue: 0.9}
  });
  assert.deepEqual(result.answers.score, {
    type: 'score',
    score: 1.3,
    legend: {0: 'low', 1: 'mid', 2: 'high'},
    probabilities: {0: 0.2, 1: 0.3, 2: 0.5},
    confidence: 0.65
  });
  assert.deepEqual(result.answers.noul, {type: 'noul', noul: 0.9877});
});

test('supports upstream one-option Choice and Score plus duplicate Score levels', () => {
  const prepared = prepare('state', {
    choice: {type: 'choice', criteria: {only: null}},
    score: {type: 'score', criteria: ['same', 'same']}
  });
  const {rows} = packKev08Request(createTokenizer(), prepared.request, limits, prepared.packing);

  assert.equal(rows[0].options.length, 1);
  assert.equal(rows[1].options.length, 2);
  assert.deepEqual(rows[1].options, ['same', 'same']);
  assert.doesNotThrow(() => validateKev08Row(rows[0], limits.maxInputTokens));
  assert.doesNotThrow(() => validateKev08Row(rows[1], limits.maxInputTokens));
  assert.throws(() => packKev08Request(createTokenizer(), prepared.request, limits), {code: 'INVALID_INPUT'});
});

test('respects the 1–255 typed option and level limits without truncating', () => {
  const one = prepare('state', {
    choice: {type: 'choice', criteria: {only: null}},
    score: {type: 'score', criteria: ['one']}
  });
  assert.equal(one.request.questions[0].options.length, 1);
  assert.equal(one.request.questions[1].options.length, 1);

  const maximum = prepare('state', {
    choice: {type: 'choice', criteria: Object.fromEntries(Array.from({length: 255}, (_, i) => [`c${i}`, null]))},
    score: {type: 'score', criteria: Array.from({length: 255}, (_, i) => `level ${i}`)}
  });
  assert.equal(maximum.request.questions[0].options.length, 255);
  assert.equal(maximum.request.questions[1].options.length, 255);

  for (const count of [0, 256]) {
    assert.throws(
      () =>
        prepare('state', {
          choice: {type: 'choice', criteria: Object.fromEntries(Array.from({length: count}, (_, i) => [`c${i}`, null]))}
        }),
      {code: 'INVALID_INPUT'}
    );
    assert.throws(
      () => prepare('state', {score: {type: 'score', criteria: Array.from({length: count}, (_, i) => `level ${i}`)}}),
      {code: 'INVALID_INPUT'}
    );
  }
});

test('rejects accessors, sparse arrays, invalid schemas, cycles, and non-finite JSON', () => {
  const getter = {};
  Object.defineProperty(getter, 'type', {
    enumerable: true,
    get() {
      throw Error('must not run');
    }
  });
  assert.throws(() => prepare('state', {bad: getter}), {code: 'INVALID_INPUT'});

  const sparse = new Array(2);
  assert.throws(() => prepare('state', {score: {type: 'score', criteria: sparse}}), {code: 'INVALID_INPUT'});
  assert.throws(() => prepare('state', {choice: {type: 'choice', criteria: []}}), {code: 'INVALID_INPUT'});
  assert.throws(() => prepare('state', {choice: {type: 'choice', criteria: {}}}), {code: 'INVALID_INPUT'});
  assert.throws(() => prepare('state', {bad: {type: 'noul', criteria: []}}), {code: 'INVALID_INPUT'});
  assert.throws(() => prepare('state', {bad: {type: 'other'}}), {code: 'INVALID_INPUT'});

  const cycle = {};
  cycle.self = cycle;
  assert.throws(() => prepare(cycle, {pick: {type: 'choice', criteria: {yes: null, no: null}}}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => prepare({value: Number.NaN}, {pick: {type: 'noul'}}), {code: 'INVALID_INPUT'});
});

test('systemOne snapshots typed requests, skips queued work on abort, and disposal waits for active work', async () => {
  const {session, events} = sessionFixture();
  let signalStarted;
  const started = new Promise((resolve) => {
    signalStarted = resolve;
  });
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  let calls = 0;
  session.run = async () => {
    calls++;
    signalStarted();
    await gate;
    return {
      tokenCount: 1,
      answers: [
        {selected: 'yes', selectedIndex: 0, probabilities: [{probability: 1, logit: 0}]},
        {
          selected: 'yes',
          selectedIndex: 1,
          probabilities: [
            {probability: 0.1, logit: 0},
            {probability: 0.9, logit: 1}
          ]
        }
      ]
    };
  };
  const questions = {
    first: {type: 'choice', criteria: {yes: null}},
    second: {type: 'noul'}
  };
  const controller = new AbortController();
  const result = session.systemOne('state', questions, {signal: controller.signal});
  questions.first.criteria.yes = 'changed';
  await started;
  controller.abort(Error('cancel typed request'));
  await assert.rejects(result, /cancel typed request/);
  const disposal = session.dispose();
  assert.deepEqual(events, []);
  release();
  await disposal;
  assert.equal(calls, 1);
  assert.deepEqual(events, ['end', 'dispose']);
});
