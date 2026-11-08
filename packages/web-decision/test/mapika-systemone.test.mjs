import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareMapikaSystemOne, finishMapikaSystemOne} from '../src/mapika/system-one.mjs';
import {ExperimentalMapikaSession} from '../src/mapika/session.js';

function createTokenizer() {
  return {
    encode(text) {
      return {ids: Array.from(text, (character) => character.codePointAt(0) + 200)};
    }
  };
}

function createLabels() {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const names = [...alphabet];
  for (const first of alphabet) {
    for (const second of alphabet) {
      names.push(first + second);
      if (names.length === 255) break;
    }
    if (names.length === 255) break;
  }
  return {
    names,
    ids: Array.from({length: 255}, (_, index) => 1000 + index),
    open: [900]
  };
}

const limits = {maxInputTokens: 8192, maxStateTokens: 4096, controlIds: [999999], temperature: 1.03};

function prepare(state, questions) {
  return prepareMapikaSystemOne(state, questions, createTokenizer(), createLabels(), limits);
}

function answer(probabilities) {
  return {
    answers: [
      {
        probabilities: probabilities.map((probability, index) => ({option: String(index), probability, logit: 0}))
      }
    ]
  };
}

function sessionFixture() {
  const events = [];
  const lifecycle = {loss: null, disposed: false};
  const session = new ExperimentalMapikaSession({
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
    labels: createLabels(),
    limits,
    uncaptured: [],
    lifecycle,
    modelId: 'mapika-0.8b-q4f16'
  });
  return {session, events};
}

test('JSON state annotation handles long arrays, nested objects, and an existing _index key', () => {
  const records = Array.from({length: 8}, (_, index) => ({id: index, _index: `source ${index}`, text: '한글'}));
  const state = {records, short: [1, 2]};
  const expected = JSON.stringify(
    {
      records: records.map(({id, _index, text}) => ({_index, id, text})),
      short: [1, 2]
    },
    null,
    0
  )
    .replaceAll(',', ', ')
    .replaceAll(':', ': ');
  const prepared = prepare(state, {pick: {instructions: 'Choose a record.', criteria: {first: null, second: null}}});
  const stateIds = createTokenizer().encode(`Context:\n${expected}`).ids;
  assert.deepEqual(prepared.rows[0].packed.ids.slice(0, stateIds.length), stateIds);
});

test('string state is kept verbatim and JSON state keeps Unicode and insertion order', () => {
  const state = {second: '안녕', first: 2};
  const prepared = prepare(state, {pick: {question: 'Choose.', options: ['안녕', '다음']}});
  const expected = '{"second": "안녕", "first": 2}';
  assert.deepEqual(
    prepared.rows[0].packed.ids.slice(0, createTokenizer().encode(`Context:\n${expected}`).ids.length),
    createTokenizer().encode(`Context:\n${expected}`).ids
  );

  const text = ' raw state stays as provided ';
  const raw = prepare(text, {pick: {instructions: 'Choose.', criteria: {one: null, two: null}}});
  assert.deepEqual(
    raw.rows[0].packed.ids.slice(0, createTokenizer().encode(`Context:\n${text}`).ids.length),
    createTokenizer().encode(`Context:\n${text}`).ids
  );
});

test('JSON state formats extreme finite numbers like Python json and list option keys like Python str', () => {
  const state = {small: 1e-5, large: 1e16, negativeZero: -0};
  const expected = '{"small": 1e-05, "large": 1e+16, "negativeZero": -0.0}';
  const prepared = prepare(state, {
    choose: {instructions: 'Choose.', criteria: [['nested', true, null], {label: 'value'}]}
  });
  const expectedState = createTokenizer().encode(`Context:\n${expected}`).ids;
  assert.deepEqual(prepared.rows[0].packed.ids.slice(0, expectedState.length), expectedState);
  assert.deepEqual(prepared.rows[0].packed.groups[0].options, ["['nested', True, None]", "{'label': 'value'}"]);
});

test('question planning follows Mapika Choice list, isolated Score, and bool aliases', () => {
  const prepared = prepare('state', {
    choice: {type: 'choice', question: 'Pick one.', options: ['red', 'blue', 'red']},
    score: {type: 'score', instructions: 'How good?', criteria: ['0: low', '1: high']},
    flag: {type: 'bool', instructions: 'Is it true?', criteria: {true: 'yes case', false: 'no case'}}
  });
  assert.equal(prepared.rows.length, 4);
  assert.deepEqual(prepared.questions[0], {name: 'choice', type: 'choice', options: ['red', 'blue']});
  assert.match(
    String.fromCodePoint(...prepared.rows[1].packed.ids.map((id) => id - 200)),
    /Proposed answer: low\nDoes the proposed answer fit\?/
  );
  assert.match(
    String.fromCodePoint(...prepared.rows[2].packed.ids.map((id) => id - 200)),
    /Proposed answer: high\nDoes the proposed answer fit\?/
  );
  assert.deepEqual(prepared.questions[2], {name: 'flag', type: 'noul'});
  assert.equal(prepare('state', {flag: {type: 'noul', instructions: 'Check?'}}).rows.length, 1);
  assert.throws(() => prepare('state', {flag: {type: 'noul', instructions: 'Check?', criteria: []}}), {
    code: 'INVALID_INPUT'
  });
});

test('Noul can derive its question from true/false descriptions like TypeSafe', () => {
  const prepared = prepare('state', {flag: {type: 'noul', criteria: {true: 'fits', false: 'does not fit'}}});
  const prompt = String.fromCodePoint(...prepared.rows[0].packed.ids.map((id) => id - 200));
  assert.match(prompt, /Which answer fits the context\?/);
  assert.throws(() => prepare('state', {flag: {type: 'noul'}}), {code: 'INVALID_INPUT'});
});

test('typed limits retain Mapika option and score bounds without truncation', () => {
  for (const optionCount of [2, 10, 11, 255]) {
    const criteria = Array.from({length: optionCount}, (_, index) => `option-${index}`);
    const prepared = prepare('state', {pick: {instructions: 'Choose.', criteria}});
    assert.equal(prepared.rows[0].packed.groups[0].options.length, optionCount);
  }
  for (const levelCount of [2, 10]) {
    const criteria = Array.from({length: levelCount}, (_, index) => `level-${index}`);
    const prepared = prepare('state', {rate: {type: 'score', instructions: 'Rate.', criteria}});
    assert.equal(prepared.rows.length, levelCount);
  }
  for (const levelCount of [1, 11]) {
    const criteria = Array.from({length: levelCount}, (_, index) => `level-${index}`);
    assert.throws(() => prepare('state', {rate: {type: 'score', instructions: 'Rate.', criteria}}), {
      code: 'INVALID_INPUT'
    });
  }
});

test('rejects accessors, sparse JSON arrays, cycles, and typed requests beyond the row bound', () => {
  const getter = {};
  Object.defineProperty(getter, 'instructions', {
    enumerable: true,
    get() {
      throw Error('must not run');
    }
  });
  assert.throws(() => prepare('state', {bad: getter}), {code: 'INVALID_INPUT'});

  const sparse = new Array(2);
  assert.throws(() => prepare('state', {bad: {type: 'choice', instructions: 'Pick?', criteria: sparse}}), {
    code: 'INVALID_INPUT'
  });

  const cycle = {};
  cycle.self = cycle;
  assert.throws(() => prepare(cycle, {pick: {instructions: 'Pick?', criteria: {a: null, b: null}}}), {
    code: 'INVALID_INPUT'
  });

  const tooManyRows = Object.fromEntries(
    Array.from({length: 52}, (_, index) => [
      `score-${index}`,
      {type: 'score', instructions: 'Rate?', criteria: Array.from({length: 10}, (_, level) => `level-${level}`)}
    ])
  );
  assert.throws(() => prepare('state', tooManyRows), {code: 'INVALID_INPUT'});
});

test('Mapika typed results keep the model contract and round values like upstream', () => {
  const prepared = prepare('state', {
    choice: {instructions: 'Pick.', criteria: {a: null, b: null}},
    score: {type: 'score', instructions: 'Rate.', criteria: ['low', 'mid', 'high']},
    yes: {type: 'noul', instructions: 'Yes?'}
  });
  const result = finishMapikaSystemOne(
    prepared,
    [answer([0.333333, 0.666667]), answer([0.9, 0.1]), answer([0.3, 0.7]), answer([0.1, 0.9]), answer([0.2, 0.8])],
    'mapika-0.8b-q4f16'
  );
  assert.equal(result.model, 'mapika-0.8b-q4f16');
  assert.equal(result.answers.choice.choice, 'b');
  assert.equal(result.answers.choice.confidence, 0.3333);
  assert.equal(result.answers.choice.x_p_max, 0.6667);
  assert.deepEqual(result.answers.choice.probabilities, {a: 0.3333, b: 0.6667});
  assert.equal(result.answers.score.score, 1.47);
  assert.equal(result.answers.score.confidence, 0.2059);
  assert.equal(result.answers.score.x_p_max, 0.5294);
  assert.equal(result.answers.score.levelFit['2'], 0.9);
  assert.equal(result.answers.score.fitMass, 1.7);
  assert.equal(result.answers.yes.noul, 0.8);
});

test('listwise Score uses numeric legend order and TypeSafe confidence', () => {
  const prepared = prepare('state', {
    rating: {
      type: 'score',
      instructions: 'Rate the request.',
      criteria: {2: 'high', 0: 'low', 1: 'middle'},
      isolated: false
    }
  });
  const input = String.fromCodePoint(...prepared.rows[0].packed.ids.map((id) => id - 200));
  const result = finishMapikaSystemOne(prepared, [answer([0.05, 0.15, 0.8])], 'mapika-0.8b-f16');
  assert.match(input, /\(A\) 0: low\n\(B\) 1: middle\n\(C\) 2: high/);
  assert.equal(result.model, 'mapika-0.8b-f16');
  assert.deepEqual(result.answers.rating.legend, {0: 'low', 1: 'middle', 2: 'high'});
  assert.equal(result.answers.rating.score, 1.75);
  assert.equal(result.answers.rating.confidence, 0.625);
  assert.equal(result.answers.rating.x_p_max, 0.8);
});

test('systemOne abort skips rows not yet submitted and disposal waits for the queued batch', async () => {
  const {session, events} = sessionFixture();
  const started = {};
  started.promise = new Promise((resolve) => {
    started.resolve = resolve;
  });
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const runs = [];
  session.run = async (packed) => {
    runs.push(packed);
    started.resolve();
    await gate;
    return answer([0.5, 0.5]);
  };
  const controller = new AbortController();
  const reason = Error('cancel after first row');
  const result = session.systemOne(
    'state',
    {
      one: {instructions: 'First?', criteria: {yes: null, no: null}},
      two: {instructions: 'Second?', criteria: {yes: null, no: null}}
    },
    {signal: controller.signal}
  );
  await started.promise;
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  const disposal = session.dispose();
  assert.equal(events.includes('dispose'), false);
  release();
  await disposal;
  assert.equal(runs.length, 1);
  assert.deepEqual(events, ['end', 'dispose']);
});
