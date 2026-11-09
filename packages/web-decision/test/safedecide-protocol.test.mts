import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareSafeDecideSystemOne, finishSafeDecideSystemOne} from '../src/safedecide/protocol.mjs';
import {createSafeDecideSession} from '../src/safedecide/session.mjs';

function fakeTokenizer({truncateAfter = Number.POSITIVE_INFINITY} = {}) {
  const calls: {first: string; second: string}[] = [];
  return {
    calls,
    encodePair(first: string, second: string) {
      calls.push({first, second});
      return {
        inputIds: Array.from({length: Math.min(4, first.length + second.length)}, (_, index) => index + 2),
        truncated: first.length + second.length > truncateAfter
      };
    }
  };
}

function whitespacePairTokenizer(limit = 640) {
  const calls: {first: string; second: string}[] = [];
  return {
    calls,
    encodePair(first: string, second: string) {
      calls.push({first, second});
      const firstTokens = first.trim().split(/\s+/u);
      const secondTokens = second.trim().split(/\s+/u);
      const contentLimit = limit - 3;
      let truncated = false;
      while (firstTokens.length + secondTokens.length > contentLimit) {
        if (firstTokens.length > secondTokens.length) firstTokens.pop();
        else secondTokens.pop();
        truncated = true;
      }
      return {
        inputIds: [
          2,
          ...firstTokens.map((_, index) => index + 10),
          1,
          ...secondTokens.map((_, index) => index + 1000),
          1
        ],
        truncated
      };
    }
  };
}

const config: Parameters<typeof createSafeDecideSession>[0]['config'] = {
  temperatures: {choice: 1, noul: 1, score: 1},
  selective_rule: {
    alpha: 0.1,
    delta: 0.1,
    threshold: 0.7,
    statement: 'The default guarantee applies to exchangeable data.'
  },
  selective_rules: [
    {alpha: 0.05, delta: 0.1, threshold: null, statement: 'Escalate all at 5%.'},
    {alpha: 0.1, delta: 0.1, threshold: 0.7, statement: 'The default guarantee applies to exchangeable data.'},
    {alpha: 0.2, delta: 0.1, threshold: 0.5, statement: 'Escalate below 50% confidence.'}
  ]
};

test('SafeDecide packs Choice, ordered Score, and Noul pairs in question and option order', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareSafeDecideSystemOne(
    {invoice: {amount: 20, currency: 'USD'}},
    {
      route: {type: 'choice', instructions: 'Which route?', criteria: {billing: 'Invoices', support: null}},
      priority: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'medium', 'high']},
      needsHuman: {type: 'noul', instructions: 'Does this need a person?', criteria: {false: 'No', true: 'Yes'}}
    },
    tokenizer
  );

  assert.deepEqual(tokenizer.calls, [
    {
      first: '[choice] option: billing: Invoices || question: Which route?',
      second: '{"invoice": {"amount": 20, "currency": "USD"}}'
    },
    {
      first: '[choice] option: support || question: Which route?',
      second: '{"invoice": {"amount": 20, "currency": "USD"}}'
    },
    {
      first: '[score] option: 0: low || question: Rate urgency.',
      second: '{"invoice": {"amount": 20, "currency": "USD"}}'
    },
    {
      first: '[score] option: 1: medium || question: Rate urgency.',
      second: '{"invoice": {"amount": 20, "currency": "USD"}}'
    },
    {
      first: '[score] option: 2: high || question: Rate urgency.',
      second: '{"invoice": {"amount": 20, "currency": "USD"}}'
    },
    {
      first: '[noul] option: no: No || question: Does this need a person?',
      second: '{"invoice": {"amount": 20, "currency": "USD"}}'
    },
    {
      first: '[noul] option: yes: Yes || question: Does this need a person?',
      second: '{"invoice": {"amount": 20, "currency": "USD"}}'
    }
  ]);
  assert.equal(prepared.rows.length, 7);
  assert.equal(prepared.tokenCount, 28);
  assert.deepEqual(prepared.truncatedQuestions, []);
});

test('SafeDecide uses the question id when the instruction is omitted or empty', () => {
  const tokenizer = fakeTokenizer();
  prepareSafeDecideSystemOne(
    'state',
    {decision: {type: 'choice', instructions: '', criteria: {a: null, b: null}}},
    tokenizer
  );
  assert.match(tokenizer.calls[0]!.first, /question: decision$/);
});

test('SafeDecide passes over-200K structured state and option text through 640-token pair encoding', async () => {
  const longText = 'x'.repeat(200_001);
  const tokenizer = whitespacePairTokenizer();
  let runnerCalls = 0;
  const model = createSafeDecideSession({
    tokenizer,
    config,
    runner: {
      async run(rows) {
        runnerCalls++;
        assert.ok(rows.every((row) => (row as {inputIds: number[]}).inputIds.length <= 640));
        return rows.map((_, index) => index);
      },
      async dispose() {}
    }
  });

  const result = await model.systemOne(
    {payload: longText},
    {route: {type: 'choice', instructions: longText, criteria: {first: longText, second: null}}}
  );

  assert.equal(tokenizer.calls.length, 2);
  assert.ok(tokenizer.calls.every(({first, second}) => first.length > 200_000 && second.length > 200_000));
  assert.equal(result.truncated, undefined);
  assert.ok(result.tokenCount <= 640 * 2);
  assert.equal(runnerCalls, 1);
  await model.dispose();
});

test('SafeDecide right-truncates each pair and accepts requests above 512 pairs', async () => {
  const tokenizer = whitespacePairTokenizer();
  let runnerCalls = 0;
  const model = createSafeDecideSession({
    tokenizer,
    config,
    runner: {
      async run(rows) {
        runnerCalls++;
        assert.ok(rows.every((row) => (row as {inputIds: number[]}).inputIds.length <= 640));
        return rows.map((_, index) => index);
      },
      async dispose() {}
    }
  });

  const longInstructions = Array.from({length: 700}, (_, index) => `word${index}`).join(' ');
  const result = await model.systemOne('state', {
    route: {type: 'choice', instructions: longInstructions, criteria: {first: null, second: null}}
  });
  assert.equal(result.truncated, true);
  assert.ok(result.tokenCount <= 640 * 2);
  assert.equal(runnerCalls, 1);

  const makeCriteria = (count: number, prefix: string) =>
    Object.fromEntries(Array.from({length: count}, (_, index) => [`${prefix}-${index}`, null]));
  const questions = {
    first: {type: 'choice', criteria: makeCriteria(256, 'first')},
    second: {type: 'choice', criteria: makeCriteria(254, 'second')},
    third: {
      type: 'choice',
      criteria: makeCriteria(3, 'third')
    }
  };
  const resultAboveFormerCap = await model.systemOne('state', questions);
  assert.equal(resultAboveFormerCap.answers.third!.type, 'choice');
  assert.equal(tokenizer.calls.length, 2 + 256 + 254 + 3);
  assert.equal(runnerCalls, 2);
  await model.dispose();
});

test('SafeDecide accepts more than 512 options in one question', () => {
  const criteria = (count: number) =>
    Object.fromEntries(Array.from({length: count}, (_, index) => [`option-${index}`, null]));
  const tokenizer = fakeTokenizer();
  const prepared = prepareSafeDecideSystemOne('state', {wide: {type: 'choice', criteria: criteria(513)}}, tokenizer);

  assert.equal(prepared.questions.length, 1);
  assert.equal(prepared.rows.length, 513);
  assert.equal(tokenizer.calls.length, 513);
});

test('SafeDecide accepts more than 256 two-option questions', () => {
  const questions = (count: number) =>
    Object.fromEntries(
      Array.from({length: count}, (_, index) => [`q${index}`, {type: 'choice', criteria: {first: null, second: null}}])
    );
  const tokenizer = fakeTokenizer();
  const prepared = prepareSafeDecideSystemOne('state', questions(257), tokenizer);

  assert.equal(prepared.questions.length, 257);
  assert.equal(prepared.rows.length, 514);
  assert.equal(tokenizer.calls.length, 514);
});

test('SafeDecide generic decide accepts more than 512 pairs', async () => {
  const tokenizer = fakeTokenizer();
  let runnerCalls = 0;
  const model = createSafeDecideSession({
    tokenizer,
    config,
    runner: {
      async run(rows) {
        runnerCalls++;
        return rows.map(() => 0);
      },
      async dispose() {}
    }
  });
  const wideOptions = Array.from({length: 513}, (_, index) => `option-${index}`);
  const wide = await model.decide({
    state: 'state',
    questions: [{instructions: 'Choose.', options: wideOptions}]
  });
  assert.equal(wide.answers[0]!.probabilities.length, 513);

  const manyQuestions = Array.from({length: 257}, (_, index) => ({
    instructions: `Choose ${index}.`,
    options: ['first', 'second']
  }));
  const many = await model.decide({state: 'state', questions: manyQuestions});
  assert.equal(many.answers.length, 257);
  assert.equal(runnerCalls, 2);

  assert.equal(tokenizer.calls.length, 513 + 257 * 2);
  assert.equal(runnerCalls, 2);
  await model.dispose();
});

test('SafeDecide preserves option cardinality across preflight and snapshot', () => {
  const criteria = (count: number, prefix: string, reads: Record<string, number>, snapshotCount = count) => {
    const target = Object.fromEntries(
      Array.from({length: Math.max(count, snapshotCount)}, (_, index) => [`${prefix}-${index}`, null])
    );
    let ownKeysCalls = 0;
    return new Proxy(target, {
      ownKeys(value) {
        reads[prefix] = (reads[prefix] ?? 0) + 1;
        ownKeysCalls++;
        const visibleCount = ownKeysCalls === 1 ? count : snapshotCount;
        return Reflect.ownKeys(value).slice(0, visibleCount);
      }
    });
  };
  const questions = (thirdCount: number, reads: Record<string, number>, thirdSnapshotCount = thirdCount) => ({
    first: {type: 'choice', criteria: criteria(256, 'first', reads)},
    second: {type: 'choice', criteria: criteria(254, 'second', reads)},
    third: {type: 'choice', criteria: criteria(thirdCount, 'third', reads, thirdSnapshotCount)}
  });
  const boundaryReads = {};
  const tokenizer = fakeTokenizer();

  const prepared = prepareSafeDecideSystemOne('state', questions(2, boundaryReads), tokenizer);
  assert.equal(prepared.rows.length, 512);
  assert.equal(tokenizer.calls.length, 512);
  assert.deepEqual(boundaryReads, {first: 2, second: 2, third: 2});

  const overFormerCapReads = {};
  const overFormerCapTokenizer = fakeTokenizer();
  const overFormerCap = prepareSafeDecideSystemOne('state', questions(3, overFormerCapReads), overFormerCapTokenizer);
  assert.equal(overFormerCap.rows.length, 513);
  assert.equal(overFormerCapTokenizer.calls.length, 513);
  assert.deepEqual(overFormerCapReads, {first: 2, second: 2, third: 2});

  const changedReads = {};
  const changedTokenizer = fakeTokenizer();
  assert.throws(() => prepareSafeDecideSystemOne('state', questions(2, changedReads, 3), changedTokenizer), {
    code: 'INVALID_INPUT',
    message: /changed after preflight/
  });
  assert.deepEqual(changedReads, {first: 2, second: 2, third: 2});
  assert.equal(changedTokenizer.calls.length, 0);
});

test('SafeDecide rejects a changed Score length before reading values or tokenizing', () => {
  let lengthReads = 0,
    valueReads = 0;
  const levels = new Proxy(Array(513).fill(null), {
    get(target, key, receiver) {
      if (key === 'length') {
        lengthReads++;
        return lengthReads === 1 ? 513 : 1_000_000;
      }
      return Reflect.get(target, key, receiver);
    },
    getOwnPropertyDescriptor(target, key) {
      if (typeof key === 'string' && /^(0|[1-9]\d*)$/.test(key)) valueReads++;
      return Reflect.getOwnPropertyDescriptor(target, key);
    }
  });
  const tokenizer = fakeTokenizer();

  assert.throws(() => prepareSafeDecideSystemOne('state', {score: {type: 'score', criteria: levels}}, tokenizer), {
    code: 'INVALID_INPUT'
  });
  assert.equal(lengthReads, 2);
  assert.equal(valueReads, 0);
  assert.equal(tokenizer.calls.length, 0);

  const boundaryTokenizer = fakeTokenizer();
  const prepared = prepareSafeDecideSystemOne(
    'state',
    {score: {type: 'score', criteria: Array(513).fill(null)}},
    boundaryTokenizer
  );
  assert.equal(prepared.rows.length, 513);
  assert.equal(boundaryTokenizer.calls.length, 513);
});

test('SafeDecide rejects invalid question types, criteria, duplicate fields, and unsafe object values', () => {
  const tokenizer = fakeTokenizer();
  assert.throws(
    () =>
      prepareSafeDecideSystemOne(
        'state',
        {q: {type: 'rank', instructions: 'Choose', criteria: {a: null, b: null}}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareSafeDecideSystemOne(
        'state',
        {q: {type: 'choice', instructions: 'Choose', criteria: ['a', 'b']}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareSafeDecideSystemOne(
        'state',
        {q: {type: 'noul', instructions: 'True?', criteria: {maybe: 'unknown'}}},
        tokenizer
      ),
    {code: 'INVALID_INPUT'}
  );
  const accessor = {};
  Object.defineProperty(accessor, 'type', {enumerable: true, get: () => 'choice'});
  assert.throws(() => prepareSafeDecideSystemOne('state', {q: accessor}, tokenizer), {code: 'INVALID_INPUT'});
});

test('SafeDecide accepts JSON state values and rejects unsafe shapes before pair tokenization', () => {
  const tokenizer = fakeTokenizer();
  const questions = {q: {type: 'choice', criteria: {a: null, b: null}}};
  for (const state of ['text', [null, false, {nested: 'value'}], {nested: {value: 1}}])
    assert.equal(prepareSafeDecideSystemOne(state, questions, tokenizer).rows.length, 2);

  const cyclic: {self?: unknown} = {};
  cyclic.self = cyclic;
  // This test requires a missing array element, not an undefined value.
  // oxlint-disable-next-line unicorn/no-new-array
  const sparse = new Array(2);
  sparse[1] = 'value';
  const arrayWithExtra: string[] & {extra?: boolean} = ['value'];
  arrayWithExtra.extra = true;
  const accessor = {};
  Object.defineProperty(accessor, 'nested', {
    enumerable: true,
    get() {
      throw Error('state getter invoked');
    }
  });
  const symbolValue = {[Symbol('metadata')]: true};
  const hiddenValue = {};
  Object.defineProperty(hiddenValue, 'hidden', {value: 'not JSON'});
  const invalidStates = [
    null,
    true,
    7,
    Number.NaN,
    undefined,
    () => {},
    cyclic,
    {notFinite: Number.NaN},
    {unsupported: undefined},
    sparse,
    arrayWithExtra,
    accessor,
    symbolValue,
    hiddenValue,
    new Date(0)
  ];
  const callsBeforeInvalid = tokenizer.calls.length;
  for (const state of invalidStates)
    assert.throws(() => prepareSafeDecideSystemOne(state, questions, tokenizer), {code: 'INVALID_INPUT'});
  assert.equal(tokenizer.calls.length, callsBeforeInvalid);
});

test('SafeDecide rejects malformed question and criteria shapes during preflight', () => {
  const tokenizer = fakeTokenizer();
  const choice = {type: 'choice', instructions: 'Choose', criteria: {a: null, b: null}};
  const invalidQuestions = [
    [],
    {},
    {'': choice},
    {q: null},
    {q: {...choice, unsupported: true}},
    {q: {...choice, instructions: 7}},
    {q: {type: 'choice', criteria: ['a', 'b']}},
    {q: {type: 'choice', criteria: {only: null}}},
    {q: {type: 'choice', criteria: {a: 1, b: null}}},
    {q: {type: 'score', criteria: {low: 'Low', high: 'High'}}},
    {q: {type: 'score', criteria: ['only']}},
    {q: {type: 'score', criteria: [7, null]}},
    {q: {type: 'noul', criteria: {maybe: 'perhaps'}}},
    {q: {type: 'noul', criteria: 'not an object'}}
  ];
  const callsBeforeInvalid = tokenizer.calls.length;
  for (const questions of invalidQuestions)
    assert.throws(() => prepareSafeDecideSystemOne('state', questions, tokenizer), {code: 'INVALID_INPUT'});
  assert.equal(tokenizer.calls.length, callsBeforeInvalid);

  const defaultChoice = prepareSafeDecideSystemOne('state', {q: {criteria: {a: null, b: null}}}, tokenizer);
  assert.equal(defaultChoice.questions[0]!.type, 'choice');
  const defaultNoul = prepareSafeDecideSystemOne('state', {q: {type: 'noul'}}, tokenizer);
  assert.deepEqual(
    defaultNoul.questions[0]!.options.map(({text}) => text),
    ['no', 'yes']
  );
  assert.throws(() => prepareSafeDecideSystemOne('state', {q: choice}, {}), TypeError);
  assert.throws(() => prepareSafeDecideSystemOne('state', {q: choice}, null), TypeError);
  assert.throws(() => prepareSafeDecideSystemOne(new Date(0), {q: choice}, tokenizer), {code: 'INVALID_INPUT'});
  assert.throws(
    () =>
      finishSafeDecideSystemOne(
        prepareSafeDecideSystemOne('state', {q: choice}, tokenizer),
        [1, 0],
        config,
        undefined,
        0.3 as Parameters<typeof finishSafeDecideSystemOne>[4]
      ),
    {code: 'INVALID_INPUT'}
  );
});

test('SafeDecide accepts certified call options and preserves abort and disposal priority', async () => {
  const tokenizer = fakeTokenizer();
  let runnerCalls = 0;
  const model = createSafeDecideSession({
    tokenizer,
    config,
    runner: {
      async run(rows) {
        runnerCalls++;
        return rows.map(() => 1);
      },
      async dispose() {}
    }
  });
  const questions = {q: {type: 'choice', criteria: {a: null, b: null}}};
  await model.systemOne('state', questions, null);
  await model.systemOne('state', questions, {alpha: 0.2});

  const controller = new AbortController();
  controller.abort();
  await assert.rejects(model.systemOne('state', questions, {signal: controller.signal}), {name: 'AbortError'});
  assert.equal(runnerCalls, 2);
  await model.dispose();
  await assert.rejects(model.systemOne('state', questions), {code: 'SESSION_CLOSED'});
});

test('SafeDecide rejects malformed call options and generic requests before tokenization', async () => {
  const tokenizer = fakeTokenizer();
  let runnerCalls = 0;
  const model = createSafeDecideSession({
    tokenizer,
    config,
    runner: {
      async run(rows) {
        runnerCalls++;
        return rows.map(() => 1);
      },
      async dispose() {}
    }
  });
  const typedQuestions = {q: {type: 'choice', criteria: {a: null, b: null}}};
  await assert.rejects(model.systemOne('state', typedQuestions, 'invalid'), {code: 'INVALID_INPUT'});
  await assert.rejects(model.systemOne('state', typedQuestions, {signal: {}}), {code: 'INVALID_INPUT'});
  await assert.rejects(model.systemOne('state', typedQuestions, {alpha: 0.3}), {code: 'INVALID_INPUT'});

  let getterCalls = 0;
  const accessorOptions = {};
  Object.defineProperty(accessorOptions, 'alpha', {
    enumerable: true,
    get() {
      getterCalls++;
      throw Error('alpha getter invoked');
    }
  });
  await assert.rejects(model.systemOne('state', typedQuestions, accessorOptions), {code: 'INVALID_INPUT'});
  assert.equal(getterCalls, 0);

  for (const request of [
    {state: 'state', questions: []},
    {state: 'state', questions: [{instructions: 'Choose', options: ['only one']}]},
    {state: 'state', questions: [{instructions: 'Choose', options: ['same', 'same']}]},
    {state: 'state', questions: [{instructions: 1, options: ['a', 'b']}]}
  ])
    await assert.rejects(model.decide(request), {code: 'INVALID_INPUT'});

  assert.equal(tokenizer.calls.length, 0);
  assert.equal(runnerCalls, 0);
  await model.dispose();
});

test('SafeDecide preserves calibrated probabilities, expected Score, Noul truth probability, action, and guarantee', () => {
  const prepared = {
    tokenCount: 15,
    rows: Array(7).fill({}),
    truncatedQuestions: [],
    questions: [
      {id: 'choice', type: 'choice', options: [{name: 'a'}, {name: 'b'}]},
      {id: 'score', type: 'score', options: [{name: '0'}, {name: '1'}, {name: '2'}]},
      {id: 'noul', type: 'noul', options: [{name: 'false'}, {name: 'true'}]}
    ]
  };
  const result = finishSafeDecideSystemOne(
    prepared as unknown as Parameters<typeof finishSafeDecideSystemOne>[0],
    [2, 0, 0, 0, 0, 0, 1],
    config
  );

  assert.deepEqual(result.guarantee, {
    alpha: 0.1,
    delta: 0.1,
    statement: 'The default guarantee applies to exchangeable data.'
  });
  assert.equal(result.outputTokenCount, 0);
  const choiceAnswer = result.answers.choice;
  const scoreAnswer = result.answers.score;
  const noulAnswer = result.answers.noul;
  assert.ok(choiceAnswer?.type === 'choice' && scoreAnswer?.type === 'score' && noulAnswer?.type === 'noul');
  assert.equal(choiceAnswer.choice, 'a');
  assert.equal(choiceAnswer.action, 'automate');
  assert.ok(Math.abs(scoreAnswer.score - 1) < 1e-12);
  assert.equal(scoreAnswer.choice, '0');
  assert.equal(scoreAnswer.action, 'escalate');
  assert.ok(Math.abs(noulAnswer.noul - 0.7310585786300049) < 1e-12);
  assert.equal(noulAnswer.action, 'automate');
});

test('SafeDecide applies certified alpha thresholds including the never-automate level', () => {
  const prepared = {
    tokenCount: 5,
    rows: [{}, {}],
    truncatedQuestions: [],
    questions: [{id: 'q', type: 'choice', options: [{name: 'a'}, {name: 'b'}]}]
  };
  const tied = finishSafeDecideSystemOne(
    prepared as unknown as Parameters<typeof finishSafeDecideSystemOne>[0],
    [0, 0],
    config,
    undefined,
    0.2
  );
  const strict = finishSafeDecideSystemOne(
    prepared as unknown as Parameters<typeof finishSafeDecideSystemOne>[0],
    [10, -10],
    config,
    undefined,
    0.05
  );
  assert.equal(tied.answers.q!.action, 'automate');
  assert.equal(strict.answers.q!.action, 'escalate');
  assert.equal(strict.guarantee.alpha, 0.05);
});

test('SafeDecide rejects malformed option scores', () => {
  const prepared = {
    tokenCount: 5,
    rows: [{}, {}],
    truncatedQuestions: [],
    questions: [{id: 'q', type: 'choice', options: [{name: 'a'}, {name: 'b'}]}]
  };
  assert.throws(
    () =>
      finishSafeDecideSystemOne(prepared as unknown as Parameters<typeof finishSafeDecideSystemOne>[0], [1], config),
    {code: 'INFERENCE_ERROR'}
  );
  assert.throws(
    () =>
      finishSafeDecideSystemOne(
        prepared as unknown as Parameters<typeof finishSafeDecideSystemOne>[0],
        [1, Number.NaN],
        config
      ),
    {code: 'INFERENCE_ERROR'}
  );
});
