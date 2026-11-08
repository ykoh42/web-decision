import test from 'node:test';
import assert from 'node:assert/strict';
import {DecisionModel} from '../src/model.js';
import {ExperimentalMapikaSession} from '../src/mapika/session.js';
import {ExperimentalKev08Session} from '../src/kev08/session.js';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {ExperimentalStrandsSession} from '../src/strands/session.mjs';
import {createGlinerMobileSession} from '../src/gliner-mobile/session.mjs';
import {GLINER_MAX_OPTIONS} from '../src/gliner-mobile/contract.mjs';
import {createSkejjaSession} from '../src/skejja/session.mjs';
import {SKEJJA_INT8_PROFILE} from '../src/skejja/contract.mjs';
import {createLayaSession} from '../src/laya/session.mjs';
import {LAYA_MAX_OPTIONS} from '../src/laya/contract.mjs';
import {createNariSession} from '../src/nari/session.mjs';
import {createEttinSession} from '../src/ettin/session.mjs';
import {ETTIN_MAX_OPTIONS} from '../src/ettin/contract.mjs';
import {createTamevSession} from '../src/tamev/session.mjs';
import {createClefSession} from '../src/clef/session.mjs';
import {CLEF_MAX_OPTIONS} from '../src/clef/protocol.mjs';
import {createFlashSession} from '../src/pplx/session.mjs';
import {FLASH_MAX_OPTIONS} from '../src/pplx/protocol.mjs';
import {createMapika4Session} from '../src/mapika4b/session.mjs';
import {MAPIKA4_MAX_OPTIONS} from '../src/mapika4b/contract.mjs';
import {createVerdictSession} from '../src/verdict/session.mjs';
import {VERDICT_MAX_CANDIDATES} from '../src/verdict/contract.mjs';
import {createTinyDecideSession} from '../src/tinydecide/session.mjs';
import {TINYDECIDE_MAX_CHOICES} from '../src/tinydecide/contract.mjs';
import {createH2oSession} from '../src/h2o/session.mjs';
import {MAX_H2O_OPTIONS} from '../src/h2o/protocol.mjs';
import {readFile} from 'node:fs/promises';

const classes = [
  DecisionModel,
  ExperimentalMapikaSession,
  ExperimentalKev08Session,
  ExperimentalDebertaSession,
  ExperimentalStrandsSession
];
const request = () => ({state: 'text', questions: [{instructions: 'Choose', options: ['A', 'B']}]});

test('all staged public session families reject unsupported fields before tokenization or runner', async () => {
  for (const Session of classes) {
    let tokens = 0,
      calls = 0;
    const session = Object.assign(Object.create(Session.prototype), {
      closed: false,
      queue: Promise.resolve(),
      tokenizer: {
        encode() {
          tokens++;
          throw Error('unexpected tokenizer');
        }
      },
      runner: {
        run() {
          calls++;
          throw Error('unexpected runner');
        }
      },
      run() {
        calls++;
        throw Error('unexpected runner');
      }
    });
    for (const [input, options] of [
      [{...request(), image: 'ignored'}, {}],
      [{...request(), questions: [{...request().questions[0], type: 'score'}]}, {}],
      [{...request(), questions: [{...request().questions[0], criteria: 'x'}]}, {}],
      [request(), {temperature: 2}]
    ])
      await assert.rejects(session.decide(input, options), {code: 'INVALID_INPUT'});
    assert.equal(tokens, 0, Session.name);
    assert.equal(calls, 0, Session.name);
  }
});

test('closed and previously aborted sessions retain their existing error priority', async () => {
  for (const Session of classes) {
    const session = Object.assign(Object.create(Session.prototype), {closed: true, runner: {}});
    await assert.rejects(session.decide({...request(), image: 1}), {code: 'SESSION_CLOSED'});
    session.closed = false;
    const controller = new AbortController(),
      reason = Error('aborted');
    controller.abort(reason);
    await assert.rejects(session.decide({...request(), image: 1}, {signal: controller.signal}), (e) => e === reason);
    await assert.rejects(session.decide(request(), {signal: controller.signal, temperature: 2}), (e) => e === reason);
    let getterCalls = 0;
    const options = {signal: controller.signal};
    Object.defineProperty(options, 'image', {
      get() {
        getterCalls++;
        throw Error('unsupported getter');
      }
    });
    await assert.rejects(session.decide(request(), options), (e) => e === reason);
    assert.equal(getterCalls, 0);
  }
});

test('sparse own question and option entries fail before any tokenization', async () => {
  for (const Session of classes) {
    let tokens = 0,
      calls = 0;
    const session = Object.assign(Object.create(Session.prototype), {
      closed: false,
      queue: Promise.resolve(),
      tokenizer: {
        encode() {
          tokens++;
          return {ids: [1]};
        }
      },
      runner: {
        run() {
          calls++;
        }
      },
      run() {
        calls++;
      }
    });
    const choices = new Array(3);
    choices[0] = 'A';
    choices[2] = 'C';
    const optionProto = Object.assign(Object.create(Array.prototype), {1: 'B'});
    Object.setPrototypeOf(choices, optionProto);
    const questions = new Array(2);
    questions[0] = request().questions[0];
    const questionProto = Object.assign(Object.create(Array.prototype), {1: request().questions[0]});
    Object.setPrototypeOf(questions, questionProto);
    for (const bad of [
      {state: 'text', questions: [{instructions: 'Choose', options: choices}]},
      {state: 'text', questions}
    ])
      await assert.rejects(session.decide(bad), {code: 'INVALID_INPUT'});
    assert.equal(tokens, 0, Session.name);
    assert.equal(calls, 0, Session.name);
  }
});

test('all staged sessions reject 256 options before reading the first entry or tokenizing', async () => {
  for (const Session of classes) {
    let reads = 0,
      tokens = 0,
      calls = 0;
    const session = Object.assign(Object.create(Session.prototype), {
      closed: false,
      queue: Promise.resolve(),
      tokenizer: {
        encode() {
          tokens++;
          throw Error('unexpected tokenizer');
        }
      },
      runner: {
        run() {
          calls++;
          throw Error('unexpected runner');
        }
      },
      run() {
        calls++;
        throw Error('unexpected runner');
      }
    });
    const options = Array.from({length: 256}, (_, i) => `option ${i}`);
    Object.defineProperty(options, '0', {
      get() {
        reads++;
        throw Error('unexpected option getter');
      }
    });
    await assert.rejects(session.decide({state: 'text', questions: [{instructions: 'Choose', options}]}), {
      code: 'INVALID_INPUT'
    });
    assert.equal(reads, 0, Session.name);
    assert.equal(tokens, 0, Session.name);
    assert.equal(calls, 0, Session.name);
  }
});

test('DeBERTa rejects accessors before getters on its uncapped question-count path', async () => {
  let reads = 0,
    tokens = 0,
    calls = 0;
  const questions = Array.from({length: 513}, () => request().questions[0]);
  Object.defineProperty(questions, '0', {
    get() {
      reads++;
      throw Error('unexpected question getter');
    }
  });
  const session = Object.assign(Object.create(ExperimentalDebertaSession.prototype), {
    closed: false,
    queue: Promise.resolve(),
    tokenizer: {
      encode() {
        tokens++;
        return {ids: [3]};
      }
    },
    run() {
      calls++;
      throw Error('unexpected inference');
    }
  });
  const input = {state: 'text', questions};
  await assert.rejects(session.decide(input), {code: 'INVALID_INPUT'});
  assert.equal(reads, 0);
  assert.equal(tokens, 0);
  assert.equal(calls, 0);

  const ordinaryQuestions = Array.from({length: 513}, () => request().questions[0]);
  await assert.rejects(session.decide({state: 'text', questions: ordinaryQuestions}), {code: 'INVALID_INPUT'});
  assert.ok(tokens > 0);
  assert.equal(calls, 0);
  const controller = new AbortController(),
    reason = Error('aborted');
  controller.abort(reason);
  await assert.rejects(session.decide(input, {signal: controller.signal}), (error) => error === reason);
  session.closed = true;
  await assert.rejects(session.decide(input, {signal: controller.signal}), {code: 'SESSION_CLOSED'});
  assert.equal(reads, 0);
  assert.equal(calls, 0);
});

test('question count cannot be shortened by a getter before any model tokenizes', async () => {
  for (const Session of classes) {
    let tokens = 0;
    const session = Object.assign(Object.create(Session.prototype), {
      closed: false,
      queue: Promise.resolve(),
      tokenizer: {
        encode() {
          tokens++;
          throw Error('unexpected tokenizer');
        }
      },
      runner: {}
    });
    const questions = [request().questions[0], request().questions[0]];
    Object.defineProperty(questions, '0', {
      get() {
        questions.length = 1;
        return request().questions[0];
      }
    });
    await assert.rejects(session.decide({state: 'text', questions}), {code: 'INVALID_INPUT'});
    assert.equal(tokens, 0, Session.name);
  }
});

test('native option caps reject before reading caller option entries', async () => {
  const counts = {tokenizer: 0, runner: 0};
  const tokenizer = {
    encode() {
      counts.tokenizer++;
      return {ids: []};
    },
    decode() {
      counts.tokenizer++;
      return '';
    },
    encodePair() {
      counts.tokenizer++;
      return {};
    },
    encodeContext() {
      counts.tokenizer++;
      return [];
    },
    encodeOption() {
      counts.tokenizer++;
      return [];
    }
  };
  const runner = {
    async run() {
      counts.runner++;
      throw Error('unexpected inference');
    },
    async dispose() {},
    async releaseCachedMemory() {}
  };
  const constructors = [
    ['GLiNER mobile', GLINER_MAX_OPTIONS, () => createGlinerMobileSession({tokenizer, runner})],
    ['Skejja', SKEJJA_INT8_PROFILE.maxChoices, () => createSkejjaSession({runner, tokenizer})],
    [
      'Laya',
      LAYA_MAX_OPTIONS,
      () => createLayaSession({runner, tokenizer, temperatures: {byType: [1, 1, 1], byOptions: {}}})
    ],
    ['Nari', 64, () => createNariSession({tokenizer, controlIds: [1], runner})],
    ['Ettin', ETTIN_MAX_OPTIONS, () => createEttinSession({runner, tokenizer})],
    ['Clef', CLEF_MAX_OPTIONS, () => createClefSession({tokenizer, controlIds: [1], runner})],
    ['PPLX', FLASH_MAX_OPTIONS, () => createFlashSession({tokenizer, config: {}, runner})],
    ['Mapika 4B', MAPIKA4_MAX_OPTIONS, () => createMapika4Session({tokenizer, labels: {}, limits: {}, runner})],
    ['Verdict', VERDICT_MAX_CANDIDATES, () => createVerdictSession({runner, tokenizer, calibrator: {}})],
    ['TinyDecide', TINYDECIDE_MAX_CHOICES, () => createTinyDecideSession({engine: {}})]
  ];

  const plan = JSON.parse(await readFile(new URL('./fixtures/h2o-labels.json', import.meta.url)));
  constructors.push([
    'H2O',
    MAX_H2O_OPTIONS,
    async () => createH2oSession({tokenizer, plan, addedTokens: ['<reserved>'], runner})
  ]);

  for (const [name, maxOptions, createSession] of constructors) {
    const session = await createSession();
    const atLimit = new Array(maxOptions);
    let boundaryReads = 0;
    const boundaryError = Error('boundary option getter invoked');
    Object.defineProperty(atLimit, '0', {
      get() {
        boundaryReads++;
        throw boundaryError;
      }
    });
    await assert.rejects(
      session.decide({state: 'state', questions: [{instructions: 'Choose', options: atLimit}]}),
      (error) => error === boundaryError,
      name
    );
    assert.equal(boundaryReads, 1, name);
    assert.equal(counts.tokenizer, 0, name);
    assert.equal(counts.runner, 0, name);

    const options = Array.from({length: maxOptions + 1}, () => 'unused');
    let reads = 0;
    Object.defineProperty(options, '0', {
      get() {
        reads++;
        throw Error('option getter invoked');
      }
    });
    await assert.rejects(
      session.decide({state: 'state', questions: [{instructions: 'Choose', options}]}),
      {code: 'INVALID_INPUT'},
      name
    );
    assert.equal(reads, 0, name);
    assert.equal(counts.tokenizer, 0, name);
    assert.equal(counts.runner, 0, name);
    await session.dispose?.();
  }
});

test('Tamev accepts 78 data options and rejects accessor-backed options before reading them', async () => {
  const counts = {tokenizer: 0, runner: 0, getter: 0};
  const tokenizer = {
    encodeContext() {
      counts.tokenizer++;
      return {ids: [1], truncated: false};
    },
    encodeOption() {
      counts.tokenizer++;
      return {ids: [2], truncated: false};
    }
  };
  const runner = {
    async run(_contextIds, optionRows) {
      counts.runner++;
      const probabilities = optionRows.map((_, index) => (index === optionRows.length - 1 ? 1 : 0));
      return {logits: probabilities, probabilities};
    },
    async dispose() {},
    async releaseCachedMemory() {}
  };
  const session = createTamevSession({runner, tokenizer, config: {calibration_by_k: {}}});
  const options = Array.from({length: 78}, (_, index) => `option ${index}`);
  const result = await session.decide({state: 'state', questions: [{instructions: 'Choose', options}]});
  assert.equal(result.answers[0].selected, options.at(-1));
  assert.equal(counts.runner, 1);

  const accessorOptions = [...options];
  Object.defineProperty(accessorOptions, '77', {
    get() {
      counts.getter++;
      throw Error('option getter invoked');
    }
  });
  const tokenCount = counts.tokenizer;
  await assert.rejects(
    session.decide({state: 'state', questions: [{instructions: 'Choose', options: accessorOptions}]}),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(counts.getter, 0);
  assert.equal(counts.tokenizer, tokenCount);
  assert.equal(counts.runner, 1);
  await session.dispose();
});

test('request-wide option caps reject before reading any option entries', async () => {
  const counts = {tokenizer: 0, runner: 0};
  const tokenizer = {
    encode() {
      counts.tokenizer++;
      return {ids: []};
    },
    encodePair() {
      counts.tokenizer++;
      return {};
    }
  };
  const runner = {
    async run() {
      counts.runner++;
      throw Error('unexpected inference');
    },
    async dispose() {},
    async releaseCachedMemory() {}
  };
  const cases = [
    {
      name: 'Nari',
      totalLimit: 64,
      exactCounts: [32, 32],
      overCounts: [32, 33],
      createSession: () => createNariSession({tokenizer, controlIds: [1], runner})
    },
    {
      name: 'Clef',
      totalLimit: CLEF_MAX_OPTIONS,
      exactCounts: [32, 32],
      overCounts: [32, 33],
      createSession: () => createClefSession({tokenizer, controlIds: [1], runner})
    }
  ];

  for (const {name, totalLimit, exactCounts, overCounts, createSession} of cases) {
    assert.equal(
      exactCounts.reduce((sum, count) => sum + count, 0),
      totalLimit,
      name
    );
    assert.equal(
      overCounts.reduce((sum, count) => sum + count, 0),
      totalLimit + 1,
      name
    );
    const session = await createSession();
    const optionsFor = (count, error, reads) => {
      const options = Array.from({length: count}, (_, index) => `option-${index}`);
      Object.defineProperty(options, '0', {
        get() {
          reads.count++;
          throw error;
        }
      });
      return options;
    };
    const makeRequest = (optionCounts, error, reads) => ({
      state: 'state',
      questions: optionCounts.map((count, index) => ({
        instructions: `q${index}`,
        options: optionsFor(count, error, reads)
      }))
    });
    const atLimitError = Error(`${name} aggregate boundary reached snapshot`);
    const atLimitReads = {count: 0};
    await assert.rejects(
      session.decide(makeRequest(exactCounts, atLimitError, atLimitReads)),
      (error) => error === atLimitError,
      name
    );
    assert.equal(atLimitReads.count, 1, name);
    assert.equal(counts.tokenizer, 0, name);
    assert.equal(counts.runner, 0, name);

    const overLimitError = Error(`${name} aggregate overflow read an option`);
    const overLimitReads = {count: 0};
    await assert.rejects(
      session.decide(makeRequest(overCounts, overLimitError, overLimitReads)),
      {code: 'INVALID_INPUT'},
      name
    );
    assert.equal(overLimitReads.count, 0, name);
    assert.equal(counts.tokenizer, 0, name);
    assert.equal(counts.runner, 0, name);
    await session.dispose();
  }
});
