import test from 'node:test';
import assert from 'node:assert/strict';
import {DecisionModel} from '../src/model.js';
import {ExperimentalMapikaSession} from '../src/mapika/session.js';
import {ExperimentalKev08Session} from '../src/kev08/session.js';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {ExperimentalStrandsSession} from '../src/strands/session.mjs';
import {createFixtureImajevSession} from '../src/imajev/fixture-session.mjs';
import {fixture as imajevFixture} from './fixtures/imajev.mjs';
import {Tokenizer as StrandsTokenizer} from './strands-fake-tokenizer.mjs';

type SessionConstructor =
  | typeof DecisionModel
  | typeof ExperimentalMapikaSession
  | typeof ExperimentalKev08Session
  | typeof ExperimentalDebertaSession
  | typeof ExperimentalStrandsSession;
type TestSession = {
  tokenizer: {encode(text: string): {ids: number[]}};
  decide(request: unknown, options?: {signal?: AbortSignal}): Promise<unknown>;
  dispose(): Promise<void>;
};

const families: SessionConstructor[] = [
  DecisionModel,
  ExperimentalMapikaSession,
  ExperimentalKev08Session,
  ExperimentalDebertaSession,
  ExperimentalStrandsSession
];
const request = () => ({state: 'state', questions: [{instructions: 'choose', options: ['a', 'b']}]});

function sessionFor(Session: SessionConstructor): {session: TestSession; events: string[]} {
  const events: string[] = [],
    lifecycle = {disposed: false, loss: null};
  const runtime = {
    endScope() {
      events.push('end');
    },
    dispose() {
      events.push('dispose');
      lifecycle.disposed = true;
    }
  };
  const tokenizer =
    Session === ExperimentalStrandsSession
      ? new StrandsTokenizer()
      : {
          encode(text: string) {
            return {ids: Array.from(text, (c) => c.codePointAt(0)! + 200)};
          }
        };
  const runner = {
    async run() {
      events.push('run');
      return [0, 0];
    },
    async dispose() {
      events.push('dispose');
    },
    async releaseCachedMemory() {}
  };
  const session = Object.assign(Object.create(Session.prototype), {
    closed: false,
    disposal: null,
    queue: Promise.resolve(),
    runtime,
    lifecycle,
    tokenizer,
    runner,
    limits: {maxStateTokens: 384, maxInputTokens: 1024, controlIds: [999999], temperature: 1.03},
    labels: {
      names: Array.from({length: 255}, (_, i) => String(i)),
      ids: Array.from({length: 255}, (_, i) => i),
      open: [3]
    },
    weights: [],
    encoderWeights: [],
    headWeights: [],
    uncaptured: [],
    run() {
      events.push('run');
      return 'unexpected';
    }
  }) as unknown as TestSession;
  return {session, events};
}

for (const Session of families) {
  test(`${Session.name}: ${
    Session === ExperimentalDebertaSession
      ? 'rejects an accessor request without invoking its getter'
      : 'disposal from a request getter prevents work from being queued'
  }`, async () => {
    const {session, events} = sessionFor(Session);
    let closing: Promise<void> | undefined;
    let getterCalls = 0;
    const input = request();
    Object.defineProperty(input, 'state', {
      get() {
        getterCalls++;
        closing = session.dispose();
        return 'state';
      }
    });
    if (Session === ExperimentalDebertaSession) {
      await assert.rejects(session.decide(input), {code: 'INVALID_INPUT'});
      assert.equal(getterCalls, 0);
      await session.dispose();
    } else {
      await assert.rejects(session.decide(input), {code: 'SESSION_CLOSED'});
      await closing;
      assert.equal(getterCalls, 1);
    }
    assert.equal(events.includes('run'), false);
    assert.equal(events.filter((event) => event === 'dispose').length, 1);
  });

  test(`${Session.name}: disposal during tokenization prevents work from being queued`, async () => {
    const {session, events} = sessionFor(Session);
    let closing: Promise<void> | undefined;
    const encode = session.tokenizer.encode.bind(session.tokenizer);
    session.tokenizer.encode = (...args: Parameters<TestSession['tokenizer']['encode']>) => {
      closing ??= session.dispose();
      return encode(...args);
    };
    await assert.rejects(session.decide(request()), {code: 'SESSION_CLOSED'});
    await closing;
    assert.equal(events.includes('run'), false);
    assert.equal(events.filter((event) => event === 'dispose').length, 1);
  });

  test(`${Session.name}: failed abort listener registration never schedules inference`, async () => {
    const {session, events} = sessionFor(Session);
    const signal = new AbortController().signal;
    signal.addEventListener = () => {
      throw Error('listener setup failed');
    };
    await assert.rejects(session.decide(request(), {signal}), /listener setup failed/);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(events.includes('run'), false);
    await session.dispose();
  });

  test(`${Session.name}: disposal during abort listener registration prevents inference`, async () => {
    const {session, events} = sessionFor(Session);
    const signal = new AbortController().signal,
      add = signal.addEventListener.bind(signal);
    let closing: Promise<void> | undefined;
    signal.addEventListener = (...args: Parameters<AbortSignal['addEventListener']>) => {
      closing = session.dispose();
      return add(...args);
    };
    await assert.rejects(session.decide(request(), {signal}), {code: 'SESSION_CLOSED'});
    await closing;
    assert.equal(events.includes('run'), false);
  });

  test(`${Session.name}: an accepted decision finishes before reentrant disposal`, async () => {
    const {session, events} = sessionFor(Session);
    const first = session.decide(request());
    const signal = new AbortController().signal,
      add = signal.addEventListener.bind(signal);
    let closing: Promise<void> | undefined;
    signal.addEventListener = (...args: Parameters<AbortSignal['addEventListener']>) => {
      closing = session.dispose();
      return add(...args);
    };
    await assert.rejects(session.decide(request(), {signal}), {code: 'SESSION_CLOSED'});
    await first;
    await closing;
    assert.equal(events.filter((event) => event === 'run').length, 1);
    assert.ok(events.indexOf('run') < events.indexOf('dispose'));
  });
}

test('ImajevSession: disposal from a request getter prevents work from being queued', async () => {
  const f = imajevFixture(createFixtureImajevSession);
  let closing: Promise<void> | undefined;
  Object.defineProperty(f.request, 'state', {
    get() {
      closing = f.model.dispose();
      return 'fake';
    }
  });
  await assert.rejects(f.model.decide(f.request), {code: 'SESSION_CLOSED'});
  await closing;
  assert.equal(f.calls(), 0);
});

test('ImajevSession: disposal during preparation prevents work from being queued', async () => {
  const f = imajevFixture(createFixtureImajevSession);
  let closing: Promise<void> | undefined;
  const processor = f.model.processor!;
  f.model.processor = {
    prepare(request: unknown) {
      const prepared = processor.prepare(request);
      closing = f.model.dispose();
      return prepared;
    },
    finish: processor.finish.bind(processor)
  };
  await assert.rejects(f.model.decide(f.request), {code: 'SESSION_CLOSED'});
  await closing;
  assert.equal(f.calls(), 0);
});

test('ImajevSession: failed abort listener registration never schedules inference', async () => {
  const f = imajevFixture(createFixtureImajevSession);
  const signal = new AbortController().signal;
  signal.addEventListener = () => {
    throw Error('listener setup failed');
  };
  await assert.rejects(f.model.decide(f.request, {signal}), /listener setup failed/);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.calls(), 0);
  await f.model.dispose();
});

test('ImajevSession: disposal during abort listener registration prevents inference', async () => {
  const f = imajevFixture(createFixtureImajevSession);
  const signal = new AbortController().signal,
    add = signal.addEventListener.bind(signal);
  let closing: Promise<void> | undefined;
  signal.addEventListener = (...args: Parameters<AbortSignal['addEventListener']>) => {
    closing = f.model.dispose();
    return add(...args);
  };
  await assert.rejects(f.model.decide(f.request, {signal}), {code: 'SESSION_CLOSED'});
  await closing;
  assert.equal(f.calls(), 0);
});

test('ImajevSession: an accepted decision finishes before reentrant disposal', async () => {
  const f = imajevFixture(createFixtureImajevSession);
  const first = f.model.decide(f.request);
  const signal = new AbortController().signal,
    add = signal.addEventListener.bind(signal);
  let closing: Promise<void> | undefined;
  signal.addEventListener = (...args: Parameters<AbortSignal['addEventListener']>) => {
    closing = f.model.dispose();
    return add(...args);
  };
  await assert.rejects(f.model.decide(f.request, {signal}), {code: 'SESSION_CLOSED'});
  await first;
  await closing;
  assert.equal(f.calls(), 2);
  assert.equal(f.events.filter((event) => event === 'dispose').length, 1);
});
