import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {
  createQwen35Session,
  type Qwen35OwnedRunner,
  type Qwen35SessionOptions
} from '../src/strands/qwen35-session.mts';
import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from '../src/strands/qwen35-graph-input.mts';
import {evaluateQwen35Request} from '../src/strands/qwen35-session-adapter.mts';

type RunnerAction = (
  chunk: Qwen35GraphInputChunk,
  input: Qwen35GraphInputRecord,
  call: number
) => unknown | PromiseLike<unknown>;

const utf8 = new TextEncoder();
const specialTokens = [
  ['<|image_pad|>', 248056],
  ['<|vision_end|>', 248054],
  ['<|endoftext|>', 248044],
  ['<bos>', 248043]
] as const;

function tokenizerFixture() {
  const byteDecoder = Object.fromEntries(
    Array.from({length: 256}, (_, byte) => [String.fromCharCode(byte + 0x100), byte])
  );
  const addedTokenIds = new Set(specialTokens.map(([, id]) => id));
  return {
    decoder: {byte_decoder: byteDecoder},
    get_added_tokens_decoder: () => addedTokenIds,
    encode(text: string, {add_special_tokens}: {readonly add_special_tokens: boolean}) {
      const ids: number[] = [];
      const tokens: string[] = [];
      if (add_special_tokens) {
        ids.push(248043);
        tokens.push('<bos>');
      }
      let cursor = 0;
      while (cursor < text.length) {
        const special = specialTokens.find(([token]) => text.startsWith(token, cursor));
        if (special) {
          tokens.push(special[0]);
          ids.push(special[1]);
          cursor += special[0].length;
          continue;
        }
        let end = cursor + 1;
        while (end < text.length && !specialTokens.some(([token]) => text.startsWith(token, end))) end += 1;
        for (const byte of utf8.encode(text.slice(cursor, end).normalize('NFC'))) {
          tokens.push(String.fromCharCode(byte + 0x100));
          ids.push(byte + 1);
        }
        cursor = end;
      }
      return {ids, tokens};
    }
  };
}

function questions(count = 3) {
  const typed = [
    ['route', {type: 'choice', instructions: 'Choose a direction.', criteria: {north: null, south: null}}],
    ['quality', {type: 'score', instructions: 'Rate quality.', criteria: ['low', 'mid', 'high']}],
    ['valid', {type: 'noul', instructions: 'Is this true?'}]
  ] as const;
  return Object.fromEntries(Array.from({length: count}, (_, index) => typed[index % typed.length]!));
}

function probabilities(chunk: Qwen35GraphInputChunk): readonly (readonly number[])[] {
  return chunk.questions.map(({type, options}) => {
    if (type === 'choice') return options.map(({label}) => (label === 'south' ? 0.75 : label === 'north' ? 0.25 : 0));
    if (type === 'score') return options.map((_, index) => [0.2, 0.3, 0.5][index]!);
    return [0.1, 0.9];
  });
}

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<Value>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return {promise, resolve, reject};
}

function runnerFixture(action: RunnerAction = (chunk) => probabilities(chunk)) {
  const events: string[] = [];
  let call = 0;
  const runner: Qwen35OwnedRunner = {
    async run(chunk, input) {
      call += 1;
      const current = call;
      events.push(`run:${current}:start`);
      try {
        const result = await action(chunk, input, current);
        events.push(`run:${current}:end`);
        return result;
      } catch (error) {
        events.push(`run:${current}:error`);
        throw error;
      }
    },
    releaseCachedMemory() {
      events.push('release');
    },
    dispose() {
      events.push('dispose');
    }
  };
  return {
    runner,
    events,
    get callCount() {
      return call;
    }
  };
}

function createSession(runner: Qwen35OwnedRunner, graphOptions: Qwen35SessionOptions['graphOptions'] = {}) {
  return createQwen35Session({
    tokenizer: tokenizerFixture(),
    runner,
    modelName: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
    graphOptions
  });
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  assert.fail('Timed out waiting for the queued Qwen operation.');
}

test('returns the exact typed Strands response from the native adapter', async () => {
  const fixture = runnerFixture();
  const session = createSession(fixture.runner);
  const request = {state: 'A river flows south.', questions: questions()};
  const source = await evaluateQwen35Request(tokenizerFixture(), runnerFixture().runner, request, {
    modelName: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610'
  });
  const result = await session.systemOne(request);

  assert.deepEqual(result, source);
  assert.equal(result.model, 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610');
  assert.deepEqual(Object.keys(result.answers), ['route', 'quality', 'valid']);
  const route = result.answers.route;
  const quality = result.answers.quality;
  const valid = result.answers.valid;
  if (route?.type !== 'choice' || quality?.type !== 'score' || valid?.type !== 'noul')
    assert.fail('Qwen typed answers must retain their source variants.');
  assert.deepEqual(route, {
    type: 'choice',
    choice: 'south',
    probabilities: {north: 0.25, south: 0.75},
    confidence: route.confidence
  });
  assert.ok(Number.isFinite(route.confidence));
  assert.deepEqual(quality.legend, {0: 'low', 1: 'mid', 2: 'high'});
  assert.equal(valid.noul, 0.9);
  assert.deepEqual(result.usage, {input_tokens: result.usage.input_tokens, output_tokens: 3});
  assert.equal(Object.isFrozen(result), true);
  await session.dispose();
});

test('Qwen public session result preserves Python dict order without adding response fields', async () => {
  const fixture = runnerFixture((chunk) =>
    chunk.questions.map(({type}) => (type === 'choice' ? [0.9, 0.1] : [0.1, 0.9]))
  );
  const session = createSession(fixture.runner);
  const result = await session.systemOne({
    state: 'Keep the explicit source order.',
    questions: new Map([
      ['10', {type: 'noul', instructions: 'First numeric question.'}],
      [
        '2',
        {
          type: 'choice',
          instructions: 'Keep numeric option order.',
          criteria: new Map([
            ['10', null],
            ['2', null]
          ])
        }
      ]
    ])
  });

  assert.deepEqual(Object.keys(result), ['model', 'answers', 'usage']);
  assert.deepEqual(Object.keys(result.answers), ['10', '2']);
  assert.deepEqual(Object.keys(result.answers['10']!), ['type', 'noul']);
  const choice = result.answers['2'];
  if (choice?.type !== 'choice') assert.fail('The numeric-key question must retain its choice answer.');
  assert.deepEqual(Object.keys(choice), ['type', 'choice', 'probabilities', 'confidence']);
  assert.equal(choice.choice, '10');
  assert.deepEqual(Object.keys(choice.probabilities), ['10', '2']);

  const serialized = JSON.stringify(result.answers);
  assert.ok(serialized.startsWith('{"10":'));
  assert.ok(serialized.indexOf('"2":') > serialized.indexOf('"10":'));
  assert.ok(serialized.includes('"probabilities":{"10":0.9,"2":0.1}'));
  assert.equal('abstain' in result.answers['10']!, false);
  await session.dispose();
});

test('Qwen session rejects context above the pinned checkpoint limit before graph execution', async () => {
  const fixture = runnerFixture();
  const session = createSession(fixture.runner, {maxLength: 4097});

  await assert.rejects(
    session.systemOne({state: 'state', questions: questions(1)}),
    (error: unknown) => error instanceof RangeError && /4096/u.test(error.message)
  );
  assert.equal(fixture.callCount, 0);
  await session.dispose();
});

test('Qwen systemOne exposes the source strict-window policy and validates its type', async () => {
  const fixture = runnerFixture();
  const session = createSession(fixture.runner);
  const request = {
    state: 's'.repeat(5000),
    questions: {q: {type: 'noul', instructions: 'Is this true?'}}
  };

  await assert.rejects(
    session.systemOne(request, {strictWindow: true}),
    (error: unknown) => error instanceof RangeError && /context window of 4096 tokens/u.test(error.message)
  );
  assert.equal(fixture.callCount, 0);
  await assert.rejects(session.systemOne(request, {strictWindow: 1 as never}), {
    code: 'INVALID_INPUT'
  });
  assert.equal(fixture.callCount, 0);
  await session.dispose();
});

test('preserves integer-like question order when the request uses a Map', async () => {
  let sourceOrder: string[] = [];
  const fixture = runnerFixture((chunk, input) => {
    sourceOrder = input.request.questions.map(({name}) => name);
    return probabilities(chunk);
  });
  const session = createSession(fixture.runner);
  await session.systemOne({
    state: 'Ordered questions.',
    questions: new Map([
      ['10', {type: 'noul', instructions: 'Is ten true?'}],
      ['2', {type: 'noul', instructions: 'Is two true?'}]
    ])
  });
  assert.deepEqual(sourceOrder, ['10', '2']);
  await session.dispose();
});

test('serializes inference and memory release in request order', async () => {
  const firstRun = deferred<unknown>();
  const fixture = runnerFixture((_chunk, _input, call) => (call === 1 ? firstRun.promise : probabilities(_chunk)));
  const session = createSession(fixture.runner);
  const first = session.systemOne({state: 'state', questions: questions(1)});
  await waitUntil(() => fixture.events.includes('run:1:start'));
  const second = session.systemOne({state: 'state', questions: questions(1)});
  const release = session.releaseCachedMemory();
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(fixture.events, ['run:1:start']);

  firstRun.resolve([[0.1, 0.9]]);
  await Promise.all([first, second, release]);
  assert.deepEqual(fixture.events, ['run:1:start', 'run:1:end', 'run:2:start', 'run:2:end', 'release']);
  await session.dispose();
  assert.equal(fixture.events.at(-1), 'dispose');
});

test('snapshots queued request data when systemOne is called', async () => {
  const firstRun = deferred<unknown>();
  let queuedStateText = '';
  const fixture = runnerFixture((chunk, input, call) => {
    if (call === 1) return firstRun.promise;
    queuedStateText = input.request.stateText;
    return probabilities(chunk);
  });
  const session = createSession(fixture.runner);
  const first = session.systemOne({state: 'active request', questions: questions(1)});
  await waitUntil(() => fixture.events.includes('run:1:start'));

  const queuedRequest = {state: 'state at call time', questions: questions(1)};
  const second = session.systemOne(queuedRequest);
  queuedRequest.state = 'state after call';
  firstRun.resolve([[0.1, 0.9]]);

  await Promise.all([first, second]);
  assert.match(queuedStateText, /state at call time/u);
  assert.doesNotMatch(queuedStateText, /state after call/u);
  await session.dispose();
});

test('aborting queued inference prevents its graph call', async () => {
  const firstRun = deferred<unknown>();
  const fixture = runnerFixture((chunk, _input, call) => (call === 1 ? firstRun.promise : probabilities(chunk)));
  const session = createSession(fixture.runner);
  const first = session.systemOne({state: 'state', questions: questions(1)});
  await waitUntil(() => fixture.events.includes('run:1:start'));
  const controller = new AbortController();
  const queued = session.systemOne({state: 'state', questions: questions(1)}, {signal: controller.signal});
  controller.abort(new Error('queued request cancelled'));
  await assert.rejects(queued, /queued request cancelled/u);

  firstRun.resolve([[0.1, 0.9]]);
  await first;
  await session.releaseCachedMemory();
  assert.equal(fixture.callCount, 1);
  assert.equal(fixture.events.at(-1), 'release');
  await session.dispose();
});

test('aborting an in-flight chunk stops later chunks after the current run settles', async () => {
  const firstRun = deferred<unknown>();
  const fixture = runnerFixture((chunk, _input, call) => (call === 1 ? firstRun.promise : probabilities(chunk)));
  const session = createSession(fixture.runner);
  const controller = new AbortController();
  const inference = session.systemOne(
    {
      state: 'state',
      questions: {first: {type: 'noul', instructions: 'true?'}, second: {type: 'noul', instructions: 'also true?'}}
    },
    {maxBatch: 1, signal: controller.signal}
  );
  await waitUntil(() => fixture.events.includes('run:1:start'));
  controller.abort(new Error('active request cancelled'));
  await assert.rejects(inference, /active request cancelled/u);

  firstRun.resolve([[0.1, 0.9]]);
  await session.releaseCachedMemory();
  assert.equal(fixture.callCount, 1);
  assert.deepEqual(fixture.events, ['run:1:start', 'run:1:end', 'release']);
  await session.dispose();
});

test('recovers the queue after runner failure', async () => {
  const fixture = runnerFixture((chunk, _input, call) => {
    if (call === 1) throw new Error('temporary graph failure');
    return probabilities(chunk);
  });
  const session = createSession(fixture.runner);
  await assert.rejects(session.systemOne({state: 'state', questions: questions(1)}), /temporary graph failure/u);
  const result = await session.systemOne({state: 'state', questions: questions(1)});
  assert.equal(result.answers.route?.type, 'choice');
  assert.deepEqual(fixture.events, ['run:1:start', 'run:1:error', 'run:2:start', 'run:2:end']);
  await session.dispose();
});

test('disposal waits for active inference and rejects later calls', async () => {
  const activeRun = deferred<unknown>();
  const fixture = runnerFixture(() => activeRun.promise);
  const session = createSession(fixture.runner);
  const inference = session.systemOne({state: 'state', questions: questions(1)});
  await waitUntil(() => fixture.events.includes('run:1:start'));
  const disposal = session.dispose();
  assert.strictEqual(session.dispose(), disposal);
  await assert.rejects(session.systemOne({state: 'state', questions: questions(1)}), (error: unknown) => {
    return error instanceof Error && 'code' in error && error.code === 'SESSION_CLOSED';
  });
  await assert.rejects(session.releaseCachedMemory(), (error: unknown) => {
    return error instanceof Error && 'code' in error && error.code === 'SESSION_CLOSED';
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(fixture.events, ['run:1:start']);

  activeRun.resolve([[0.1, 0.9]]);
  await Promise.all([inference, disposal]);
  assert.deepEqual(fixture.events, ['run:1:start', 'run:1:end', 'dispose']);
});

test('requires all owned-runner lifecycle methods', () => {
  assert.throws(
    () => createQwen35Session({tokenizer: tokenizerFixture(), runner: {run() {}} as never, modelName: 'qwen'}),
    /owned graph runner/u
  );
});

test('Qwen image requests recover after malformed input and cancellation', async () => {
  const gate = deferred<void>();
  const started = deferred<void>();
  const fixture = runnerFixture(async (chunk, input, call) => {
    assert.equal(input.imageCount, 1);
    if (call === 1) {
      started.resolve();
      await gate.promise;
    }
    return probabilities(chunk);
  });
  const session = createSession(fixture.runner);
  const encoded = (await readFile(new URL('./fixtures/qwen35-image-input/single-pixel.png', import.meta.url))).toString(
    'base64'
  );
  const request = {state: 'image', images: [encoded], questions: questions(1)};
  try {
    await assert.rejects(session.systemOne({...request, images: ['%%%']}), /base64/i);
    assert.equal(fixture.callCount, 0);
    const controller = new AbortController();
    const first = session.systemOne(request, {signal: controller.signal});
    await started.promise;
    controller.abort();
    await assert.rejects(first, {name: 'AbortError'});
    gate.resolve();
    const result = await session.systemOne(request);
    assert.equal(result.usage.output_tokens, 1);
    assert.equal(fixture.callCount, 2);
  } finally {
    gate.resolve();
    await session.dispose();
  }
});
