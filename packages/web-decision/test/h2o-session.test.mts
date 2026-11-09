import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createH2oSession} from '../src/h2o/session.mjs';

type SessionOptions = Parameters<typeof createH2oSession>[0];
type Runner = NonNullable<SessionOptions['runner']>;
type GraphInputs = Parameters<Runner['run']>[0];
type Request = NonNullable<Parameters<Awaited<ReturnType<typeof createH2oSession>>['decide']>[0]>;
const plan = JSON.parse(await readFile(new URL('./fixtures/h2o-labels.json', import.meta.url), 'utf8')) as NonNullable<
  SessionOptions['plan']
>;
const request = () => ({state: 'record', questions: [{instructions: 'Choose', options: ['first', 'second']}]});

async function sessionWith(runner: Runner, encode: (prompt: string) => {ids: number[]} = () => ({ids: [1, 2, 3]})) {
  return createH2oSession({tokenizer: {encode}, plan, addedTokens: ['<reserved>'], runner});
}

function setReentrantHook(
  hook: string,
  input: Request,
  options: {signal?: AbortSignal},
  signal: AbortSignal,
  close: () => void
) {
  if (hook === 'state')
    Object.defineProperty(input, 'state', {
      enumerable: true,
      get() {
        close();
        return 'record';
      }
    });
  if (hook === 'signal-option')
    Object.defineProperty(options, 'signal', {
      enumerable: true,
      get() {
        close();
        return signal;
      }
    });
  else if (hook.startsWith('signal-')) options.signal = signal;
  if (hook === 'signal-validation')
    Object.defineProperty(signal, 'addEventListener', {
      get() {
        close();
        return AbortSignal.prototype.addEventListener;
      }
    });
  if (hook === 'signal-listener')
    signal.addEventListener = function (...args: Parameters<AbortSignal['addEventListener']>) {
      close();
      return AbortSignal.prototype.addEventListener.apply(this, args);
    };
  if (hook === 'signal-aborted')
    Object.defineProperty(signal, 'aborted', {
      get() {
        close();
        return false;
      }
    });
  if (hook === 'signal-check') signal.throwIfAborted = () => close();
}

test('H2O choice session packs independent questions and reports temperature-scaled results', async () => {
  const calls: GraphInputs[] = [],
    prompts: string[] = [],
    events: string[] = [];
  const session = await sessionWith(
    {
      async run(input: GraphInputs) {
        calls.push(input);
        const scores = new Float32Array(255);
        scores[1] = 1;
        return scores;
      },
      async releaseCachedMemory() {
        events.push('release');
      },
      async dispose() {
        events.push('dispose');
      }
    },
    (prompt) => {
      prompts.push(prompt);
      return {ids: [1, 2, 3]};
    }
  );
  const result = await session.decide({
    state: 'record',
    questions: [request().questions[0]!, {instructions: 'Again', options: ['third', 'fourth']}]
  });
  assert.equal(result.tokenCount, 6);
  assert.deepEqual(
    result.answers.map((answer) => answer.selected),
    ['second', 'fourth']
  );
  assert.ok(Math.abs(result.answers[0]!.probabilities[1]!.probability - 1 / (1 + Math.exp(-1 / 0.75))) < 1e-12);
  assert.match(prompts[0]!, /record: record\nquestion: Choose/);
  assert.deepEqual(
    calls[0]!.map((input) => input.dtype),
    ['int32', 'int32', 'float16', 'float16', 'int32']
  );
  assert.deepEqual([...calls[0]![2]!.data], Array(9).fill(0));
  assert.deepEqual([...calls[0]![3]!.data], [0x3c00, 0x3c00, 0x3c00]);
  assert.deepEqual([...calls[0]![4]!.data], [2]);
  await session.releaseCachedMemory();
  await session.dispose();
  assert.deepEqual(events, ['release', 'dispose']);
  await assert.rejects(session.decide(request()), {code: 'SESSION_CLOSED'});
});

test('invalid common fields, sparse values, reserved tokens and bad tokens fail before inference', async () => {
  let calls = 0,
    tokenizations = 0;
  const runner = {
    async run() {
      calls++;
      return new Float32Array(255);
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = await sessionWith(runner, () => {
    tokenizations++;
    return {ids: [1, 2]};
  });
  const sparseOptions = ['a', 'b', 'c'];
  delete sparseOptions[1];
  for (const [input, options] of [
    [{...request(), image: 'ignored'}, {}],
    [{...request(), questions: [{...request().questions[0]!, criteria: 'ignored'}]}, {}],
    [request(), {temperature: 2}],
    [{state: 'record', questions: [{instructions: 'Choose', options: sparseOptions}]}, {}],
    [{state: '<reserved>', questions: request().questions}, {}]
  ] as Array<[Request, Record<string, unknown>]>)
    await assert.rejects(session.decide(input, options), {code: 'INVALID_INPUT'});
  assert.equal(tokenizations, 0);
  assert.equal(calls, 0);
  await session.dispose();
  const sparseIds = [1, 2, 3];
  delete sparseIds[1];
  const bad = await sessionWith(runner, () => ({ids: sparseIds}));
  await assert.rejects(bad.decide(request()), {code: 'INVALID_INPUT'});
  assert.equal(calls, 0);
  await bad.dispose();
});

test('H2O accepts 257 independent questions and executes each full graph sequentially', async () => {
  let calls = 0,
    tokenizations = 0,
    active = 0,
    maxActive = 0;
  const session = await sessionWith(
    {
      async run(inputs: GraphInputs) {
        calls++;
        active++;
        maxActive = Math.max(maxActive, active);
        assert.deepEqual(inputs[0]!.shape, [1, 2]);
        await Promise.resolve();
        active--;
        return new Float32Array(255);
      },
      async releaseCachedMemory() {},
      async dispose() {}
    },
    () => {
      tokenizations++;
      return {ids: [1, 2]};
    }
  );
  const questions = Array.from({length: 257}, (_, index) => ({
    instructions: `Choose route ${index}`,
    options: ['first', 'second']
  }));
  const result = await session.decide({state: 'record', questions});
  assert.equal(result.answers.length, 257);
  assert.equal(result.tokenCount, 514);
  assert.equal(tokenizations, 257);
  assert.equal(calls, 257);
  assert.equal(maxActive, 1);
  await session.dispose();
});

test('queued inputs are snapshotted and abort does not release in-flight work', async () => {
  let finish!: (scores: Float32Array) => void;
  const events: string[] = [],
    prompts: string[] = [];
  const runner = {
    run() {
      events.push('run');
      return new Promise<Float32Array>((resolve) => {
        finish = resolve;
      });
    },
    async releaseCachedMemory() {},
    async dispose() {
      events.push('dispose');
    }
  };
  const session = await sessionWith(runner, (prompt) => {
    prompts.push(prompt);
    return {ids: [1, 2]};
  });
  const input = request(),
    controller = new AbortController();
  const result = session.decide(input, {signal: controller.signal});
  input.questions[0]!.options[1] = 'changed';
  await Promise.resolve();
  controller.abort();
  await assert.rejects(result, {name: 'AbortError'});
  const disposal = session.dispose();
  assert.deepEqual(events, ['run']);
  assert.match(prompts[0]!, /second: second/);
  finish(new Float32Array(255));
  await disposal;
  assert.deepEqual(events, ['run', 'dispose']);
});

test('invalid signals and failed abort setup never schedule inference', async () => {
  let calls = 0,
    tokenizations = 0;
  const session = await sessionWith(
    {
      async run() {
        calls++;
        return new Float32Array(255);
      },
      async releaseCachedMemory() {},
      async dispose() {}
    },
    () => {
      tokenizations++;
      return {ids: [1, 2]};
    }
  );
  for (const signal of [{throwIfAborted() {}}, Object.create(AbortSignal.prototype), false] as unknown as AbortSignal[])
    await assert.rejects(session.decide(request(), {signal}), {code: 'INVALID_INPUT'});
  assert.equal(tokenizations, 0);

  const signal = new AbortController().signal;
  signal.addEventListener = () => {
    throw new Error('listener setup failed');
  };
  await assert.rejects(session.decide(request(), {signal}), /listener setup failed/);
  await session.releaseCachedMemory();
  assert.equal(calls, 0);
  await session.dispose();
});

test('reentrant disposal during request or abort setup prevents scheduling inference', async () => {
  for (const hook of [
    'state',
    'signal-option',
    'signal-validation',
    'signal-listener',
    'signal-aborted',
    'signal-check',
    'tokenizer'
  ]) {
    const events: string[] = [];
    let session!: Awaited<ReturnType<typeof createH2oSession>>,
      disposal: Promise<void> | undefined,
      tokenizations = 0;
    const close = () => {
      if (!disposal) disposal = session.dispose();
    };
    const runner = {
      async run() {
        events.push('run');
        return new Float32Array(255);
      },
      async releaseCachedMemory() {},
      async dispose() {
        events.push('dispose-start');
        await Promise.resolve();
        events.push('dispose-end');
      }
    };
    session = await sessionWith(runner, () => {
      tokenizations++;
      if (hook === 'tokenizer') close();
      return {ids: [1, 2]};
    });
    const input = request(),
      signal = new AbortController().signal,
      options = {};
    setReentrantHook(hook, input, options, signal, close);
    await assert.rejects(session.decide(input, options), {code: 'SESSION_CLOSED'}, hook);
    assert.ok(disposal, hook);
    await disposal;
    assert.deepEqual(events, ['dispose-start', 'dispose-end'], hook);
    if (['state', 'signal-option', 'signal-validation', 'signal-check'].includes(hook))
      assert.equal(tokenizations, 0, hook);
  }
});

test('a decision queued before disposal completes before the runner is disposed', async () => {
  const events: string[] = [];
  const session = await sessionWith({
    async run() {
      events.push('run');
      return new Float32Array(255);
    },
    async releaseCachedMemory() {},
    async dispose() {
      events.push('dispose');
    }
  });
  const decision = session.decide(request());
  const disposal = session.dispose();
  await decision;
  await disposal;
  assert.deepEqual(events, ['run', 'dispose']);
});

test('reentrant disposal during an accepted signal check waits for inference', async () => {
  const events: string[] = [],
    signal = new AbortController().signal;
  let session!: Awaited<ReturnType<typeof createH2oSession>>,
    disposal: Promise<void> | undefined,
    finish!: (scores: Float32Array) => void,
    checks = 0;
  session = await sessionWith({
    run() {
      events.push('run');
      return new Promise<Float32Array>((resolve) => {
        finish = resolve;
      });
    },
    async releaseCachedMemory() {},
    async dispose() {
      events.push('dispose');
    }
  });
  signal.throwIfAborted = () => {
    if (++checks === 3) disposal = session.dispose();
  };
  const decision = session.decide(request(), {signal});
  await Promise.resolve();
  assert.ok(disposal);
  assert.deepEqual(events, ['run']);
  finish(new Float32Array(255));
  await decision;
  await disposal;
  assert.deepEqual(events, ['run', 'dispose']);
});

test('cancelling a queued decision skips it without interrupting earlier work', async () => {
  let finish!: (scores: Float32Array) => void;
  const events: string[] = [];
  const session = await sessionWith({
    run() {
      events.push('run');
      if (events.length === 1)
        return new Promise<Float32Array>((resolve) => {
          finish = resolve;
        });
      return Promise.resolve(new Float32Array(255));
    },
    async releaseCachedMemory() {
      events.push('release');
    },
    async dispose() {
      events.push('dispose');
    }
  });
  const first = session.decide(request());
  await Promise.resolve();
  const controller = new AbortController();
  const cancelled = session.decide(request(), {signal: controller.signal});
  controller.abort();
  await assert.rejects(cancelled, {name: 'AbortError'});
  const release = session.releaseCachedMemory();
  finish(new Float32Array(255));
  await first;
  await release;
  await session.dispose();
  assert.deepEqual(events, ['run', 'release', 'dispose']);
});

test('each question keeps its own tokens when the tokenizer reuses an output array', async () => {
  const shared = [1, 2],
    seen: number[][] = [];
  let next = 0;
  const runner = {
    async run(input: GraphInputs) {
      seen.push([...input[0]!.data]);
      return new Float32Array(255);
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = await sessionWith(runner, () => {
    shared[0] = ++next;
    return {ids: shared};
  });
  const result = session.decide({
    state: 'record',
    questions: [request().questions[0]!, {instructions: 'Again', options: ['third', 'fourth']}]
  });
  shared[0] = 99;
  await result;
  assert.deepEqual(seen, [
    [1, 2],
    [2, 2]
  ]);
  await session.dispose();
});

test('invalid head identity and model outputs fail closed', async () => {
  const runner = {
    async run() {
      return new Float32Array(254);
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  await assert.rejects(
    createH2oSession({
      tokenizer: {encode() {}} as unknown as NonNullable<SessionOptions['tokenizer']>,
      plan: {...plan, labelIds: [...plan.labelIds].reverse()},
      addedTokens: [],
      runner
    }),
    {code: 'INVALID_ASSET'}
  );
  const session = await sessionWith(runner);
  await assert.rejects(session.decide(request()), {code: 'INFERENCE_ERROR'});
  await session.dispose();
});
