import assert from 'node:assert/strict';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {createFlashSession} from '../src/pplx/session.mjs';

const marker = '<|fim_suffix|>';
const config = {
  codes: Array.from({length: 255}, (_, index) => String(index)),
  markerId: 248062,
  escapedTokens: [marker],
  addedTokens: [marker],
  addedTokenIds: [248062],
  maxInputTokens: 512
};
const tokenizer = {
  encode(text: string) {
    if (!text.startsWith('<|im_start|>')) return {ids: [3]};
    return {ids: [1, ...Array.from(text.matchAll(/<\|fim_suffix\|>/g), () => 248062), 2]};
  }
};
const request = () => ({
  state: 'state',
  questions: [
    {instructions: 'first?', options: ['a', 'b']},
    {instructions: 'second?', options: ['c', 'd']}
  ]
});

test('runs the whole joint request once and retains question order', async () => {
  const calls: Array<{groups: readonly {readonly index: number}[]}> = [],
    events: string[] = [];
  const runner = {
    async run(packed: {groups: readonly {readonly index: number}[]}) {
      calls.push(packed);
      const first = new Float32Array(255),
        second = new Float32Array(255);
      first[1] = 1;
      second[0] = 1;
      return [first, second];
    },
    async releaseCachedMemory() {
      events.push('release');
    },
    async dispose() {
      events.push('dispose');
    }
  };
  const session = createFlashSession({tokenizer, config, runner});
  const result = await session.decide(request());
  assert.equal(calls.length, 1);
  assert.deepEqual(
    calls[0]!.groups.map((group) => group.index),
    [1, 2]
  );
  assert.deepEqual(
    result.answers.map((answer) => answer.selected),
    ['b', 'c']
  );
  assert.equal(result.tokenCount, 4);
  await session.releaseCachedMemory();
  await session.dispose();
  assert.deepEqual(events, ['release', 'dispose']);
  await assert.rejects(session.decide(request()), {code: 'SESSION_CLOSED'});
});

test('aborting the caller keeps later inference and disposal behind active work', async () => {
  let finish!: () => void,
    started = false,
    calls = 0;
  const runner = {
    async run() {
      calls++;
      if (calls === 1) {
        started = true;
        await new Promise<void>((resolve) => {
          finish = resolve;
        });
      }
      return [new Float32Array(255), new Float32Array(255)];
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = createFlashSession({tokenizer, config, runner});
  const controller = new AbortController();
  const first = session.decide(request(), {signal: controller.signal});
  while (!started) await nextTurn();
  controller.abort();
  await assert.rejects(first, {name: 'AbortError'});
  const second = session.decide(request());
  await nextTurn();
  assert.equal(calls, 1);
  finish();
  await second;
  assert.equal(calls, 2);
  await session.dispose();
});

test('rejects unsupported request fields and excessive questions before running', async () => {
  let calls = 0;
  const runner = {
    async run() {
      calls++;
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = createFlashSession({
    tokenizer,
    config,
    runner: runner as unknown as Parameters<typeof createFlashSession>[0]['runner']
  });
  await assert.rejects(session.decide({...request(), image: 'unsupported'}), {code: 'INVALID_INPUT'});
  await assert.rejects(session.decide({state: 'x', questions: Array(256).fill(request().questions[0])}), {
    code: 'INVALID_INPUT'
  });
  assert.equal(calls, 0);
  await session.dispose();
});

test('a one-option choice returns that option with probability one', async () => {
  const runner = {
    async run() {
      return [new Float32Array(255)];
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = createFlashSession({tokenizer, config, runner});
  const result = await session.decide({state: 'record', questions: [{instructions: 'Choose', options: ['only']}]});
  assert.equal(result.answers[0]!.selected, 'only');
  assert.equal(result.answers[0]!.probabilities[0]!.probability, 1);
  await session.dispose();
});
