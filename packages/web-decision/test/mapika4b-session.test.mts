import assert from 'node:assert/strict';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {createMapika4Session} from '../src/mapika4b/session.mjs';

const tokenizer = {encode: () => ({ids: [1]})};
const labels = {
  names: Array.from({length: 255}, (_, i) => String(i)),
  ids: Array.from({length: 255}, (_, i) => i),
  open: [2]
};
const limits = {controlIds: [17], maxStateTokens: 128, maxInputTokens: 128, temperature: 2.098};
const request = () => ({
  state: 'state',
  questions: [
    {instructions: 'first?', options: ['a', 'b']},
    {instructions: 'second?', options: ['c', 'd']}
  ]
});

test('runs one joint request and applies the model temperature', async () => {
  const calls: Array<{groups: readonly {readonly slot: number}[]}> = [],
    events: string[] = [];
  const runner = {
    async run(packed: {groups: readonly {readonly slot: number}[]}) {
      calls.push(packed);
      const first = new Float32Array(255),
        second = new Float32Array(255);
      first[1] = 2.098;
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
  const session = createMapika4Session({tokenizer, labels, limits, runner});
  const result = await session.decide(request());
  assert.equal(calls.length, 1);
  assert.deepEqual(
    calls[0]!.groups.map((group) => group.slot),
    [1, 2]
  );
  assert.deepEqual(
    result.answers.map((answer) => answer.selected),
    ['b', 'c']
  );
  assert.ok(Math.abs(result.answers[0]!.probabilities[1]!.probability - Math.E / (1 + Math.E)) < 1e-7);
  await session.releaseCachedMemory();
  await session.dispose();
  assert.deepEqual(events, ['release', 'dispose']);
  await assert.rejects(session.decide(request()), {code: 'SESSION_CLOSED'});
});

test('caller cancellation does not let later work overtake the active GPU operation', async () => {
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
  const session = createMapika4Session({tokenizer, labels, limits, runner});
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
