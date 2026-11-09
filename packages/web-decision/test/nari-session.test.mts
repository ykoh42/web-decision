import assert from 'node:assert/strict';
import test from 'node:test';
import {createNariSession} from '../src/nari/session.mjs';

const tokenizer = {
  encode(text: string) {
    return {ids: Array.from({length: Math.ceil(text.length / 3)}, (_, i) => text.codePointAt(i * 3))};
  }
};
const request = {state: 'x', questions: [{instructions: 'x', options: ['a', 'b']}]};

test('session submits all questions jointly and returns model choices', async () => {
  let calls = 0;
  const runner = {
    async run(input: {options: string[][]}) {
      calls++;
      assert.equal(input.options.length, 2);
      assert.deepEqual(input.options, [
        ['a', 'b'],
        ['c', 'd']
      ]);
      return {headScores: Float32Array.of(0, 2, 3, 0), letterScores: new Float32Array(52)};
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = createNariSession({tokenizer, controlIds: [248045], runner});
  const result = await session.decide({
    state: 'x',
    questions: [request.questions[0], {instructions: 'x', options: ['c', 'd']}]
  });
  assert.equal(calls, 1);
  assert.deepEqual(
    result.answers.map((value) => value.selected),
    ['b', 'c']
  );
  await session.dispose();
});

test('rejects excessive questions, options and tokens before the graph', async () => {
  let calls = 0;
  const runner = {
    async run() {
      calls++;
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = createNariSession({tokenizer, controlIds: [248045], runner});
  await assert.rejects(session.decide({state: '', questions: Array(9).fill(request.questions[0])}), {
    code: 'INVALID_INPUT'
  });
  await assert.rejects(
    session.decide({
      state: '',
      questions: [{instructions: 'x', options: Array.from({length: 65}, (_, i) => String(i))}]
    }),
    {code: 'INVALID_INPUT'}
  );
  await assert.rejects(session.decide({...request, state: 'x'.repeat(1500)}), {code: 'INVALID_INPUT'});
  assert.equal(calls, 0);
  await session.dispose();
});

test('cancellation cannot release a pending GPU run before it settles', async () => {
  let finish!: (result: {headScores: Float32Array; letterScores: Float32Array}) => void,
    started!: () => void,
    disposed = false;
  const entered = new Promise<void>((resolve) => {
    started = resolve;
  });
  const runner = {
    run() {
      started();
      return new Promise<{headScores: Float32Array; letterScores: Float32Array}>((resolve) => {
        finish = resolve;
      });
    },
    async releaseCachedMemory() {},
    async dispose() {
      disposed = true;
    }
  };
  const session = createNariSession({tokenizer, controlIds: [248045], runner});
  const controller = new AbortController();
  const pending = session.decide(request, {signal: controller.signal});
  await entered;
  controller.abort();
  await assert.rejects(pending, {name: 'AbortError'});
  const disposing = session.dispose();
  await Promise.resolve();
  assert.equal(disposed, false);
  finish({headScores: new Float32Array(2), letterScores: new Float32Array(26)});
  await disposing;
  assert.equal(disposed, true);
});
