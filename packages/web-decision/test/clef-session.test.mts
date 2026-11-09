import test from 'node:test';
import assert from 'node:assert/strict';
import {createClefSession} from '../src/clef/session.mjs';
import {approximateSoftmax} from '../src/clef/softmax.mjs';
import {packChoiceRequest, unpackChoiceScores} from '../src/clef/protocol.mjs';

type ClefSessionOptions = Parameters<typeof createClefSession>[0];
type ClefRunner = ClefSessionOptions['runner'];
type ClefTokenizer = ClefSessionOptions['tokenizer'];
type ClefDecisionOptions = NonNullable<Parameters<ReturnType<typeof createClefSession>['decide']>[1]>;

const tokenizer: ClefTokenizer = {encode: (text) => ({ids: Array.from({length: Math.ceil(text.length / 4)}, () => 1)})};
const request = () => ({state: 's', questions: [{instructions: 'choose', options: ['a', 'b']}]});
const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return {promise, resolve};
};
const tick = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));

function makeSession(run: ClefRunner['run'], events: string[] = [], inputTokenizer: ClefTokenizer = tokenizer) {
  return createClefSession({
    tokenizer: inputTokenizer,
    controlIds: [2],
    runner: {
      run,
      async releaseCachedMemory() {
        events.push('release');
      },
      async dispose() {
        events.push('dispose');
      }
    }
  });
}

test('abort returns promptly while active work, queued work and dispose remain serialized', async () => {
  const started = deferred(),
    gate = deferred(),
    events: string[] = [];
  let calls = 0;
  const session = makeSession(async () => {
    events.push('run');
    if (++calls === 1) {
      started.resolve();
      await gate.promise;
    }
    return [Float32Array.of(1, 0)];
  }, events);
  const controller = new AbortController();
  const first = session.decide(request(), {signal: controller.signal});
  await started.promise;
  controller.abort();
  await assert.rejects(first, {name: 'AbortError'});
  const second = session.decide(request());
  const release = session.releaseCachedMemory();
  const disposal = session.dispose();
  await tick();
  assert.deepEqual(events, ['run']);
  gate.resolve();
  assert.equal((await second).answers[0]!.selected, 'a');
  await release;
  await disposal;
  assert.deepEqual(events, ['run', 'run', 'release', 'dispose']);
  assert.equal(session.dispose(), disposal);
  await assert.rejects(session.decide(request()), {code: 'SESSION_CLOSED'});
});

test('a cancelled queued decision does not invoke the runner', async () => {
  const started = deferred(),
    gate = deferred(),
    events: string[] = [];
  const session = makeSession(async () => {
    events.push('run');
    started.resolve();
    await gate.promise;
    return [Float32Array.of(0, 0)];
  }, events);
  const first = session.decide(request());
  await started.promise;
  const controller = new AbortController(),
    second = session.decide(request(), {signal: controller.signal});
  controller.abort();
  await assert.rejects(second, {name: 'AbortError'});
  const disposal = session.dispose();
  gate.resolve();
  await first;
  await disposal;
  assert.deepEqual(events, ['run', 'dispose']);
});

test('the default request path has no one-megabyte text-byte cap', async () => {
  const state = 'x'.repeat(1_048_577);
  const compressingTokenizer: ClefTokenizer = {
    encode(text) {
      return {ids: text === state ? [] : Array(Math.ceil(text.length / 4)).fill(1)};
    }
  };
  let calls = 0;
  const session = makeSession(
    async (packed) => {
      calls++;
      assert.ok(packed.ids.length <= 512);
      return [Float32Array.of(1, 0)];
    },
    [],
    compressingTokenizer
  );

  const result = await session.decide({...request(), state});
  assert.equal(result.answers[0]!.selected, 'a');
  assert.equal(calls, 1);
  await session.dispose();
});

test('the session keeps the 512-token native graph boundary before inference', async () => {
  const characterTokenizer: ClefTokenizer = {encode: (text) => ({ids: Array(text.length).fill(1)})};
  const questions = [{instructions: 'q', options: ['a', 'b']}];
  const base = packChoiceRequest(characterTokenizer, {state: '', questions}, [2]);
  const stateLength = 512 - base.ids.length;
  let calls = 0;
  const session = makeSession(
    async (packed) => {
      calls++;
      assert.equal(packed.ids.length, 512);
      return [Float32Array.of(1, 0)];
    },
    [],
    characterTokenizer
  );

  const result = await session.decide({state: 's'.repeat(stateLength), questions});
  assert.equal(result.tokenCount, 512);
  assert.equal(calls, 1);
  await assert.rejects(session.decide({state: 's'.repeat(stateLength + 1), questions}), {code: 'INVALID_INPUT'});
  assert.equal(calls, 1);
  await session.dispose();
});

test('joint questions, invalid scores, request validation and near-tie difference', async () => {
  const combined = {
    state: 's',
    questions: [
      {instructions: 'q1', options: ['a', 'b']},
      {instructions: 'q2', options: ['c', 'd']}
    ]
  };
  let calls = 0;
  const session = makeSession(async (packed) => {
    calls++;
    assert.equal(packed.questions.length, 2);
    return [Float32Array.of(0, 1), Float32Array.of(2, 0)];
  });
  const result = await session.decide(combined);
  assert.deepEqual(
    result.answers.map((answer) => answer.selected),
    ['b', 'c']
  );
  await assert.rejects(session.decide({...combined, images: []}), {code: 'INVALID_INPUT'});
  await assert.rejects(session.decide(combined, {temperature: 1} as unknown as ClefDecisionOptions), {
    code: 'INVALID_INPUT'
  });
  assert.equal(calls, 1);
  await session.dispose();
  const bad = makeSession(async () => [Float32Array.of(NaN, 0)]);
  await assert.rejects(bad.decide(request()), {code: 'INFERENCE_ERROR'});
  await bad.dispose();
  const packed = packChoiceRequest(
    tokenizer,
    {state: 's', questions: [{instructions: 'q', options: ['a', 'b', 'c']}]},
    [2]
  );
  const logits = [Float32Array.of(0.6212127208709717, 0.6212127804756165, -0.6041103601455688)];
  const probs = approximateSoftmax(logits);
  assert.equal(probs[0]![0], probs[0]![1]);
  assert.equal(unpackChoiceScores(packed, logits, probs).answers[0]!.selected, 'a');
  // Recorded Torch CPU FP32 probabilities differ here: it selects b.
  assert.equal(
    unpackChoiceScores(packed, logits, [Float32Array.of(0.4359838366508484, 0.4359838664531708, 0.12803223729133606)])
      .answers[0]!.selected,
    'b'
  );
});
