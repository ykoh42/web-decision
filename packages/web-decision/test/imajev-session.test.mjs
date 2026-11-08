import assert from 'node:assert/strict';
import test from 'node:test';
import {createFixtureImajevSession} from '../src/imajev/fixture-session.mjs';
import {fixture} from './fixtures/imajev.mjs';
import {decodeChoice} from '../src/imajev/protocol.mjs';
import {safeImajevAnswer} from '../src/imajev/safe-outcome.mjs';

test('the copied real protocol and session preserve answered and abstained results on fake rows', async () => {
  const f = fixture(createFixtureImajevSession),
    result = await f.model.decide(f.request);
  assert.equal(result.tokenCount, 2);
  assert.equal(f.calls(), 2);
  const [unknown, known] = result.answers;
  assert.equal(unknown.abstained, true);
  assert.equal(unknown.selected, null);
  assert.equal(unknown.selectedIndex, null);
  assert.deepEqual(unknown.bestKnown, {option: 'red', index: 0});
  assert.equal(known.abstained, false);
  assert.equal(known.selected, 'yes');
  assert.equal(known.selectedIndex, 0);
  assert.ok(unknown.unknownProbability > 0 && unknown.unknownProbability < 1);
  assert.ok(Math.abs(unknown.probabilities.reduce((sum, item) => sum + item.probability, 0) - 1) < 1e-12);
  await f.model.dispose();
  assert.equal(f.events.filter((event) => event === 'dispose').length, 1);
});

test('unknown winner can abstain even when its probability is below one half', async () => {
  const row = new Float32Array(256);
  row.set([Math.log(0.3), Math.log(0.3), Math.log(0.4)]);
  const f = fixture(createFixtureImajevSession, {rows: [row]});
  f.request.questions.length = 1;
  const answer = (await f.model.decide(f.request)).answers[0];
  assert.equal(answer.abstained, true);
  assert.equal(answer.selected, null);
  assert.ok(Math.abs(answer.unknownProbability - 0.4) < 1e-7);
  assert.ok(answer.unknownProbability < 0.5);
  assert.equal(answer.bestKnown.option, 'red');
  await f.model.dispose();
});

test('unknown dominance retains the relative known probabilities and best known choice', async () => {
  const row = new Float32Array(256);
  row.set([-1000, -900, 1000]);
  const f = fixture(createFixtureImajevSession, {rows: [row]});
  f.request.questions.length = 1;
  const answer = (await f.model.decide(f.request)).answers[0];
  assert.equal(answer.abstained, true);
  assert.equal(answer.selected, null);
  assert.equal(answer.selectedIndex, null);
  assert.deepEqual(answer.bestKnown, {option: 'blue', index: 1});
  assert.equal(answer.unknownProbability, 1);
  assert.ok(Math.abs(answer.probabilities[0].probability - Math.exp(-100)) < 1e-55);
  assert.equal(answer.probabilities[1].probability, 1);
  await f.model.dispose();
});

test('unsupported request and decision fields reject before graph execution', async () => {
  const f = fixture(createFixtureImajevSession);
  for (const field of ['image', 'criteria', 'temperature']) {
    await assert.rejects(f.model.decide({...f.request, [field]: 'unsupported'}), {code: 'INVALID_INPUT'});
    const question = {...f.request.questions[0], [field]: 'unsupported'};
    await assert.rejects(f.model.decide({...f.request, questions: [question]}), {code: 'INVALID_INPUT'});
    await assert.rejects(f.model.decide(f.request, {[field]: 'unsupported'}), {code: 'INVALID_INPUT'});
  }
  const inherited = Object.assign(Object.create({image: 'unsupported'}), f.request.questions[0]);
  await assert.rejects(f.model.decide({...f.request, questions: [inherited]}), {code: 'INVALID_INPUT'});
  assert.equal(f.calls(), 0);
  await f.model.dispose();
});

test('Imajev rejects an over-limit generic option array before reading it', async () => {
  const f = fixture(createFixtureImajevSession);
  const options = new Array(256);
  let reads = 0;
  Object.defineProperty(options, '0', {
    get() {
      reads++;
      throw Error('option getter invoked');
    }
  });
  await assert.rejects(f.model.decide({...f.request, questions: [{...f.request.questions[0], options}]}), {
    code: 'INVALID_INPUT'
  });
  assert.equal(reads, 0);
  assert.equal(f.calls(), 0);
  await f.model.dispose();
});

test('token-ID ties preserve the first known choice and can select abstention', () => {
  const calibration = {schema_version: '1.0', calibration_version: 'synthetic', temperatures: {}};
  const cases = [
    {options: ['red', 'blue'], logits: [1, 0, 1], labelIds: [5, 6, 4], choice: 'red', abstained: true},
    {options: ['red', 'blue'], logits: [1, 1, -1], labelIds: [10, 5, 20], choice: 'red', abstained: false}
  ];
  for (const row of cases) {
    const answer = safeImajevAnswer(decodeChoice(row.options, row.logits, calibration, row.labelIds));
    assert.equal(answer.abstained, row.abstained);
    if (row.abstained) {
      assert.equal(answer.selected, null);
      assert.equal(answer.bestKnown.option, row.choice);
    } else assert.equal(answer.selected, row.choice);
  }
});

test('active fake cancellation returns early while disposal waits for pending work', async () => {
  let start, release;
  const started = new Promise((resolve) => {
    start = resolve;
  });
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const f = fixture(createFixtureImajevSession, {syncGate: gate, syncStarted: start});
  const controller = new AbortController(),
    reason = new Error('fake cancellation');
  const pending = f.model.decide(f.request, {signal: controller.signal});
  await started;
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  const closing = f.model.dispose();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.calls(), 1);
  assert.equal(f.events.includes('dispose'), false);
  release();
  await closing;
  assert.equal(f.events.filter((event) => event === 'dispose').length, 1);
});

test('failed syncs retain the scope until a delayed readback writes, then prevent reuse and disposal', async () => {
  const f = fixture(createFixtureImajevSession);
  f.request.questions.length = 1;
  let finishReadback,
    startSync,
    syncs = 0,
    writes = 0;
  const readback = new Promise((resolve) => {
    finishReadback = resolve;
  });
  const started = new Promise((resolve) => {
    startSync = resolve;
  });
  const runtime = f.model.runtime;
  const empty = runtime.empty.bind(runtime);
  runtime.empty = (shape, dtype, device) => {
    const tensor = empty(shape, dtype, device);
    if (device === 'cpu') {
      const copy = tensor.copyFrom.bind(tensor);
      tensor.copyFrom = (value) => {
        copy(value);
        runtime.lib.webGPUContext.pendingGPUToCPUCopy = readback.then(() => {
          assert.equal(f.events.includes('end'), false, 'readback wrote after scope closure');
          assert.equal(f.events.includes('dispose'), false, 'readback wrote after runtime disposal');
          writes++;
        });
        return tensor;
      };
    }
    return tensor;
  };
  f.model.device.sync = async () => {
    syncs++;
    runtime.lib.webGPUContext.pendingGPUToCPUCopy = null;
    startSync();
    throw new Error('sync failed');
  };
  const controller = new AbortController();
  const reason = new Error('cancelled');
  const first = f.model.decide(f.request, {signal: controller.signal});
  await started;
  const queued = f.model.decide(f.request);
  const releasing = f.model.releaseCachedMemory();
  const disposing = f.model.dispose();
  controller.abort(reason);
  await assert.rejects(first, (error) => error === reason);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(syncs, 1);
  assert.equal(f.events.includes('end'), false);
  assert.equal(f.events.includes('dispose'), false);
  finishReadback();
  await assert.rejects(queued, {code: 'INFERENCE_ERROR'});
  await assert.rejects(releasing, {code: 'INFERENCE_ERROR'});
  await assert.rejects(disposing, {code: 'INFERENCE_ERROR'});
  assert.equal(writes, 1);
  assert.equal(syncs, 2);
  assert.equal(f.calls(), 1);
  assert.equal(f.events.includes('end'), false);
  assert.equal(f.events.includes('dispose'), false);
});

test('a pre-sync graph failure makes one shared fallback sync and retains its original cause', async () => {
  const f = fixture(createFixtureImajevSession),
    cause = new Error('synthetic graph failure');
  f.request.questions.length = 1;
  f.model.fn = () => {
    throw cause;
  };
  await assert.rejects(f.model.decide(f.request), (error) => error.cause === cause);
  assert.equal(f.events.filter((event) => event === 'sync').length, 1);
  assert.ok(f.events.indexOf('sync') < f.events.indexOf('end'));
  await f.model.dispose();
});

test('a failed fallback keeps the original error and retains uncertain resources', async () => {
  const f = fixture(createFixtureImajevSession);
  let syncs = 0;
  f.model.fn = () => {
    throw false;
  };
  f.model.device.sync = async () => {
    syncs++;
    throw 0;
  };
  await assert.rejects(f.model.decide(f.request), (error) => Object.hasOwn(error, 'cause') && error.cause === false);
  assert.equal(syncs, 1);
  assert.equal(f.events.includes('end'), false);
  await assert.rejects(f.model.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(f.events.includes('dispose'), false);
});

test('an observed device loss skips fallback and preserves the graph error', async () => {
  const f = fixture(createFixtureImajevSession),
    cause = new Error('fake graph failure');
  f.model.fn = () => {
    f.model.lifecycle.loss = {reason: 'unknown'};
    throw cause;
  };
  await assert.rejects(f.model.decide(f.request), (error) => error.cause === cause);
  assert.equal(f.events.filter((event) => event === 'sync').length, 0);
  await assert.rejects(f.model.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(f.events.includes('dispose'), false);
});

test('abort returns while a pre-sync failure drains, and disposal waits', async () => {
  let start, release;
  const started = new Promise((resolve) => {
    start = resolve;
  });
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const f = fixture(createFixtureImajevSession, {syncGate: gate, syncStarted: start});
  f.model.fn = () => {
    throw new Error('fake graph failure');
  };
  const controller = new AbortController(),
    reason = new Error('fake cancellation');
  const pending = f.model.decide(f.request, {signal: controller.signal});
  await started;
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  const closing = f.model.dispose();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.events.includes('end'), false);
  assert.equal(f.events.includes('dispose'), false);
  release();
  await closing;
  assert.equal(f.events.filter((event) => event === 'sync').length, 1);
  assert.equal(f.events.filter((event) => event === 'dispose').length, 1);
});
