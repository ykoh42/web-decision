import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture} from './fixtures/imajev.mjs';
import {createFixtureImajevSession} from '../src/imajev/fixture-session.mjs';

test('a failed root scope close retains resources and never disposes the runtime', async () => {
  const {model, events} = fixture(createFixtureImajevSession);
  const cause = new Error('root scope close failed');
  const runtime = model.runtime;
  let retained = 0;
  model.disposalGuard.retain = () => retained++;
  runtime.endScope = () => {
    throw cause;
  };

  await assert.rejects(
    model.dispose(),
    (error) => (error as {code: string}).code === 'INFERENCE_ERROR' && (error as {cause: unknown}).cause === cause
  );
  await assert.rejects(model.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(retained, 1);
  assert.equal(model.runtime, runtime);
  assert.deepEqual(model.weights, ['fake']);
  assert.equal(events.includes('dispose'), false);
});

test('device loss before or during root scope close retains resources', async () => {
  for (const phase of ['before', 'during']) {
    const {model, events} = fixture(createFixtureImajevSession);
    const runtime = model.runtime;
    let retained = 0;
    model.disposalGuard.retain = () => retained++;
    if (phase === 'before') model.lifecycle.loss = {reason: 'unknown'};
    else
      runtime.endScope = () => {
        model.lifecycle.loss = {reason: 'unknown'};
      };

    await assert.rejects(model.dispose(), {code: 'INFERENCE_ERROR'});
    assert.equal(retained, 1);
    assert.equal(model.runtime, runtime);
    assert.equal(events.includes('dispose'), false);
  }
});

test('a failing runtime disposal retains session ownership', async () => {
  const {model} = fixture(createFixtureImajevSession);
  const runtime = model.runtime;
  const cause = new Error('runtime disposal failed');
  let retained = 0;
  model.disposalGuard.retain = () => retained++;
  runtime.dispose = () => {
    throw cause;
  };
  await assert.rejects(
    model.dispose(),
    (error) => (error as {code: string}).code === 'INFERENCE_ERROR' && (error as {cause: unknown}).cause === cause
  );
  assert.equal(retained, 1);
  assert.equal(model.runtime, runtime);
  assert.deepEqual(model.weights, ['fake']);
});
