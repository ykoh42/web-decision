import assert from 'node:assert/strict';
import test from 'node:test';
import {observeH2oTeardown} from '../src/h2o/teardown.mjs';

test('H2O teardown reports device loss before disposal', async () => {
  const guard = observeH2oTeardown({lost: Promise.resolve({reason: 'unknown'})});
  await assert.rejects(guard.beforeDispose(), /lost before disposal/u);
  await assert.rejects(guard.afterDispose(), /Unexpected GPU device loss/u);
});

test('H2O teardown preserves a rejected device-loss cause', async () => {
  const reason = new Error('device lost');
  const guard = observeH2oTeardown({lost: Promise.reject(reason)});
  await assert.rejects(guard.afterDispose(), {message: 'Unexpected GPU device loss during teardown.', cause: reason});
});

test('H2O teardown handles malformed loss info and stalled device destruction', async () => {
  const guard = observeH2oTeardown({
    lost: Promise.resolve({
      get reason() {
        throw new Error('unreadable');
      }
    })
  });
  await assert.rejects(guard.afterDispose(), /Unexpected GPU device loss/u);
  const stalled = observeH2oTeardown({lost: new Promise(() => {})});
  await assert.rejects(stalled.afterDispose(1), /did not settle/u);
});
