import assert from 'node:assert/strict';
import test from 'node:test';
import {prepareCandidatePlan} from '../src/h2o/plan.mjs';

test('plan abort during the first digest stops before the next digest or metadata parsing', async (t) => {
  let resolveDigest!: (value: ArrayBuffer) => void;
  let calls = 0;
  const pendingDigest = new Promise<ArrayBuffer>((resolve) => {
    resolveDigest = resolve;
  });
  t.mock.method(crypto.subtle, 'digest', () => {
    calls++;
    return pendingDigest;
  });
  const controller = new AbortController(),
    reason = new Error('synthetic cancellation'),
    pending = prepareCandidatePlan(Uint8Array.of(1), Uint8Array.of(2), controller.signal);
  controller.abort(reason);
  resolveDigest(new Uint8Array(32).buffer);
  await assert.rejects(pending, (error) => error === reason);
  assert.equal(calls, 1);
});
