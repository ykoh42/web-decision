import assert from 'node:assert/strict';
import test from 'node:test';
import {waitForAbort} from '../src/abort.js';
import {createOfflinePreparation} from '../src/mapika2b/offline-preparation.mts';

function pending() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => (resolve = done));
  return {resolve, promise};
}

test('shared preparation survives one waiter cancelling and becomes ready only when done', async () => {
  const started = pending();
  const downloaded = pending();
  let attempts = 0;
  const preparation = createOfflinePreparation(false, async (signal) => {
    attempts++;
    started.resolve();
    await waitForAbort(() => downloaded.promise, signal);
  });
  const cancellation = new AbortController();
  const first = preparation.prepareOffline({signal: cancellation.signal});
  const second = preparation.prepareOffline();
  await started.promise;
  cancellation.abort();
  await assert.rejects(first, {name: 'AbortError'});
  assert.equal(preparation.complete, false);
  downloaded.resolve();
  await second;
  assert.equal(attempts, 1);
  assert.equal(preparation.complete, true);
});

test('a failed attempt is retryable and a disposed session cancels preparation', async () => {
  const retryStarted = pending();
  const failure = new Error('quota');
  let attempts = 0;
  const preparation = createOfflinePreparation(true, async (signal) => {
    if (++attempts === 1) throw failure;
    retryStarted.resolve();
    await waitForAbort(() => pending().promise, signal);
  });
  await assert.rejects(preparation.prepareOffline(), (cause) => cause === failure);
  assert.equal(preparation.complete, false);
  const retry = preparation.prepareOffline();
  await retryStarted.promise;
  preparation.dispose();
  await assert.rejects(retry, {code: 'SESSION_CLOSED'});
  await assert.rejects(preparation.prepareOffline(), {code: 'SESSION_CLOSED'});
});

test('a progress callback error rejects its own waiter', async () => {
  const progress = {loadedBytes: 1, totalBytes: 1, fraction: 1} as const;
  const failure = new Error('progress');
  const preparation = createOfflinePreparation(false, async (_signal, onProgress) => onProgress(progress));
  const first = preparation.prepareOffline({
    onProgress() {
      throw failure;
    }
  });
  const seen: unknown[] = [];
  await Promise.all([
    assert.rejects(first, (cause) => cause === failure),
    preparation.prepareOffline({onProgress: (value) => seen.push(value)})
  ]);
  assert.deepEqual(seen, [progress]);
  assert.equal(preparation.complete, true);
});
