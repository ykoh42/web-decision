import test from 'node:test';
import assert from 'node:assert/strict';
import {readAssetBatch} from '../src/asset-batch.js';
import {readVerifiedAsset, sha256} from '../src/assets.js';
import {responseFromReader} from './stream-helpers.mjs';

const deferred = <T = void,>() => {
  let resolve!: (value: T) => void, reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((a, b) => {
    resolve = a;
    reject = b;
  });
  return {promise, resolve, reject};
};

test('returns successful initial assets in requested order', async () => {
  const result = await readAssetBatch(['slow', 'fast'], async (name) => {
    if (name === 'slow') await new Promise((resolve) => setImmediate(resolve));
    return name;
  });
  assert.deepEqual(result, ['slow', 'fast']);
});

test('a failed asset stops sibling reads and preserves the original error', async (t) => {
  const pending = deferred<ArrayBuffer>(),
    started = deferred(),
    reason = Error('failure'),
    signals: AbortSignal[] = [],
    progress: number[] = [];
  const bytes = new TextEncoder().encode('model'),
    digest = await sha256(bytes);
  const manifest = {
    files: {'bad.bin': {bytes: bytes.length, sha256: digest}, 'slow.bin': {bytes: bytes.length, sha256: digest}}
  };
  t.mock.method(globalThis, 'fetch', async (url: string, {signal}: {signal: AbortSignal}) => {
    signals.push(signal);
    if (url.endsWith('slow.bin')) {
      started.resolve();
      return responseFromReader(() => pending.promise.then((value) => ({done: false, value: new Uint8Array(value)})));
    }
    await started.promise;
    throw reason;
  });
  const read = (name: string, signal: AbortSignal) =>
    readVerifiedAsset(new URL('https://example.test/'), manifest, name, {
      signal,
      onBytes: (count) => progress.push(count)
    });
  const task = readAssetBatch(['bad.bin', 'slow.bin'], read);
  await assert.rejects(
    task,
    (error) =>
      (error as {code: string; cause: unknown}).code === 'ASSET_LOAD_FAILED' &&
      (error as {cause: unknown}).cause === reason
  );
  assert.equal(signals.length, 2);
  assert(signals.every((signal) => signal.aborted));
  pending.resolve(bytes.buffer);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(progress, []);
});

test('caller cancellation propagates its exact reason and cleans up the listener', async () => {
  const controller = new AbortController(),
    reason = Error('cancel'),
    started = deferred();
  const signal = controller.signal,
    add = signal.addEventListener.bind(signal),
    remove = signal.removeEventListener.bind(signal);
  let listeners = 0;
  signal.addEventListener = (...args: Parameters<typeof add>) => {
    listeners++;
    return add(...args);
  };
  signal.removeEventListener = (...args: Parameters<typeof remove>) => {
    listeners--;
    return remove(...args);
  };
  const task = readAssetBatch(
    ['one'],
    (_name, internal) =>
      new Promise((resolve, reject) => {
        internal.addEventListener('abort', () => reject(internal.reason));
        started.resolve();
      }),
    signal
  );
  await started.promise;
  controller.abort(reason);
  await assert.rejects(task, (error) => error === reason);
  assert.equal(listeners, 0);
});

test('already cancelled input starts no reads', async () => {
  const controller = new AbortController(),
    reason = Error('cancel'),
    calls: unknown[][] = [];
  controller.abort(reason);
  await assert.rejects(
    readAssetBatch(['one'], (...args) => calls.push(args), controller.signal),
    (error) => error === reason
  );
  assert.deepEqual(calls, []);
});

test('caller abort immediately after the final read still rejects the batch', async () => {
  const controller = new AbortController();
  const reason = Error('cancel completed batch');
  const pending = deferred<string>();
  const entered = deferred();
  const job = readAssetBatch(
    ['one'],
    () => {
      entered.resolve();
      return pending.promise;
    },
    controller.signal
  );
  await entered.promise;
  pending.resolve('success');
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(job, (error) => error === reason);
  assert.equal(controller.signal.aborted, true);
});
