import test from 'node:test';
import assert from 'node:assert/strict';
import {
  commitCachedManifest,
  openAssetCache,
  readManifestResponse,
  readVerifiedAsset,
  sha256,
  waitForAbort
} from '../src/assets.js';
import {responseFromReader} from './stream-helpers.mjs';

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return {promise, resolve};
}

test('an already cancelled operation never starts', async () => {
  const controller = new AbortController(),
    reason = Error('already stopped');
  controller.abort(reason);
  let started = false;
  assert.throws(
    () =>
      waitForAbort(() => {
        started = true;
      }, controller.signal),
    (error) => error === reason
  );
  assert.equal(started, false);
});

test('the abort listener is removed on success, failure and cancellation', async () => {
  for (const mode of ['resolve', 'reject', 'cancel']) {
    const controller = new AbortController(),
      signal = controller.signal,
      started = deferred(),
      release = deferred();
    const originalAdd = signal.addEventListener.bind(signal),
      originalRemove = signal.removeEventListener.bind(signal);
    let listeners = 0;
    signal.addEventListener = (...args) => {
      listeners++;
      return originalAdd(...args);
    };
    signal.removeEventListener = (...args) => {
      listeners--;
      return originalRemove(...args);
    };
    const task = waitForAbort(async () => {
      started.resolve();
      await release.promise;
      if (mode === 'reject') throw Error('failure');
      return 'ok';
    }, signal);
    const result = task.then(
      (value) => value,
      (error) => error
    );
    await started.promise;
    if (mode === 'cancel') controller.abort(Error('cancel'));
    else release.resolve();
    const value = await result;
    assert.equal(listeners, 0);
    if (mode === 'resolve') assert.equal(value, 'ok');
    else assert(value instanceof Error);
    release.resolve();
  }
});

test('a late cache write may finish after cancellation and stays verifiable', async (t) => {
  const started = deferred(),
    release = deferred(),
    completed = deferred(),
    controller = new AbortController(),
    reason = Error('cancel write');
  const data = new TextEncoder().encode('valid data'),
    hash = await sha256(data),
    entries = new Map();
  const manifest = {files: {'file.bin': {bytes: data.length, sha256: hash}}};
  const cache = {
    match: async (key) => entries.get(String(key))?.clone(),
    put: async (key, response) => {
      started.resolve();
      await release.promise;
      entries.set(String(key), response.clone());
      completed.resolve();
    }
  };
  t.mock.method(globalThis, 'fetch', async () => new Response(data));
  const read = readVerifiedAsset(new URL('https://example.test/model/'), manifest, 'file.bin', {
    cache,
    signal: controller.signal
  });
  const observed = read.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    assert.equal(
      await Promise.race([observed, new Promise((resolve) => setTimeout(() => resolve('still pending'), 100))]),
      reason
    );
  } finally {
    release.resolve();
    await observed;
    await completed.promise;
  }
  const recovered = await readVerifiedAsset(new URL('https://example.test/model/'), manifest, 'file.bin', {
    cache,
    offline: true
  });
  assert.equal(new TextDecoder().decode(recovered), 'valid data');
});

test('a cancelled manifest write cannot overtake a later cooperating update', async () => {
  const started = deferred(),
    release = deferred(),
    commits = [],
    calls = [];
  const cache = {
    put: async (_url, response) => {
      const value = await response.json();
      if (value.version === 'old') {
        started.resolve();
        await release.promise;
      }
      commits.push(value.version);
    }
  };
  let tail = Promise.resolve();
  const locks = {
    request: (name, options, operation) => {
      calls.push(name);
      const task = tail.then(() => {
        options.signal?.throwIfAborted();
        return operation();
      });
      tail = task.catch(() => {});
      return task;
    }
  };
  const controller = new AbortController(),
    reason = Error('stop old update');
  const old = commitCachedManifest(
    cache,
    'https://example.test/model/manifest.json',
    {version: 'old'},
    controller.signal,
    locks
  );
  const oldResult = old.then(
    () => 'resolved',
    (error) => error
  );
  await started.promise;
  controller.abort(reason);
  assert.equal(await oldResult, reason);
  const newer = commitCachedManifest(
    cache,
    'https://example.test/model/manifest.json',
    {version: 'new'},
    undefined,
    locks
  );
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(commits, []);
  release.resolve();
  await newer;
  assert.deepEqual(commits, ['old', 'new']);
  assert.equal(calls.length, 2);
  assert.equal(calls[0], calls[1]);
});

test('without Web Locks a cancelled caller waits for the manifest write to finish', async () => {
  const started = deferred(),
    release = deferred(),
    controller = new AbortController(),
    reason = Error('stop update');
  const cache = {
    put: async () => {
      started.resolve();
      await release.promise;
    }
  };
  const task = commitCachedManifest(
    cache,
    'https://example.test/model/manifest.json',
    {version: 'old'},
    controller.signal,
    null
  );
  const observed = task.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    assert.equal(
      await Promise.race([observed, new Promise((resolve) => setTimeout(() => resolve('still pending'), 100))]),
      'still pending'
    );
  } finally {
    release.resolve();
  }
  assert.equal(await observed, reason);
});

test('a stalled cache lookup does not delay a cancelled model-file read', async () => {
  const started = deferred(),
    release = deferred(),
    controller = new AbortController(),
    reason = Error('stop loading');
  const cache = {
    match: async () => {
      started.resolve();
      await release.promise;
      return undefined;
    }
  };
  const manifest = {files: {'file.bin': {bytes: 1, sha256: '0'.repeat(64)}}};
  const read = readVerifiedAsset(new URL('https://example.test/model/'), manifest, 'file.bin', {
    cache,
    signal: controller.signal,
    offline: true
  });
  const observed = read.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    const result = await Promise.race([
      observed,
      new Promise((resolve) => setTimeout(() => resolve('still pending'), 100))
    ]);
    assert.equal(result, reason, 'The read must reject with the caller reason while the cache remains stalled');
  } finally {
    release.resolve();
    await observed;
  }
});

test('a stalled cached response body stops delaying cancellation', async () => {
  const started = deferred(),
    release = deferred(),
    controller = new AbortController(),
    reason = Error('cancel body');
  const cache = {
    match: async () =>
      responseFromReader(async () => {
        started.resolve();
        await release.promise;
        throw Error('late body failure');
      })
  };
  const manifest = {files: {'file.bin': {bytes: 1, sha256: '0'.repeat(64)}}};
  const read = readVerifiedAsset(new URL('https://example.test/model/'), manifest, 'file.bin', {
    cache,
    signal: controller.signal,
    offline: true
  });
  const observed = read.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    assert.equal(
      await Promise.race([observed, new Promise((resolve) => setTimeout(() => resolve('still pending'), 100))]),
      reason
    );
  } finally {
    release.resolve();
    await observed;
  }
  await new Promise((resolve) => setImmediate(resolve));
});

test('a stalled verified-asset cache write stops delaying cancellation', async (t) => {
  const started = deferred(),
    release = deferred(),
    completed = deferred(),
    controller = new AbortController(),
    reason = Error('cancel write');
  const data = new TextEncoder().encode('valid model asset');
  const manifest = {files: {'file.bin': {bytes: data.length, sha256: await sha256(data)}}};
  const cache = {
    match: async () => undefined,
    put: async () => {
      started.resolve();
      await release.promise;
      completed.resolve();
    }
  };
  t.mock.method(globalThis, 'fetch', async () => new Response(data));
  const read = readVerifiedAsset(new URL('https://example.test/model/'), manifest, 'file.bin', {
    cache,
    signal: controller.signal
  });
  const observed = read.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    assert.equal(
      await Promise.race([observed, new Promise((resolve) => setTimeout(() => resolve('still pending'), 100))]),
      reason
    );
  } finally {
    release.resolve();
    await observed;
    await completed.promise;
  }
});

test('a stalled offline manifest body stops delaying cancellation', async () => {
  const started = deferred(),
    release = deferred(),
    controller = new AbortController(),
    reason = Error('cancel manifest');
  const response = responseFromReader(async () => {
    started.resolve();
    await release.promise;
    throw Error('late manifest failure');
  });
  const read = readManifestResponse(response, controller.signal);
  const observed = read.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    assert.equal(
      await Promise.race([observed, new Promise((resolve) => setTimeout(() => resolve('still pending'), 100))]),
      reason
    );
  } finally {
    release.resolve();
    await observed;
  }
  await new Promise((resolve) => setImmediate(resolve));
});

test('a stalled cache open stops delaying cancellation', async () => {
  const started = deferred(),
    release = deferred(),
    controller = new AbortController(),
    reason = Error('cancel cache open');
  const old = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', {
    value: {
      open: async () => {
        started.resolve();
        await release.promise;
        throw Error('late open failure');
      }
    },
    configurable: true
  });
  const read = openAssetCache(true, controller.signal);
  const observed = read.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    assert.equal(
      await Promise.race([observed, new Promise((resolve) => setTimeout(() => resolve('still pending'), 100))]),
      reason
    );
  } finally {
    release.resolve();
    await observed;
    if (old) Object.defineProperty(globalThis, 'caches', old);
    else delete globalThis.caches;
  }
  await new Promise((resolve) => setImmediate(resolve));
});
