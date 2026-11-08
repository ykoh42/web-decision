import test from 'node:test';
import assert from 'node:assert/strict';
import {assetUrl, readManifestResponse, readVerifiedAsset, sha256} from '../src/assets.js';
import {MAX_ASSET_BYTES, MAX_VERIFIED_ASSET_BYTES} from '../src/bounded-response.js';
import {responseFromReader} from './stream-helpers.mjs';

function memoryCache() {
  const entries = new Map();
  return {
    entries,
    match: async (key) => entries.get(String(key))?.clone(),
    put: async (key, value) => {
      entries.set(String(key), value.clone());
    },
    delete: async (key) => entries.delete(String(key))
  };
}

test('asset paths cannot escape the configured directory', () => {
  const base = new URL('https://example.test/models/kev/');
  assert.equal(assetUrl(base, 'weights/a.bin'), 'https://example.test/models/kev/weights/a.bin');
  assert.throws(() => assetUrl(base, '../other'), {code: 'INVALID_ASSET'});
  assert.throws(() => assetUrl(base, 'https://other.test/file'), {code: 'INVALID_ASSET'});
  assert.throws(() => assetUrl(base, 'weights/a.bin?version=1'), {code: 'INVALID_ASSET'});
});

test('empty URL delimiters do not prevent asset resolution or mutate the base', () => {
  for (const suffix of ['?', '#', '?#']) {
    const base = new URL(`https://example.test/models/%3F%23/${suffix}`),
      original = base.href;
    assert.equal(assetUrl(base, 'weights/a.bin'), 'https://example.test/models/%3F%23/weights/a.bin');
    assert.equal(assetUrl(base, 'weights/%3F%23.bin'), 'https://example.test/models/%3F%23/weights/%3F%23.bin');
    assert.equal(base.href, original);
    for (const path of [
      '../escape',
      '%2e%2e/escape',
      'https://other.test/file',
      'https://example.test/models/%3F%23-other/file',
      'weights/a?query'
    ]) {
      assert.throws(() => assetUrl(base, path), {code: 'INVALID_ASSET'});
      assert.equal(base.href, original);
    }
  }
  for (const url of [
    'https://example.test/models/?v=1',
    'https://example.test/models/?v=1#',
    'https://example.test/models/#part',
    'https://example.test/models/?#part',
    'https://user:pass@example.test/models/',
    'https://user:pass@example.test/models/?'
  ]) {
    const base = new URL(url),
      original = base.href;
    assert.throws(() => assetUrl(base, 'weights/a.bin'), {code: 'INVALID_ASSET'});
    assert.equal(base.href, original);
  }
});

test('corrupted cached assets are replaced and verified before use', async (t) => {
  const good = new TextEncoder().encode('correct');
  const hash = await sha256(good);
  const manifest = {files: {'file.bin': {bytes: good.byteLength, sha256: hash}}};
  let saved = false;
  const cache = {
    match: async () => new Response('wrong'),
    put: async () => {
      saved = true;
    }
  };
  t.mock.method(globalThis, 'fetch', async () => new Response(good));
  const result = await readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {cache});
  assert.equal(new TextDecoder().decode(result), 'correct');
  assert.equal(saved, true);
});

test('rejects a corrupted asset from the server', async (t) => {
  const good = new TextEncoder().encode('correct');
  const manifest = {files: {'file.bin': {bytes: good.byteLength, sha256: await sha256(good)}}};
  t.mock.method(globalThis, 'fetch', async () => new Response('invalid'));
  await assert.rejects(readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin'), {
    code: 'INVALID_ASSET'
  });
});

test('allows pinned larger assets only with an explicit bounded size limit', async (t) => {
  const bytes = MAX_ASSET_BYTES + 1;
  const manifest = {files: {'encoder.onnx': {bytes, sha256: '0'.repeat(64)}}};
  let fetched = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetched++;
    return new Response(new Uint8Array());
  });
  const base = new URL('https://example.test/model/');
  await assert.rejects(readVerifiedAsset(base, manifest, 'encoder.onnx'), {code: 'INVALID_ASSET'});
  assert.equal(fetched, 0);
  await assert.rejects(readVerifiedAsset(base, manifest, 'encoder.onnx', {maxAssetBytes: bytes}), {
    code: 'INVALID_ASSET'
  });
  assert.equal(fetched, 1);
  await assert.rejects(
    readVerifiedAsset(base, manifest, 'encoder.onnx', {maxAssetBytes: MAX_VERIFIED_ASSET_BYTES + 1}),
    {code: 'INVALID_ASSET'}
  );
  assert.equal(fetched, 1);
});

test('allows the pinned Laya Typed-Decisions weights within its explicit one-gibibyte cap', async (t) => {
  const bytes = 842_587_136;
  const manifest = {files: {'weights.onnx_data': {bytes, sha256: '0'.repeat(64)}}};
  let fetched = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetched++;
    throw new Error('Stop before reading model bytes.');
  });
  const base = new URL('https://example.test/model/');

  await assert.rejects(readVerifiedAsset(base, manifest, 'weights.onnx_data', {maxAssetBytes: bytes - 1}), {
    code: 'INVALID_ASSET'
  });
  assert.equal(fetched, 0);
  await assert.rejects(readVerifiedAsset(base, manifest, 'weights.onnx_data', {maxAssetBytes: bytes}), {
    code: 'ASSET_LOAD_FAILED'
  });
  assert.equal(fetched, 1);
});

test('offline assets never fetch and still reject same-length corruption', async (t) => {
  const good = new TextEncoder().encode('correct');
  const manifest = {files: {'file.bin': {bytes: good.byteLength, sha256: await sha256(good)}}};
  let fetched = 0;
  const cache = memoryCache();
  t.mock.method(globalThis, 'fetch', async () => {
    fetched++;
    return new Response(good);
  });
  await readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {cache});
  const [key] = cache.entries.keys();
  cache.entries.set(key, new Response('corrEct'));
  fetched = 0;
  await assert.rejects(
    readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {cache, offline: true}),
    {code: 'ASSET_LOAD_FAILED'}
  );
  assert.equal(cache.entries.has(key), true);
  assert.equal(fetched, 0);
  cache.entries.set(key, new Response(good));
  const value = await readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {
    cache,
    offline: true
  });
  assert.equal(new TextDecoder().decode(value), 'correct');
  assert.equal(fetched, 0);
});

test('preserves cancellation when a model file fetch is aborted', async (t) => {
  const controller = new AbortController();
  const manifest = {files: {'file.bin': {bytes: 0, sha256: '0'.repeat(64)}}};
  t.mock.method(globalThis, 'fetch', async () => {
    controller.abort();
    throw new DOMException('Aborted', 'AbortError');
  });
  await assert.rejects(
    readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {signal: controller.signal}),
    {name: 'AbortError'}
  );
});

test('stops waiting on a fetch wrapper that ignores abort and consumes its late rejection', async (t) => {
  let rejectFetch, started;
  const pending = new Promise((_, reject) => {
    rejectFetch = reject;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = new Error('stop fetching');
  const manifest = {files: {'file.bin': {bytes: 0, sha256: '0'.repeat(64)}}};
  t.mock.method(globalThis, 'fetch', () => {
    started();
    return pending;
  });
  const load = readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {signal: controller.signal});
  await seen;
  controller.abort(reason);
  await assert.rejects(load, (error) => error === reason);
  rejectFetch(new Error('late fetch failure'));
  await new Promise((resolve) => setImmediate(resolve));
});

test('stops waiting on a noncooperative response body without writing it to cache', async (t) => {
  let resolveBody, started;
  const body = new Promise((resolve) => {
    resolveBody = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const good = new TextEncoder().encode('correct'),
    digest = await sha256(good);
  const manifest = {files: {'file.bin': {bytes: good.byteLength, sha256: digest}}};
  const controller = new AbortController(),
    reason = new Error('stop reading');
  let writes = 0;
  const cache = {
    match: async () => undefined,
    put: async () => {
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async () =>
    responseFromReader(() => {
      started();
      return body.then((value) => ({done: false, value: new Uint8Array(value)}));
    })
  );
  const load = readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {
    cache,
    signal: controller.signal
  });
  await seen;
  controller.abort(reason);
  await assert.rejects(load, (error) => error === reason);
  resolveBody(good.buffer);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(writes, 0);
});

test('preserves cancellation while the manifest response body is being read', async () => {
  const controller = new AbortController();
  const response = responseFromReader(async () => {
    controller.abort();
    throw new DOMException('Aborted', 'AbortError');
  });
  await assert.rejects(readManifestResponse(response, controller.signal), {name: 'AbortError'});
});

test('reports malformed manifest JSON as an invalid asset', async () => {
  const response = new Response('{bad JSON');
  await assert.rejects(readManifestResponse(response), {code: 'INVALID_ASSET'});
});

test('keeps cached versions at one URL independent across an interrupted update', async (t) => {
  const base = new URL('https://example.test/m/');
  const old = new TextEncoder().encode('old asset'),
    next = new TextEncoder().encode('new asset');
  const oldManifest = {files: {'file.bin': {bytes: old.length, sha256: await sha256(old)}}};
  const newManifest = {files: {'file.bin': {bytes: next.length, sha256: await sha256(next)}}};
  const entries = new Map();
  const cache = {
    match: async (key) => entries.get(String(key))?.clone(),
    put: async (key, value) => {
      entries.set(String(key), value.clone());
    },
    delete: async (key) => entries.delete(String(key))
  };
  let contents = old,
    cacheStatus = [];
  t.mock.method(globalThis, 'fetch', async () => new Response(contents));
  await readVerifiedAsset(base, oldManifest, 'file.bin', {cache, onCacheStatus: (value) => cacheStatus.push(value)});
  contents = next;
  await readVerifiedAsset(base, newManifest, 'file.bin', {cache, onCacheStatus: (value) => cacheStatus.push(value)});
  // An update may fail after caching this new file; the old manifest must still work offline.
  const restored = await readVerifiedAsset(base, oldManifest, 'file.bin', {cache, offline: true});
  assert.equal(new TextDecoder().decode(restored), 'old asset');
  assert.equal(entries.size, 2);
  assert.deepEqual(cacheStatus, [true, true]);
});

test('a new manifest with the same hash but a wrong size cannot delete the old cached asset', async (t) => {
  const base = new URL('https://example.test/m/');
  const data = new TextEncoder().encode('existing asset');
  const digest = await sha256(data);
  const valid = {files: {'file.bin': {bytes: data.length, sha256: digest}}};
  const invalid = {files: {'file.bin': {bytes: data.length + 1, sha256: digest}}};
  const entries = new Map();
  const cache = {
    match: async (key) => entries.get(String(key))?.clone(),
    put: async (key, value) => {
      entries.set(String(key), value.clone());
    },
    delete: async (key) => entries.delete(String(key))
  };
  t.mock.method(globalThis, 'fetch', async () => new Response(data));
  await readVerifiedAsset(base, valid, 'file.bin', {cache});
  await assert.rejects(readVerifiedAsset(base, invalid, 'file.bin', {cache}), {code: 'INVALID_ASSET'});
  const restored = await readVerifiedAsset(base, valid, 'file.bin', {cache, offline: true});
  assert.equal(new TextDecoder().decode(restored), 'existing asset');
});

test('reads and migrates a verified legacy entry without deleting it', async () => {
  const base = new URL('https://example.test/m/');
  const data = new TextEncoder().encode('old asset');
  const manifest = {files: {'file.bin': {bytes: data.length, sha256: await sha256(data)}}};
  const entries = new Map([['https://example.test/m/file.bin', new Response(data)]]);
  const cache = {
    match: async (key) => entries.get(String(key))?.clone(),
    put: async (key, value) => {
      entries.set(String(key), value.clone());
    },
    delete: async (key) => entries.delete(String(key))
  };
  const value = await readVerifiedAsset(base, manifest, 'file.bin', {cache, offline: true});
  assert.equal(new TextDecoder().decode(value), 'old asset');
  assert.equal(entries.size, 2);
  assert.ok(entries.has('https://example.test/m/file.bin'));
});

test('reports failed cache writes without breaking an online read', async (t) => {
  const data = new TextEncoder().encode('asset');
  const manifest = {files: {'file.bin': {bytes: data.length, sha256: await sha256(data)}}};
  const cache = {
    match: async () => undefined,
    put: async () => {
      throw Error('quota');
    }
  };
  t.mock.method(globalThis, 'fetch', async () => new Response(data));
  let stored;
  const value = await readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin', {
    cache,
    onCacheStatus: (status) => {
      stored = status;
    }
  });
  assert.equal(new TextDecoder().decode(value), 'asset');
  assert.equal(stored, false);
});

test('reports a failed response body as an asset load error and preserves its cause', async (t) => {
  const failure = new Error('connection closed');
  t.mock.method(globalThis, 'fetch', async () =>
    responseFromReader(async () => {
      throw failure;
    })
  );
  const manifest = {files: {'file.bin': {bytes: 0, sha256: '0'.repeat(64)}}};
  await assert.rejects(
    readVerifiedAsset(new URL('https://example.test/m/'), manifest, 'file.bin'),
    (error) => error.code === 'ASSET_LOAD_FAILED' && error.cause === failure
  );
});

test('reuses verified identical content across model directories on the same origin', async (t) => {
  const data = new TextEncoder().encode('shared model shard');
  const digest = await sha256(data),
    cache = memoryCache(),
    fetched = [],
    stored = [],
    progress = [];
  const first = {files: {'weights/a.bin': {bytes: data.length, sha256: digest}}};
  const second = {files: {'weights/b.bin': {bytes: data.length, sha256: digest}}};
  t.mock.method(globalThis, 'fetch', async (url) => {
    fetched.push(String(url));
    return new Response(data);
  });
  await readVerifiedAsset(new URL('https://example.test/baseline/'), first, 'weights/a.bin', {
    cache,
    onCacheStatus: (value) => stored.push(value),
    onBytes: (value) => progress.push(value)
  });
  const result = await readVerifiedAsset(new URL('https://example.test/k32/'), second, 'weights/b.bin', {
    cache,
    onCacheStatus: (value) => stored.push(value),
    onBytes: (value) => progress.push(value)
  });
  assert.equal(new TextDecoder().decode(result), 'shared model shard');
  assert.deepEqual(fetched, ['https://example.test/baseline/weights/a.bin']);
  assert.deepEqual(stored, [true, true]);
  assert.deepEqual(progress, [data.length, data.length]);
  assert.equal(cache.entries.size, 1);
});

test('does not reuse content across different asset origins', async (t) => {
  const data = new TextEncoder().encode('shared data'),
    digest = await sha256(data),
    cache = memoryCache();
  const manifest = {files: {'file.bin': {bytes: data.length, sha256: digest}}};
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response(data);
  });
  await readVerifiedAsset(new URL('https://one.test/a/'), manifest, 'file.bin', {cache});
  await readVerifiedAsset(new URL('https://two.test/b/'), manifest, 'file.bin', {cache});
  assert.equal(requests, 2);
  assert.equal(cache.entries.size, 2);
});

test('migrates a preview.12 hashed URL entry and shares it without deleting the old entry', async (t) => {
  const base = new URL('https://example.test/old/'),
    data = new TextEncoder().encode('cached shard');
  const digest = await sha256(data),
    manifest = {files: {'file.bin': {bytes: data.length, sha256: digest}}};
  const legacyKey = new URL('file.bin', base);
  legacyKey.searchParams.set('__web_decision_sha256', digest);
  const cache = memoryCache();
  cache.entries.set(legacyKey.href, new Response(data));
  let requests = 0,
    cached;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    throw Error('Unexpected request');
  });
  await readVerifiedAsset(base, manifest, 'file.bin', {
    cache,
    offline: true,
    onCacheStatus: (value) => {
      cached = value;
    }
  });
  const shared = await readVerifiedAsset(new URL('https://example.test/new/'), manifest, 'file.bin', {
    cache,
    offline: true
  });
  assert.equal(new TextDecoder().decode(shared), 'cached shard');
  assert.equal(cached, true);
  assert.equal(requests, 0);
  assert.equal(cache.entries.size, 2);
  assert(cache.entries.has(legacyKey.href));
});

test('a failed legacy migration does not prevent an offline read from the old URL', async (t) => {
  const base = new URL('https://example.test/old/'),
    data = new TextEncoder().encode('cached shard');
  const digest = await sha256(data),
    manifest = {files: {'file.bin': {bytes: data.length, sha256: digest}}};
  const legacyKey = new URL('file.bin', base);
  legacyKey.searchParams.set('__web_decision_sha256', digest);
  const cache = memoryCache();
  cache.entries.set(legacyKey.href, new Response(data));
  cache.put = async () => {
    throw Error('quota');
  };
  let cached;
  t.mock.method(globalThis, 'fetch', async () => {
    throw Error('Unexpected request');
  });
  const value = await readVerifiedAsset(base, manifest, 'file.bin', {
    cache,
    offline: true,
    onCacheStatus: (result) => {
      cached = result;
    }
  });
  assert.equal(new TextDecoder().decode(value), 'cached shard');
  assert.equal(cached, false);
  await assert.rejects(
    readVerifiedAsset(new URL('https://example.test/new/'), manifest, 'file.bin', {cache, offline: true}),
    {code: 'ASSET_LOAD_FAILED'}
  );
  assert(cache.entries.has(legacyKey.href));
});

test('does not use a corrupted shared entry and repairs it online', async (t) => {
  const data = new TextEncoder().encode('correct'),
    digest = await sha256(data),
    cache = memoryCache();
  const manifest = {files: {'file.bin': {bytes: data.length, sha256: digest}}};
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response(data);
  });
  await readVerifiedAsset(new URL('https://example.test/a/'), manifest, 'file.bin', {cache});
  const [key] = cache.entries.keys();
  cache.entries.set(key, new Response('corrEct'));
  await assert.rejects(
    readVerifiedAsset(new URL('https://example.test/b/'), manifest, 'file.bin', {cache, offline: true}),
    {code: 'ASSET_LOAD_FAILED'}
  );
  assert.equal(requests, 1);
  await readVerifiedAsset(new URL('https://example.test/b/'), manifest, 'file.bin', {cache});
  const restored = await readVerifiedAsset(new URL('https://example.test/a/'), manifest, 'file.bin', {
    cache,
    offline: true
  });
  assert.equal(new TextDecoder().decode(restored), 'correct');
  assert.equal(requests, 2);
});

test('an offline reader cannot delete a concurrently repaired shared entry', async (t) => {
  const data = new TextEncoder().encode('correct'),
    digest = await sha256(data),
    cache = memoryCache();
  const manifest = {files: {'file.bin': {bytes: data.length, sha256: digest}}};
  const base = new URL('https://example.test/a/');
  t.mock.method(globalThis, 'fetch', async () => new Response(data));
  await readVerifiedAsset(base, manifest, 'file.bin', {cache});
  const [key] = cache.entries.keys();
  cache.entries.set(key, new Response('corrEct'));
  let releaseRead, startRead;
  const readStarted = new Promise((resolve) => {
    startRead = resolve;
  });
  const gate = new Promise((resolve) => {
    releaseRead = resolve;
  });
  const originalMatch = cache.match;
  let delayed = false;
  cache.match = async (requested) => {
    if (!delayed && String(requested) === key) {
      delayed = true;
      const stale = cache.entries.get(key).clone();
      let delivered = false;
      return responseFromReader(async () => {
        if (delivered) return {done: true};
        startRead();
        await gate;
        delivered = true;
        return {done: false, value: new Uint8Array(await stale.arrayBuffer())};
      });
    }
    return originalMatch(requested);
  };
  const staleRead = readVerifiedAsset(base, manifest, 'file.bin', {cache, offline: true});
  await readStarted;
  // Another request successfully replaces the corrupted item before this read finishes.
  cache.entries.set(key, new Response(data));
  releaseRead();
  await assert.rejects(staleRead, {code: 'ASSET_LOAD_FAILED'});
  assert.equal(
    new TextDecoder().decode(await readVerifiedAsset(base, manifest, 'file.bin', {cache, offline: true})),
    'correct'
  );
});
