import test from 'node:test';
import assert from 'node:assert/strict';
import {readManifestResponse, readVerifiedAsset, sha256} from '../src/assets.js';
import {MAX_ASSET_BYTES, MAX_MANIFEST_BYTES, readBoundedResponse, ResponseSizeError} from '../src/bounded-response.js';
import {prepareModelSource} from '../src/model-source.js';
import {readPinnedManifest} from '../src/pinned-manifest.js';
import {responseFromReader} from './stream-helpers.mjs';

const base = new URL('https://example.test/model/');
const bytes = (text) => new TextEncoder().encode(text);
const manifest = async (text) => ({
  files: {'file.bin': {bytes: bytes(text).length, sha256: await sha256(bytes(text))}}
});

test('bounded reads preserve exact bytes and allow empty assets', async () => {
  const response = new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(bytes('ab'));
        controller.enqueue(bytes('cd'));
        controller.close();
      }
    })
  );
  assert.equal(new TextDecoder().decode(await readBoundedResponse(response, 4, {exact: true})), 'abcd');
  assert.equal((await readBoundedResponse(new Response(null), 0, {exact: true})).byteLength, 0);
  await assert.rejects(readBoundedResponse(new Response('abc'), 4, {exact: true}), ResponseSizeError);
});

test('an oversized streamed asset is cancelled before more data or cache writes', async (t) => {
  let reads = 0,
    cancelled = 0,
    writes = 0;
  const response = responseFromReader(
    async () => {
      reads++;
      return {done: false, value: bytes('more than four bytes')};
    },
    {
      cancel: () => {
        cancelled++;
      }
    }
  );
  t.mock.method(globalThis, 'fetch', async () => response);
  const cache = {
    match: async () => undefined,
    put: async () => {
      writes++;
    }
  };
  await assert.rejects(readVerifiedAsset(base, await manifest('four'), 'file.bin', {cache}), {code: 'INVALID_ASSET'});
  assert.equal(reads, 1);
  assert.equal(cancelled, 1);
  assert.equal(writes, 0);
});

test('a huge declared asset is rejected before fetch or allocation', async (t) => {
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response('unused');
  });
  const declared = {files: {'file.bin': {bytes: MAX_ASSET_BYTES + 1, sha256: '0'.repeat(64)}}};
  await assert.rejects(readVerifiedAsset(base, declared, 'file.bin'), {code: 'INVALID_ASSET'});
  assert.equal(requests, 0);
});

test('oversized cached data is cancelled, then repaired online', async (t) => {
  let cancelled = 0,
    requests = 0,
    writes = 0;
  const oversized = () =>
    responseFromReader(async () => ({done: false, value: bytes('too many bytes')}), {
      cancel: () => {
        cancelled++;
      }
    });
  const cache = {
    match: async () => oversized(),
    put: async () => {
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response('good');
  });
  const value = await readVerifiedAsset(base, await manifest('good'), 'file.bin', {cache});
  assert.equal(new TextDecoder().decode(value), 'good');
  assert.equal(cancelled, 3);
  assert.equal(requests, 1);
  assert.equal(writes, 1);
});

test('oversized manifest streams are rejected and cancelled', async () => {
  let cancelled = 0;
  const response = responseFromReader(async () => ({done: false, value: new Uint8Array(MAX_MANIFEST_BYTES + 1)}), {
    cancel: () => {
      cancelled++;
    }
  });
  await assert.rejects(readManifestResponse(response), {code: 'INVALID_ASSET'});
  assert.equal(cancelled, 1);
});

test('manifest readers preserve AbortError detection for primitive causes with prototype names', async () => {
  const primitiveCause = 'stream failure';
  const originalName = Object.getOwnPropertyDescriptor(String.prototype, 'name');
  Object.defineProperty(String.prototype, 'name', {configurable: true, value: 'AbortError'});
  const failedResponse = () => responseFromReader(() => Promise.reject(primitiveCause));
  try {
    assert.equal(primitiveCause?.name, 'AbortError');
    let caught;
    try {
      await readManifestResponse(failedResponse());
    } catch (cause) {
      caught = cause;
    }
    assert.equal(caught, primitiveCause);

    caught = undefined;
    try {
      await readPinnedManifest(failedResponse(), undefined, 'unused', 'test');
    } catch (cause) {
      caught = cause;
    }
    assert.equal(caught, primitiveCause);
  } finally {
    if (originalName) Object.defineProperty(String.prototype, 'name', originalName);
    else delete String.prototype.name;
  }
});

test('late fetch responses are cancelled after the caller aborts', async (t) => {
  let resolveFetch,
    started,
    cancelled = 0;
  const pending = new Promise((resolve) => {
    resolveFetch = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop');
  t.mock.method(globalThis, 'fetch', () => {
    started();
    return pending;
  });
  const load = readVerifiedAsset(base, await manifest('good'), 'file.bin', {signal: controller.signal});
  await seen;
  controller.abort(reason);
  await assert.rejects(load, (cause) => cause === reason);
  resolveFetch(
    responseFromReader(async () => ({done: true}), {
      cancel: () => {
        cancelled++;
      }
    })
  );
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(cancelled, 1);
});

test('a response delivered just before abort is cancelled before reading', async (t) => {
  let resolveFetch,
    started,
    cancelled = 0,
    reads = 0;
  const pending = new Promise((resolve) => {
    resolveFetch = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop in delivery');
  t.mock.method(globalThis, 'fetch', () => {
    started();
    return pending;
  });
  const load = readVerifiedAsset(base, await manifest('good'), 'file.bin', {signal: controller.signal});
  await seen;
  resolveFetch(
    responseFromReader(
      async () => {
        reads++;
        return {done: true};
      },
      {
        cancel: () => {
          cancelled++;
        }
      }
    )
  );
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(load, (cause) => cause === reason);
  assert.equal(reads, 0);
  assert.equal(cancelled, 1);
});

test('a failing asset response delivered just before abort preserves the caller reason', async (t) => {
  let resolveFetch,
    started,
    cancelled = 0;
  const pending = new Promise((resolve) => {
    resolveFetch = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop before HTTP failure');
  t.mock.method(globalThis, 'fetch', () => {
    started();
    return pending;
  });
  const load = readVerifiedAsset(base, await manifest('good'), 'file.bin', {signal: controller.signal});
  await seen;
  resolveFetch(
    responseFromReader(async () => ({done: true}), {
      ok: false,
      status: 404,
      cancel: () => {
        cancelled++;
      }
    })
  );
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(load, (cause) => cause === reason);
  assert.equal(cancelled, 1);
});

function mockGpu(t) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {gpu: {requestAdapter: async () => ({features: new Set(['shader-f16'])})}}
  });
  t.after(() => (original ? Object.defineProperty(globalThis, 'navigator', original) : delete globalThis.navigator));
}

test('a failing manifest response delivered just before abort preserves the caller reason', async (t) => {
  let resolveFetch,
    started,
    cancelled = 0;
  const pending = new Promise((resolve) => {
    resolveFetch = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop before manifest failure');
  mockGpu(t);
  t.mock.method(globalThis, 'fetch', () => {
    started();
    return pending;
  });
  const load = prepareModelSource(
    {assetsUrl: base.href, cache: false, signal: controller.signal},
    {offlineMissingMessage: 'missing'}
  );
  await seen;
  resolveFetch(
    responseFromReader(async () => ({done: true}), {
      ok: false,
      status: 404,
      cancel: () => {
        cancelled++;
      }
    })
  );
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(load, (cause) => cause === reason);
  assert.equal(cancelled, 1);
});

test('an offline manifest cache miss delivered just before abort preserves the caller reason', async (t) => {
  let resolveMatch, started;
  const pending = new Promise((resolve) => {
    resolveMatch = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop before offline miss');
  const originalCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  mockGpu(t);
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: {
      open: async () => ({
        match: () => {
          started();
          return pending;
        }
      })
    }
  });
  t.after(() => {
    if (originalCaches) Object.defineProperty(globalThis, 'caches', originalCaches);
    else delete globalThis.caches;
  });
  const load = prepareModelSource(
    {assetsUrl: base.href, offline: true, signal: controller.signal},
    {offlineMissingMessage: 'missing'}
  );
  await seen;
  resolveMatch(null);
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(load, (cause) => cause === reason);
});

test('asset hash mismatch completed just before abort preserves the caller reason', async (t) => {
  let resolveDigest, started;
  const pending = new Promise((resolve) => {
    resolveDigest = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop after asset hash');
  const metadata = await manifest('good');
  t.mock.method(globalThis, 'fetch', async () => new Response('good'));
  t.mock.method(crypto.subtle, 'digest', () => {
    started();
    return pending;
  });
  const load = readVerifiedAsset(base, metadata, 'file.bin', {signal: controller.signal});
  await seen;
  resolveDigest(new Uint8Array(32).buffer);
  queueMicrotask(() => queueMicrotask(() => controller.abort(reason)));
  await assert.rejects(load, (cause) => cause === reason);
});

test('a pinned manifest hash completed just before abort cannot succeed', async (t) => {
  let resolveDigest, started;
  const pending = new Promise((resolve) => {
    resolveDigest = resolve;
  });
  const seen = new Promise((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop after manifest hash');
  const source = '{"model":"x"}',
    expected = await sha256(bytes(`${source}\n`));
  const digest = Uint8Array.from(expected.match(/../g), (part) => Number.parseInt(part, 16));
  t.mock.method(crypto.subtle, 'digest', () => {
    started();
    return pending;
  });
  const load = readPinnedManifest(new Response(source), controller.signal, expected, 'test');
  await seen;
  resolveDigest(digest.buffer);
  queueMicrotask(() => queueMicrotask(() => controller.abort(reason)));
  await assert.rejects(load, (cause) => cause === reason);
});

test('abort from either progress callback stops subsequent work and preserves its reason', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('good'));
  const metadata = await manifest('good');
  for (const stage of ['cache', 'bytes']) {
    const controller = new AbortController(),
      reason = Error(`stop at ${stage}`);
    let progressCalls = 0;
    const load = readVerifiedAsset(base, metadata, 'file.bin', {
      signal: controller.signal,
      onCacheStatus: () => {
        if (stage === 'cache') controller.abort(reason);
      },
      onBytes: () => {
        progressCalls++;
        if (stage === 'bytes') controller.abort(reason);
      }
    });
    await assert.rejects(load, (cause) => cause === reason);
    assert.equal(progressCalls, stage === 'cache' ? 0 : 1);
  }
});
