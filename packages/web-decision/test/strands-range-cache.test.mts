import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import type {AssetCache} from '../src/assets.js';
import {RangeCacheTransaction} from '../src/strands/range-cache.mjs';
import {readVerifiedRange} from '../src/strands/range-reader.mjs';

const base = new URL('https://fixture.invalid/model/');
const digest = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

function reusableChunks() {
  const shared = new Uint8Array(4);
  let count = 0;
  return {
    getReader() {
      return {
        async read() {
          if (count === 2) return {done: true};
          shared.fill(++count);
          return {done: false, value: shared};
        },
        async cancel() {},
        releaseLock() {}
      };
    }
  };
}

test('snapshots reused response chunks before the next read for network and offline cache', async (t) => {
  const expected = Uint8Array.of(1, 1, 1, 1, 2, 2, 2, 2);
  const spec = {path: 'fixture.bin', totalBytes: 8, offset: 0, length: 8, sha256: digest(expected)};
  let requests = 0,
    writes = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return {status: 206, headers: new Headers({'content-range': 'bytes 0-7/8'}), body: reusableChunks()};
  });
  const cache = {
    async match() {
      return writes ? {body: reusableChunks()} : undefined;
    },
    async put(_key: string, response: Response) {
      assert.deepEqual(new Uint8Array(await response.arrayBuffer()), expected);
      writes++;
    }
  };
  const online = await readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache});
  const offline = await readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache, offline: true});
  assert.deepEqual(new Uint8Array(online.bytes), expected);
  assert.deepEqual(new Uint8Array(offline.bytes), expected);
  assert.equal(offline.cachedByHash, true);
  assert.equal(requests, 1);
  assert.equal(writes, 1);
});

test('short and oversized range bodies fail without caching', async (t) => {
  const expected = Uint8Array.of(1, 2, 3, 4);
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: digest(expected)};
  let body: unknown;
  let requests = 0;
  let writes = 0,
    released = 0,
    cancelled = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return {
      status: 206,
      headers: new Headers({'content-range': 'bytes 0-3/4'}),
      body
    };
  });
  const cache = {
    async match() {
      return undefined;
    },
    async put() {
      writes++;
    }
  };
  for (const data of [Uint8Array.of(1, 2, 3), Uint8Array.of(1, 2, 3, 4, 5)]) {
    let consumed = false;
    body = {
      getReader() {
        return {
          async read() {
            if (consumed) return {done: true};
            consumed = true;
            return {done: false, value: data};
          },
          cancel() {
            cancelled++;
          },
          releaseLock() {
            released++;
          }
        };
      }
    };
    await assert.rejects(readVerifiedRange(base, spec, {cache}), {code: 'ASSET_LOAD_FAILED'});
  }
  assert.equal(writes, 0);
  assert.equal(requests, 2, 'short and oversized body lengths must not retry');
  assert.equal(released, 2);
  assert.equal(cancelled, 1);
});

test('retries a transient range body error and caches only the fully verified retry', async (t) => {
  const expected = Uint8Array.of(1, 2, 3, 4);
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: digest(expected)};
  const requestRanges: string[] = [];
  let requests = 0;
  let writes = 0;
  const cache = {
    async match() {
      return null;
    },
    async put(_key: string, response: Response) {
      assert.deepEqual(new Uint8Array(await response.arrayBuffer()), expected);
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async (_input: RequestInfo | URL, init?: RequestInit) => {
    requests++;
    requestRanges.push(new Headers(init?.headers).get('Range') ?? '');
    if (requests === 1) {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(expected.subarray(0, 2));
          controller.error(new TypeError('net::ERR_HTTP2_PROTOCOL_ERROR'));
        }
      });
      return new Response(body, {
        status: 206,
        headers: {'content-range': 'bytes 0-3/4', 'content-length': '4'}
      });
    }
    assert.equal(writes, 0, 'partial bytes from the failed body must not be cached');
    return new Response(expected, {
      status: 206,
      headers: {'content-range': 'bytes 0-3/4', 'content-length': '4'}
    });
  });

  const result = await readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache});

  assert.deepEqual(new Uint8Array(result.bytes), expected);
  assert.equal(result.cachedByHash, true);
  assert.deepEqual(requestRanges, ['bytes=0-3', 'bytes=0-3']);
  assert.equal(requests, 2);
  assert.equal(writes, 1);
});

test('aborting after a transient range body error prevents the retry and preserves the reason', async (t) => {
  const expected = Uint8Array.of(1, 2, 3, 4);
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: digest(expected)};
  const controller = new AbortController();
  const reason = new Error('stop before range retry');
  let markBodyReadFailed!: () => void;
  const bodyReadFailed = new Promise<void>((resolve) => {
    markBodyReadFailed = resolve;
  });
  let requests = 0;
  let writes = 0;
  const cache = {
    async match() {
      return null;
    },
    async put() {
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    const body = {
      getReader() {
        return {
          read() {
            return Promise.reject(new TypeError('transient body transport failure'));
          },
          cancel() {},
          releaseLock() {
            markBodyReadFailed();
          }
        };
      }
    };
    return {
      status: 206,
      headers: new Headers({'content-range': 'bytes 0-3/4', 'content-length': '4'}),
      body
    } as unknown as Response;
  });

  const pending = readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache, signal: controller.signal});
  const assertion = assert.rejects(pending, (error) => error === reason);
  await bodyReadFailed;
  await nextTurn();
  controller.abort(reason);

  await assertion;
  assert.equal(requests, 1);
  assert.equal(writes, 0);
});

test('mixed fetch and body transport failures exhaust three total attempts without caching', async (t) => {
  const expected = Uint8Array.of(1, 2, 3, 4);
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: digest(expected)};
  let requests = 0;
  let writes = 0;
  const cache = {
    async match() {
      return null;
    },
    async put() {
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    if (requests === 2) {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.error(new TypeError('transient body transport failure'));
        }
      });
      return new Response(body, {
        status: 206,
        headers: {'content-range': 'bytes 0-3/4', 'content-length': '4'}
      });
    }
    throw new TypeError('temporary fetch transport failure');
  });

  await assert.rejects(readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache}), (error: unknown) => {
    assert.equal((error as {code?: unknown}).code, 'ASSET_LOAD_FAILED');
    assert.match((error as Error).message, /Could not fetch range/u);
    return true;
  });

  assert.equal(requests, 3);
  assert.equal(writes, 0);
});

test('invalid content hashes fail without transport retry', async (t) => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: '0'.repeat(64)};
  let requests = 0;
  let writes = 0;
  const cache = {
    async match() {
      return null;
    },
    async put() {
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response(bytes, {
      status: 206,
      headers: {'content-range': 'bytes 0-3/4', 'content-length': '4'}
    });
  });

  await assert.rejects(readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache}), {code: 'INVALID_ASSET'});
  assert.equal(requests, 1);
  assert.equal(writes, 0);
});

test('malformed range body chunks fail without transport retry', async (t) => {
  const expected = Uint8Array.of(1, 2, 3, 4);
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: digest(expected)};
  let requests = 0;
  let writes = 0;
  const cache = {
    async match() {
      return null;
    },
    async put() {
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue('not a byte chunk' as unknown as Uint8Array);
      }
    });
    return new Response(body, {
      status: 206,
      headers: {'content-range': 'bytes 0-3/4', 'content-length': '4'}
    });
  });

  await assert.rejects(readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache}), {
    code: 'ASSET_LOAD_FAILED'
  });
  assert.equal(requests, 1);
  assert.equal(writes, 0);
});

test('aborting a pending range body preserves the reason and cancels its reader', async (t) => {
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: '0'.repeat(64)};
  const controller = new AbortController(),
    reason = Error('stop range');
  let started!: () => void;
  let cancelled = 0,
    writes = 0;
  const reading = new Promise<void>((resolve) => {
    started = resolve;
  });
  const body = {
    getReader() {
      return {
        read() {
          started();
          return new Promise(() => {});
        },
        cancel() {
          cancelled++;
        },
        releaseLock() {}
      };
    }
  };
  t.mock.method(globalThis, 'fetch', async () => ({
    status: 206,
    headers: new Headers({'content-range': 'bytes 0-3/4'}),
    body
  }));
  const cache = {
    async match() {
      return undefined;
    },
    async put() {
      writes++;
    }
  };
  const promise = readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache, signal: controller.signal});
  await reading;
  controller.abort(reason);
  await assert.rejects(promise, (error) => error === reason);
  assert.equal(cancelled, 1);
  assert.equal(writes, 0);
});

test('oversized cached ranges fail offline and are replaced online', async (t) => {
  const expected = Uint8Array.of(1, 2, 3, 4);
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: digest(expected)};
  let requests = 0,
    writes = 0;
  const cache = {
    async match() {
      return new Response(Uint8Array.of(1, 2, 3, 4, 5));
    },
    async put(_key: string, value: Response) {
      assert.deepEqual(new Uint8Array(await value.arrayBuffer()), expected);
      writes++;
    }
  };
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response(expected, {status: 206, headers: {'content-range': 'bytes 0-3/4'}});
  });
  await assert.rejects(readVerifiedRange(base, spec, {cache, offline: true}), {code: 'ASSET_LOAD_FAILED'});
  assert.equal(requests, 0);
  const result = await readVerifiedRange(base, spec, {cache});
  assert.deepEqual(new Uint8Array(result.bytes), expected);
  assert.equal(requests, 1);
  assert.equal(writes, 1);
});

test('aborting a pending cached body preserves the reason without fetching', async (t) => {
  const spec = {path: 'fixture.bin', totalBytes: 4, offset: 0, length: 4, sha256: '0'.repeat(64)};
  const controller = new AbortController(),
    reason = Error('stop cache');
  let started!: () => void;
  let cancelled = 0,
    requests = 0;
  const reading = new Promise<void>((resolve) => {
    started = resolve;
  });
  const body = {
    getReader() {
      return {
        read() {
          started();
          return new Promise(() => {});
        },
        cancel() {
          cancelled++;
        },
        releaseLock() {}
      };
    }
  };
  const cache = {
    async match() {
      return {body};
    }
  };
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    throw Error('unexpected fetch');
  });
  const promise = readVerifiedRange(base, spec, {cache: cache as unknown as AssetCache, signal: controller.signal});
  await reading;
  controller.abort(reason);
  await assert.rejects(promise, (error) => error === reason);
  assert.equal(cancelled, 1);
  assert.equal(requests, 0);
});

test('aborted offline marker read cancels a response resolved during cancellation', async () => {
  const controller = new AbortController(),
    reason = new Error('stop marker');
  let resolveMatch!: (response: Response) => void;
  let cancelled = 0;
  const response = new Response(
    new ReadableStream({
      cancel() {
        cancelled++;
        return Promise.reject(new Error('late cancel'));
      }
    })
  );
  const cache = {
    match: () =>
      new Promise<Response>((resolve) => {
        resolveMatch = resolve;
      })
  };
  const tx = new RangeCacheTransaction({
    base: new URL('https://fixture.invalid/model/'),
    cache: cache as unknown as AssetCache,
    offline: true,
    signal: controller.signal,
    markerId: 'fixture'
  });
  const open = tx.open();
  resolveMatch(response);
  queueMicrotask(() => controller.abort(reason));
  await assert.rejects(open, (error) => error === reason);
  await nextTurn();
  assert.equal(cancelled, 1);
});

test('verified range is cached and rechecked offline without a network request', async () => {
  const originalFetch = globalThis.fetch,
    entries = new Map<string, Response>(),
    bytes = new TextEncoder().encode('test');
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const spec = {path: 'fixture.bin', totalBytes: 8, offset: 2, length: 4, sha256};
  const cache = {
    async match(key: string) {
      return entries.get(key)?.clone();
    },
    async put(key: string, value: Response) {
      entries.set(key, value.clone());
    }
  };
  let requests = 0;
  globalThis.fetch = async (_url, options) => {
    requests++;
    assert.equal(new Headers(options?.headers).get('Range'), 'bytes=2-5');
    return new Response(bytes, {status: 206, headers: {'content-range': 'bytes 2-5/8', 'content-length': '4'}});
  };
  try {
    const base = new URL('https://fixture.invalid/model/');
    const first = await readVerifiedRange(base, spec, {cache});
    const second = await readVerifiedRange(base, spec, {cache, offline: true});
    assert.deepEqual(new Uint8Array(first.bytes), bytes);
    assert.deepEqual(new Uint8Array(second.bytes), bytes);
    assert.equal(requests, 1);
    for (const key of entries.keys()) entries.set(key, new Response('bad!'));
    await assert.rejects(readVerifiedRange(base, spec, {cache, offline: true}), {code: 'ASSET_LOAD_FAILED'});
    assert.equal(requests, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('range response must include the exact declared Content-Range', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response('test', {status: 206, headers: {'content-range': 'bytes 0-3/8'}});
  try {
    await assert.rejects(
      readVerifiedRange(new URL('https://fixture.invalid/'), {
        path: 'fixture.bin',
        totalBytes: 8,
        offset: 2,
        length: 4,
        sha256: '0'.repeat(64)
      }),
      {code: 'INVALID_ASSET'}
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
