import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import type {AssetCache} from '../src/assets.js';
import {RangeCacheTransaction} from '../src/strands/range-cache.mjs';
import type {RangeCacheSpec} from '../src/strands/range-cache.mjs';

const BASE = new URL('https://fixture.invalid/model/');

function rangeSpec(path: string, data: Uint8Array): RangeCacheSpec {
  return {
    path,
    totalBytes: data.byteLength,
    offset: 0,
    length: data.byteLength,
    sha256: createHash('sha256').update(data).digest('hex')
  };
}

function trackedSignal(controller: AbortController, name: string, events: string[]): AbortSignal {
  const signal = controller.signal;
  return {
    get aborted() {
      return signal.aborted;
    },
    get reason() {
      return signal.reason;
    },
    throwIfAborted() {
      signal.throwIfAborted();
    },
    addEventListener(
      type: string,
      listener: EventListenerOrEventListenerObject,
      options?: boolean | AddEventListenerOptions
    ) {
      events.push(`${name}.add:${type}`);
      signal.addEventListener(type, listener, options);
    },
    removeEventListener(
      type: string,
      listener: EventListenerOrEventListenerObject,
      options?: boolean | EventListenerOptions
    ) {
      events.push(`${name}.remove:${type}`);
      signal.removeEventListener(type, listener, options);
    }
  } as unknown as AbortSignal;
}

function lateResponse(events: string[]): Response {
  return {
    get body() {
      return {
        cancel() {
          events.push('late-body.cancel');
          return Promise.resolve();
        }
      };
    }
  } as unknown as Response;
}

test('reads constructor getters in order and commits only after all verified ranges are cached', async (t) => {
  const events: string[] = [];
  const firstBytes = Uint8Array.of(1, 2, 3, 4);
  const secondBytes = Uint8Array.of(5, 6, 7, 8);
  const first = rangeSpec('first.bin', firstBytes);
  const second = rangeSpec('second.bin', secondBytes);
  const files = new Map([
    [first.path, {spec: first, bytes: firstBytes}],
    [second.path, {spec: second, bytes: secondBytes}]
  ]);
  const entries = new Map<string, Response>();
  let failSecondWrite = true;
  const cache: AssetCache & Pick<Cache, 'keys'> = {
    async keys() {
      events.push('cache.keys');
      return [...entries.keys()].map((key) => new Request(key));
    },
    async match(key) {
      events.push(`cache.match:${new URL(key).searchParams.get('digest')}`);
      return entries.get(key)?.clone() ?? null;
    },
    async put(key, response) {
      const url = new URL(key);
      if (url.pathname.endsWith('strands-offline-complete-v1.json')) {
        events.push('cache.put:marker');
        entries.set(key, response.clone());
        return;
      }
      const expected = url.searchParams.get('digest');
      if (failSecondWrite && expected === second.sha256) {
        events.push('cache.put:range-failed');
        throw new Error('cache write failed');
      }
      events.push('cache.put:range');
      entries.set(key, response.clone());
    }
  };
  t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const path = url.pathname.slice(new URL(BASE).pathname.length);
    const entry = files.get(path);
    assert.ok(entry);
    assert.equal(new Headers(init?.headers).get('range'), `bytes=0-${entry.spec.length - 1}`);
    events.push(`fetch:${path}`);
    return new Response(entry.bytes, {
      status: 206,
      headers: {
        'content-range': `bytes 0-${entry.spec.length - 1}/${entry.spec.totalBytes}`,
        'content-length': String(entry.spec.length)
      }
    });
  });

  const options = {
    get base() {
      events.push('constructor.base');
      return BASE;
    },
    get cache() {
      events.push('constructor.cache');
      return cache;
    },
    get offline() {
      events.push('constructor.offline');
      return false;
    },
    get signal() {
      events.push('constructor.signal');
      return undefined;
    },
    get markerId() {
      events.push('constructor.markerId');
      return 'fixture';
    }
  };
  const transaction = new RangeCacheTransaction(options);
  assert.deepEqual(events.splice(0), [
    'constructor.base',
    'constructor.cache',
    'constructor.offline',
    'constructor.signal',
    'constructor.markerId'
  ]);

  await transaction.open();
  await assert.rejects(transaction.commit(), {
    code: 'INVALID_ASSET',
    message: 'The range transaction was not sealed.'
  });
  transaction.require([first, second]);
  transaction.seal();

  assert.equal((await transaction.read(BASE, first)).cachedByHash, true);
  assert.equal((await transaction.read(BASE, second)).cachedByHash, false);
  assert.equal(await transaction.commit(), false);
  assert.equal(events.includes('cache.put:marker'), false);

  failSecondWrite = false;
  assert.equal((await transaction.read(BASE, second)).cachedByHash, true);
  assert.equal(await transaction.commit(), true);
  assert.equal(events.at(-1), 'cache.put:marker');
  assert.deepEqual(
    events.filter((event) => event.startsWith('fetch:')),
    ['fetch:first.bin', 'fetch:second.bin']
  );

  const reopened = new RangeCacheTransaction(options);
  await reopened.open();
  reopened.require([first, second]);
  reopened.seal();
  events.length = 0;
  assert.equal(await reopened.hasCommittedRanges(), true);
  assert.deepEqual(events, ['cache.match:null', 'cache.keys']);
  const firstKey = [...entries.keys()].find((key) => new URL(key).searchParams.get('digest') === first.sha256)!;
  entries.delete(firstKey);
  assert.equal(await reopened.hasCommittedRanges(), false);
});

test('serves a resident verified snapshot after the cache entry is replaced or deleted', async (t) => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const corruptBytes = Uint8Array.of(5, 6, 7, 8);
  const spec = rangeSpec('mutable.bin', bytes);
  const entries = new Map<string, Response>();
  const cache: AssetCache = {
    async match(key) {
      return entries.get(key)?.clone() ?? null;
    },
    async put(key, response) {
      entries.set(key, response.clone());
    }
  };
  let fetches = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response(bytes, {
      status: 206,
      headers: {
        'content-range': `bytes 0-${bytes.byteLength - 1}/${bytes.byteLength}`,
        'content-length': String(bytes.byteLength)
      }
    });
  });

  const transaction = new RangeCacheTransaction({base: BASE, cache, markerId: 'resident-range'});
  await transaction.open();
  transaction.require([spec]);
  transaction.seal();
  assert.equal((await transaction.read(BASE, spec)).cachedByHash, true);
  const rangeKey = [...entries.keys()].find((key) => new URL(key).pathname === '/.web-decision-cache/sha256/');
  assert.ok(rangeKey);

  entries.set(rangeKey, new Response(corruptBytes));
  assert.deepEqual(new Uint8Array((await transaction.read(BASE, spec)).bytes), bytes);
  entries.delete(rangeKey);
  assert.deepEqual(new Uint8Array((await transaction.read(BASE, spec)).bytes), bytes);
  assert.equal(fetches, 1, 'the verified in-memory snapshot avoids another fetch');
  assert.equal(await transaction.commit(), false, 'a missing CacheStorage entry cannot complete offline caching');
  assert.equal(transaction.isCached(spec), false);
  assert.equal((await transaction.read(BASE, spec)).cachedByHash, true);
  assert.equal(fetches, 1, 'the verified snapshot can repair the missing entry');
  assert.equal(await transaction.commit(), true);
});

test('failed rereads clear prior cache verification before a completion marker can be committed', async (t) => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const corruptBytes = Uint8Array.of(5, 6, 7, 8);
  const spec = rangeSpec('mutable.bin', bytes);
  const fillers = Array.from({length: 4}, (_, index) => {
    const data = Uint8Array.of(index + 10, 2, 3, 4);
    return {spec: rangeSpec(`filler-${index}.bin`, data), data};
  });
  const payloads = new Map(fillers.map(({spec: filler, data}) => [filler.path, data]));
  const entries = new Map<string, Response>();
  const cache: AssetCache = {
    async match(key) {
      return entries.get(key)?.clone() ?? null;
    },
    async put(key, response) {
      entries.set(key, response.clone());
    }
  };
  let responseBytes = bytes;
  t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL) => {
    const path = new URL(String(input)).pathname.split('/').at(-1)!;
    const data = path === spec.path ? responseBytes : payloads.get(path)!;
    return new Response(data, {
      status: 206,
      headers: {
        'content-range': `bytes 0-${data.byteLength - 1}/${data.byteLength}`,
        'content-length': String(data.byteLength)
      }
    });
  });

  const transaction = new RangeCacheTransaction({base: BASE, cache, markerId: 'mutable-range'});
  await transaction.open();
  transaction.require([spec, ...fillers.map(({spec: filler}) => filler)]);
  transaction.seal();
  assert.equal((await transaction.read(BASE, spec)).cachedByHash, true);
  for (const {spec: filler} of fillers) await transaction.read(BASE, filler);

  const rangeKey = [...entries.keys()].find((key) => new URL(key).pathname === '/.web-decision-cache/sha256/');
  assert.ok(rangeKey);
  entries.set(rangeKey, new Response(corruptBytes));
  responseBytes = corruptBytes;
  await assert.rejects(transaction.read(BASE, spec), {
    code: 'INVALID_ASSET',
    message: 'Range integrity check failed.'
  });

  assert.equal(await transaction.commit(), false);
  assert.equal(entries.has(new URL('strands-offline-complete-v1.json', BASE).href), false);
});

test('removes both combined-signal listeners and cancels a late response after abort', async (t) => {
  const events: string[] = [];
  const transactionController = new AbortController();
  const operationController = new AbortController();
  const transactionSignal = trackedSignal(transactionController, 'transaction', events);
  const operationSignal = trackedSignal(operationController, 'operation', events);
  const reason = new Error('stop combined range read');
  const data = Uint8Array.of(1, 2, 3, 4);
  const spec = rangeSpec('pending.bin', data);
  let fetchSignal: AbortSignal | null | undefined;
  let resolveFetch!: (response: Response) => void;
  let signalFetchStarted!: () => void;
  const fetchStarted = new Promise<void>((resolve) => {
    signalFetchStarted = resolve;
  });
  t.mock.method(globalThis, 'fetch', (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    events.push('fetch');
    fetchSignal = init?.signal;
    signalFetchStarted();
    return new Promise<Response>((resolve) => {
      resolveFetch = resolve;
    });
  });

  const transaction = new RangeCacheTransaction({
    base: BASE,
    cache: null,
    signal: transactionSignal,
    markerId: 'fixture'
  });
  await transaction.open();
  transaction.require([spec]);
  transaction.seal();
  const readOptions = {
    get signal() {
      events.push('read.signal');
      return operationSignal;
    }
  };
  const pending = transaction.read(BASE, spec, readOptions);
  await fetchStarted;
  assert.ok(fetchSignal);
  assert.notEqual(fetchSignal, transactionSignal);
  assert.notEqual(fetchSignal, operationSignal);
  operationController.abort(reason);

  await assert.rejects(pending, (error: unknown) => error === reason);
  assert.deepEqual(events, [
    'transaction.add:abort',
    'transaction.add:abort',
    'transaction.remove:abort',
    'read.signal',
    'transaction.add:abort',
    'operation.add:abort',
    'fetch',
    'transaction.remove:abort',
    'operation.remove:abort'
  ]);
  resolveFetch(lateResponse(events));
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(events.at(-1), 'late-body.cancel');
  transaction.dispose();
  assert.equal(events.at(-1), 'transaction.remove:abort', 'disposing removes the transaction lifetime listener');
});

test('manual disposal cancels a pending cache check before it can publish the completion marker', async (t) => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const spec = rangeSpec('pending-commit.bin', bytes);
  const entries = new Map<string, Response>();
  let pendingKey: string | undefined, resolveMatch!: (response: Response | null) => void, markMatchStarted!: () => void;
  const matchStarted = new Promise<void>((resolve) => {
    markMatchStarted = resolve;
  });
  const cache: AssetCache = {
    async match(key) {
      if (key === pendingKey) {
        markMatchStarted();
        return new Promise<Response | null>((resolve) => {
          resolveMatch = resolve;
        });
      }
      return entries.get(key)?.clone() ?? null;
    },
    async put(key, response) {
      entries.set(key, response.clone());
    }
  };
  t.mock.method(
    globalThis,
    'fetch',
    async () =>
      new Response(bytes, {
        status: 206,
        headers: {
          'content-range': `bytes 0-${bytes.byteLength - 1}/${bytes.byteLength}`,
          'content-length': String(bytes.byteLength)
        }
      })
  );

  const transaction = new RangeCacheTransaction({base: BASE, cache, markerId: 'dispose-pending-commit'});
  await transaction.open();
  transaction.require([spec]);
  transaction.seal();
  await transaction.read(BASE, spec);
  pendingKey = [...entries.keys()][0];
  assert.ok(pendingKey);

  const pending = transaction.commit();
  await matchStarted;
  transaction.dispose();
  await assert.rejects(pending, {code: 'SESSION_CLOSED'});
  resolveMatch(entries.get(pendingKey)?.clone() ?? null);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(entries.has(new URL('strands-offline-complete-v1.json', BASE).href), false);
});
