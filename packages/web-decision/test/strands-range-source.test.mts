import assert from 'node:assert/strict';
import test from 'node:test';
import type {AssetCache} from '../src/assets.js';
import {MANIFEST_BYTES} from '../src/strands/manifest.mjs';
import {openPinnedRangeSource} from '../src/strands/range-source.mjs';
import type {PinnedRangeSourceOptions} from '../src/strands/range-source.mjs';

const BASE = 'https://example.test/strands/';
const FIRST_PATH = 'fused-ordered-manifest-4096-candidate-v1.json';

function invalidRangeResponse(events: string[]): Response {
  return {
    status: 200,
    headers: {
      get(name: string) {
        events.push(`headers.get:${name}`);
        return null;
      }
    },
    body: {
      cancel() {
        events.push('body.cancel');
        return Promise.resolve();
      }
    }
  } as unknown as Response;
}

function trackingCache(events: string[]): AssetCache {
  return {
    get match() {
      events.push('cache.match.get');
      return async (key: string) => {
        events.push(`cache.match:${new URL(key).pathname}`);
        return null;
      };
    },
    async put() {
      events.push('cache.put');
    }
  };
}

test('reads source options and requests the pinned manifest range before cancelling an invalid response', async (t) => {
  const events: string[] = [];
  const cache = trackingCache(events);
  const options = {
    get base() {
      events.push('options.base');
      return new URL(BASE);
    },
    get cache() {
      events.push('options.cache');
      return cache;
    },
    get offline() {
      events.push('options.offline');
      return false;
    },
    get signal() {
      events.push('options.signal');
      return undefined;
    }
  } satisfies PinnedRangeSourceOptions;

  t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    events.push('fetch');
    assert.equal(String(input), `${BASE}${FIRST_PATH}`);
    assert.equal(new Headers(init?.headers).get('range'), `bytes=0-${MANIFEST_BYTES - 1}`);
    assert.equal(init?.signal, undefined);
    return invalidRangeResponse(events);
  });

  await assert.rejects(openPinnedRangeSource(options), {
    code: 'INVALID_ASSET',
    message: 'Invalid or inaccessible Content-Range or Content-Length response.'
  });
  assert.deepEqual(events, [
    'options.base',
    'options.cache',
    'options.offline',
    'options.signal',
    'cache.match.get',
    'cache.match:/.web-decision-cache/sha256/',
    'fetch',
    'headers.get:content-length',
    'body.cancel'
  ]);
});

test('propagates abort from the first range and cancels its late response body', async (t) => {
  const events: string[] = [];
  const controller = new AbortController();
  const reason = new Error('cancel first pinned range');
  let fetchSignal: AbortSignal | null | undefined;
  let resolveFetch!: (response: Response) => void;
  let signalFetchStarted!: () => void;
  const fetchStarted = new Promise<void>((resolve) => {
    signalFetchStarted = resolve;
  });
  const options: PinnedRangeSourceOptions = {
    base: new URL(BASE),
    signal: controller.signal
  };

  t.mock.method(globalThis, 'fetch', (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    events.push('fetch');
    fetchSignal = init?.signal;
    signalFetchStarted();
    return new Promise<Response>((resolve) => {
      resolveFetch = resolve;
    });
  });

  const pending = openPinnedRangeSource(options);
  await fetchStarted;
  assert.equal(fetchSignal, controller.signal);
  controller.abort(reason);
  await assert.rejects(pending, (error: unknown) => error === reason);
  resolveFetch(invalidRangeResponse(events));
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(events, ['fetch', 'body.cancel']);
});

test('does not inspect cache or fetch when already aborted', async (t) => {
  const events: string[] = [];
  const controller = new AbortController();
  const reason = new Error('cancel before opening range transaction');
  controller.abort(reason);
  const cache = trackingCache(events);
  const options = {
    get base() {
      events.push('options.base');
      return new URL(BASE);
    },
    get cache() {
      events.push('options.cache');
      return cache;
    },
    get offline() {
      events.push('options.offline');
      return false;
    },
    get signal() {
      events.push('options.signal');
      return controller.signal;
    }
  } satisfies PinnedRangeSourceOptions;
  t.mock.method(globalThis, 'fetch', async () => {
    events.push('fetch');
    return invalidRangeResponse(events);
  });

  await assert.rejects(openPinnedRangeSource(options), (error: unknown) => error === reason);
  assert.deepEqual(events, ['options.base', 'options.cache', 'options.offline', 'options.signal']);
});
