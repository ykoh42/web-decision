import assert from 'node:assert/strict';
import test from 'node:test';
import type {TestContext} from 'node:test';
import {CACHE_NAME} from '../src/assets.js';
import {prepareModelSource} from '../src/model-source.js';

const assetsUrl = 'https://example.test/model';
const globalsToRestore = new WeakMap<TestContext, Map<PropertyKey, PropertyDescriptor | undefined>>();

function setGlobal(t: TestContext, name: PropertyKey, value: unknown) {
  let originals = globalsToRestore.get(t);
  if (!originals) {
    originals = new Map();
    globalsToRestore.set(t, originals);
    t.after(() => {
      for (const [key, descriptor] of originals!) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete (globalThis as Record<PropertyKey, unknown>)[key];
      }
    });
  }
  if (!originals.has(name)) originals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, {
    configurable: true,
    enumerable: descriptor?.enumerable ?? false,
    writable: true,
    value
  });
}

function gpuWith(
  features: string[] = ['shader-f16'],
  requestAdapter: () => Promise<{features: Set<string>} | null> = async () => ({features: new Set(features)})
) {
  return {requestAdapter};
}

function installEnvironment(t: TestContext, options: {gpu?: unknown; crypto?: unknown} = {}) {
  const gpu = Object.hasOwn(options, 'gpu') ? options.gpu : gpuWith();
  const crypto = Object.hasOwn(options, 'crypto') ? options.crypto : globalThis.crypto;
  setGlobal(t, 'navigator', {gpu});
  setGlobal(t, 'crypto', crypto);
}

test('source options reject invalid values before checking browser support', async () => {
  const invalidOptions: unknown[] = [
    undefined,
    {},
    {assetsUrl: ''},
    {assetsUrl, offline: 'true'},
    {assetsUrl, offline: true, cache: false},
    {assetsUrl, maxPrefetchBytes: -1},
    {assetsUrl, maxPrefetchBytes: 1.5}
  ];
  for (const options of invalidOptions) {
    await assert.rejects(prepareModelSource(options as Parameters<typeof prepareModelSource>[0], {}), {
      code: 'INVALID_INPUT'
    });
  }
});

test('model option validation runs before environment checks', async () => {
  let calls = 0;
  await assert.rejects(
    prepareModelSource(
      {assetsUrl},
      {
        validateModelOptions() {
          calls++;
          const error = Error('invalid model option');
          (error as Error & {code: string}).code = 'INVALID_INPUT';
          throw error;
        }
      }
    ),
    {code: 'INVALID_INPUT', message: 'invalid model option'}
  );
  assert.equal(calls, 1);
});

test('source validation rejects missing WebGPU and Web Crypto without fetching', async (t: TestContext) => {
  let fetches = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response('unused');
  });
  setGlobal(t, 'navigator', {});
  setGlobal(t, 'crypto', globalThis.crypto);
  await assert.rejects(prepareModelSource({assetsUrl}, {}), {
    code: 'UNSUPPORTED_ENVIRONMENT',
    message: /WebGPU is unavailable/
  });

  installEnvironment(t, {crypto: undefined});
  await assert.rejects(prepareModelSource({assetsUrl}, {}), {
    code: 'UNSUPPORTED_ENVIRONMENT',
    message: /Web Crypto is unavailable/
  });
  assert.equal(fetches, 0);
});

test('source URL validation rejects malformed and non-directory origins before fetch', async (t: TestContext) => {
  installEnvironment(t);
  let fetches = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response('unused');
  });
  for (const value of [
    '%',
    'file:///tmp/model/',
    'https://example.test/model/?revision=1',
    'https://example.test/model/#fragment',
    'https://user:pass@example.test/model/'
  ]) {
    await assert.rejects(prepareModelSource({assetsUrl: value}, {}), {code: 'INVALID_INPUT'});
  }
  assert.equal(fetches, 0);
});

test('valid online source adds a directory slash and returns model options', async (t: TestContext) => {
  installEnvironment(t);
  const requests: Array<[RequestInfo | URL, RequestInit | undefined]> = [];
  t.mock.method(globalThis, 'fetch', async (url: RequestInfo | URL, init?: RequestInit) => {
    requests.push([url, init]);
    return new Response('synthetic manifest');
  });
  const prepared = await prepareModelSource(
    {assetsUrl: 'https://example.test/model', cache: false, maxPrefetchBytes: 8, mode: 'test'},
    {validateModelOptions: (options) => options.mode}
  );
  assert.equal(prepared.base.href, 'https://example.test/model/');
  assert.equal(prepared.manifestUrl, 'https://example.test/model/manifest.json');
  assert.equal(prepared.maxPrefetchBytes, 8);
  assert.equal(prepared.modelOptions, 'test');
  assert.equal(await prepared.response.text(), 'synthetic manifest');
  assert.deepEqual(requests, [[prepared.manifestUrl, {signal: undefined, cache: 'no-cache'}]]);
});

test('adapter failures stop source loading before cache or fetch', async (t: TestContext) => {
  let fetches = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response('unused');
  });
  const noAdapter = gpuWith([], async () => null);
  installEnvironment(t, {gpu: noAdapter});
  await assert.rejects(prepareModelSource({assetsUrl, cache: false}, {}), {
    code: 'MODEL_LOAD_FAILED',
    message: /Could not initialize/
  });

  installEnvironment(t, {gpu: gpuWith([])});
  await assert.rejects(prepareModelSource({assetsUrl, cache: false}, {}), {
    code: 'UNSUPPORTED_ENVIRONMENT',
    message: /shader-f16/
  });
  assert.equal(fetches, 0);
});

test('online manifest fetch errors preserve their load error details', async (t: TestContext) => {
  installEnvironment(t);
  const failure = Error('synthetic network failure');
  t.mock.method(globalThis, 'fetch', async () => {
    throw failure;
  });
  await assert.rejects(prepareModelSource({assetsUrl, cache: false}, {}), (error: unknown) => {
    assert.equal((error as {code?: unknown}).code, 'ASSET_LOAD_FAILED');
    assert.equal((error as {message?: unknown}).message, 'Could not fetch manifest.json.');
    assert.equal((error as {cause?: unknown}).cause, failure);
    return true;
  });
});

test('online manifest HTTP errors cancel the body and report the status', async (t: TestContext) => {
  installEnvironment(t);
  let cancelled = 0;
  t.mock.method(globalThis, 'fetch', async () => ({
    ok: false,
    status: 503,
    body: {
      cancel() {
        cancelled++;
      }
    }
  }));
  await assert.rejects(prepareModelSource({assetsUrl, cache: false}, {}), {
    code: 'ASSET_LOAD_FAILED',
    message: 'Could not fetch manifest.json: HTTP 503.'
  });
  assert.equal(cancelled, 1);
});

test('offline source returns a cached manifest without network access', async (t: TestContext) => {
  installEnvironment(t);
  let opened;
  let fetches = 0;
  const response = new Response('cached manifest');
  setGlobal(t, 'caches', {
    async open(name: string) {
      opened = name;
      return {
        async match(url: string) {
          assert.equal(url, 'https://example.test/model/manifest.json');
          return response;
        }
      };
    }
  });
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response('unused');
  });

  const prepared = await prepareModelSource({assetsUrl, offline: true}, {});
  assert.equal(opened, CACHE_NAME);
  assert.equal(prepared.response, response);
  assert.equal(fetches, 0);
});

test('offline cache misses and failures are reported without fetching', async (t: TestContext) => {
  installEnvironment(t);
  let fetches = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response('unused');
  });
  setGlobal(t, 'caches', {
    async open() {
      return {
        async match() {
          return null;
        }
      };
    }
  });
  await assert.rejects(
    prepareModelSource({assetsUrl, offline: true}, {offlineMissingMessage: 'synthetic offline miss'}),
    {code: 'ASSET_LOAD_FAILED', message: 'synthetic offline miss'}
  );

  const failure = Error('synthetic cache read failure');
  setGlobal(t, 'caches', {
    async open() {
      return {
        async match() {
          throw failure;
        }
      };
    }
  });
  await assert.rejects(prepareModelSource({assetsUrl, offline: true}, {}), (error: unknown) => {
    assert.equal((error as {code?: unknown}).code, 'ASSET_LOAD_FAILED');
    assert.equal((error as {message?: unknown}).message, 'Could not read the cached model manifest.');
    assert.equal((error as {cause?: unknown}).cause, failure);
    return true;
  });
  assert.equal(fetches, 0);
});

test('offline loading reports an unavailable cache', async (t: TestContext) => {
  installEnvironment(t);
  setGlobal(t, 'caches', undefined);
  await assert.rejects(prepareModelSource({assetsUrl, offline: true}, {}), {
    code: 'ASSET_LOAD_FAILED',
    message: 'The model cache is unavailable for offline loading.'
  });
});

test('caller cancellation before adapter setup is preserved', async (t: TestContext) => {
  let adapterRequests = 0;
  installEnvironment(t, {
    gpu: gpuWith(['shader-f16'], async () => {
      adapterRequests++;
      return {features: new Set(['shader-f16'])};
    })
  });
  const controller = new AbortController();
  const reason = Error('cancel before source setup');
  controller.abort(reason);
  await assert.rejects(prepareModelSource({assetsUrl, signal: controller.signal}, {}), (error) => error === reason);
  assert.equal(adapterRequests, 0);
});

test('caller cancellation while opening the asset cache is preserved', async (t: TestContext) => {
  let openStarted!: () => void;
  let resolveOpen!: (value: {match: () => Promise<null>}) => void;
  const started = new Promise<void>((resolve) => {
    openStarted = resolve;
  });
  const pending = new Promise<{match: () => Promise<null>}>((resolve) => {
    resolveOpen = resolve;
  });
  let fetches = 0;
  installEnvironment(t);
  setGlobal(t, 'caches', {
    open() {
      openStarted();
      return pending;
    }
  });
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response('unused');
  });
  const controller = new AbortController();
  const reason = Error('cancel while opening cache');
  const loading = prepareModelSource({assetsUrl, signal: controller.signal}, {});
  await started;
  controller.abort(reason);
  await assert.rejects(loading, (error) => error === reason);
  resolveOpen({match: async () => null});
  assert.equal(fetches, 0);
});
