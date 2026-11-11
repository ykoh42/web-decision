import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createDirectKevPackageSource, resolveDirectKevAssetUrl} from '../src/kev06/direct-source.mjs';
import {assetUrl} from '../src/assets.js';

test('direct Kev assets resolve under the catalog-pinned artifact directory', () => {
  const root = 'https://huggingface.co/example/kev/resolve/0123456789abcdef0123456789abcdef01234567/';

  assert.equal(resolveDirectKevAssetUrl(root, 'manifest.json'), `${root}manifest.json`);
  assert.equal(resolveDirectKevAssetUrl(root, 'weights/params_shard_0000.bin'), `${root}weights/params_shard_0000.bin`);
  assert.equal(resolveDirectKevAssetUrl(root, 'token files/tokenizer.json'), `${root}token%20files/tokenizer.json`);
});

test('direct Kev assets reject roots and paths that can escape the pinned directory', async (t) => {
  for (const [root, name] of [
    ['http://assets.example.test/kev/', 'manifest.json'],
    ['https://assets.example.test/kev?mutable=1', 'manifest.json'],
    ['https://user:password@assets.example.test/kev/', 'manifest.json'],
    ['https://assets.example.test/kev/', '../manifest.json'],
    ['https://assets.example.test/kev/', 'weights/../../manifest.json'],
    ['https://assets.example.test/kev/', '/manifest.json'],
    ['https://assets.example.test/kev/', 'weights\\shard.bin'],
    ['https://assets.example.test/kev/', 'weights/%2e%2e/shard.bin'],
    ['https://assets.example.test/kev/', 'manifest.json?alternate=1']
  ] as const) {
    await t.test(`${root} + ${name}`, () => {
      assert.throws(() => resolveDirectKevAssetUrl(root, name), {code: 'INVALID_ASSET'});
    });
  }
});

test('direct Kev source fetches from its pinned root and reuses the cached manifest offline', async () => {
  const root = 'https://assets.example.test/kev/resolve/0123456789abcdef0123456789abcdef01234567/';
  const requestedUrls: string[] = [];
  const originalGlobals = {
    fetch: globalThis.fetch,
    caches: Object.getOwnPropertyDescriptor(globalThis, 'caches'),
    location: Object.getOwnPropertyDescriptor(globalThis, 'location'),
    navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  };
  const cacheEntries = new Map<string, Response>();
  const cache = {
    match: async (key: RequestInfo | URL) => cacheEntries.get(String(key))?.clone(),
    put: async (key: RequestInfo | URL, response: Response) => cacheEntries.set(String(key), response.clone())
  } as unknown as Cache;

  Object.defineProperty(globalThis, 'location', {configurable: true, value: {origin: 'https://app.example.test'}});
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {gpu: {requestAdapter: async () => ({features: new Set(['shader-f16'])})}}
  });
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: {open: async () => cache}
  });
  globalThis.fetch = async (input) => {
    requestedUrls.push(String(input));
    return new Response('{}');
  };

  try {
    const source = createDirectKevPackageSource('a'.repeat(64), root);
    assert.equal(source.assetLoadPhase, 'model-assets');
    const prepared = await source.prepare({cache: true});
    const modelUrl = assetUrl(prepared.base, 'weights/params_shard_0000.bin');
    await source.fetch?.(modelUrl);
    await cache.put(prepared.manifestUrl, prepared.response.clone());
    const offline = await source.prepare({cache: true, offline: true});
    assert.equal(await offline.response.text(), '{}');
  } finally {
    globalThis.fetch = originalGlobals.fetch;
    if (originalGlobals.caches) Object.defineProperty(globalThis, 'caches', originalGlobals.caches);
    else Reflect.deleteProperty(globalThis, 'caches');
    if (originalGlobals.location) Object.defineProperty(globalThis, 'location', originalGlobals.location);
    else Reflect.deleteProperty(globalThis, 'location');
    if (originalGlobals.navigator) Object.defineProperty(globalThis, 'navigator', originalGlobals.navigator);
    else Reflect.deleteProperty(globalThis, 'navigator');
  }

  assert.deepEqual(requestedUrls, [`${root}manifest.json`, `${root}weights/params_shard_0000.bin`]);
});
