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
  ]) {
    await t.test(`${root} + ${name}`, () => {
      assert.throws(() => resolveDirectKevAssetUrl(root, name), {code: 'INVALID_ASSET'});
    });
  }
});

test('direct Kev source fetches the manifest and model files from the pinned root', async () => {
  const root = 'https://assets.example.test/kev/resolve/0123456789abcdef0123456789abcdef01234567/';
  const requestedUrls: string[] = [];
  const originalGlobals = {
    fetch: globalThis.fetch,
    location: Object.getOwnPropertyDescriptor(globalThis, 'location'),
    navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  };

  Object.defineProperty(globalThis, 'location', {configurable: true, value: {origin: 'https://app.example.test'}});
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {gpu: {requestAdapter: async () => ({features: new Set(['shader-f16'])})}}
  });
  globalThis.fetch = async (input) => {
    requestedUrls.push(String(input));
    return new Response('{}');
  };

  try {
    const source = createDirectKevPackageSource('a'.repeat(64), root);
    const prepared = await source.prepare({cache: false});
    const modelUrl = assetUrl(prepared.base, 'weights/params_shard_0000.bin');
    await source.fetch?.(modelUrl);
  } finally {
    globalThis.fetch = originalGlobals.fetch;
    if (originalGlobals.location) Object.defineProperty(globalThis, 'location', originalGlobals.location);
    else Reflect.deleteProperty(globalThis, 'location');
    if (originalGlobals.navigator) Object.defineProperty(globalThis, 'navigator', originalGlobals.navigator);
    else Reflect.deleteProperty(globalThis, 'navigator');
  }

  assert.deepEqual(requestedUrls, [`${root}manifest.json`, `${root}weights/params_shard_0000.bin`]);
});
