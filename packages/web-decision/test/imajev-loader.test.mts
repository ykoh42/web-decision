import assert from 'node:assert/strict';
import test from 'node:test';
import {selectModelFamily} from '../src/model-family.js';
import {loadExperimentalImajev} from '../src/imajev/loader.mjs';
import {rejectUnverifiedImajevAssets} from '../src/imajev/contract.mjs';

test('the model router selects the explicit Imajev candidate', () => {
  assert.equal(selectModelFamily({model: 'experimental-imajev-4b-candidate'}), 'imajev');
  assert.equal(selectModelFamily({}), 'kev');
});

test('unverified candidate manifests cannot reach fetch, cache or GPU', async () => {
  const oldFetch = Object.getOwnPropertyDescriptor(globalThis, 'fetch');
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const oldCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  let fetches = 0,
    gpuReads = 0,
    cacheReads = 0;
  try {
    const manifests = [
      '{}',
      '{not-json',
      JSON.stringify({
        model: 'experimental-imajev-4b-candidate',
        executable: true,
        abi: ['fake'],
        ranges: Array.from({length: 407}, () => '0'.repeat(64))
      })
    ];
    for (const manifest of manifests) {
      Object.defineProperty(globalThis, 'fetch', {
        configurable: true,
        value: async () => {
          fetches++;
          return new Response(manifest);
        }
      });
      Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: {
          get gpu() {
            gpuReads++;
            return {};
          }
        }
      });
      Object.defineProperty(globalThis, 'caches', {
        configurable: true,
        get() {
          cacheReads++;
          return {};
        }
      });
      await assert.rejects(loadExperimentalImajev({assetsUrl: 'https://fixture.invalid/'}), {
        code: 'UNSUPPORTED_ENVIRONMENT'
      });
      assert.throws(() => (rejectUnverifiedImajevAssets as (value: unknown) => never)(manifest), {
        code: 'UNSUPPORTED_ENVIRONMENT'
      });
    }
    assert.deepEqual([fetches, gpuReads, cacheReads], [0, 0, 0]);
  } finally {
    for (const [key, descriptor] of [
      ['fetch', oldFetch],
      ['navigator', oldNavigator],
      ['caches', oldCaches]
    ] as const) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete (globalThis as unknown as Record<string, unknown>)[key];
    }
  }
});

test('Imajev candidate validates common options and preserves pre-abort reasons', async () => {
  for (const options of [
    {},
    {assetsUrl: 'https://fixture.invalid/', offline: 'yes'},
    {assetsUrl: 'https://fixture.invalid/', offline: true, cache: false},
    {assetsUrl: 'https://fixture.invalid/', maxPrefetchBytes: -1},
    {assetsUrl: 'javascript:alert(1)'},
    {assetsUrl: 'https://fixture.invalid/?x=1'},
    {assetsUrl: 'https://user:pass@fixture.invalid/'},
    {assetsUrl: 'https://fixture.invalid/#fragment'}
  ])
    await assert.rejects(loadExperimentalImajev(options), {code: 'INVALID_INPUT'});
  for (const reason of [undefined, null, false, 0, '']) {
    const controller = new AbortController();
    controller.abort(reason);
    let caught = false,
      observed;
    try {
      await loadExperimentalImajev({assetsUrl: 'https://fixture.invalid/', signal: controller.signal});
    } catch (error) {
      caught = true;
      observed = error;
    }
    assert.equal(caught, true);
    assert.equal(observed, controller.signal.reason);
  }
});
