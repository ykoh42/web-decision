import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import test from 'node:test';
import {loadExperimentalMapikaQ4} from '../src/mapika-q4/loader.js';

test('Mapika Q4 rejects offline loading when the cache is disabled', async () => {
  await assert.rejects(
    loadExperimentalMapikaQ4({assetsUrl: 'https://example.invalid/model/', offline: true, cache: false}),
    {code: 'INVALID_INPUT', message: 'Offline loading requires the model cache.'}
  );
});

test('Mapika Q4 stops a cancelled load before requesting a GPU or assets', async (t) => {
  const previous = new Map(
    ['navigator', 'crypto', 'fetch'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)])
  );
  t.after(() => {
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
  });
  let requests = 0;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        requestAdapter() {
          requests++;
        }
      }
    }
  });
  Object.defineProperty(globalThis, 'crypto', {configurable: true, value: webcrypto});
  Object.defineProperty(globalThis, 'fetch', {
    configurable: true,
    value() {
      requests++;
    }
  });
  const controller = new AbortController();
  const reason = new Error('cancelled before loading');
  controller.abort(reason);
  await assert.rejects(
    loadExperimentalMapikaQ4({assetsUrl: 'https://example.invalid/model/', signal: controller.signal}),
    (error) => error === reason
  );
  assert.equal(requests, 0);
});
