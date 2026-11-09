import assert from 'node:assert/strict';
import test from 'node:test';
import {loadExperimentalStrandsFromSource} from '../src/strands/loader.mjs';

test('the Strands source entry rejects invalid asset settings before requesting an adapter', async () => {
  let adapterRequests = 0;
  const gpuApi = {
    async requestAdapter() {
      adapterRequests++;
      return null;
    }
  };
  await assert.rejects(
    loadExperimentalStrandsFromSource({assetsUrl: 'https://example.test/model/', gpuApi, maxPrefetchBytes: 1}),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(adapterRequests, 0);
});

test('the Strands source entry preserves a cancellation before requesting an adapter', async () => {
  const controller = new AbortController();
  const reason = new Error('cancelled before adapter request');
  controller.abort(reason);
  let adapterRequests = 0;
  const gpuApi = {
    async requestAdapter() {
      adapterRequests++;
      return null;
    }
  };
  await assert.rejects(
    loadExperimentalStrandsFromSource({assetsUrl: 'https://example.test/model/', gpuApi, signal: controller.signal}),
    (error: unknown) => error === reason
  );
  assert.equal(adapterRequests, 0);
});
