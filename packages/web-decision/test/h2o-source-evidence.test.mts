import assert from 'node:assert/strict';
import test from 'node:test';
import {readH2oInventory} from '../src/h2o/source-evidence.mjs';

test('H2O inventory rejects invalid asset directories before any network request', async (t) => {
  const originalFetch = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = async () => {
    requests++;
    throw new Error('unexpected request');
  };
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  for (const url of [
    'not a URL',
    'file:///tmp/',
    'https://user:pass@example.invalid/',
    'https://example.invalid/?query=1',
    'https://example.invalid/#fragment',
    'https://example.invalid/missing-trailing-slash'
  ]) {
    await assert.rejects(readH2oInventory(url), {code: 'INVALID_ASSET'});
  }
  assert.equal(requests, 0);
});

test('H2O inventory preserves cancellation before reading the source', async () => {
  const controller = new AbortController();
  const reason = new Error('inventory cancelled');
  controller.abort(reason);
  await assert.rejects(readH2oInventory('https://example.invalid/', controller.signal), (error) => error === reason);
});
