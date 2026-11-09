import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {openD1OmniSource} from '../src/d1-omni/source.mjs';
import type {D1OmniSpec} from '../src/d1-omni/contract.mjs';

test('d1-omni source verifies bytes and reports completed asset progress', async () => {
  const bytes = new TextEncoder().encode('{"model":"d1-omni-test"}');
  const metadata = {
    bytes: bytes.byteLength,
    sha256: createHash('sha256').update(bytes).digest('hex')
  };
  const files = {'config.json': metadata};
  const spec: D1OmniSpec = {
    model: 'd1-omni-test',
    repository: 'test/d1-omni',
    revision: 'pinned-test-revision',
    sourceRepository: 'test/d1-omni-source',
    sourceRevision: 'pinned-test-source-revision',
    files,
    manifest: {files},
    totalBytes: bytes.byteLength,
    maxAssetBytes: bytes.byteLength,
    maxTokens: 1,
    temperatures: {}
  };
  const originalFetch = globalThis.fetch;
  const progress: {loadedBytes: number; totalBytes: number; fraction: number}[] = [];
  globalThis.fetch = async () => new Response(bytes.slice());

  try {
    const source = await openD1OmniSource({
      base: new URL('https://assets.example/d1-omni/'),
      cacheEnabled: false,
      offline: false,
      onProgress(update) {
        progress.push(update);
      },
      spec
    });
    const result = await source.read('config.json');

    assert.deepEqual(new Uint8Array(result), bytes);
    assert.deepEqual(progress, [{loadedBytes: bytes.byteLength, totalBytes: bytes.byteLength, fraction: 1}]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('d1-omni source defaults to its pinned manifest', async () => {
  const source = await openD1OmniSource({
    base: new URL('https://assets.example/d1-omni/'),
    cacheEnabled: false,
    offline: true
  });

  await assert.rejects(source.read('not-listed.json'), (error: unknown) => {
    return error instanceof Error && error.message === 'Asset not listed in manifest: not-listed.json';
  });
});
