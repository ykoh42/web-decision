import assert from 'node:assert/strict';
import test from 'node:test';
import type {ManifestSourceSpec} from '../src/asset-source-types.js';
import {openSafeDecideSource} from '../src/safedecide/source.mjs';

test('SafeDecide source reads the manifest before asset-limit getters', async () => {
  const accesses: string[] = [];
  const spec: ManifestSourceSpec = {
    get manifest() {
      accesses.push('manifest');
      return {files: {}};
    },
    get maxAssetBytes() {
      accesses.push('maxAssetBytes');
      return 16;
    },
    totalBytes: 16
  };
  const source = await openSafeDecideSource({base: new URL('https://example.test/assets/'), cacheEnabled: false, spec});

  await assert.rejects(source.read('missing.bin'));
  assert.deepEqual(accesses, ['manifest', 'maxAssetBytes']);
});

test('SafeDecide source preserves manifest getter short-circuit and exception identity', async () => {
  const accesses: string[] = [];
  const failure = new Error('manifest getter failed');
  const spec: ManifestSourceSpec = {
    get manifest(): ManifestSourceSpec['manifest'] {
      accesses.push('manifest');
      throw failure;
    },
    get maxAssetBytes() {
      accesses.push('maxAssetBytes');
      return 16;
    },
    totalBytes: 16
  };
  const source = await openSafeDecideSource({base: new URL('https://example.test/assets/'), cacheEnabled: false, spec});

  await assert.rejects(source.read('missing.bin'), (error: unknown) => error === failure);
  assert.deepEqual(accesses, ['manifest']);
});
