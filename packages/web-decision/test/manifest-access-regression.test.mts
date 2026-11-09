import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {validateManifest} from '../src/assets.js';

test('manifest file metadata is rechecked if an accessor changes during validation', async () => {
  const manifest: Record<string, unknown> = JSON.parse(
    await readFile(new URL('./fixtures/kev06-manifest.json', import.meta.url), 'utf8')
  );
  const files = manifest.files;
  let reads = 0;
  Object.defineProperty(manifest, 'files', {
    enumerable: true,
    get() {
      reads++;
      return reads === 1 ? files : undefined;
    }
  });
  assert.throws(() => validateManifest(manifest), {code: 'INVALID_ASSET', message: 'The asset manifest has no files.'});
  assert.equal(reads, 2);
});
