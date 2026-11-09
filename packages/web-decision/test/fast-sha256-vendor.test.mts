import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import test from 'node:test';

// Keep the reviewed ESM adaptation tied to the upstream provenance record.
test('the vendored fast-sha256 implementation matches its pinned digest', () => {
  const bytes = readFileSync(new URL('../src/vendor/fast-sha256/sha256.mjs', import.meta.url));
  assert.equal(
    createHash('sha256').update(bytes).digest('hex'),
    'c096ba1501c4a1f6cf1f093a0cdb03671a9df9f4024cde490013799a2c7b25d5'
  );
});
