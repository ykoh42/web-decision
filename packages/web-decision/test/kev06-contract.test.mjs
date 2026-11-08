import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {KEV06_MANIFEST_SHA256, readKev06ManifestResponse} from '../src/kev06/contract.js';

const manifestBytes = await readFile(new URL('./fixtures/kev06-manifest.json', import.meta.url));

test('accepts the exact Kev 0.6B asset manifest', async () => {
  const manifest = await readKev06ManifestResponse(new Response(manifestBytes));
  assert.equal(manifest.model, 'kev-0.6b-q4f16');
  assert.equal(Object.hasOwn(manifest, 'maxInputTokens'), false);
  assert.equal(manifest.maxStateTokens, 8192);
  assert.equal(manifest.maxBranchTokens, 8192);
  assert.match(KEV06_MANIFEST_SHA256, /^[0-9a-f]{64}$/);
});

test('accepts a cached reserialization of the pinned manifest', async () => {
  const cached = JSON.stringify(JSON.parse(manifestBytes.toString('utf8')));
  const manifest = await readKev06ManifestResponse(new Response(cached));
  assert.equal(manifest.model, 'kev-0.6b-q4f16');
});

test('rejects a manifest that changes its declared model assets', async () => {
  const changed = JSON.parse(manifestBytes.toString('utf8'));
  changed.files['tokenizer.json'].sha256 = '0'.repeat(64);
  await assert.rejects(readKev06ManifestResponse(new Response(JSON.stringify(changed))), {code: 'INVALID_ASSET'});
});
