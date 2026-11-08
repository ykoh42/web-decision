import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {KEV06_MANIFEST_SHA256, readKev06ManifestResponse} from '../src/kev06/contract.js';
import {
  EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256
} from '../src/kev06/upstream-contract.js';

type ManifestFile = {bytes: number; sha256: string};
type KevManifest = {
  experimental: boolean;
  files: Record<string, ManifestFile>;
  maxBranchTokens: number;
  maxStateTokens: number;
  model: string;
};
type ShardMap = {
  source: {sha256: string};
  target: {shards: {bytes: number; path: string; sha256: string}[]};
};

const manifestBytes: Buffer = await readFile(new URL('./fixtures/kev06-manifest.json', import.meta.url));
const stableManifestBytes: Buffer = await readFile(new URL('../assets/kev06-stable/manifest.json', import.meta.url));
const stableWasmBytes: Buffer = await readFile(new URL('../assets/kev06-stable/model.wasm', import.meta.url));
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

function parseManifest(bytes: Buffer): KevManifest {
  return JSON.parse(bytes.toString('utf8')) as KevManifest;
}

function requiredFile(manifest: KevManifest, name: string): ManifestFile {
  const metadata = manifest.files[name];
  assert.ok(metadata, 'The manifest must contain ' + name + '.');
  return metadata;
}

function responseFromBytes(bytes: Uint8Array): Response {
  const body = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(body).set(bytes);
  return new Response(body);
}

test('stable bundled Kev WASM and manifest match the current-main pins', async () => {
  const manifest = (await readKev06ManifestResponse(responseFromBytes(stableManifestBytes))) as KevManifest;
  const model = requiredFile(manifest, 'model.wasm');
  assert.equal(sha256(stableWasmBytes), model.sha256);
  assert.equal(stableWasmBytes.byteLength, model.bytes);
  assert.equal(sha256(stableWasmBytes), '68fe0331180dc1adfb3d1e4295c06133815b3f5bfc5ad8dbd73ae66f3a02b8fb');
  assert.equal(KEV06_MANIFEST_SHA256, '8577804e638051cb056af51fee89eb18ea5de85ccec4f196594d830c16fdc767');
  assert.equal(manifest.experimental, true);
});

test('stable shards and reused bundle metadata match their pinned manifest', async () => {
  const manifest = parseManifest(stableManifestBytes);
  const metadataPaths = ['tokenizer.json', 'tokenizer_config.json', 'model_config.json', 'weights/tensor-cache.json'];
  for (const path of metadataPaths) {
    const bytes: Buffer = await readFile(new URL('../assets/experimental-kev-upstream/' + path, import.meta.url));
    const metadata = requiredFile(manifest, path);
    assert.equal(bytes.byteLength, metadata.bytes, path);
    assert.equal(sha256(bytes), metadata.sha256, path);
  }

  const mapBytes: Buffer = await readFile(
    new URL('../assets/experimental-kev-upstream/source-shard-map.json', import.meta.url)
  );
  const map = JSON.parse(mapBytes.toString('utf8')) as ShardMap;
  assert.equal(sha256(mapBytes), EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256);
  assert.equal(map.source.sha256, EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256);
  const modelConfig = JSON.parse(
    (await readFile(new URL('../assets/experimental-kev-upstream/model_config.json', import.meta.url))).toString('utf8')
  ) as {max_position_embeddings: number};
  assert.equal(modelConfig.max_position_embeddings, 32768);
  for (const shard of map.target.shards)
    assert.deepEqual(manifest.files[shard.path], {bytes: shard.bytes, sha256: shard.sha256}, shard.path);
});

test('accepts the exact Kev 0.6B asset manifest', async () => {
  const manifest = (await readKev06ManifestResponse(responseFromBytes(manifestBytes))) as KevManifest;
  assert.equal(manifest.model, 'kev-0.6b-q4f16');
  assert.equal(Object.hasOwn(manifest, 'maxInputTokens'), false);
  assert.equal(manifest.maxStateTokens, 8192);
  assert.equal(manifest.maxBranchTokens, 8192);
  assert.match(KEV06_MANIFEST_SHA256, /^[0-9a-f]{64}$/);
});

test('accepts a cached reserialization of the pinned manifest', async () => {
  const cached = JSON.stringify(parseManifest(manifestBytes));
  const manifest = (await readKev06ManifestResponse(new Response(cached))) as KevManifest;
  assert.equal(manifest.model, 'kev-0.6b-q4f16');
});

test('rejects a manifest that changes its declared model assets', async () => {
  const changed = parseManifest(manifestBytes);
  const tokenizer = requiredFile(changed, 'tokenizer.json');
  changed.files['tokenizer.json'] = {...tokenizer, sha256: '0'.repeat(64)};
  await assert.rejects(readKev06ManifestResponse(new Response(JSON.stringify(changed))), {code: 'INVALID_ASSET'});
});
