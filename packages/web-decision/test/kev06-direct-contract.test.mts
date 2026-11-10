import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {test} from 'node:test';
import {
  DIRECT_KEV06_BASE,
  DIRECT_KEV06_CHECKPOINT,
  DIRECT_KEV06_PARAMETER_COUNT,
  hasDirectKev06CatalogPin,
  readDirectKev06ManifestResponse
} from '../src/kev06/direct-contract.js';

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonical((value as Record<string, unknown>)[key])])
  );
}

type ManifestFixture = {
  formatVersion: number;
  model: string;
  backend: string;
  parameterCount: number;
  maxStateTokens: number;
  maxBranchTokens: number;
  checkpoint: {repository: string; revision: string};
  base: {repository: string; revision: string};
  files: Record<string, {bytes: number; sha256: string}>;
};

function fixture(): ManifestFixture {
  return {
    formatVersion: 1,
    model: 'kev-0.6b-q4f16',
    backend: 'tvm-webgpu',
    parameterCount: DIRECT_KEV06_PARAMETER_COUNT,
    maxStateTokens: 8192,
    maxBranchTokens: 8192,
    checkpoint: {...DIRECT_KEV06_CHECKPOINT},
    base: {...DIRECT_KEV06_BASE},
    files: {
      'model.wasm': {bytes: 1, sha256: 'a'.repeat(64)},
      'tokenizer.json': {bytes: 1, sha256: 'b'.repeat(64)},
      'tokenizer_config.json': {bytes: 1, sha256: 'c'.repeat(64)},
      'model_config.json': {bytes: 1, sha256: 'd'.repeat(64)},
      'weights/tensor-cache.json': {bytes: 1, sha256: 'e'.repeat(64)},
      'weights/params_shard_0000.bin': {bytes: 4, sha256: 'f'.repeat(64)}
    }
  };
}

async function pinnedResponse(value: unknown): Promise<{readonly response: Response; readonly sha256: string}> {
  const canonicalBytes = new TextEncoder().encode(JSON.stringify(canonical(value)) + '\n');
  return {
    response: new Response(JSON.stringify(value)),
    sha256: createHash('sha256').update(canonicalBytes).digest('hex')
  };
}

test('direct Kev manifest is pinned to the source/base revisions and 511 positional inputs', async () => {
  const {response, sha256} = await pinnedResponse(fixture());
  const manifest = await readDirectKev06ManifestResponse(response, sha256);

  assert.equal(manifest.model, 'kev-0.6b-q4f16');
  assert.equal(manifest.parameterCount, 511);
  assert.deepEqual(manifest.checkpoint, DIRECT_KEV06_CHECKPOINT);
  assert.ok(manifest.files['weights/params_shard_0000.bin']);
});

test('direct Kev manifest rejects a digest outside the catalog pin', async () => {
  const {response, sha256} = await pinnedResponse(fixture());
  await assert.rejects(readDirectKev06ManifestResponse(response, '0'.repeat(64)), {code: 'INVALID_ASSET'});
  assert.match(sha256, /^[0-9a-f]{64}$/u);
});

test('direct Kev manifest rejects a legacy parameter inventory or different checkpoint', async (t) => {
  for (const [field, update] of [
    ['parameterCount', (manifest: ReturnType<typeof fixture>) => (manifest.parameterCount = 394)],
    ['baseRevision', (manifest: ReturnType<typeof fixture>) => (manifest.base.revision = '0'.repeat(40))]
  ] as const) {
    await t.test(field, async () => {
      const value = fixture();
      update(value);
      const {response, sha256} = await pinnedResponse(value);
      await assert.rejects(readDirectKev06ManifestResponse(response, sha256), {code: 'INVALID_ASSET'});
    });
  }
});

test('direct Kev manifest requires the config and raw tensor cache used by the loader', async (t) => {
  for (const path of ['model_config.json', 'weights/tensor-cache.json']) {
    await t.test(path, async () => {
      const value = fixture();
      delete value.files[path as keyof typeof value.files];
      const {response, sha256} = await pinnedResponse(value);
      await assert.rejects(readDirectKev06ManifestResponse(response, sha256), {code: 'INVALID_ASSET'});
    });
  }
});

test('curated direct loading stays gated until the catalog pins the direct checkpoint', () => {
  const oldOnnxPin = {
    modelId: 'kev-0.6b-q4f16',
    checkpoint: {repository: 'onnx-community/kev-0.6b-ONNX', revision: '5a827091b3d58fe4fdfc02a5a5b2ce364047d678'},
    assets: {manifestSha256: '0'.repeat(64)}
  };
  const directPin = {
    modelId: 'kev-0.6b-q4f16',
    checkpoint: DIRECT_KEV06_CHECKPOINT,
    assets: {
      manifestSha256: '1'.repeat(64),
      directArtifactRoot: 'https://assets.example.test/kev-0.6b-q4f16/'
    }
  };

  assert.equal(hasDirectKev06CatalogPin(oldOnnxPin), false);
  assert.equal(hasDirectKev06CatalogPin(directPin), true);
  assert.equal(hasDirectKev06CatalogPin({...directPin, assets: {manifestSha256: '1'.repeat(64)}}), false);
  assert.equal(
    hasDirectKev06CatalogPin({
      ...directPin,
      assets: {...directPin.assets, directArtifactRoot: 'http://assets.example.test/kev/'}
    }),
    false
  );
  assert.equal(hasDirectKev06CatalogPin({...directPin, modelId: 'kanana-2-v-decide-3b-2611'}), false);
  assert.equal(hasDirectKev06CatalogPin({...directPin, assets: {manifestSha256: 'not-a-manifest-digest'}}), false);
});
