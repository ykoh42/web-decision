import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {
  checkFusedAdapter,
  LARGEST_BINDING,
  MANIFEST_BYTES,
  MANIFEST_SHA256,
  parsePinnedManifest,
  WASM_BYTES,
  WASM_SHA256,
  uploadPinnedWeights
} from '../src/strands/manifest.mjs';

function adapter() {
  return {
    features: new Set(['shader-f16']),
    limits: {
      maxBufferSize: LARGEST_BINDING,
      maxStorageBufferBindingSize: LARGEST_BINDING,
      maxStorageBuffersPerShaderStage: 8
    }
  };
}

const candidateManifest = new URL(
  '../../../models/strands-integration-20261030/fused-ordered-manifest-4096-candidate-v1.json',
  import.meta.url
);
const candidateWasm = new URL(
  '../../../models/strands-integration-20261030/full-fused-tvm-4096-candidate-v1/model.wasm',
  import.meta.url
);
const candidateSkip =
  existsSync(candidateManifest) && existsSync(candidateWasm)
    ? false
    : 'The locally built Qwen 4096 graph and manifest are not included in the source repository.';

test('Strands accepts an adapter at its documented limits', () => {
  assert.doesNotThrow(() => checkFusedAdapter(adapter()));
});

test('Strands rejects fallback adapters and missing or insufficient GPU capabilities', () => {
  const valid = adapter();
  const invalid = [
    null,
    {...valid, isFallbackAdapter: true},
    {...valid, info: {isFallbackAdapter: true}},
    {...valid, features: new Set()},
    {...valid, limits: {...valid.limits, maxBufferSize: LARGEST_BINDING - 1}},
    {...valid, limits: {...valid.limits, maxStorageBufferBindingSize: LARGEST_BINDING - 1}},
    {...valid, limits: {...valid.limits, maxStorageBuffersPerShaderStage: 7}},
    {...valid, limits: {...valid.limits, maxBufferSize: NaN}},
    {...valid, limits: {}}
  ];
  for (const candidate of invalid) {
    assert.throws(() => checkFusedAdapter(candidate), {code: 'UNSUPPORTED_ENVIRONMENT'});
  }
});

test('Strands rejects unverified manifests before parsing or uploading weights', async () => {
  await assert.rejects(parsePinnedManifest(new Uint8Array()), {code: 'INVALID_ASSET'});
  await assert.rejects(parsePinnedManifest(new TextEncoder().encode('{}')), {code: 'INVALID_ASSET'});
  let allocated = false;
  await assert.rejects(uploadPinnedWeights({records: []}, {allocate: () => (allocated = true), upload() {}}), {
    code: 'INVALID_ASSET'
  });
  assert.equal(allocated, false);
});

test(
  'pinned candidate manifest matches the compiled 4096 graph and all 328 parameters',
  {skip: candidateSkip},
  async () => {
    const manifestBytes = new Uint8Array(await readFile(candidateManifest));
    const wasmBytes = await readFile(candidateWasm);
    const manifest = await parsePinnedManifest(manifestBytes);

    assert.equal(MANIFEST_SHA256, '74eb7f501ab64d5906a498dac20e161044616cfb67607ddd5c24030887132209');
    assert.equal(WASM_SHA256, '020720bae0d99b592b3985130daac08a2c41adadad76143ab649f4afe109bc6e');
    assert.equal(manifestBytes.byteLength, MANIFEST_BYTES);
    assert.equal(wasmBytes.byteLength, WASM_BYTES);
    assert.equal(createHash('sha256').update(wasmBytes).digest('hex'), WASM_SHA256);
    assert.equal(manifest.inputContract.sequence, '2..4096');
    assert.equal(manifest.records.length, 328);
    assert.equal(manifest.totalBytes, 3776255872);
  }
);
