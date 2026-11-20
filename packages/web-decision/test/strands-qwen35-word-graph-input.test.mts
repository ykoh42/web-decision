import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {
  materializeQwen35WordGraphInput,
  planQwen35WordGraphInput,
  type Qwen35WordGraphBinding
} from '../src/strands/qwen35-word-graph-input.mts';
import {type Qwen35WordAssetManifest, type Qwen35WordAssetSource} from '../src/strands/qwen35-assets.mts';

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function bytesFromWords(...words: number[]): Uint8Array {
  const bytes = new Uint8Array(words.length * 4);
  const view = new DataView(bytes.buffer);
  words.forEach((word, index) => view.setUint32(index * 4, word, true));
  return bytes;
}

function sourceFor(manifest: Qwen35WordAssetManifest, ranges: readonly Uint8Array[]): Qwen35WordAssetSource {
  return {
    manifest,
    async readRange(_asset, index) {
      const range = ranges[index];
      if (!range) throw new Error('Unexpected range index: ' + index);
      return range.slice().buffer;
    },
    async commit() {
      return false;
    }
  };
}

function syntheticManifest(
  payload: Uint8Array,
  packedTensor: Uint8Array,
  ranges: readonly Uint8Array[],
  rangeOffsets: readonly number[],
  sourceDtype: 'BF16' | 'F32',
  shape: readonly number[],
  sourceOffset = 0
): Qwen35WordAssetManifest {
  const rangeChunks = ranges.map((range, index) => ({
    offset: rangeOffsets[index]!,
    bytes: range.byteLength,
    sha256: sha256(range)
  }));
  const tensor = {
    name: 'test.weight',
    shape,
    dtype: sourceDtype,
    elementCount: shape.reduce((count, dimension) => count * dimension, 1),
    sourceDataByteOffset: sourceOffset,
    storageByteOffset: sourceOffset,
    storageWordOffset: sourceOffset / 4,
    sourceByteCount: payload.byteLength,
    packedByteCount: packedTensor.byteLength,
    paddingByteCount: packedTensor.byteLength - payload.byteLength,
    sourcePayloadSha256: sha256(payload),
    packedTensorSha256: sha256(packedTensor)
  };
  const asset = {
    source: {
      repository: 'synthetic/test',
      revision: 'test-revision',
      file: 'test.safetensors',
      bytes: payload.byteLength,
      sha256: sha256(payload),
      pinnedSha256: sha256(payload),
      headerBytes: 0,
      headerSha256: '0'.repeat(64),
      payloadBytes: payload.byteLength
    },
    packed: {
      file: 'test.words.u32le',
      bytes: rangeChunks.reduce((total, range) => total + range.bytes, 0),
      sha256: sha256(ranges.reduce((joined, range) => concat(joined, range), new Uint8Array())),
      storageDtype: 'uint32' as const,
      endianness: 'little' as const,
      rangeChunkBytes: 8,
      rangeChunks
    },
    tensorCount: 1,
    dtypeCounts: {[sourceDtype]: 1},
    tensors: [tensor]
  };
  return {
    format: 'web-decision-qwen35-pinned-word-assets-v2',
    storage: {
      wordDtype: 'uint32',
      endianness: 'little',
      bf16Layout: 'flatten row-major; element 2n in low16, element 2n+1 in high16',
      f32Layout: 'one unchanged 32-bit source value per word',
      tensorBoundaryPadding: 'zero bytes to next 4-byte boundary; excluded by logical shape',
      maxChunkBytes: 8 * 1024 * 1024
    },
    assets: {base: asset, lora: asset, head: asset}
  } as unknown as Qwen35WordAssetManifest;
}

function concat(left: Uint8Array, right: Uint8Array): Uint8Array {
  const joined = new Uint8Array(left.byteLength + right.byteLength);
  joined.set(left);
  joined.set(right, left.byteLength);
  return joined;
}

test('materializes a pinned-loader F32→BF16 RNE→F32 input across verified range boundaries', async () => {
  const payload = bytesFromWords(0x3f808000, 0x3f818000);
  const loadedBf16 = new Uint8Array([0x80, 0x3f, 0x82, 0x3f]);
  const packedTensor = payload;
  const firstRange = concat(new Uint8Array([0xaa, 0xbb, 0xcc, 0xdd]), payload.subarray(0, 4));
  const secondRange = payload.subarray(4);
  const manifest = syntheticManifest(payload, packedTensor, [firstRange, secondRange], [0, 8], 'F32', [2], 4);
  const source = sourceFor(manifest, [firstRange, secondRange]);
  const binding: Qwen35WordGraphBinding = {
    asset: 'base',
    sourceName: 'test.weight',
    targetName: 'torso.base_model.test.weight',
    relaxArgumentName: 'p_torso_base_model_test_weight',
    shape: [2],
    sourceDtype: 'float32',
    loadedDtype: 'bfloat16',
    relaxDtype: 'float32',
    loadedPayloadSha256: sha256(loadedBf16)
  };

  const plan = planQwen35WordGraphInput(manifest, binding);
  const graphInput = await materializeQwen35WordGraphInput(source, binding);

  assert.equal(plan.materialization, 'base-f32-rne-bf16-exact-widen');
  assert.equal(plan.byteLength, 8);
  assert.equal(graphInput.sourceDataByteOffset, 4);
  assert.equal(graphInput.storageByteOffset, 4);
  assert.equal(graphInput.sourcePayloadSha256, sha256(payload));
  assert.equal(graphInput.loadedPayloadSha256, sha256(loadedBf16));
  assert.equal(graphInput.materialization, 'base-f32-rne-bf16-exact-widen');
  assert.deepEqual(graphInput.bytes, bytesFromWords(0x3f800000, 0x3f820000));
});

test('widens every BF16 word to its exact float32 bit pattern and excludes zero padding', async () => {
  const payload = new Uint8Array([
    0x80,
    0x3f, // 1.0
    0x01,
    0x00, // smallest positive BF16 subnormal
    0x00,
    0x80, // negative zero
    0x80,
    0x7f, // +infinity
    0xc1,
    0x7f // NaN payload
  ]);
  const packedTensor = concat(payload, new Uint8Array([0, 0]));
  const manifest = syntheticManifest(payload, packedTensor, [packedTensor], [0], 'BF16', [5]);
  const binding: Qwen35WordGraphBinding = {
    asset: 'base',
    sourceName: 'test.weight',
    targetName: 'torso.base_model.test.weight',
    relaxArgumentName: 'p_torso_base_model_test_weight',
    shape: [5],
    sourceDtype: 'bfloat16',
    loadedDtype: 'bfloat16',
    relaxDtype: 'float32'
  };
  const graphInput = await materializeQwen35WordGraphInput(sourceFor(manifest, [packedTensor]), binding);

  assert.equal(graphInput.loadedPayloadSha256, sha256(payload));
  assert.equal(graphInput.materialization, 'bf16-exact-widen');
  assert.deepEqual(graphInput.bytes, bytesFromWords(0x3f800000, 0x00010000, 0x80000000, 0x7f800000, 0x7fc10000));
});

test('rejects a corrupted range before exposing tensor bytes', async () => {
  const payload = bytesFromWords(0x3f800000);
  const packedTensor = payload;
  const manifest = syntheticManifest(payload, packedTensor, [payload], [0], 'F32', [1]);
  const corrupted = payload.slice();
  corrupted[0] = corrupted[0]! ^ 1;
  const binding: Qwen35WordGraphBinding = {
    asset: 'lora',
    sourceName: 'test.weight',
    targetName: 'torso.test.weight',
    relaxArgumentName: 'p_torso_test_weight',
    shape: [1],
    sourceDtype: 'float32',
    loadedDtype: 'float32',
    relaxDtype: 'float32'
  };

  await assert.rejects(materializeQwen35WordGraphInput(sourceFor(manifest, [corrupted]), binding), {
    code: 'INVALID_ASSET',
    message: 'Qwen3.5 word range hash mismatch for lora.'
  });
});
