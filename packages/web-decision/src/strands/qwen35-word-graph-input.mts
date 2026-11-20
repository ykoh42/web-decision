import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {
  Qwen35WordAssetId,
  Qwen35WordAssetManifest,
  Qwen35WordAssetSource,
  Qwen35WordTensor
} from './qwen35-assets.mjs';

export interface Qwen35WordGraphBinding {
  readonly asset: Qwen35WordAssetId;
  readonly sourceName: string;
  readonly targetName: string;
  readonly relaxArgumentName: string;
  readonly shape: readonly number[];
  readonly sourceDtype: 'bfloat16' | 'float32';
  readonly loadedDtype: 'bfloat16' | 'float32';
  readonly relaxDtype: 'float32';
  /** Required for the pinned base F32→BF16 RNE path; comes from its validated source-cast proof. */
  readonly loadedPayloadSha256?: string | undefined;
}

export interface Qwen35MaterializedWordGraphInput {
  readonly asset: Qwen35WordAssetId;
  readonly sourceName: string;
  readonly targetName: string;
  readonly argumentName: string;
  readonly shape: readonly number[];
  readonly sourceDtype: 'bfloat16' | 'float32';
  readonly loadedDtype: 'bfloat16' | 'float32';
  readonly dtype: 'float32';
  readonly sourceDataByteOffset: number;
  readonly storageByteOffset: number;
  readonly sourcePayloadSha256: string;
  readonly loadedPayloadSha256: string;
  readonly materialization: 'source-f32' | 'bf16-exact-widen' | 'base-f32-rne-bf16-exact-widen';
  /** Little-endian graph input bytes after only the pinned loader's dtype steps. */
  readonly bytes: Uint8Array;
  readonly sha256: string;
}

export interface Qwen35WordGraphInputPlan {
  readonly tensor: Qwen35WordTensor;
  readonly dtype: 'float32';
  readonly byteLength: number;
  readonly materialization: Qwen35MaterializedWordGraphInput['materialization'];
}

const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);

function wordDtype(tensor: Qwen35WordTensor): 'bfloat16' | 'float32' {
  return tensor.dtype === 'BF16' ? 'bfloat16' : 'float32';
}

function sameShape(actual: readonly number[], expected: readonly number[], name: string): void {
  if (actual.length !== expected.length || actual.some((dimension, index) => dimension !== expected[index]))
    throw invalid('Qwen3.5 graph ABI shape mismatch for ' + name + '.');
}

function float32BytesToBf16Rne(source: Uint8Array, elementCount: number): Uint8Array {
  if (source.byteLength !== elementCount * 4) throw invalid('Invalid Qwen3.5 base F32 source payload.');
  const input = new DataView(source.buffer, source.byteOffset, source.byteLength);
  const output = new Uint8Array(elementCount * 2);
  const result = new DataView(output.buffer);
  for (let index = 0; index < elementCount; index += 1) {
    const bits = input.getUint32(index * 4, true);
    const upper = bits >>> 16;
    const exponent = bits & 0x7f800000;
    const mantissa = bits & 0x007fffff;
    const rounded = exponent === 0x7f800000 && mantissa !== 0 ? upper | 0x0040 : (bits + 0x7fff + (upper & 1)) >>> 16;
    result.setUint16(index * 2, rounded, true);
  }
  return output;
}

function widenBf16ToFloat32(source: Uint8Array, elementCount: number): Uint8Array {
  if (source.byteLength !== elementCount * 2) throw invalid('Invalid Qwen3.5 loaded BF16 payload.');
  const input = new DataView(source.buffer, source.byteOffset, source.byteLength);
  const output = new Uint8Array(elementCount * 4);
  const result = new DataView(output.buffer);
  for (let index = 0; index < elementCount; index += 1)
    result.setUint32(index * 4, input.getUint16(index * 2, true) << 16, true);
  return output;
}

function findTensor(manifest: Qwen35WordAssetManifest, binding: Qwen35WordGraphBinding): Qwen35WordTensor {
  const asset = manifest.assets[binding.asset];
  const tensor = asset?.tensors.find(({name}) => name === binding.sourceName);
  if (!tensor) throw invalid('Unknown Qwen3.5 word tensor: ' + binding.sourceName + '.');
  sameShape(binding.shape, tensor.shape, binding.sourceName);
  if (binding.sourceDtype !== wordDtype(tensor))
    throw invalid('Qwen3.5 graph ABI source dtype differs from word manifest for ' + binding.sourceName + '.');
  const baseBf16 = binding.asset === 'base' && binding.loadedDtype === 'bfloat16';
  const adapterF32 =
    (binding.asset === 'lora' || binding.asset === 'head') &&
    binding.sourceDtype === 'float32' &&
    binding.loadedDtype === 'float32';
  if (!baseBf16 && !adapterF32)
    throw invalid('Qwen3.5 word tensor dtype path differs from its pinned loader: ' + binding.sourceName + '.');
  if (binding.relaxDtype !== 'float32')
    throw invalid('Qwen3.5 pinned Relax ABI requires float32 for ' + binding.sourceName + '.');
  if (binding.sourceDtype === 'float32' && binding.loadedDtype === 'bfloat16') {
    if (
      binding.asset !== 'base' ||
      typeof binding.loadedPayloadSha256 !== 'string' ||
      !/^[0-9a-f]{64}$/.test(binding.loadedPayloadSha256)
    )
      throw invalid(
        'Qwen3.5 base F32-to-BF16 conversion lacks its pinned source-cast proof: ' + binding.sourceName + '.'
      );
  } else if (binding.loadedPayloadSha256 !== undefined && binding.loadedPayloadSha256 !== tensor.sourcePayloadSha256) {
    throw invalid(
      'Qwen3.5 unchanged tensor payload hash differs from the pinned manifest: ' + binding.sourceName + '.'
    );
  }
  return tensor;
}

function validateBindingShape(binding: Qwen35WordGraphBinding): void {
  if (
    !binding ||
    typeof binding.sourceName !== 'string' ||
    !binding.sourceName ||
    typeof binding.targetName !== 'string' ||
    !binding.targetName ||
    typeof binding.relaxArgumentName !== 'string' ||
    !binding.relaxArgumentName ||
    !Array.isArray(binding.shape) ||
    binding.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1)
  )
    throw invalid('Invalid Qwen3.5 graph weight binding.');
}

/** Check a manifest/ABI binding without reading a range or allocating tensor data. */
export function planQwen35WordGraphInput(
  manifest: Qwen35WordAssetManifest,
  binding: Qwen35WordGraphBinding
): Qwen35WordGraphInputPlan {
  validateBindingShape(binding);
  const tensor = findTensor(manifest, binding);
  const materialization =
    binding.sourceDtype === 'float32' && binding.loadedDtype === 'bfloat16'
      ? 'base-f32-rne-bf16-exact-widen'
      : binding.loadedDtype === 'bfloat16'
        ? 'bf16-exact-widen'
        : 'source-f32';
  return Object.freeze({
    tensor,
    dtype: 'float32',
    byteLength: tensor.elementCount * 4,
    materialization
  });
}

async function readPackedTensor(
  source: Qwen35WordAssetSource,
  assetId: Qwen35WordAssetId,
  tensor: Qwen35WordTensor
): Promise<Uint8Array> {
  const asset = source.manifest.assets[assetId];
  if (
    !asset ||
    source.manifest.storage.wordDtype !== 'uint32' ||
    source.manifest.storage.endianness !== 'little' ||
    tensor.storageByteOffset !== tensor.sourceDataByteOffset ||
    tensor.packedByteCount !== tensor.sourceByteCount + tensor.paddingByteCount ||
    tensor.storageByteOffset > asset.packed.bytes - tensor.packedByteCount
  )
    throw invalid('Invalid Qwen3.5 packed tensor layout for ' + tensor.name + '.');

  const start = tensor.storageByteOffset;
  const end = start + tensor.packedByteCount;
  const packed = new Uint8Array(tensor.packedByteCount);
  let copied = 0;
  for (let index = 0; index < asset.packed.rangeChunks.length; index += 1) {
    const range = asset.packed.rangeChunks[index]!;
    const overlapStart = Math.max(start, range.offset);
    const overlapEnd = Math.min(end, range.offset + range.bytes);
    if (overlapStart >= overlapEnd) continue;
    const buffer = await source.readRange(assetId, index);
    const chunk = new Uint8Array(buffer);
    if (chunk.byteLength !== range.bytes || (await sha256(chunk)) !== range.sha256)
      throw invalid('Qwen3.5 word range hash mismatch for ' + assetId + '.');
    const sourceOffset = overlapStart - range.offset;
    const targetOffset = overlapStart - start;
    const length = overlapEnd - overlapStart;
    packed.set(chunk.subarray(sourceOffset, sourceOffset + length), targetOffset);
    copied += length;
  }
  if (copied !== tensor.packedByteCount) throw invalid('Qwen3.5 word ranges do not cover ' + tensor.name + '.');
  if ((await sha256(packed)) !== tensor.packedTensorSha256)
    throw invalid('Qwen3.5 packed tensor hash mismatch: ' + tensor.name + '.');

  const payload = packed.subarray(0, tensor.sourceByteCount);
  for (const value of packed.subarray(tensor.sourceByteCount))
    if (value !== 0) throw invalid('Qwen3.5 packed tensor padding is not zero: ' + tensor.name + '.');
  if ((await sha256(payload)) !== tensor.sourcePayloadSha256)
    throw invalid('Qwen3.5 source tensor hash mismatch: ' + tensor.name + '.');
  return payload.slice();
}

/** Materialize only the dtype transitions in the validated pinned Qwen loader and Relax ABI. */
export async function materializeQwen35WordGraphInput(
  source: Qwen35WordAssetSource,
  binding: Qwen35WordGraphBinding
): Promise<Qwen35MaterializedWordGraphInput> {
  const plan = planQwen35WordGraphInput(source.manifest, binding);
  const tensor = plan.tensor;
  const sourceBytes = await readPackedTensor(source, binding.asset, tensor);
  let loadedBytes: Uint8Array;
  let materialization: Qwen35MaterializedWordGraphInput['materialization'];
  if (binding.sourceDtype === 'float32' && binding.loadedDtype === 'bfloat16') {
    loadedBytes = float32BytesToBf16Rne(sourceBytes, tensor.elementCount);
    if ((await sha256(loadedBytes)) !== binding.loadedPayloadSha256)
      throw invalid('Qwen3.5 base F32-to-BF16 values differ from the pinned source-cast proof: ' + tensor.name + '.');
    materialization = 'base-f32-rne-bf16-exact-widen';
  } else {
    loadedBytes = sourceBytes;
    if ((await sha256(loadedBytes)) !== tensor.sourcePayloadSha256)
      throw invalid('Qwen3.5 loaded tensor hash mismatch: ' + tensor.name + '.');
    materialization = binding.loadedDtype === 'bfloat16' ? 'bf16-exact-widen' : 'source-f32';
  }
  const bytes = binding.loadedDtype === 'bfloat16' ? widenBf16ToFloat32(loadedBytes, tensor.elementCount) : loadedBytes;
  if (bytes.byteLength !== plan.byteLength)
    throw invalid('Qwen3.5 graph input byte count mismatch: ' + tensor.name + '.');
  return Object.freeze({
    asset: binding.asset,
    sourceName: binding.sourceName,
    targetName: binding.targetName,
    argumentName: binding.relaxArgumentName,
    shape: tensor.shape,
    sourceDtype: binding.sourceDtype,
    loadedDtype: binding.loadedDtype,
    dtype: 'float32',
    sourceDataByteOffset: tensor.sourceDataByteOffset,
    storageByteOffset: tensor.storageByteOffset,
    sourcePayloadSha256: tensor.sourcePayloadSha256,
    loadedPayloadSha256: await sha256(loadedBytes),
    materialization,
    bytes,
    sha256: await sha256(bytes)
  });
}
