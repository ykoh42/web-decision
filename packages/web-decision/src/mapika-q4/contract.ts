import {WebDecisionError} from '../kev.js';
import {readPinnedManifest} from '../pinned-manifest.js';
import {validateTensorIndex} from '../tensor-index.js';
import type {MapikaQ4Manifest, MapikaQ4ManifestCandidate, MapikaQ4WeightIndex} from './types.js';

export const MAPIKA_Q4_MODEL = 'mapika-0.8b-q4f16';
export const MAPIKA_Q4_MANIFEST_SHA256 = '11c031e276990ed8443098e558820688b80a600c489548a504ac0cf02f320d72';
export const MAPIKA_Q4_PARAMETER_BYTES = 423938940;
export const MAPIKA_Q4_MAX_SHARD_BYTES = 8 * 1024 * 1024;
export const MAPIKA_Q4_LARGEST_BUFFER = 127139840;
export const MAPIKA_Q4_LABEL_INDEX = 133;

function invalid(message: string): never {
  throw new WebDecisionError('INVALID_ASSET', message);
}

/** Accept only the exact experimental Q4 graph, tokenizer and weights checked together. */
export async function readMapikaQ4ManifestResponse(
  response: Response,
  signal?: AbortSignal
): Promise<MapikaQ4Manifest> {
  const manifest = (await readPinnedManifest(
    response,
    signal,
    MAPIKA_Q4_MANIFEST_SHA256,
    'Mapika Q4'
  )) as MapikaQ4ManifestCandidate;
  if (
    manifest.model !== MAPIKA_Q4_MODEL ||
    manifest.backend !== 'tvm-webgpu' ||
    manifest.formatVersion !== 1 ||
    manifest.experimental !== true ||
    manifest.maxInputTokens !== 128 ||
    manifest.maxStateTokens !== 128 ||
    manifest.parameterCount !== 510 ||
    manifest.vocabSize !== 248320 ||
    manifest.temperature !== 1.03 ||
    manifest.quantization?.scheme !== 'q4-offset8-group32-f16' ||
    manifest.quantization?.scaleMethod !== 'maxabs'
  )
    invalid('Unexpected Mapika Q4 model metadata.');
  return manifest as MapikaQ4Manifest;
}

/** Validate ordered tensor/shard spans and the tied-embedding/label parameter layout. */
export function validateMapikaQ4WeightIndex(manifest: unknown, index: unknown): MapikaQ4WeightIndex {
  validateTensorIndex(manifest, index, {
    parameterCount: 510,
    totalBytes: MAPIKA_Q4_PARAMETER_BYTES,
    shardCount: 51,
    maxShardBytes: MAPIKA_Q4_MAX_SHARD_BYTES,
    allowedDtypes: ['float16', 'float32', 'int32', 'uint32']
  });
  const checkedIndex = index as MapikaQ4WeightIndex;
  const labels = checkedIndex.records[MAPIKA_Q4_LABEL_INDEX]!;
  const packed = checkedIndex.records[506]!;
  const scales = checkedIndex.records[507]!;
  if (checkedIndex.records.some((record) => record.nbytes > MAPIKA_Q4_LARGEST_BUFFER))
    invalid('Mapika Q4 tensor exceeds the supported buffer limit.');
  if (labels.dtype !== 'int32' || labels.shape.length !== 1 || labels.shape[0] !== 255)
    invalid('Invalid Mapika Q4 label parameter.');
  if (packed.dtype !== 'uint32' || packed.shape.length !== 2 || packed.shape[0] !== 248320 || packed.shape[1] !== 128)
    invalid('Invalid Mapika Q4 packed embedding.');
  if (scales.dtype !== 'float16' || scales.shape.length !== 2 || scales.shape[0] !== 248320 || scales.shape[1] !== 32)
    invalid('Invalid Mapika Q4 embedding scales.');
  return checkedIndex;
}
