import {WebDecisionError} from '../kev.js';
import {readPinnedManifest} from '../pinned-manifest.js';
import {validateTensorIndex} from '../tensor-index.js';

export const MAPIKA_MODEL = 'mapika-0.8b-f16';
export const MAPIKA_MANIFEST_SHA256 = '57efcc1ca1376b8d9c501805e2a78356c7c05eb23ee18001095553fd14f7b088';
export const MAPIKA_PARAMETER_BYTES = 1504787324;
export const MAPIKA_MAX_SHARD_BYTES = 8 * 1024 * 1024;
export const MAPIKA_LARGEST_BUFFER = 508559360;

function invalid(message) {
  throw new WebDecisionError('INVALID_ASSET', message);
}

/** Only accept the exact experimental graph, tokenizer and weights checked together. */
export async function readMapikaManifestResponse(response, signal) {
  const manifest = await readPinnedManifest(response, signal, MAPIKA_MANIFEST_SHA256, 'Mapika');
  if (
    manifest.model !== MAPIKA_MODEL ||
    manifest.backend !== 'tvm-webgpu' ||
    manifest.formatVersion !== 1 ||
    manifest.experimental !== true ||
    manifest.maxInputTokens !== 512 ||
    manifest.maxStateTokens !== 512 ||
    manifest.parameterCount !== 323 ||
    manifest.vocabSize !== 248320 ||
    manifest.temperature !== 1.03
  )
    invalid('Unexpected Mapika model metadata.');
  return manifest;
}

/** Validate the ordered tensors and the tokenizer-label parameter. */
export function validateMapikaWeightIndex(manifest, index) {
  validateTensorIndex(manifest, index, {
    parameterCount: 323,
    totalBytes: MAPIKA_PARAMETER_BYTES,
    shardCount: 180,
    maxShardBytes: MAPIKA_MAX_SHARD_BYTES
  });
  const labels = index.records[320];
  if (labels.dtype !== 'int32' || labels.shape.length !== 1 || labels.shape[0] !== 255)
    invalid('Invalid Mapika label parameter.');
  return index;
}
