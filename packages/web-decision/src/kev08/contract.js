import {WebDecisionError} from '../kev.js';
import {KEV08_MARKERS} from '../kev08.js';
import {readPinnedManifest} from '../pinned-manifest.js';
import {validateTensorIndex} from '../tensor-index.js';

export const KEV08_MODEL = 'kev-0.8b-f16';
export const KEV08_MANIFEST_SHA256 = '6e4a19e49a8d3fbee4d7effc16e2778b8b7272d79adb146a563c2369e85b503a';
export const KEV08_PARAMETER_BYTES = 1506885504;
export const KEV08_LARGEST_BUFFER = 508559360;

function invalid(message) {
  throw new WebDecisionError('INVALID_ASSET', message);
}

export function validateKev08TokenizerData(data) {
  const expected = {
    '<|fim_prefix|>': KEV08_MARKERS.state,
    '<|fim_middle|>': KEV08_MARKERS.question,
    '<|box_start|>': KEV08_MARKERS.open,
    '<|box_end|>': KEV08_MARKERS.close,
    '<|fim_suffix|>': KEV08_MARKERS.decide
  };
  if (!Array.isArray(data?.added_tokens)) invalid('Missing Kev 0.8B markers.');
  for (const [content, id] of Object.entries(expected)) {
    const matches = data.added_tokens.filter((entry) => entry?.content === content);
    if (matches.length !== 1 || matches[0].id !== id) invalid('Unexpected Kev 0.8B tokenizer markers.');
  }
}

export async function readKev08ManifestResponse(response, signal) {
  const manifest = await readPinnedManifest(response, signal, KEV08_MANIFEST_SHA256, 'Kev 0.8B');
  if (
    manifest.model !== KEV08_MODEL ||
    manifest.backend !== 'tvm-webgpu' ||
    manifest.formatVersion !== 1 ||
    manifest.experimental !== true ||
    manifest.maxInputTokens !== 128 ||
    manifest.maxStateTokens !== 128 ||
    manifest.parameterCount !== 326 ||
    manifest.vocabSize !== 248320 ||
    manifest.headTemperature !== 2.3510958125672174 ||
    manifest.headOutputCalibrated !== true
  )
    invalid('Unexpected Kev 0.8B model metadata.');
  return manifest;
}

export function validateKev08WeightIndex(manifest, index) {
  validateTensorIndex(manifest, index, {
    parameterCount: 326,
    totalBytes: KEV08_PARAMETER_BYTES,
    shardCount: 404,
    maxShardBytes: 8 * 1024 * 1024
  });
  for (const [i, shape] of [
    [320, [256, 1024]],
    [321, [256]],
    [322, [256, 1024]],
    [323, [256]]
  ]) {
    const record = index.records[i];
    if (record.dtype !== 'float32' || JSON.stringify(record.shape) !== JSON.stringify(shape))
      invalid('Invalid Kev 0.8B pointer head.');
  }
  return index;
}
