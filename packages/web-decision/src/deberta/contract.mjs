import pinned from './contract.json' with {type: 'json'};
import {WebDecisionError} from '../kev.js';
import {parseJson} from '../assets.js';
import {MAX_MANIFEST_BYTES, readBoundedResponse} from '../bounded-response.js';

const invalid = (message) => {
  throw new WebDecisionError('INVALID_ASSET', message);
};
const object = (value) => value && typeof value === 'object' && !Array.isArray(value);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Accept only the local model build and exact files that were validated. */
export function validateDebertaManifest(value) {
  if (!object(value) || !object(value.files) || !object(value.checkpoint))
    invalid('Invalid experimental DeBERTa manifest.');
  for (const key of [
    'formatVersion',
    'model',
    'backend',
    'experimental',
    'maxInputTokens',
    'maxStateTokens',
    'parameterCount',
    'vocabSize'
  ]) {
    if (value[key] !== pinned[key]) invalid(`Unexpected DeBERTa manifest field: ${key}.`);
  }
  if (!same(Object.keys(value.checkpoint).sort(), Object.keys(pinned.checkpoint).sort()))
    invalid('Unexpected checkpoint fields.');
  for (const [key, expected] of Object.entries(pinned.checkpoint)) {
    if (value.checkpoint[key] !== expected) invalid(`Unexpected checkpoint field: ${key}.`);
  }
  if (!same(Object.keys(value.files).sort(), Object.keys(pinned.files).sort()))
    invalid('Unexpected DeBERTa asset list.');
  for (const [name, expected] of Object.entries(pinned.files)) {
    const entry = value.files[name];
    if (
      !object(entry) ||
      !same(Object.keys(entry).sort(), ['bytes', 'sha256']) ||
      entry.bytes !== expected.bytes ||
      entry.sha256 !== expected.sha256
    ) {
      invalid(`Unexpected DeBERTa asset metadata: ${name}.`);
    }
  }
  return value;
}

export async function readDebertaManifestResponse(response, signal) {
  try {
    const value = parseJson(await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal}), 'manifest.json');
    signal?.throwIfAborted();
    return validateDebertaManifest(value);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Invalid DeBERTa manifest JSON.', {cause});
  }
}

/** Check every tensor range before copying untrusted metadata to GPU buffers. */
export function validateDebertaTensorCache(manifest, cache) {
  if (
    !object(cache) ||
    cache.metadata?.numParams !== 343 ||
    !Array.isArray(cache.records) ||
    cache.records.length !== 10
  ) {
    invalid('Invalid DeBERTa tensor-cache metadata.');
  }
  const indices = new Set(),
    shards = new Set();
  for (const shard of cache.records) {
    if (
      !object(shard) ||
      typeof shard.dataPath !== 'string' ||
      !/^params_shard_(?:[0-9])\.bin$/.test(shard.dataPath) ||
      shards.has(shard.dataPath) ||
      shard.format !== 'raw-shard' ||
      !Array.isArray(shard.records) ||
      shard.nbytes !== manifest.files[`weights/${shard.dataPath}`]?.bytes
    )
      invalid('Invalid DeBERTa tensor shard.');
    shards.add(shard.dataPath);
    const spans = [];
    for (const record of shard.records) {
      const match = typeof record?.name === 'string' && /^param_(0|[1-9][0-9]*)$/.exec(record.name);
      const index = match ? Number(match[1]) : -1,
        width = record?.dtype === 'uint32' ? 4 : record?.dtype === 'float16' ? 2 : 0;
      if (
        !Number.isSafeInteger(index) ||
        index < 0 ||
        index >= 343 ||
        indices.has(index) ||
        !width ||
        record.format !== 'raw' ||
        !Array.isArray(record.shape) ||
        !record.shape.length ||
        record.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
        !Number.isSafeInteger(record.byteOffset) ||
        record.byteOffset < 0 ||
        record.byteOffset % 4 ||
        !Number.isSafeInteger(record.nbytes) ||
        record.nbytes <= 0 ||
        record.nbytes % 4
      )
        invalid('Invalid DeBERTa tensor metadata.');
      const size = record.shape.reduce((a, n) => a * n, width),
        end = record.byteOffset + record.nbytes;
      if (!Number.isSafeInteger(size) || size !== record.nbytes || !Number.isSafeInteger(end) || end > shard.nbytes)
        invalid('DeBERTa tensor exceeds its shard.');
      indices.add(index);
      spans.push([record.byteOffset, end]);
    }
    spans.sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < spans.length; i++) if (spans[i][0] < spans[i - 1][1]) invalid('Overlapping DeBERTa tensors.');
  }
  if (indices.size !== 343 || shards.size !== 10) invalid('Missing DeBERTa parameters or shards.');
}

/** Check the split executable's function signatures and parameter order. */
export function validateDebertaSplit(split, mapping) {
  if (
    !object(split) ||
    !object(mapping) ||
    split.encoderEntry !== 'encode' ||
    split.headEntry !== 'score_head' ||
    split.originalEntry !== 'main' ||
    !same(split.encoderInputs, ['input_ids', 'attention_mask']) ||
    !same(split.headInputs, ['hidden', 'seg', 'pair_q', 'pair_opt']) ||
    !same(split.hiddenShape, [1, 'sequence_length', 1024]) ||
    split.hiddenDtype !== 'float16' ||
    mapping.model !== 'deberta-dynamic-pairs3' ||
    mapping.sequenceLength !== 'dynamic' ||
    mapping.pairCount !== 3 ||
    mapping.cacheManifestSha256 !== pinned.files['weights/tensor-cache.json'].sha256 ||
    !Array.isArray(split.encoderWeightIndices) ||
    split.encoderWeightIndices.length !== 339 ||
    !Array.isArray(split.headWeightIndices) ||
    split.headWeightIndices.length !== 4 ||
    !Array.isArray(mapping.weightIndices)
  ) {
    invalid('Invalid DeBERTa split model metadata.');
  }
  const indices = [...split.encoderWeightIndices, ...split.headWeightIndices];
  if (
    indices.some((i) => !Number.isSafeInteger(i) || i < 0 || i >= 343) ||
    new Set(indices).size !== 343 ||
    !same(indices, mapping.weightIndices)
  ) {
    invalid('Invalid DeBERTa weight mapping.');
  }
  return {encoderIndices: [...split.encoderWeightIndices], headIndices: [...split.headWeightIndices]};
}
