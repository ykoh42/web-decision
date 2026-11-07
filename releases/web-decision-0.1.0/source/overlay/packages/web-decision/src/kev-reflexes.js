import pinned from './kev-reflexes-contract.json' with {type: 'json'};
import {WebDecisionError} from './kev.js';
import {parseJson} from './assets.js';
import {MAX_MANIFEST_BYTES, readBoundedResponse} from './bounded-response.js';

export const REFLEXES_ID = 'kev-0.6b-reflexes-q4f16';
export const REFLEXES_HEADS = Object.freeze([
  Object.freeze({index: 394, path: 'heads/param_394.bin', shape: Object.freeze([1024, 256]), bytes: 1048576}),
  Object.freeze({index: 395, path: 'heads/param_395.bin', shape: Object.freeze([1024, 256]), bytes: 1048576}),
  Object.freeze({index: 396, path: 'heads/param_396.bin', shape: Object.freeze([256]), bytes: 1024}),
  Object.freeze({index: 397, path: 'heads/param_397.bin', shape: Object.freeze([256]), bytes: 1024})
]);
const invalid = (message) => {
  throw new WebDecisionError('INVALID_ASSET', message);
};
const object = (value) => value && typeof value === 'object' && !Array.isArray(value);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Accept only the separately verified, pinned experimental checkpoint assets. */
export function validateReflexesManifest(value) {
  if (!object(value) || !object(value.files) || !object(value.checkpoint))
    invalid('Invalid experimental Kev reflexes manifest.');
  for (const key of [
    'formatVersion',
    'model',
    'backend',
    'experimental',
    'maxStateTokens',
    'maxBranchTokens',
    'parameterCount',
    'tvmVersion'
  ]) {
    if (value[key] !== pinned[key]) invalid(`Unexpected Kev reflexes manifest field: ${key}.`);
  }
  if (!same(Object.keys(value.checkpoint).sort(), Object.keys(pinned.checkpoint).sort()))
    invalid('Unexpected Kev reflexes checkpoint fields.');
  for (const [key, expected] of Object.entries(pinned.checkpoint))
    if (value.checkpoint[key] !== expected) invalid(`Unexpected Kev reflexes checkpoint field: ${key}.`);
  if (!same(Object.keys(value.files).sort(), Object.keys(pinned.files).sort()))
    invalid('Unexpected Kev reflexes asset list.');
  for (const [name, expected] of Object.entries(pinned.files)) {
    const entry = value.files[name];
    if (
      !object(entry) ||
      !same(Object.keys(entry).sort(), ['bytes', 'sha256']) ||
      entry.bytes !== expected.bytes ||
      entry.sha256 !== expected.sha256
    )
      invalid(`Unexpected Kev reflexes asset metadata: ${name}.`);
  }
  return value;
}

export async function readReflexesManifestResponse(response, signal) {
  try {
    const value = parseJson(await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal}), 'manifest.json');
    signal?.throwIfAborted();
    return validateReflexesManifest(value);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Kev reflexes manifest JSON.', {cause});
  }
}

/** Upload verified head bytes after the common Q4 shard sequence. */
export async function uploadReflexesHeads({read, runtime, device, weights, signal, uncaptured}) {
  for (const head of REFLEXES_HEADS) {
    signal?.throwIfAborted();
    const data = await read(head.path, signal);
    signal?.throwIfAborted();
    if (data.byteLength !== head.bytes) invalid(`Invalid Kev reflexes head length: ${head.index}.`);
    const tensor = runtime.empty(head.shape, 'float32', device);
    tensor.copyFromRawBytes(new Uint8Array(data));
    weights[head.index] = tensor;
    await device.sync();
    signal?.throwIfAborted();
    if (uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', uncaptured[0]);
  }
}
