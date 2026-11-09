import pinned from './kev-reflexes-contract.json' with {type: 'json'};
import {WebDecisionError} from './kev.js';
import {parseJson} from './assets.js';
import type {KevAssetManifest} from './assets.js';
import type {DLDevice, Instance, Tensor} from './runtime/tvm.js';
import {MAX_MANIFEST_BYTES, readBoundedResponse} from './bounded-response.js';

export const REFLEXES_ID = 'kev-0.6b-reflexes-q4f16';
export const REFLEXES_HEADS = Object.freeze([
  Object.freeze({index: 394, path: 'heads/param_394.bin', shape: Object.freeze([1024, 256]), bytes: 1048576}),
  Object.freeze({index: 395, path: 'heads/param_395.bin', shape: Object.freeze([1024, 256]), bytes: 1048576}),
  Object.freeze({index: 396, path: 'heads/param_396.bin', shape: Object.freeze([256]), bytes: 1024}),
  Object.freeze({index: 397, path: 'heads/param_397.bin', shape: Object.freeze([256]), bytes: 1024})
]);
const invalid = (message: string): never => {
  throw new WebDecisionError('INVALID_ASSET', message);
};
const object = (value: unknown): value is Record<string, unknown> =>
  (value && typeof value === 'object' && !Array.isArray(value)) as boolean;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Accept only the separately verified, pinned experimental checkpoint assets. */
export function validateReflexesManifest(value: unknown) {
  if (!object(value) || !object(value.files) || !object(value.checkpoint))
    invalid('Invalid experimental Kev reflexes manifest.');
  const record = value as {files: Record<string, unknown>; checkpoint: Record<string, unknown>} & Record<
    string,
    unknown
  >;
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
    if (record[key] !== pinned[key as keyof typeof pinned]) invalid(`Unexpected Kev reflexes manifest field: ${key}.`);
  }
  if (!same(Object.keys(record.checkpoint).sort(), Object.keys(pinned.checkpoint).sort()))
    invalid('Unexpected Kev reflexes checkpoint fields.');
  for (const [key, expected] of Object.entries(pinned.checkpoint))
    if (record.checkpoint[key] !== expected) invalid(`Unexpected Kev reflexes checkpoint field: ${key}.`);
  if (!same(Object.keys(record.files).sort(), Object.keys(pinned.files).sort()))
    invalid('Unexpected Kev reflexes asset list.');
  for (const [name, expected] of Object.entries(pinned.files)) {
    const entry = record.files[name];
    if (
      !object(entry) ||
      !same(Object.keys(entry).sort(), ['bytes', 'sha256']) ||
      entry.bytes !== expected.bytes ||
      entry.sha256 !== expected.sha256
    )
      invalid(`Unexpected Kev reflexes asset metadata: ${name}.`);
  }
  return value as KevAssetManifest;
}

export async function readReflexesManifestResponse(response: Response, signal?: AbortSignal) {
  try {
    const value = parseJson(
      await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal} as {signal: AbortSignal}),
      'manifest.json'
    );
    signal?.throwIfAborted();
    return validateReflexesManifest(value);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Kev reflexes manifest JSON.', {cause});
  }
}

/** Upload verified head bytes after the common Q4 shard sequence. */
export async function uploadReflexesHeads({
  read,
  runtime,
  device,
  weights,
  signal,
  uncaptured
}: {
  read: (path: string, signal?: AbortSignal) => Promise<ArrayBuffer>;
  runtime: Instance;
  device: DLDevice;
  weights: Tensor[];
  signal?: AbortSignal | undefined;
  uncaptured: readonly string[];
}) {
  for (const head of REFLEXES_HEADS) {
    signal?.throwIfAborted();
    const data = await read(head.path, signal);
    signal?.throwIfAborted();
    if (data.byteLength !== head.bytes) invalid(`Invalid Kev reflexes head length: ${head.index}.`);
    const tensor = runtime.empty(head.shape as unknown as number[], 'float32', device);
    tensor.copyFromRawBytes(new Uint8Array(data));
    weights[head.index] = tensor;
    await device.sync();
    signal?.throwIfAborted();
    if (uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', uncaptured[0]!);
  }
}
