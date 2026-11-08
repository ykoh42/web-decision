import pinnedJson from './contract.json' with {type: 'json'};
import {WebDecisionError} from '../kev.js';
import {parseJson} from '../assets.js';
import {MAX_MANIFEST_BYTES, readBoundedResponse} from '../bounded-response.js';

type UnknownRecord = Record<string, unknown>;
type AssetMetadata = Readonly<{bytes: number; sha256: string}>;
type DebertaManifest = UnknownRecord &
  Readonly<{
    files: Readonly<Record<string, AssetMetadata>>;
    checkpoint: Readonly<Record<string, unknown>>;
  }>;
type DebertaManifestPin = Readonly<{
  readonly formatVersion: number;
  readonly model: string;
  readonly backend: string;
  readonly experimental: boolean;
  readonly maxInputTokens: number;
  readonly maxStateTokens: number;
  readonly parameterCount: number;
  readonly vocabSize: number;
  readonly checkpoint: Readonly<Record<string, unknown>>;
  readonly files: Readonly<Record<string, AssetMetadata>>;
}>;
type BoundedResponseReader = (
  response: Response,
  maxBytes: number,
  options: {readonly signal?: AbortSignal | undefined}
) => Promise<ArrayBuffer>;
type JsonParser = (bytes: ArrayBuffer, name: string) => unknown;
type TensorSpan = readonly [start: number, end: number];

const pinned = pinnedJson as DebertaManifestPin;
const readBoundedResponseForModel = readBoundedResponse as BoundedResponseReader;
const parseJsonForModel = parseJson as JsonParser;

const invalid = (message: string): never => {
  throw new WebDecisionError('INVALID_ASSET', message);
};

function object(value: unknown): value is UnknownRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function same(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

/** Accept only the local model build and exact files that were validated. */
export function validateDebertaManifest(value: unknown): DebertaManifest {
  if (!object(value) || !object(value.files) || !object(value.checkpoint))
    invalid('Invalid experimental DeBERTa manifest.');
  const manifest = value as DebertaManifest;
  for (const key of [
    'formatVersion',
    'model',
    'backend',
    'experimental',
    'maxInputTokens',
    'maxStateTokens',
    'parameterCount',
    'vocabSize'
  ] as const) {
    if (manifest[key] !== pinned[key]) invalid(`Unexpected DeBERTa manifest field: ${key}.`);
  }
  if (!same(Object.keys(manifest.checkpoint).sort(), Object.keys(pinned.checkpoint).sort()))
    invalid('Unexpected checkpoint fields.');
  for (const [key, expected] of Object.entries(pinned.checkpoint)) {
    if (manifest.checkpoint[key] !== expected) invalid(`Unexpected checkpoint field: ${key}.`);
  }
  if (!same(Object.keys(manifest.files).sort(), Object.keys(pinned.files).sort()))
    invalid('Unexpected DeBERTa asset list.');
  for (const [name, expected] of Object.entries(pinned.files)) {
    const entry = manifest.files[name];
    if (
      !object(entry) ||
      !same(Object.keys(entry).sort(), ['bytes', 'sha256']) ||
      entry.bytes !== expected.bytes ||
      entry.sha256 !== expected.sha256
    ) {
      invalid(`Unexpected DeBERTa asset metadata: ${name}.`);
    }
  }
  return manifest;
}

export async function readDebertaManifestResponse(response: Response, signal?: AbortSignal): Promise<DebertaManifest> {
  try {
    const bytes = await readBoundedResponseForModel(response, MAX_MANIFEST_BYTES, {signal});
    const value = parseJsonForModel(bytes, 'manifest.json');
    signal?.throwIfAborted();
    return validateDebertaManifest(value);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null)?.name === 'AbortError') throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Invalid DeBERTa manifest JSON.', {cause});
  }
}

function validateCacheHeader(cache: unknown): asserts cache is {metadata: {numParams: number}; records: unknown[]} {
  if (
    !object(cache) ||
    !object(cache.metadata) ||
    cache.metadata.numParams !== 343 ||
    !Array.isArray(cache.records) ||
    cache.records.length !== 10
  )
    invalid('Invalid DeBERTa tensor-cache metadata.');
}

function validateShardHeader(
  manifest: DebertaManifest,
  shardValue: unknown,
  shards: Set<string>
): {dataPath: string; nbytes: number; records: unknown[]} {
  const shard = shardValue as {
    readonly dataPath?: unknown;
    readonly format?: unknown;
    readonly nbytes?: unknown;
    readonly records?: unknown;
  } | null;
  if (
    !object(shardValue) ||
    typeof shard?.dataPath !== 'string' ||
    !/^params_shard_(?:[0-9])\.bin$/.test(shard.dataPath) ||
    shards.has(shard.dataPath) ||
    shard.format !== 'raw-shard' ||
    !Array.isArray(shard.records) ||
    shard.nbytes !== manifest.files[`weights/${shard.dataPath}`]?.bytes
  ) {
    invalid('Invalid DeBERTa tensor shard.');
  }
  const dataPath = shard!.dataPath as string;
  shards.add(dataPath);
  return {dataPath, nbytes: shard!.nbytes as number, records: shard!.records as unknown[]};
}

function validateTensorRecord(recordValue: unknown, shardBytes: number, indices: Set<number>): TensorSpan {
  const record = recordValue as {
    readonly name?: unknown;
    readonly dtype?: unknown;
    readonly format?: unknown;
    readonly shape?: unknown;
    readonly byteOffset?: unknown;
    readonly nbytes?: unknown;
  } | null;
  const match = typeof record?.name === 'string' && /^param_(0|[1-9][0-9]*)$/.exec(record.name);
  const index = match ? Number(match[1]) : -1;
  const width = record?.dtype === 'uint32' ? 4 : record?.dtype === 'float16' ? 2 : 0;
  const shape = record?.shape;
  const byteOffset = record?.byteOffset as number;
  const nbytes = record?.nbytes as number;
  if (
    !Number.isSafeInteger(index) ||
    index < 0 ||
    index >= 343 ||
    indices.has(index) ||
    !width ||
    record?.format !== 'raw' ||
    !Array.isArray(shape) ||
    !shape.length ||
    shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1) ||
    !Number.isSafeInteger(byteOffset) ||
    byteOffset < 0 ||
    byteOffset % 4 ||
    !Number.isSafeInteger(nbytes) ||
    nbytes <= 0 ||
    nbytes % 4
  ) {
    invalid('Invalid DeBERTa tensor metadata.');
  }
  const size = (shape as number[]).reduce((product, dimension) => product * dimension, width);
  const end = byteOffset + nbytes;
  if (!Number.isSafeInteger(size) || size !== nbytes || !Number.isSafeInteger(end) || end > shardBytes)
    invalid('DeBERTa tensor exceeds its shard.');
  indices.add(index);
  return [byteOffset, end];
}

function validateNonOverlappingSpans(spans: TensorSpan[]): void {
  spans.sort((left, right) => left[0] - right[0]);
  for (let index = 1; index < spans.length; index++)
    if (spans[index]![0] < spans[index - 1]![1]) invalid('Overlapping DeBERTa tensors.');
}

function validateTensorShard(
  manifest: DebertaManifest,
  shardValue: unknown,
  indices: Set<number>,
  shards: Set<string>
): void {
  const shard = validateShardHeader(manifest, shardValue, shards);
  const spans = shard.records.map((record) => validateTensorRecord(record, shard.nbytes, indices));
  validateNonOverlappingSpans(spans);
}

/** Check every tensor range before copying untrusted metadata to GPU buffers. */
export function validateDebertaTensorCache(manifest: DebertaManifest, cache: unknown): void {
  validateCacheHeader(cache);
  const indices = new Set<number>();
  const shards = new Set<string>();
  for (const shard of cache.records) validateTensorShard(manifest, shard, indices, shards);
  if (indices.size !== 343 || shards.size !== 10) invalid('Missing DeBERTa parameters or shards.');
}

/** Check the split executable's function signatures and parameter order. */
export function validateDebertaSplit(
  splitValue: unknown,
  mappingValue: unknown
): {encoderIndices: number[]; headIndices: number[]} {
  if (!object(splitValue) || !object(mappingValue)) invalid('Invalid DeBERTa split model metadata.');
  const split = splitValue as UnknownRecord;
  const mapping = mappingValue as UnknownRecord;
  if (
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
    mapping.cacheManifestSha256 !== pinned.files['weights/tensor-cache.json']!.sha256 ||
    !Array.isArray(split.encoderWeightIndices) ||
    split.encoderWeightIndices.length !== 339 ||
    !Array.isArray(split.headWeightIndices) ||
    split.headWeightIndices.length !== 4 ||
    !Array.isArray(mapping.weightIndices)
  ) {
    invalid('Invalid DeBERTa split model metadata.');
  }
  const encoderIndices = split.encoderWeightIndices as number[];
  const headIndices = split.headWeightIndices as number[];
  const weightIndices = mapping.weightIndices as number[];
  const indices = [...encoderIndices, ...headIndices];
  if (
    indices.some((index) => !Number.isSafeInteger(index) || index < 0 || index >= 343) ||
    new Set(indices).size !== 343 ||
    !same(indices, weightIndices)
  ) {
    invalid('Invalid DeBERTa weight mapping.');
  }
  return {encoderIndices: [...encoderIndices], headIndices: [...headIndices]};
}
