import {WebDecisionError} from '../kev.js';
import {sha256} from '../assets.js';
import type {AssetCache} from '../assets.js';
import {RangeCacheTransaction} from '../strands/range-cache.mjs';
import {KANANA_ASSETS_URL} from './contract.mjs';
import {KANANA_SAFETENSORS_SPEC} from './safetensors-spec.mjs';
import {SAFETENSORS_MAX_HEADER_BYTES, convertBf16ToFp16, parseSafetensorsHeader} from './safetensors.mjs';

export {KANANA_SAFETENSORS_SPEC};

const RANGE_MAX = 8 * 1024 * 1024;
const HEX = /^[0-9a-f]{64}$/;
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);
const TVM_DTYPE = Object.freeze({
  BOOL: 'bool',
  U8: 'uint8',
  I8: 'int8',
  F16: 'float16',
  BF16: 'bfloat16',
  I16: 'int16',
  U16: 'uint16',
  F32: 'float32',
  I32: 'int32',
  U32: 'uint32',
  F64: 'float64',
  I64: 'int64',
  U64: 'uint64'
});
const DTYPE_BYTES = Object.freeze({
  BOOL: 1,
  U8: 1,
  I8: 1,
  F16: 2,
  BF16: 2,
  I16: 2,
  U16: 2,
  F32: 4,
  I32: 4,
  U32: 4,
  F64: 8,
  I64: 8,
  U64: 8
});
const KANANA_TEXT_GRAPH_WEIGHT_COUNT = 354;
const KANANA_VISION_GRAPH_WEIGHT_COUNT = 333;

type SafetensorsDtype = keyof typeof DTYPE_BYTES;
type TensorChunk = Readonly<{offset: number; length: number; sha256: string}>;
type RangeTensor = Readonly<{
  name: string;
  dtype: SafetensorsDtype;
  shape: readonly number[];
  offset: number;
  length: number;
  sha256: string;
  chunks: readonly TensorChunk[];
}>;
type ConversionCheck = Readonly<{
  sourceDtype: string;
  targetDtype: string;
  verified: boolean;
  rounding: string;
  valid: boolean;
  invalidValueCount: number;
  overflowValueCount: number;
  nonFiniteValueCount: number;
}>;
type RangeManifest = Readonly<{
  format: string;
  model?: unknown;
  repository?: unknown;
  revision?: unknown;
  path: string;
  totalBytes: number;
  headerLength: number;
  headerSha256: string;
  tensorCount: number;
  tensors: readonly (RangeTensor & {conversion?: ConversionCheck})[];
  conversionVerification?: Readonly<{
    enabled: boolean;
    sourceDtype: string;
    targetDtype: string;
    rounding: string;
    valid: boolean;
    invalidValueCount: number;
    overflowValueCount: number;
    nonFiniteValueCount: number;
    failures: readonly unknown[];
  }>;
}>;
type RequiredWeight = Readonly<{
  name: string;
  shape: readonly number[];
  dtype: string;
  length: number;
  parameterName?: string;
}>;
type OutputRecord = Readonly<{
  name: string;
  shape: readonly number[];
  dtype: string;
  length: number;
  sourceDtype: SafetensorsDtype;
  elementCount?: number;
  parameterName?: string;
}>;
type OpenSafetensorsWeightSourceOptions = Readonly<{
  base: string | URL;
  manifest: unknown;
  requiredWeights: readonly RequiredWeight[];
  cache?: AssetCache | null | undefined;
  offline?: boolean | undefined;
  signal?: AbortSignal | null | undefined;
  onBytes?: ((count: number) => void) | undefined;
}>;
type ReadTensorChunksOptions = Readonly<{
  dtype?: string | undefined;
  signal?: AbortSignal | null | undefined;
  onChunk: (bytes: Uint8Array, offset: number) => void | Promise<void>;
}>;
type ReadTensorOptions = Readonly<{dtype?: string; signal?: AbortSignal | null}>;

function checkedBase(value: string | URL): URL {
  let base;
  try {
    base = value instanceof URL ? new URL(value.href) : new URL(value);
  } catch (cause) {
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Safetensors source URL.', {cause});
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw new WebDecisionError('INVALID_INPUT', 'Safetensors source must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return base;
}

function checkedRecord(
  input: unknown,
  index: number,
  totalBytes: number,
  headerRangeBytes: number,
  names: Set<string>
): RangeTensor {
  const record = input as RangeTensor;
  if (
    !record ||
    typeof record.name !== 'string' ||
    !record.name ||
    names.has(record.name) ||
    typeof record.dtype !== 'string' ||
    !Array.isArray(record.shape) ||
    record.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0) ||
    !Number.isSafeInteger(record.offset) ||
    record.offset < headerRangeBytes ||
    !Number.isSafeInteger(record.length) ||
    record.length < 0 ||
    record.offset > totalBytes - record.length ||
    !HEX.test(record.sha256) ||
    !Array.isArray(record.chunks)
  )
    throw invalid(`Invalid Safetensors range record ${index}.`);
  names.add(record.name);

  let next = record.offset;
  const chunks = record.chunks.map((chunk, chunkIndex) => {
    if (
      !Number.isSafeInteger(chunk.offset) ||
      chunk.offset !== next ||
      !Number.isSafeInteger(chunk.length) ||
      chunk.length < 1 ||
      chunk.length > RANGE_MAX ||
      chunk.offset > totalBytes - chunk.length ||
      !HEX.test(chunk.sha256) ||
      (record.dtype === 'BF16' && (chunk.offset % 2 !== 0 || chunk.length % 2 !== 0))
    )
      throw invalid(`Invalid Safetensors chunk ${chunkIndex} for ${record.name}.`);
    next += chunk.length;
    return Object.freeze({offset: chunk.offset, length: chunk.length, sha256: chunk.sha256});
  });
  if (next !== record.offset + record.length || (record.length === 0 && chunks.length !== 0))
    throw invalid(`Safetensors chunks do not cover ${record.name}.`);

  return Object.freeze({
    name: record.name,
    dtype: record.dtype,
    shape: Object.freeze([...record.shape]),
    offset: record.offset,
    length: record.length,
    sha256: record.sha256,
    chunks: Object.freeze(chunks)
  });
}

function checkedManifest(input: unknown): RangeManifest {
  const value = input as RangeManifest;
  if (
    !value ||
    typeof value !== 'object' ||
    value.format !== 'kanana-safetensors-range-manifest-v1' ||
    typeof value.path !== 'string' ||
    !value.path ||
    !Number.isSafeInteger(value.totalBytes) ||
    value.totalBytes < 8 ||
    !Number.isSafeInteger(value.headerLength) ||
    value.headerLength < 2 ||
    value.headerLength > SAFETENSORS_MAX_HEADER_BYTES ||
    8 + value.headerLength > value.totalBytes ||
    !HEX.test(value.headerSha256) ||
    !Number.isSafeInteger(value.tensorCount) ||
    value.tensorCount < 1 ||
    !Array.isArray(value.tensors) ||
    value.tensors.length !== value.tensorCount
  )
    throw invalid('Invalid Safetensors range manifest.');

  const headerRangeBytes = 8 + value.headerLength;
  const names = new Set<string>();
  const tensors = value.tensors.map((record, index) =>
    checkedRecord(record, index, value.totalBytes, headerRangeBytes, names)
  );
  const ordered = [...tensors].sort((left, right) => left.offset - right.offset || left.length - right.length);
  let nextOffset = headerRangeBytes;
  for (const tensor of ordered) {
    if (tensor.offset !== nextOffset) throw invalid('Safetensors tensor ranges overlap or leave a gap.');
    nextOffset += tensor.length;
  }
  if (nextOffset !== value.totalBytes) throw invalid('Safetensors tensor ranges do not cover the source file.');

  return Object.freeze({
    format: value.format,
    model: value.model,
    repository: value.repository,
    revision: value.revision,
    path: value.path,
    totalBytes: value.totalBytes,
    headerLength: value.headerLength,
    headerSha256: value.headerSha256,
    tensorCount: value.tensorCount,
    tensors: Object.freeze(tensors)
  });
}

function compareHeader(manifest: RangeManifest, parsed: ReturnType<typeof parseSafetensorsHeader>): void {
  if (parsed.headerLength !== manifest.headerLength || parsed.tensorCount !== manifest.tensorCount)
    throw invalid('Safetensors header does not match its range manifest.');
  const expected = new Map(manifest.tensors.map((tensor) => [tensor.name, tensor]));
  for (const tensor of parsed.tensors) {
    const record = expected.get(tensor.name);
    if (
      !record ||
      record.dtype !== tensor.dtype ||
      record.offset !== tensor.offset ||
      record.length !== tensor.length ||
      record.shape.length !== tensor.shape.length ||
      record.shape.some((dimension, index) => dimension !== tensor.shape[index])
    )
      throw invalid(`Safetensors header metadata differs for ${tensor.name}.`);
  }
}

function selectRequiredWeights(
  requiredWeights: readonly RequiredWeight[],
  sourceRecords: readonly RangeTensor[]
): readonly Readonly<{
  name: string;
  shape: readonly number[];
  dtype: string;
  length: number;
  sourceDtype: SafetensorsDtype;
  parameterName?: string;
}>[] {
  if (!Array.isArray(requiredWeights) || !requiredWeights.length)
    throw invalid('An explicit graph-required weight list is required.');
  const sourceByName = new Map(sourceRecords.map((record) => [record.name, record]));
  const seen = new Set();
  const parameterNames = new Set();
  return requiredWeights.map((required, index) => {
    if (
      !required ||
      typeof required.name !== 'string' ||
      !required.name ||
      seen.has(required.name) ||
      !Array.isArray(required.shape) ||
      !Number.isSafeInteger(required.length) ||
      required.length < 0 ||
      typeof required.dtype !== 'string' ||
      (required.parameterName !== undefined &&
        (typeof required.parameterName !== 'string' ||
          !required.parameterName ||
          parameterNames.has(required.parameterName)))
    )
      throw invalid(`Invalid graph-required Safetensors weight ${index}.`);
    seen.add(required.name);
    if (required.parameterName !== undefined) parameterNames.add(required.parameterName);

    const sourceRecord = sourceByName.get(required.name);
    if (!sourceRecord) throw invalid(`Graph-required weight is missing from Safetensors: ${required.name}.`);
    const expectedDtype = sourceRecord.dtype === 'BF16' ? 'float16' : TVM_DTYPE[sourceRecord.dtype];
    if (
      required.dtype !== expectedDtype ||
      required.length !== sourceRecord.length ||
      required.shape.length !== sourceRecord.shape.length ||
      required.shape.some(
        (dimension: number, dimensionIndex: number) => dimension !== sourceRecord.shape[dimensionIndex]
      )
    )
      throw invalid(`Graph-required weight metadata differs from Safetensors: ${required.name}.`);
    return Object.freeze({
      name: sourceRecord.name,
      shape: sourceRecord.shape,
      dtype: required.dtype,
      length: sourceRecord.length,
      sourceDtype: sourceRecord.dtype,
      ...(required.parameterName === undefined ? {} : {parameterName: required.parameterName})
    });
  });
}

function outputRecord(record: RangeTensor, dtype: string, parameterName: string | undefined): OutputRecord {
  const sourceBytesPerElement = DTYPE_BYTES[record.dtype];
  const targetBytesPerElement = Object.entries(TVM_DTYPE).find(([, tvmDtype]) => tvmDtype === dtype)?.[0];
  const targetByteWidth = targetBytesPerElement ? DTYPE_BYTES[targetBytesPerElement as SafetensorsDtype] : undefined;
  const elementCount = record.shape.reduce((count, dimension) => count * dimension, 1);
  if (!sourceBytesPerElement || sourceBytesPerElement !== targetByteWidth)
    throw invalid(`Safetensors dtype conversion changes the byte width for ${record.name}.`);
  return Object.freeze({
    name: record.name,
    shape: record.shape,
    dtype,
    length: record.length,
    sourceDtype: record.dtype,
    elementCount,
    ...(parameterName === undefined ? {} : {parameterName})
  });
}

/** Open a hash-indexed Safetensors file and expose only the explicitly selected graph weights. */
export async function openSafetensorsWeightSource({
  base,
  manifest: value,
  requiredWeights,
  cache,
  offline = false,
  signal,
  onBytes
}: OpenSafetensorsWeightSourceOptions) {
  const resolvedBase = checkedBase(base);
  const manifest = checkedManifest(value);
  const sourceByName = new Map(manifest.tensors.map((record) => [record.name, record]));
  const records = selectRequiredWeights(requiredWeights, manifest.tensors);
  const requiredByName = new Map(records.map((record) => [record.name, record]));
  const headerRange = Object.freeze({
    path: manifest.path,
    totalBytes: manifest.totalBytes,
    offset: 0,
    length: 8 + manifest.headerLength,
    sha256: manifest.headerSha256
  });
  const rangeSpecs = [
    headerRange,
    ...records.flatMap((required) =>
      sourceByName.get(required.name)!.chunks.map(({offset, length, sha256}) => ({
        path: manifest.path,
        totalBytes: manifest.totalBytes,
        offset,
        length,
        sha256
      }))
    )
  ];
  const requiredIdentity = await sha256(
    new TextEncoder().encode(
      JSON.stringify(
        records.map(({name, shape, dtype, length, parameterName}) => [name, shape, dtype, length, parameterName])
      )
    )
  );
  const transaction = new RangeCacheTransaction({
    base: resolvedBase,
    cache,
    offline,
    signal,
    markerId: `kanana-safetensors:${manifest.headerSha256}:${requiredIdentity}`,
    markerPath: 'kanana-offline-complete-v1.json'
  });
  await transaction.open();
  transaction.require(rangeSpecs);
  transaction.seal();

  const headerResult = await transaction.read(resolvedBase, headerRange);
  onBytes?.(headerResult.bytes.byteLength);
  const parsed = parseSafetensorsHeader(headerResult.bytes, manifest.totalBytes);
  compareHeader(manifest, parsed);
  const outputRecords = Object.freeze(
    records.map(({name, dtype, parameterName}) => outputRecord(sourceByName.get(name)!, dtype, parameterName))
  );

  function sourceTensor(name: string): {outputRecord: OutputRecord; sourceRecord: RangeTensor} {
    const outputRecord = requiredByName.get(name);
    const sourceRecord = sourceByName.get(name);
    if (!outputRecord || !sourceRecord)
      throw invalid(`Unknown or unused graph-required Safetensors tensor: ${String(name)}.`);
    return {outputRecord, sourceRecord};
  }

  async function readTensorChunks(
    name: string,
    {dtype: requestedDtype, signal: readSignal, onChunk}: ReadTensorChunksOptions
  ): Promise<OutputRecord> {
    if (typeof onChunk !== 'function') throw new TypeError('readTensorChunks requires an onChunk callback.');
    const {outputRecord: selectedRecord, sourceRecord: record} = sourceTensor(name);
    const dtype = requestedDtype ?? selectedRecord.dtype;
    if (dtype !== selectedRecord.dtype) throw invalid(`Requested dtype differs from graph weight ${record.name}.`);
    for (const chunk of record.chunks) {
      readSignal?.throwIfAborted();
      const spec = {
        path: manifest.path,
        totalBytes: manifest.totalBytes,
        offset: chunk.offset,
        length: chunk.length,
        sha256: chunk.sha256
      };
      const {bytes} = await transaction.read(resolvedBase, spec, {signal: readSignal});
      onBytes?.(bytes.byteLength);
      const sourceBytes = new Uint8Array(bytes);
      const output = record.dtype === 'BF16' && dtype === 'float16' ? convertBf16ToFp16(sourceBytes) : sourceBytes;
      await onChunk(output, chunk.offset - record.offset);
    }
    return selectedRecord;
  }

  async function readTensor(
    name: string,
    options: ReadTensorOptions = {}
  ): Promise<Readonly<{record: OutputRecord; bytes: Uint8Array}>> {
    const {outputRecord: record} = sourceTensor(name);
    const dtype = options.dtype ?? record.dtype;
    if (dtype !== record.dtype) throw invalid(`Requested dtype differs from graph weight ${record.name}.`);
    const bytes = new Uint8Array(record.length);
    await readTensorChunks(name, {
      dtype,
      signal: options.signal,
      onChunk(chunk, offset) {
        bytes.set(chunk, offset);
      }
    });
    return Object.freeze({record, bytes});
  }

  return Object.freeze({
    records: outputRecords,
    readTensor,
    readTensorChunks,
    commit(options?: {signal?: AbortSignal | null}) {
      return transaction.commit(options);
    }
  });
}

function validatePinnedManifest(input: unknown): asserts input is RangeManifest {
  const manifest = input as RangeManifest;
  const spec = KANANA_SAFETENSORS_SPEC;
  if (
    manifest?.model !== spec.model ||
    manifest?.repository !== spec.repository ||
    manifest?.revision !== spec.revision ||
    manifest?.path !== spec.path ||
    manifest?.totalBytes !== spec.totalBytes ||
    manifest?.headerLength !== spec.headerLength ||
    manifest?.headerSha256 !== spec.headerSha256 ||
    manifest?.tensorCount !== spec.tensorCount
  )
    throw invalid('Safetensors manifest is not for the pinned Kanana checkpoint.');
  const conversion = manifest.conversionVerification;
  if (
    !conversion ||
    conversion.enabled !== true ||
    conversion.sourceDtype !== 'BF16' ||
    conversion.targetDtype !== 'float16' ||
    conversion.rounding !== 'nearest-even' ||
    conversion.valid !== true ||
    conversion.invalidValueCount !== 0 ||
    conversion.overflowValueCount !== 0 ||
    conversion.nonFiniteValueCount !== 0 ||
    !Array.isArray(conversion.failures) ||
    conversion.failures.length !== 0 ||
    !Array.isArray(manifest.tensors) ||
    manifest.tensors.some(
      (tensor) =>
        tensor.dtype === 'BF16' &&
        (!tensor.conversion ||
          tensor.conversion.sourceDtype !== 'BF16' ||
          tensor.conversion.targetDtype !== 'float16' ||
          tensor.conversion.verified !== true ||
          tensor.conversion.rounding !== 'nearest-even' ||
          tensor.conversion.valid !== true ||
          tensor.conversion.invalidValueCount !== 0 ||
          tensor.conversion.overflowValueCount !== 0 ||
          tensor.conversion.nonFiniteValueCount !== 0)
    )
  )
    throw invalid('Safetensors manifest must verify finite BF16 to FP16 nearest-even conversion for every tensor.');
}

type PinnedWeightSourceOptions = Readonly<{
  base?: string | URL;
  manifest: unknown;
  requiredWeights: readonly RequiredWeight[];
  cache?: AssetCache | null | undefined;
  offline?: boolean | undefined;
  signal?: AbortSignal | null | undefined;
}>;

/** Open the revision-pinned upstream BF16 checkpoint, given its generated per-range hash manifest. */
export async function openKananaWeightSource({
  base = KANANA_ASSETS_URL,
  manifest,
  requiredWeights,
  cache,
  offline = false,
  signal
}: PinnedWeightSourceOptions) {
  validatePinnedManifest(manifest);
  if (!Array.isArray(requiredWeights) || requiredWeights.length !== KANANA_TEXT_GRAPH_WEIGHT_COUNT)
    throw invalid(`The current Kanana text TVM graph requires exactly ${KANANA_TEXT_GRAPH_WEIGHT_COUNT} weights.`);
  return openSafetensorsWeightSource({base, manifest, requiredWeights, cache, offline, signal});
}

/** Open only the revision-pinned Safetensors ranges used by the native TVM vision graph. */
export async function openKananaVisionWeightSource({
  base = KANANA_ASSETS_URL,
  manifest,
  requiredWeights,
  cache,
  offline = false,
  signal
}: PinnedWeightSourceOptions) {
  validatePinnedManifest(manifest);
  if (!Array.isArray(requiredWeights) || requiredWeights.length !== KANANA_VISION_GRAPH_WEIGHT_COUNT)
    throw invalid(`The current Kanana vision TVM graph requires exactly ${KANANA_VISION_GRAPH_WEIGHT_COUNT} weights.`);
  return openSafetensorsWeightSource({base, manifest, requiredWeights, cache, offline, signal});
}

/** Open one verified cache transaction covering both pinned Kanana TVM graphs. */
export async function openKananaModelWeightSource({
  base = KANANA_ASSETS_URL,
  manifest,
  textWeights,
  visionWeights,
  cache,
  offline = false,
  signal,
  onBytes
}: Readonly<{
  base?: string | URL;
  manifest: unknown;
  textWeights: readonly RequiredWeight[];
  visionWeights: readonly RequiredWeight[];
  cache?: AssetCache | null | undefined;
  offline?: boolean | undefined;
  signal?: AbortSignal | null | undefined;
  onBytes?: ((count: number) => void) | undefined;
}>) {
  validatePinnedManifest(manifest);
  if (!Array.isArray(textWeights) || textWeights.length !== KANANA_TEXT_GRAPH_WEIGHT_COUNT)
    throw invalid(`The current Kanana text TVM graph requires exactly ${KANANA_TEXT_GRAPH_WEIGHT_COUNT} weights.`);
  if (!Array.isArray(visionWeights) || visionWeights.length !== KANANA_VISION_GRAPH_WEIGHT_COUNT)
    throw invalid(`The current Kanana vision TVM graph requires exactly ${KANANA_VISION_GRAPH_WEIGHT_COUNT} weights.`);
  const names = new Set(textWeights.map(({name}) => name));
  if (visionWeights.some(({name}) => names.has(name)))
    throw invalid('Kanana text and vision TVM graph weight names must be disjoint.');
  return openSafetensorsWeightSource({
    base,
    manifest,
    requiredWeights: [...textWeights, ...visionWeights],
    cache,
    offline,
    signal,
    onBytes
  });
}
