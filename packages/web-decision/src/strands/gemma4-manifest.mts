const PARAMETER_COUNT = 1248;
const MAX_SHARD_BYTES = 128 * 1024 * 1024;
const SHA256 = /^[0-9a-f]{64}$/u;
const SHARD_PATH = /^params_shard_(0|[1-9][0-9]*)\.bin$/u;
const invalid = () => new Error('Invalid Gemma 4 shard plan metadata.');

type Gemma4DType = 'float16' | 'bfloat16' | 'float32';
type Gemma4DataRecord = Record<string, unknown>;

interface Gemma4TensorShape {
  readonly shape: readonly number[];
  readonly dtype: Gemma4DType;
}

interface Gemma4Tensor extends Gemma4TensorShape {
  readonly nbytes: number;
}

interface Gemma4ExpectedParameter extends Gemma4TensorShape {
  readonly index: number;
}

interface Gemma4FileInfo {
  readonly bytes: number;
  readonly sha256: string;
}

export interface Gemma4PlannedRecord extends Gemma4Tensor {
  readonly name: string;
  readonly byteOffset: number;
}

export interface Gemma4PlannedShard {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly records: readonly Gemma4PlannedRecord[];
  readonly expectedParameters: readonly Gemma4ExpectedParameter[];
}

export interface Gemma4ShardPlan {
  readonly shards: readonly Gemma4PlannedShard[];
  readonly expectedParameters: readonly Gemma4ExpectedParameter[];
}

export interface Gemma4ShardPlanInput {
  readonly inputSpecs: unknown;
  readonly cacheManifest: unknown;
  readonly cacheIndex: unknown;
  readonly sourceMap: unknown;
  readonly expected: unknown;
}

function array(value: unknown, length?: number): unknown[] {
  if (!Array.isArray(value) || (length !== undefined && value.length !== length)) throw invalid();
  for (let i = 0; i < value.length; i++) if (!Object.hasOwn(value, i)) throw invalid();
  return value;
}

function positive(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) throw invalid();
  return value as number;
}

function digest(value: unknown): string {
  if (typeof value !== 'string' || !SHA256.test(value)) throw invalid();
  return value;
}

function tensor(shape: unknown, dtype: Gemma4DType): Gemma4Tensor {
  const width = dtype === 'float16' || dtype === 'bfloat16' ? 2 : dtype === 'float32' ? 4 : 0;
  if (!width) throw invalid();
  let bytes = width;
  const dimensions = array(shape).map((dimension) => {
    bytes *= positive(dimension);
    if (!Number.isSafeInteger(bytes)) throw invalid();
    return dimension as number;
  });
  return {shape: Object.freeze(dimensions), dtype, nbytes: bytes};
}

function sameTensor(left: Gemma4TensorShape, right: Gemma4TensorShape): boolean {
  return (
    left.dtype === right.dtype &&
    left.shape.length === right.shape.length &&
    left.shape.every((value, index) => value === right.shape[index])
  );
}

function checkIdentity(manifest: Gemma4DataRecord, metadata: Gemma4DataRecord, expected: Gemma4DataRecord): void {
  if (!manifest || !metadata || !expected) throw invalid();
  for (const key of ['model', 'baseModel', 'baseRevision']) {
    if (
      typeof expected[key] !== 'string' ||
      !expected[key] ||
      manifest[key] !== expected[key] ||
      metadata[key] !== expected[key]
    )
      throw invalid();
  }
  if (
    typeof expected.checkpointRevision !== 'string' ||
    !expected.checkpointRevision ||
    manifest.schemaVersion !== 1 ||
    manifest.checkpointRevision !== expected.checkpointRevision ||
    metadata.revision !== expected.checkpointRevision ||
    metadata.format !== 'raw' ||
    manifest.tensorSourceMap !== 'tensor-source-map.json' ||
    metadata.wasmSha256 !== digest(expected.graphSha256) ||
    metadata.inputSpecsSha256 !== digest(expected.inputSpecsSha256) ||
    manifest.sourceBaseSha256 !== digest(expected.sourceBaseSha256) ||
    metadata.baseWeightsSha256 !== expected.sourceBaseSha256
  )
    throw invalid();
}

function manifestFiles(manifest: Gemma4DataRecord, expected: Gemma4DataRecord): Map<string, Gemma4FileInfo> {
  const files = new Map<string, Gemma4FileInfo>(),
    shardHashes = new Set<string>();
  for (const file of array(manifest.files)) {
    const path = (file as Gemma4DataRecord | null | undefined)?.path;
    if (
      typeof path !== 'string' ||
      (!SHARD_PATH.test(path) && path !== 'tensor-cache.json' && path !== 'tensor-source-map.json') ||
      files.has(path)
    )
      throw invalid();
    const bytes = positive((file as Gemma4DataRecord).bytes),
      sha256 = digest((file as Gemma4DataRecord).sha256);
    if (SHARD_PATH.test(path)) {
      if (bytes > MAX_SHARD_BYTES || shardHashes.has(sha256)) throw invalid();
      shardHashes.add(sha256);
    }
    files.set(path, {bytes, sha256});
  }
  for (const [path, sha, bytes] of [
    ['tensor-cache.json', expected.cacheIndexSha256, expected.cacheIndexBytes],
    ['tensor-source-map.json', expected.sourceMapSha256, expected.sourceMapBytes]
  ]) {
    const entry = files.get(path as string);
    if (!entry || entry.sha256 !== digest(sha) || entry.bytes !== positive(bytes)) throw invalid();
    files.delete(path as string);
  }
  return files;
}

function graphParameters(specs: unknown, sourceMap: unknown): readonly Gemma4ExpectedParameter[] {
  array(specs, PARAMETER_COUNT + 3);
  array(sourceMap, PARAMETER_COUNT);
  const expected: Gemma4ExpectedParameter[] = [],
    targets = new Set<string>(),
    args = new Set<string>();
  let parameters = 0,
    buffers = 0,
    constants = 0;
  for (let index = 0; index < PARAMETER_COUNT; index++) {
    const spec = (specs as unknown[])[index] as Gemma4DataRecord,
      source = (sourceMap as unknown[])[index] as Gemma4DataRecord;
    if (
      !spec ||
      !source ||
      !['InputKind.PARAMETER', 'InputKind.BUFFER', 'InputKind.CONSTANT_TENSOR'].includes(spec.kind as string) ||
      typeof spec.target !== 'string' ||
      !spec.target ||
      targets.has(spec.target) ||
      typeof spec.arg !== 'string' ||
      !spec.arg ||
      args.has(spec.arg) ||
      source.name !== `param_${index}` ||
      source.target !== spec.target
    )
      throw invalid();
    parameters += Number(spec.kind === 'InputKind.PARAMETER');
    buffers += Number(spec.kind === 'InputKind.BUFFER');
    constants += Number(spec.kind === 'InputKind.CONSTANT_TENSOR');
    targets.add(spec.target);
    args.add(spec.arg);
    const item = tensor(source.shape, source.dtype as Gemma4DType);
    if (source.nbytes !== item.nbytes) throw invalid();
    if (index === 1207) {
      if (
        spec.kind !== 'InputKind.CONSTANT_TENSOR' ||
        spec.target !== 'torso.embed_tokens.embed_scale' ||
        item.shape.length !== 0 ||
        item.dtype !== 'float16' ||
        source.sourceKind !== 'derived' ||
        source.sourceTensor !== 'main_embed_scale' ||
        source.sourceValue !== 39.1875 ||
        source.conversion !== 'none' ||
        source.sourceDtype !== 'float16'
      )
        throw invalid();
    }
    digest(source.sourceSha256);
    expected.push(Object.freeze({index, shape: item.shape, dtype: item.dtype}));
  }
  if (parameters !== 1207 || buffers !== 40 || constants !== 1) throw invalid();
  if (((specs as unknown[])[1207] as Gemma4DataRecord | null | undefined)?.target !== 'torso.embed_tokens.embed_scale')
    throw invalid();
  for (const [offset, arg] of ['token_ids', 'answer_index', 'option_indices'].entries()) {
    const item = (specs as unknown[])[PARAMETER_COUNT + offset] as Gemma4DataRecord | null | undefined;
    if (item?.kind !== 'InputKind.USER_INPUT' || item.target !== null || item.arg !== arg || args.has(arg))
      throw invalid();
  }
  return Object.freeze(expected);
}

function shardRecords(
  shard: Gemma4DataRecord,
  file: Gemma4FileInfo,
  parameters: readonly Gemma4ExpectedParameter[],
  seen: Set<number>
): {readonly records: readonly Gemma4PlannedRecord[]; readonly expectedParameters: readonly Gemma4ExpectedParameter[]} {
  const records: Gemma4PlannedRecord[] = [],
    expectedParameters: Gemma4ExpectedParameter[] = [];
  let offset = 0;
  for (const record of array(shard.records)) {
    const match = /^param_(0|[1-9][0-9]*)$/u.exec((record as Gemma4DataRecord | null | undefined)?.name as string);
    if (!match) throw invalid();
    const index = Number(match[1]),
      specification = parameters[index];
    if (!Number.isSafeInteger(index) || !specification || seen.has(index)) throw invalid();
    const item = tensor((record as Gemma4DataRecord).shape, (record as Gemma4DataRecord).dtype as Gemma4DType);
    if (
      (record as Gemma4DataRecord).format !== 'raw' ||
      !sameTensor(item, specification) ||
      (record as Gemma4DataRecord).nbytes !== item.nbytes ||
      !Number.isSafeInteger((record as Gemma4DataRecord).byteOffset) ||
      (record as Gemma4DataRecord).byteOffset !== offset ||
      item.nbytes > file.bytes - offset
    )
      throw invalid();
    offset += item.nbytes;
    seen.add(index);
    records.push(
      Object.freeze({
        name: (record as Gemma4DataRecord).name as string,
        shape: item.shape,
        dtype: item.dtype,
        nbytes: item.nbytes,
        byteOffset: (record as Gemma4DataRecord).byteOffset as number
      })
    );
    expectedParameters.push(specification);
  }
  if (!records.length || offset !== file.bytes) throw invalid();
  return {records: Object.freeze(records), expectedParameters: Object.freeze(expectedParameters)};
}

function shardPlan(
  cache: Gemma4DataRecord,
  files: Map<string, Gemma4FileInfo>,
  parameters: readonly Gemma4ExpectedParameter[]
): readonly Gemma4PlannedShard[] {
  const seen = new Set<number>(),
    shards: Gemma4PlannedShard[] = [];
  for (const shard of array(cache.records)) {
    const path = (shard as Gemma4DataRecord | null | undefined)?.dataPath as string,
      file = files.get(path);
    if (
      !file ||
      (shard as Gemma4DataRecord).format !== 'raw-shard' ||
      (shard as Gemma4DataRecord).nbytes !== file.bytes ||
      file.bytes > MAX_SHARD_BYTES
    )
      throw invalid();
    files.delete(path);
    const entries = shardRecords(shard as Gemma4DataRecord, file, parameters, seen);
    shards.push(Object.freeze({path, bytes: file.bytes, sha256: file.sha256, ...entries}));
  }
  if (files.size || seen.size !== PARAMETER_COUNT) throw invalid();
  return Object.freeze(shards);
}

/** Parse caller-authenticated JSON objects; expected hashes must describe the bytes the caller verified. */
export function parseGemma4ShardPlan({
  inputSpecs,
  cacheManifest,
  cacheIndex,
  sourceMap,
  expected
}: Gemma4ShardPlanInput): Gemma4ShardPlan {
  checkIdentity(
    cacheManifest as Gemma4DataRecord,
    (cacheIndex as {readonly metadata?: unknown} | null | undefined)?.metadata as Gemma4DataRecord,
    expected as Gemma4DataRecord
  );
  if (
    (cacheManifest as Gemma4DataRecord).sourceWeightDtype !== 'bfloat16' ||
    (cacheManifest as Gemma4DataRecord).computeWeightDtype !== 'float16' ||
    (cacheManifest as Gemma4DataRecord).weightConversion !== 'bfloat16-to-float16-lossy' ||
    (cacheManifest as Gemma4DataRecord).derivedBufferConversion !== 'float32-to-float16-lossy' ||
    ((cacheIndex as Gemma4DataRecord).metadata as Gemma4DataRecord).sourceWeightDtype !== 'bfloat16' ||
    ((cacheIndex as Gemma4DataRecord).metadata as Gemma4DataRecord).computeWeightDtype !== 'float16' ||
    ((cacheIndex as Gemma4DataRecord).metadata as Gemma4DataRecord).weightConversion !== 'bfloat16-to-float16-lossy' ||
    ((cacheIndex as Gemma4DataRecord).metadata as Gemma4DataRecord).derivedBufferConversion !==
      'float32-to-float16-lossy'
  )
    throw invalid();
  const files = manifestFiles(cacheManifest as Gemma4DataRecord, expected as Gemma4DataRecord);
  const expectedParameters = graphParameters(inputSpecs, sourceMap);
  const shards = shardPlan(cacheIndex as Gemma4DataRecord, files, expectedParameters);
  return Object.freeze({shards, expectedParameters});
}
