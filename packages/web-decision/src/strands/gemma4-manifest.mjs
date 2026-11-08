const PARAMETER_COUNT = 1248;
const MAX_SHARD_BYTES = 128 * 1024 * 1024;
const SHA256 = /^[0-9a-f]{64}$/u;
const SHARD_PATH = /^params_shard_(0|[1-9][0-9]*)\.bin$/u;
const invalid = () => new Error('Invalid Gemma 4 shard plan metadata.');

function array(value, length) {
  if (!Array.isArray(value) || (length !== undefined && value.length !== length)) throw invalid();
  for (let i = 0; i < value.length; i++) if (!Object.hasOwn(value, i)) throw invalid();
  return value;
}

function positive(value) {
  if (!Number.isSafeInteger(value) || value <= 0) throw invalid();
  return value;
}

function digest(value) {
  if (typeof value !== 'string' || !SHA256.test(value)) throw invalid();
  return value;
}

function tensor(shape, dtype) {
  const width = dtype === 'float16' || dtype === 'bfloat16' ? 2 : dtype === 'float32' ? 4 : 0;
  if (!width) throw invalid();
  let bytes = width;
  const dimensions = array(shape).map((dimension) => {
    bytes *= positive(dimension);
    if (!Number.isSafeInteger(bytes)) throw invalid();
    return dimension;
  });
  return {shape: Object.freeze(dimensions), dtype, nbytes: bytes};
}

function sameTensor(left, right) {
  return (
    left.dtype === right.dtype &&
    left.shape.length === right.shape.length &&
    left.shape.every((value, index) => value === right.shape[index])
  );
}

function checkIdentity(manifest, metadata, expected) {
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

function manifestFiles(manifest, expected) {
  const files = new Map(),
    shardHashes = new Set();
  for (const file of array(manifest.files)) {
    const path = file?.path;
    if (
      typeof path !== 'string' ||
      (!SHARD_PATH.test(path) && path !== 'tensor-cache.json' && path !== 'tensor-source-map.json') ||
      files.has(path)
    )
      throw invalid();
    const bytes = positive(file.bytes),
      sha256 = digest(file.sha256);
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
    const entry = files.get(path);
    if (!entry || entry.sha256 !== digest(sha) || entry.bytes !== positive(bytes)) throw invalid();
    files.delete(path);
  }
  return files;
}

function graphParameters(specs, sourceMap) {
  array(specs, PARAMETER_COUNT + 3);
  array(sourceMap, PARAMETER_COUNT);
  const expected = [],
    targets = new Set(),
    args = new Set();
  let parameters = 0,
    buffers = 0,
    constants = 0;
  for (let index = 0; index < PARAMETER_COUNT; index++) {
    const spec = specs[index],
      source = sourceMap[index];
    if (
      !spec ||
      !source ||
      !['InputKind.PARAMETER', 'InputKind.BUFFER', 'InputKind.CONSTANT_TENSOR'].includes(spec.kind) ||
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
    const item = tensor(source.shape, source.dtype);
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
  if (specs[1207]?.target !== 'torso.embed_tokens.embed_scale') throw invalid();
  for (const [offset, arg] of ['token_ids', 'answer_index', 'option_indices'].entries()) {
    const item = specs[PARAMETER_COUNT + offset];
    if (item?.kind !== 'InputKind.USER_INPUT' || item.target !== null || item.arg !== arg || args.has(arg))
      throw invalid();
  }
  return Object.freeze(expected);
}

function shardRecords(shard, file, parameters, seen) {
  const records = [],
    expectedParameters = [];
  let offset = 0;
  for (const record of array(shard.records)) {
    const match = /^param_(0|[1-9][0-9]*)$/u.exec(record?.name);
    if (!match) throw invalid();
    const index = Number(match[1]),
      specification = parameters[index];
    if (!Number.isSafeInteger(index) || !specification || seen.has(index)) throw invalid();
    const item = tensor(record.shape, record.dtype);
    if (
      record.format !== 'raw' ||
      !sameTensor(item, specification) ||
      record.nbytes !== item.nbytes ||
      !Number.isSafeInteger(record.byteOffset) ||
      record.byteOffset !== offset ||
      item.nbytes > file.bytes - offset
    )
      throw invalid();
    offset += item.nbytes;
    seen.add(index);
    records.push(
      Object.freeze({
        name: record.name,
        shape: item.shape,
        dtype: item.dtype,
        nbytes: item.nbytes,
        byteOffset: record.byteOffset
      })
    );
    expectedParameters.push(specification);
  }
  if (!records.length || offset !== file.bytes) throw invalid();
  return {records: Object.freeze(records), expectedParameters: Object.freeze(expectedParameters)};
}

function shardPlan(cache, files, parameters) {
  const seen = new Set(),
    shards = [];
  for (const shard of array(cache.records)) {
    const path = shard?.dataPath,
      file = files.get(path);
    if (!file || shard.format !== 'raw-shard' || shard.nbytes !== file.bytes || file.bytes > MAX_SHARD_BYTES)
      throw invalid();
    files.delete(path);
    const entries = shardRecords(shard, file, parameters, seen);
    shards.push(Object.freeze({path, bytes: file.bytes, sha256: file.sha256, ...entries}));
  }
  if (files.size || seen.size !== PARAMETER_COUNT) throw invalid();
  return Object.freeze(shards);
}

/** Parse caller-authenticated JSON objects; expected hashes must describe the bytes the caller verified. */
export function parseGemma4ShardPlan({inputSpecs, cacheManifest, cacheIndex, sourceMap, expected}) {
  checkIdentity(cacheManifest, cacheIndex?.metadata, expected);
  if (
    cacheManifest.sourceWeightDtype !== 'bfloat16' ||
    cacheManifest.computeWeightDtype !== 'float16' ||
    cacheManifest.weightConversion !== 'bfloat16-to-float16-lossy' ||
    cacheManifest.derivedBufferConversion !== 'float32-to-float16-lossy' ||
    cacheIndex.metadata.sourceWeightDtype !== 'bfloat16' ||
    cacheIndex.metadata.computeWeightDtype !== 'float16' ||
    cacheIndex.metadata.weightConversion !== 'bfloat16-to-float16-lossy' ||
    cacheIndex.metadata.derivedBufferConversion !== 'float32-to-float16-lossy'
  )
    throw invalid();
  const files = manifestFiles(cacheManifest, expected);
  const expectedParameters = graphParameters(inputSpecs, sourceMap);
  const shards = shardPlan(cacheIndex, files, expectedParameters);
  return Object.freeze({shards, expectedParameters});
}
