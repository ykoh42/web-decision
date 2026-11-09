import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {readAssetBatch} from '../asset-batch.js';
import {cancelResponse, MAX_ASSET_BYTES, readBoundedResponse} from '../bounded-response.js';
import {WebDecisionError} from '../kev.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {OPENDECIDER_NANO_MANIFEST_SHA256, OPENDECIDER_NANO_PARAMETER_COUNT} from './contract.mjs';

export interface OpenDeciderNanoDigestEntry {
  readonly bytes: number;
  readonly sha256: string;
}

export interface OpenDeciderNanoManifestFile extends OpenDeciderNanoDigestEntry {
  readonly path: string;
}

export interface OpenDeciderNanoManifestShard extends OpenDeciderNanoDigestEntry {
  readonly path: string;
}

export interface OpenDeciderNanoManifest {
  readonly kind?: unknown;
  readonly schemaVersion?: unknown;
  readonly model?: unknown;
  readonly source?: {
    readonly repository?: unknown;
    readonly revision?: unknown;
    readonly modelSha256?: unknown;
    readonly tokenizerSha256?: unknown;
  } | null;
  readonly build?: {
    readonly tvm?: unknown;
    readonly target?: unknown;
    readonly host?: unknown;
    readonly integerDtype?: unknown;
    readonly parameterCount?: unknown;
    readonly parameterBytes?: unknown;
  } | null;
  readonly tensorCache?: {readonly parameterCount?: unknown} | null;
  readonly files?: unknown;
  readonly shards?: unknown;
}

export interface OpenDeciderNanoTensorRecord {
  readonly name: string;
  readonly dtype: string;
  readonly shape: number[];
  readonly byteOffset: number;
  readonly nbytes: number;
  readonly format: string;
}

export interface OpenDeciderNanoTensorShard {
  readonly dataPath: string;
  readonly format: string;
  readonly nbytes: number;
  readonly records: OpenDeciderNanoTensorRecord[];
}

export interface OpenDeciderNanoTensorCache {
  readonly metadata?: {readonly numParams?: unknown} | null;
  readonly records: OpenDeciderNanoTensorShard[];
}

export interface OpenDeciderNanoProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

export interface OpenDeciderNanoSourceOptions {
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((progress: OpenDeciderNanoProgress) => void) | undefined;
}

export interface OpenDeciderNanoSource {
  readonly manifest: OpenDeciderNanoManifest;
  readonly wasm: Uint8Array;
  readonly tokenizer: unknown;
  readonly tokenizerConfig: unknown;
  readonly tensorCache: OpenDeciderNanoTensorCache;
  readonly totalBytes: number;
  readShard(dataPath: string, signal?: AbortSignal | null | undefined): Promise<Uint8Array>;
}

const MANIFEST_BYTES = 6262;
const FILE_PATHS = ['model.wasm', 'tokenizer.json', 'tokenizer_config.json', 'weights/tensor-cache.json'] as const;
const WIDTHS: Readonly<Record<string, number>> = Object.freeze({float16: 2, float32: 4, int32: 4, int64: 8, bool: 1});
const PARAMETER_BYTES = 791668895;
type WebDecisionErrorCode = ConstructorParameters<typeof WebDecisionError>[0];
type NanoFilePath = (typeof FILE_PATHS)[number];
interface NanoByteInterval {
  readonly index: number;
  readonly start: number;
  readonly end: number;
  readonly bytes: number;
}

const fail = (code: WebDecisionErrorCode, message: string, cause?: unknown) =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);
const invalidAsset = (message: string) => fail('INVALID_ASSET', message);

async function readFile(
  base: URL,
  path: string,
  entry: OpenDeciderNanoDigestEntry,
  {
    signal,
    onBytes
  }: {readonly signal?: AbortSignal | null | undefined; readonly onBytes?: ((count: number) => void) | undefined} = {}
): Promise<Uint8Array> {
  let response;
  try {
    response = await waitForAbort(() => fetch(assetUrl(base, path), {signal} as RequestInit), signal, cancelResponse);
    if (response.status !== 200) {
      cancelResponse(response);
      throw new Error(`Unexpected HTTP status ${response.status}`);
    }
    if (!Number.isSafeInteger(entry.bytes) || entry.bytes < 0 || entry.bytes > MAX_ASSET_BYTES)
      throw invalidAsset('OpenDecider-nano asset is outside the supported file size.');
    const bytes = await readBoundedResponse(response, entry.bytes, {exact: true, signal} as unknown as Parameters<
      typeof readBoundedResponse
    >[2]);
    if ((await waitForAbort(() => sha256(bytes), signal)) !== entry.sha256)
      throw invalidAsset('OpenDecider-nano asset hash differs.');
    signal?.throwIfAborted();
    onBytes?.(entry.bytes);
    return new Uint8Array(bytes);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError) throw cause;
    throw fail('ASSET_LOAD_FAILED', `Could not load OpenDecider-nano asset ${path}.`, cause);
  }
}

function checkDigest(value: unknown): value is string {
  return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
}

function validateManifest(manifest: OpenDeciderNanoManifest | null | undefined): {
  readonly files: Map<string, OpenDeciderNanoManifestFile>;
  readonly shards: Map<string, OpenDeciderNanoManifestShard>;
} {
  if (
    manifest?.kind !== 'web-decision-opendecider-nano-tvm-assets' ||
    manifest.schemaVersion !== 1 ||
    manifest.model !== 'experimental-opendecider-nano-fp16' ||
    manifest.source?.repository !== 'manjunathshiva/opendecider-nano-ONNX' ||
    manifest.source?.revision !== '5137ae7b4f1cf8c55f1d05d3f5cdd3694d92da3e' ||
    manifest.source?.modelSha256 !== 'ce3b92ed327c394570123867d48a73383e748b015c08ca5b735670d6601ab44c' ||
    manifest.source?.tokenizerSha256 !== 'fd4146f0082d98fd1fc03bf41349f4d1eed478987ce9a64addc1654dfcad14ee' ||
    manifest.build?.tvm !== '0.27.0.post1' ||
    manifest.build?.target !== 'webgpu' ||
    manifest.build?.host !== 'wasm32-unknown-unknown-wasm' ||
    manifest.build?.integerDtype !== 'int32' ||
    manifest.build?.parameterCount !== OPENDECIDER_NANO_PARAMETER_COUNT ||
    manifest.build?.parameterBytes !== 791668895 ||
    manifest.tensorCache?.parameterCount !== OPENDECIDER_NANO_PARAMETER_COUNT ||
    !Array.isArray(manifest.files) ||
    manifest.files.length !== FILE_PATHS.length ||
    !Array.isArray(manifest.shards) ||
    manifest.shards.length !== 29
  )
    throw invalidAsset('OpenDecider-nano manifest does not match the pinned graph.');

  const files = new Map<string, OpenDeciderNanoManifestFile>();
  for (const entry of manifest.files as OpenDeciderNanoManifestFile[]) {
    if (
      !entry ||
      !FILE_PATHS.includes(entry.path as NanoFilePath) ||
      files.has(entry.path) ||
      !Number.isSafeInteger(entry.bytes) ||
      entry.bytes < 1 ||
      entry.bytes > MAX_ASSET_BYTES ||
      !checkDigest(entry.sha256)
    )
      throw invalidAsset('OpenDecider-nano manifest has an invalid file entry.');
    files.set(entry.path, entry);
  }
  if (FILE_PATHS.some((path) => !files.has(path))) throw invalidAsset('OpenDecider-nano manifest is missing a file.');

  const shards = new Map<string, OpenDeciderNanoManifestShard>();
  for (const shard of manifest.shards as OpenDeciderNanoManifestShard[]) {
    if (
      !shard ||
      typeof shard.path !== 'string' ||
      !/^weights\/params_shard_\d+\.bin$/.test(shard.path) ||
      shards.has(shard.path) ||
      !Number.isSafeInteger(shard.bytes) ||
      shard.bytes < 1 ||
      shard.bytes > MAX_ASSET_BYTES ||
      !checkDigest(shard.sha256)
    )
      throw invalidAsset('OpenDecider-nano manifest has an invalid weight shard.');
    shards.set(shard.path, shard);
  }
  for (let index = 0; index < manifest.shards.length; index++)
    if (!shards.has(`weights/params_shard_${index}.bin`))
      throw invalidAsset('OpenDecider-nano weight shard order is incomplete.');
  return {files, shards};
}

function validateTensorRecord(
  record: OpenDeciderNanoTensorRecord | null | undefined,
  shard: Pick<OpenDeciderNanoTensorShard, 'nbytes'>,
  seen: Set<number>
): NanoByteInterval {
  if (!record || typeof record.name !== 'string' || !Array.isArray(record.shape))
    throw invalidAsset('OpenDecider-nano tensor metadata is invalid.');
  const match = /^param_(\d+)$/.exec(record.name);
  const index = match ? Number(match[1]) : -1;
  const width = WIDTHS[record.dtype];
  const elements = record.shape.reduce((count, size) => count * size, 1);
  const end = record.byteOffset + record.nbytes;
  if (
    index < 0 ||
    index >= OPENDECIDER_NANO_PARAMETER_COUNT ||
    seen.has(index) ||
    !width ||
    record.format !== 'raw' ||
    record.shape.some((size) => !Number.isSafeInteger(size) || size < 0) ||
    !Number.isSafeInteger(elements) ||
    !Number.isSafeInteger(record.byteOffset) ||
    record.byteOffset < 0 ||
    !Number.isSafeInteger(record.nbytes) ||
    record.nbytes < 1 ||
    elements * width !== record.nbytes ||
    !Number.isSafeInteger(end) ||
    end > shard.nbytes
  )
    throw invalidAsset('OpenDecider-nano tensor metadata is invalid.');
  seen.add(index);
  return {index, start: record.byteOffset, end, bytes: record.nbytes};
}

function validateTensorShard(
  shard: OpenDeciderNanoTensorShard | null | undefined,
  shards: Map<string, OpenDeciderNanoManifestShard>,
  seen: Set<number>,
  seenShards: Set<string>
): number {
  if (!shard || typeof shard.dataPath !== 'string')
    throw invalidAsset('OpenDecider-nano tensor cache has an invalid shard.');
  const path = `weights/${shard.dataPath}`;
  const expectedShard = shards.get(path);
  if (
    !expectedShard ||
    seenShards.has(path) ||
    shard.format !== 'raw-shard' ||
    shard.nbytes !== expectedShard.bytes ||
    !Array.isArray(shard.records) ||
    shard.records.length < 1
  )
    throw invalidAsset('OpenDecider-nano tensor cache has an invalid shard.');
  seenShards.add(path);
  const intervals = shard.records.map((record) => validateTensorRecord(record, shard, seen));
  intervals.sort((left, right) => left.start - right.start);
  for (let index = 1; index < intervals.length; index++)
    if (intervals[index - 1]!.end > intervals[index]!.start)
      throw invalidAsset('OpenDecider-nano tensor records overlap.');
  return intervals.reduce((total, interval) => total + interval.bytes, 0);
}

function validateTensorCache(
  cache: OpenDeciderNanoTensorCache | null | undefined,
  shards: Map<string, OpenDeciderNanoManifestShard>
): void {
  if (
    cache?.metadata?.numParams !== OPENDECIDER_NANO_PARAMETER_COUNT ||
    !Array.isArray(cache.records) ||
    cache.records.length !== shards.size
  )
    throw invalidAsset('OpenDecider-nano tensor cache has an invalid parameter count.');
  const seen = new Set<number>(),
    seenShards = new Set<string>();
  const parameterBytes = cache.records.reduce(
    (total, shard) => total + validateTensorShard(shard, shards, seen, seenShards),
    0
  );
  if (
    seen.size !== OPENDECIDER_NANO_PARAMETER_COUNT ||
    seenShards.size !== shards.size ||
    parameterBytes !== PARAMETER_BYTES
  )
    throw invalidAsset('OpenDecider-nano tensor parameters are incomplete.');
}

/** Read and verify the pinned graph, tokenizer and shards without model caching. */
export async function openOpenDeciderNanoSource(
  base: URL,
  {signal, onProgress}: OpenDeciderNanoSourceOptions = {}
): Promise<OpenDeciderNanoSource> {
  const manifestBytes = await readFile(
    base,
    'manifest.json',
    {bytes: MANIFEST_BYTES, sha256: OPENDECIDER_NANO_MANIFEST_SHA256},
    {signal}
  );
  let manifest: OpenDeciderNanoManifest;
  try {
    manifest = parseStrictJson(manifestBytes) as OpenDeciderNanoManifest;
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('INVALID_ASSET', 'OpenDecider-nano manifest is invalid.', cause);
  }
  const {files, shards} = validateManifest(manifest as OpenDeciderNanoManifest | null | undefined);
  const totalBytes =
    MANIFEST_BYTES +
    [...files.values()].reduce((total, entry) => total + entry.bytes, 0) +
    [...shards.values()].reduce((total, entry) => total + entry.bytes, 0);
  let loadedBytes = MANIFEST_BYTES;
  const reportProgress = (count: number) => {
    loadedBytes += count;
    onProgress?.({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes});
  };
  onProgress?.({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes});
  const [wasm, tokenizerBytes, tokenizerConfigBytes, tensorCacheBytes] = await (readAssetBatch(
    FILE_PATHS,
    (path: string, localSignal: AbortSignal) =>
      readFile(base, path, files.get(path)!, {signal: localSignal, onBytes: reportProgress}),
    signal
  ) as unknown as Promise<[Uint8Array, Uint8Array, Uint8Array, Uint8Array]>);
  let tokenizer: unknown, tokenizerConfig: unknown, tensorCache: OpenDeciderNanoTensorCache;
  try {
    tokenizer = parseStrictJson(tokenizerBytes, {maxBytes: 16 * 1024 * 1024});
    tokenizerConfig = parseStrictJson(tokenizerConfigBytes);
    tensorCache = parseStrictJson(tensorCacheBytes, {maxBytes: 1024 * 1024}) as OpenDeciderNanoTensorCache;
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('INVALID_ASSET', 'OpenDecider-nano model metadata is invalid.', cause);
  }
  validateTensorCache(tensorCache, shards);
  return {
    manifest,
    wasm,
    tokenizer,
    tokenizerConfig,
    tensorCache,
    totalBytes,
    async readShard(dataPath, readSignal = signal) {
      const path = `weights/${dataPath}`;
      const entry = shards.get(path);
      if (!entry) throw invalidAsset('Unknown OpenDecider-nano weight shard.');
      return readFile(base, path, entry, {signal: readSignal, onBytes: reportProgress});
    }
  };
}
