/// <reference types="vite/client" />

import candidateManifest from '../../assets/experimental-kev-upstream/manifest.json' with {type: 'json'};
import stableManifest from '../../assets/kev06-stable/manifest.json' with {type: 'json'};
import sourceShardMapText from '../../assets/experimental-kev-upstream/source-shard-map.json?raw';
import {cancelBody, MAX_ASSET_BYTES, ResponseSizeError} from '../bounded-response.js';
import {assetUrl, openAssetCache, sha256, waitForAbort} from '../assets.js';
import {requestF16Adapter} from '../gpu-preflight.js';
import {WebDecisionError} from '../kev.js';
import {readPinnedManifest} from '../pinned-manifest.js';
import {validatePrefetchBytes} from '../shard-prefetch.js';
import type {ModelProgress} from '../curated/types.js';
import {
  EXPERIMENTAL_KEV_UPSTREAM_MANIFEST_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_MODEL,
  EXPERIMENTAL_KEV_UPSTREAM_PARAMETER_REGION_START,
  EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
  EXPERIMENTAL_KEV_UPSTREAM_REVISION,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_BYTES,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_TENSOR_COUNT,
  EXPERIMENTAL_KEV_UPSTREAM_WASM_SHA256
} from './upstream-contract.js';

type RecordValue = Record<string, unknown>;
type PropertyRecord = Record<PropertyKey, unknown>;
type AssetMetadata = {readonly bytes: number; readonly sha256: string};
type SourceManifestBase = {
  readonly files: Readonly<Record<string, AssetMetadata>>;
  readonly [key: string]: unknown;
};
type PinnedKevManifest = SourceManifestBase & {
  readonly parameterCount: number;
  readonly maxInputTokens?: number;
  readonly maxStateTokens: number;
  readonly maxBranchTokens: number;
};
type SourceRecord = {
  readonly sourceOffset: number;
  readonly targetOffset: number;
  readonly bytes: number;
  readonly shardIndex: number;
};
type SourceShard = {readonly path: string; readonly bytes: number; readonly sha256: string};
type SourceMap = {
  readonly source: RecordValue;
  readonly records: readonly unknown[];
  readonly target: {readonly shards: readonly unknown[]};
};
type SourceDefaults = {
  readonly displayName: string;
  readonly bundledAssetExport: string;
  readonly manifest: SourceManifestBase;
  readonly sourceShardMapText: string;
  readonly sourceUrl: string;
  readonly sourceBytes: number;
  readonly sourceSha256: string;
  readonly mapSha256: string;
  readonly repository: string;
  readonly revision: string;
  readonly sourcePath: string;
  readonly parameterRegionStart: number;
  readonly tensorCount: number;
  readonly progressChunkBytes: number;
  readonly loadBundledAssets: () => Promise<unknown>;
  readonly cacheBase: URL;
};
type SourceFetcher = (url: string, init: RequestInit) => Promise<Response>;
type SourceProfile = SourceDefaults & {
  readonly bundledUrls: Readonly<Record<string, string>> | undefined;
  readonly fetchSource: SourceFetcher;
  readonly preflight: (signal: AbortSignal | undefined, gpu: GPU) => Promise<void>;
  readonly openCache: (enabled: boolean, signal: AbortSignal | undefined) => Promise<Cache | null>;
};
type SourceOverrides = Partial<SourceDefaults> & {
  readonly bundledUrls?: Readonly<Record<string, string>>;
  readonly fetchSource?: SourceFetcher;
  readonly preflight?: SourceProfile['preflight'];
  readonly openCache?: SourceProfile['openCache'];
};
type KevSourceOptions = {
  readonly model?: string;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal | undefined;
  readonly onProgress?: ((progress: ModelProgress) => void) | undefined;
  readonly maxPrefetchBytes?: number;
  readonly maxInputTokens?: number;
  readonly maxInputCharacters?: number;
  readonly debertaReadback?: unknown;
  readonly [key: string]: unknown;
};
type PreparedSource = {
  readonly base: URL;
  readonly cache: Cache | null;
  readonly signal: AbortSignal | undefined;
  readonly manifestUrl: string;
  readonly response: Response;
  readonly maxPrefetchBytes: number;
};
type SourceProgressCallback = (loadedBytes: number) => void;
type SourceLoadEntry = {
  readonly controller: AbortController;
  references: number;
  settled: boolean;
  loadedBytes?: number;
  promise: Promise<ArrayBuffer>;
  readonly progressCallbacks: Set<SourceProgressCallback>;
};
type SourceLease = {
  read(signal: AbortSignal | undefined, onProgress?: SourceProgressCallback): Promise<ArrayBuffer>;
  release(): void;
};
type KevReconstructedSource = {
  readonly manifest: SourceManifestBase;
  prepare(options: KevSourceOptions): Promise<PreparedSource>;
  fetch(url: string | URL, init?: Pick<RequestInit, 'signal'>): Promise<Response>;
  readonly bundledProgressPhase: ModelProgress['phase'];
  setProgress(callback: ((progress: ModelProgress) => void) | undefined, loadedBytes?: number): void;
  readonly assetProgressPhase: ModelProgress['phase'];
  dispose(): Promise<void>;
};

export const EXPERIMENTAL_KEV_MODEL = EXPERIMENTAL_KEV_UPSTREAM_MODEL;
export const KEV06_STABLE_MODEL = 'kev-0.6b-q4f16';

const MAX_PREFETCH_BYTES = 0;
const EXPERIMENTAL_CACHE_BASE = new URL('https://kev-assets.web-decision.invalid/experimental-kev-upstream/');
const STABLE_CACHE_BASE = new URL('https://kev-assets.web-decision.invalid/kev-0.6b-q4f16/');
const BUNDLED_ASSETS = new Set([
  'model.wasm',
  'tokenizer.json',
  'tokenizer_config.json',
  'model_config.json',
  'weights/tensor-cache.json'
]);

const SOURCE_URL = new URL(
  `https://huggingface.co/${EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY}/resolve/${EXPERIMENTAL_KEV_UPSTREAM_REVISION}/${EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH}`
).href;

const sourceLoads = new Map<string, SourceLoadEntry>();
const mappingPromises = new Map<string, Promise<ValidatedSourceMap>>();

type ValidatedSourceMap = {
  readonly records: readonly SourceRecord[];
  readonly target: {readonly shards: readonly SourceShard[]};
};

function invalidAsset(message: string, cause?: unknown): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message, cause ? {cause} : undefined);
}

function isRecord(value: unknown): value is RecordValue {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isPropertyRecord(value: unknown): value is PropertyRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isUnknownArray(value: unknown): value is readonly unknown[] {
  return Array.isArray(value);
}

function requireBundledDataUrls(value: unknown, modelName = 'experimental Kev'): Readonly<Record<string, string>> {
  if (!isRecord(value)) throw invalidAsset(`The ${modelName} bundle is missing pinned inline assets.`);
  const entries: [string, string][] = [];
  for (const [key, url] of Object.entries(value)) {
    if (typeof key !== 'string' || typeof url !== 'string' || !url.startsWith('data:'))
      throw invalidAsset(`The ${modelName} bundle is missing pinned inline assets.`);
    entries.push([key, url]);
  }
  return Object.freeze(Object.fromEntries(entries));
}

function requireKevWebGpu(): GPU {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu)
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'
    );
  return gpu;
}

function responseFromDataUrl(value: string, signal: AbortSignal | undefined): Response {
  signal?.throwIfAborted();
  const comma = value.indexOf(',');
  if (comma < 5) throw invalidAsset('Invalid bundled data URL.');
  const metadata = value.slice(5, comma);
  const payload = value.slice(comma + 1);
  const base64 = /(?:^|;)base64(?:;|$)/u.test(metadata);
  let bytes;
  if (base64) {
    if (payload.length > Math.ceil((MAX_ASSET_BYTES * 4) / 3) + 8)
      throw invalidAsset('Bundled data URL exceeds the per-asset limit.');
    const binary = atob(payload);
    if (binary.length > MAX_ASSET_BYTES) throw invalidAsset('Bundled data URL exceeds the per-asset limit.');
    bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  } else {
    let decoded;
    try {
      decoded = decodeURIComponent(payload);
    } catch (cause) {
      throw invalidAsset('Invalid bundled data URL encoding.', cause);
    }
    bytes = new TextEncoder().encode(decoded);
    if (bytes.byteLength > MAX_ASSET_BYTES) throw invalidAsset('Bundled data URL exceeds the per-asset limit.');
  }
  signal?.throwIfAborted();
  const mime = metadata.split(';', 1)[0] || 'application/octet-stream';
  return new Response(bytes, {headers: {'content-type': mime}});
}

function requireOptionRecord(
  value: unknown,
  modelName: 'Experimental Kev' | 'Stable Kev',
  allowed: ReadonlySet<string>
): PropertyRecord {
  if (!isPropertyRecord(value)) throw new WebDecisionError('INVALID_INPUT', `${modelName} load options are required.`);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !allowed.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new WebDecisionError(
        'INVALID_INPUT',
        `Unsupported ${modelName === 'Experimental Kev' ? 'experimental Kev' : 'stable Kev'} load option.`
      );
  }
  return value;
}

function isAbortSignalLike(value: unknown): boolean {
  if (value == null) return true;
  if (typeof value !== 'object' && typeof value !== 'function') return false;
  const signal = value as PropertyRecord;
  return (
    typeof signal.throwIfAborted === 'function' &&
    typeof signal.addEventListener === 'function' &&
    typeof signal.removeEventListener === 'function'
  );
}

function hasValidKevLoadOptions(options: PropertyRecord, model: string): boolean {
  return (
    options.model === model &&
    (options.cache === undefined || typeof options.cache === 'boolean') &&
    (options.offline === undefined || typeof options.offline === 'boolean') &&
    !(options.offline && options.cache === false) &&
    isAbortSignalLike(options.signal) &&
    (options.onProgress === undefined || typeof options.onProgress === 'function')
  );
}

function validateExperimentalLoadOptions(value: unknown): number {
  const allowed = new Set([
    'model',
    'cache',
    'offline',
    'signal',
    'onProgress',
    'maxPrefetchBytes',
    'maxInputTokens',
    'maxInputCharacters'
  ]);
  const options = requireOptionRecord(value, 'Experimental Kev', allowed);
  if (
    !hasValidKevLoadOptions(options, EXPERIMENTAL_KEV_MODEL) ||
    validatePrefetchBytes(options.maxPrefetchBytes) !== MAX_PREFETCH_BYTES
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid experimental Kev load option values.');
  return MAX_PREFETCH_BYTES;
}

function validateStableLoadOptions(value: unknown): number {
  const allowed = new Set([
    'model',
    'cache',
    'offline',
    'signal',
    'onProgress',
    'maxPrefetchBytes',
    'maxInputTokens',
    'maxInputCharacters',
    'debertaReadback'
  ]);
  const options = requireOptionRecord(value, 'Stable Kev', allowed);
  const maxPrefetchBytes = validatePrefetchBytes(options.maxPrefetchBytes);
  if (!hasValidKevLoadOptions(options, KEV06_STABLE_MODEL) || options.debertaReadback !== undefined)
    throw new WebDecisionError('INVALID_INPUT', 'Invalid stable Kev load option values.');
  return maxPrefetchBytes;
}

function isSafeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value);
}

function isSourceRecord(value: unknown): value is SourceRecord {
  if (!isRecord(value)) return false;
  return (
    isSafeInteger(value.sourceOffset) &&
    isSafeInteger(value.targetOffset) &&
    isSafeInteger(value.bytes) &&
    isSafeInteger(value.shardIndex)
  );
}

function isSourceShard(value: unknown): value is SourceShard {
  if (!isRecord(value)) return false;
  return typeof value.path === 'string' && isSafeInteger(value.bytes) && typeof value.sha256 === 'string';
}

function isAssetMetadata(value: unknown): value is AssetMetadata {
  if (!isRecord(value)) return false;
  return isSafeInteger(value.bytes) && typeof value.sha256 === 'string';
}

function isSourceManifest(value: unknown): value is PinnedKevManifest {
  if (!isRecord(value) || !isRecord(value.files)) return false;
  if (
    !isSafeInteger(value.parameterCount) ||
    !isSafeInteger(value.maxStateTokens) ||
    !isSafeInteger(value.maxBranchTokens)
  )
    return false;
  if (value.maxInputTokens !== undefined && !isSafeInteger(value.maxInputTokens)) return false;
  return Object.values(value.files).every(isAssetMetadata);
}

function hasSourceMapStructure(value: unknown): value is SourceMap {
  if (!isRecord(value) || !isRecord(value.source) || !isRecord(value.target)) return false;
  return isUnknownArray(value.records) && isUnknownArray(value.target.shards);
}

function hasExpectedSourceMapMetadata(mapping: SourceMap, profile: SourceProfile): boolean {
  const expectedSource = {
    repository: profile.repository,
    revision: profile.revision,
    path: profile.sourcePath,
    bytes: profile.sourceBytes,
    sha256: profile.sourceSha256,
    parameterRegionStart: profile.parameterRegionStart
  };
  return (
    Object.entries(expectedSource).every(([key, expected]) => mapping.source[key] === expected) &&
    mapping.records.length === profile.tensorCount &&
    mapping.target.shards.length > 0
  );
}

function requireSourceMap(value: unknown, profile: SourceProfile): SourceMap {
  if (!hasSourceMapStructure(value) || !hasExpectedSourceMapMetadata(value, profile))
    throw invalidAsset(`The ${profile.displayName} source map metadata is invalid.`);
  return value;
}

function requireSourceRecords(values: readonly unknown[], profile: SourceProfile): SourceRecord[] {
  const records: SourceRecord[] = [];
  for (const value of values) {
    if (!isSourceRecord(value))
      throw invalidAsset(`The ${profile.displayName} source map has a source gap or overlap.`);
    records.push(value);
  }
  return records;
}

function validateSourceCoverage(records: readonly SourceRecord[], shardCount: number, profile: SourceProfile): void {
  const sortedRecords = [...records].sort((left, right) => left.sourceOffset - right.sourceOffset);
  let sourceCursor = profile.parameterRegionStart;
  for (const record of sortedRecords) {
    if (
      record.bytes < 1 ||
      record.sourceOffset !== sourceCursor ||
      record.sourceOffset + record.bytes > profile.sourceBytes ||
      record.shardIndex < 0 ||
      record.shardIndex >= shardCount
    )
      throw invalidAsset(`The ${profile.displayName} source map has a source gap or overlap.`);
    sourceCursor += record.bytes;
  }
  if (sourceCursor !== profile.sourceBytes)
    throw invalidAsset(`The ${profile.displayName} source map has incomplete coverage.`);
}

function requireSourceShards(values: readonly unknown[]): SourceShard[] {
  const shards: SourceShard[] = [];
  for (const value of values) {
    if (!isSourceShard(value))
      throw invalidAsset('The source map does not match pinned target metadata for an invalid shard path.');
    shards.push(value);
  }
  return shards;
}

function validateTargetCoverage(
  shards: readonly SourceShard[],
  records: readonly SourceRecord[],
  profile: SourceProfile
): void {
  for (let shardIndex = 0; shardIndex < shards.length; shardIndex++) {
    const shard = shards[shardIndex];
    if (!shard) throw invalidAsset('The source map does not match pinned target metadata for an invalid shard path.');
    const metadata = profile.manifest.files[shard.path];
    if (!metadata || metadata.bytes !== shard.bytes || metadata.sha256 !== shard.sha256)
      throw invalidAsset(`The source map does not match pinned target metadata for ${shard.path}.`);
    const shardRecords = records
      .filter((record) => record.shardIndex === shardIndex)
      .sort((left, right) => left.targetOffset - right.targetOffset);
    let targetCursor = 0;
    for (const record of shardRecords) {
      if (record.targetOffset !== targetCursor)
        throw invalidAsset(`The source map has a target gap or overlap for ${shard.path}.`);
      targetCursor += record.bytes;
    }
    if (targetCursor !== shard.bytes) throw invalidAsset(`The source map has incomplete coverage for ${shard.path}.`);
  }
}

async function loadValidatedMapping(profile: SourceProfile): Promise<ValidatedSourceMap> {
  if ((await sha256(new TextEncoder().encode(profile.sourceShardMapText))) !== profile.mapSha256)
    throw invalidAsset(`The ${profile.displayName} source map does not match its pinned digest.`);
  const parsed: unknown = JSON.parse(profile.sourceShardMapText);
  const mapping = requireSourceMap(parsed, profile);
  const records = requireSourceRecords(mapping.records, profile);
  validateSourceCoverage(records, mapping.target.shards.length, profile);
  const shards = requireSourceShards(mapping.target.shards);
  validateTargetCoverage(shards, records, profile);
  return {records, target: {shards}};
}

async function validatedMapping(signal: AbortSignal | undefined, profile: SourceProfile): Promise<ValidatedSourceMap> {
  let mappingPromise = mappingPromises.get(profile.mapSha256);
  if (!mappingPromise) {
    mappingPromise = loadValidatedMapping(profile);
    mappingPromises.set(profile.mapSha256, mappingPromise);
    mappingPromise.catch(() => {
      if (mappingPromises.get(profile.mapSha256) === mappingPromise) mappingPromises.delete(profile.mapSha256);
    });
  }
  return waitForAbort(() => mappingPromise, signal);
}

type SourceStreamState = {length: number; lastReported: number; done: boolean};

function validateSourceResponseLength(response: Response, profile: SourceProfile): void {
  const declaredLength = response.headers.get('content-length');
  if (declaredLength !== null && (!/^\d+$/u.test(declaredLength) || Number(declaredLength) !== profile.sourceBytes)) {
    cancelBody(response.body);
    throw invalidAsset('The pinned ONNX external-data response has an unexpected Content-Length.');
  }
}

function reportSourceProgress(
  state: SourceStreamState,
  profile: SourceProfile,
  onProgress: ((loadedBytes: number) => void) | undefined
): void {
  if (state.length === profile.sourceBytes || state.length - state.lastReported >= profile.progressChunkBytes) {
    state.lastReported = state.length;
    onProgress?.(state.length);
  }
}

function appendSourceChunk(
  chunk: Uint8Array,
  buffer: Uint8Array,
  state: SourceStreamState,
  onProgress: ((loadedBytes: number) => void) | undefined,
  profile: SourceProfile
): void {
  if (!ArrayBuffer.isView(chunk) || chunk.BYTES_PER_ELEMENT !== 1)
    throw invalidAsset('The pinned ONNX external-data stream did not return bytes.');
  if (chunk.byteLength > profile.sourceBytes - state.length)
    throw new ResponseSizeError('ONNX external-data body exceeds its pinned size.');
  buffer.set(chunk, state.length);
  state.length += chunk.byteLength;
  reportSourceProgress(state, profile, onProgress);
}

async function readSourceBytes(
  response: Response,
  signal: AbortSignal | undefined,
  onProgress: ((loadedBytes: number) => void) | undefined,
  profile: SourceProfile
): Promise<ArrayBuffer> {
  validateSourceResponseLength(response, profile);
  const body = response.body;
  if (!body) throw invalidAsset('The pinned ONNX external-data response has no body.');
  const reader = body.getReader();
  const bytes = new ArrayBuffer(profile.sourceBytes);
  const buffer = new Uint8Array(bytes);
  const state: SourceStreamState = {length: 0, lastReported: 0, done: false};
  try {
    for (;;) {
      const part = await waitForAbort(() => reader.read(), signal);
      if (part.done) {
        state.done = true;
        break;
      }
      appendSourceChunk(part.value, buffer, state, onProgress, profile);
    }
    signal?.throwIfAborted();
    if (state.length !== profile.sourceBytes)
      throw new ResponseSizeError('ONNX external-data body is shorter than its pinned size.');
    const digest = await waitForAbort(() => sha256(bytes), signal);
    signal?.throwIfAborted();
    if (digest !== profile.sourceSha256)
      throw invalidAsset('The pinned ONNX external-data source failed SHA-256 verification.');
    return bytes;
  } finally {
    if (!state.done) cancelBody(reader);
    try {
      reader.releaseLock();
    } catch {}
  }
}

function hasErrorName(value: unknown, name: string): boolean {
  if (value === null || (typeof value !== 'object' && typeof value !== 'function')) return false;
  return (value as PropertyRecord).name === name;
}

async function fetchAndVerifySource(
  signal: AbortSignal,
  onProgress: ((loadedBytes: number) => void) | undefined,
  profile: SourceProfile
): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await waitForAbort(
      () => profile.fetchSource(profile.sourceUrl, {signal, credentials: 'omit', cache: 'no-store'}),
      signal,
      (lateResponse: Response | undefined) => cancelBody(lateResponse?.body)
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not fetch pinned Kev source weights.', {cause});
  }
  if (!response.ok) {
    cancelBody(response.body);
    signal?.throwIfAborted();
    throw new WebDecisionError(
      'ASSET_LOAD_FAILED',
      `Could not fetch pinned Kev source weights: HTTP ${response.status}.`
    );
  }
  try {
    return await readSourceBytes(response, signal, onProgress, profile);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || hasErrorName(cause, 'AbortError')) throw cause;
    throw invalidAsset('Could not verify the pinned Kev source weights.', cause);
  }
}

function acquireSourceLease(profile: SourceProfile): SourceLease {
  const sourceKey = `${profile.sourceUrl}\n${profile.sourceSha256}`;
  let entry = sourceLoads.get(sourceKey);
  if (!entry) {
    const controller = new AbortController();
    const createdEntry: SourceLoadEntry = {
      controller,
      references: 0,
      settled: false,
      promise: Promise.resolve(new ArrayBuffer(0)),
      progressCallbacks: new Set()
    };
    createdEntry.promise = fetchAndVerifySource(
      controller.signal,
      (loadedBytes) => {
        createdEntry.loadedBytes = loadedBytes;
        for (const callback of createdEntry.progressCallbacks) callback(loadedBytes);
      },
      profile
    ).finally(() => {
      createdEntry.settled = true;
      if (createdEntry.references === 0 && sourceLoads.get(sourceKey) === createdEntry) sourceLoads.delete(sourceKey);
    });
    entry = createdEntry;
    entry.promise.catch(() => {});
    sourceLoads.set(sourceKey, entry);
  }
  entry.references++;
  let released = false;
  return {
    read(signal, onProgress) {
      let callbackFailure: unknown;
      const sourceProgress: SourceProgressCallback | null = onProgress
        ? (loadedBytes) => {
            if (released || signal?.aborted) return;
            try {
              onProgress(loadedBytes);
            } catch (cause) {
              callbackFailure = cause;
            }
          }
        : null;
      if (sourceProgress) {
        entry.progressCallbacks.add(sourceProgress);
        // The source transfer is complete before the first shard response is
        // consumed. Replaying its final byte count for every later shard would
        // move progress back from asset verification to the source phase.
        if (entry.loadedBytes && !entry.settled) sourceProgress(entry.loadedBytes);
      }
      return waitForAbort(() => entry.promise, signal)
        .then((bytes) => {
          if (callbackFailure) throw callbackFailure;
          return bytes;
        })
        .finally(() => {
          if (sourceProgress) entry.progressCallbacks.delete(sourceProgress);
        });
    },
    release() {
      if (released) return;
      released = true;
      entry.references--;
      if (entry.references > 0) return;
      if (!entry.settled)
        entry.controller.abort(new DOMException('No Kev loads still need the source data.', 'AbortError'));
      if (sourceLoads.get(sourceKey) === entry) sourceLoads.delete(sourceKey);
    }
  };
}

function readBundledAssetExport(module: unknown, name: string): unknown {
  return isRecord(module) ? module[name] : undefined;
}

function createKevReconstructedSourceFor(
  defaults: SourceDefaults,
  overrides: SourceOverrides,
  validateOptions: (options: unknown) => number
): KevReconstructedSource {
  const {bundledUrls: providedBundledUrls, ...sourceOverrides} = overrides;
  const profile: SourceProfile = {
    ...defaults,
    bundledUrls:
      providedBundledUrls === undefined ? undefined : requireBundledDataUrls(providedBundledUrls, defaults.displayName),
    fetchSource: (url, init) => fetch(url, init),
    preflight: (signal, gpu) => requestF16Adapter(gpu, signal),
    openCache: (enabled, signal) => openAssetCache(enabled, signal),
    ...sourceOverrides
  };
  let signal: AbortSignal | undefined;
  let sourceLease: SourceLease | undefined;
  let mapping: ValidatedSourceMap | undefined;
  let onProgress: ((progress: ModelProgress) => void) | undefined;
  let bundledUrlsPromise: Promise<Readonly<Record<string, string>>> | undefined;

  async function getBundledUrls(requestSignal: AbortSignal | undefined): Promise<Readonly<Record<string, string>>> {
    if (profile.bundledUrls) return profile.bundledUrls;
    if (!bundledUrlsPromise) {
      const pending = Promise.resolve()
        .then(() => profile.loadBundledAssets())
        .then((module) =>
          requireBundledDataUrls(readBundledAssetExport(module, profile.bundledAssetExport), profile.displayName)
        );
      bundledUrlsPromise = pending;
      pending.catch(() => {
        if (bundledUrlsPromise === pending) bundledUrlsPromise = undefined;
      });
    }
    const pending = bundledUrlsPromise;
    return waitForAbort(() => pending, requestSignal);
  }

  async function prepare(options: KevSourceOptions): Promise<PreparedSource> {
    const maxPrefetchBytes = validateOptions(options);
    signal = options.signal;
    signal?.throwIfAborted();
    const gpu = requireKevWebGpu();
    if (!globalThis.crypto?.subtle)
      throw new WebDecisionError(
        'UNSUPPORTED_ENVIRONMENT',
        `Web Crypto is required to verify the ${profile.displayName} source.`
      );
    mapping = await validatedMapping(signal, profile);
    await profile.preflight(signal, gpu);
    const cache = await profile.openCache(options.cache !== false, signal);
    signal?.throwIfAborted();
    const manifestUrl = assetUrl(profile.cacheBase, 'manifest.json');
    const response = new Response(JSON.stringify(profile.manifest), {
      headers: {'content-type': 'application/json'}
    });
    return {base: profile.cacheBase, cache, signal, manifestUrl, response, maxPrefetchBytes};
  }

  async function fetchAsset(url: string | URL, init: Pick<RequestInit, 'signal'> = {}): Promise<Response> {
    const requestSignal = init.signal ?? signal;
    let parsed;
    try {
      parsed = new URL(url);
    } catch (cause) {
      throw invalidAsset(`Invalid ${profile.displayName} asset URL.`, cause);
    }
    const base = profile.cacheBase;
    if (
      parsed.origin !== base.origin ||
      !parsed.pathname.startsWith(base.pathname) ||
      parsed.username ||
      parsed.password ||
      parsed.search ||
      parsed.hash ||
      parsed.href !== `${parsed.origin}${parsed.pathname}`
    )
      throw invalidAsset(`Unexpected ${profile.displayName} asset URL.`);
    const relative = parsed.pathname.slice(profile.cacheBase.pathname.length);
    if (BUNDLED_ASSETS.has(relative)) {
      const bundledUrls = await getBundledUrls(requestSignal);
      const bundledUrl = bundledUrls[relative];
      if (!bundledUrl) throw invalidAsset(`Missing ${profile.displayName} asset: ${relative}`);
      if (bundledUrl.startsWith('data:')) return responseFromDataUrl(bundledUrl, requestSignal);
      return fetch(bundledUrl, {
        ...(requestSignal ? {signal: requestSignal} : {}),
        credentials: 'omit'
      });
    }
    const match = /^weights\/params_shard_(\d+)\.bin$/u.exec(relative);
    if (!match) throw invalidAsset(`Unknown ${profile.displayName} asset: ${relative}`);
    const shardIndex = Number(match[1]);
    const shard = mapping!.target.shards[shardIndex];
    if (!shard || shard.path !== relative) throw invalidAsset(`Unknown ${profile.displayName} shard: ${relative}`);
    sourceLease ??= acquireSourceLease(profile);
    const sourceBytes = await sourceLease.read(requestSignal, (loadedBytes) => {
      if (!requestSignal?.aborted)
        onProgress?.({
          loadedBytes,
          totalBytes: profile.sourceBytes,
          fraction: loadedBytes / profile.sourceBytes,
          phase: 'upstream-source'
        });
    });
    const records = mapping!.records
      .filter((record) => record.shardIndex === shardIndex)
      .sort((left, right) => left.targetOffset - right.targetOffset);
    let index = 0;
    let targetCursor = 0;
    const body = new ReadableStream({
      pull(controller) {
        requestSignal?.throwIfAborted();
        const record = records[index++];
        if (!record) {
          if (targetCursor !== shard.bytes) {
            controller.error(invalidAsset(`Incomplete source map while building ${relative}.`));
            return;
          }
          controller.close();
          return;
        }
        if (record.targetOffset !== targetCursor) {
          controller.error(invalidAsset(`Invalid target offset while building ${relative}.`));
          return;
        }
        targetCursor += record.bytes;
        controller.enqueue(new Uint8Array(sourceBytes, record.sourceOffset, record.bytes));
      },
      cancel() {}
    });
    return new Response(body, {
      status: 200,
      headers: {'content-type': 'application/octet-stream', 'content-length': String(shard.bytes)}
    });
  }

  return {
    manifest: profile.manifest,
    prepare,
    fetch: fetchAsset,
    bundledProgressPhase: 'bundled-assets',
    setProgress(callback, _loadedBytes) {
      onProgress = callback;
    },
    assetProgressPhase: 'asset-verification',
    async dispose() {
      sourceLease?.release();
      sourceLease = undefined;
      onProgress = undefined;
      mapping = undefined;
    }
  };
}

export function createKevReconstructedSource(overrides: SourceOverrides = {}): KevReconstructedSource {
  return createKevReconstructedSourceFor(
    {
      displayName: 'experimental Kev',
      bundledAssetExport: 'experimentalKevBundledUrls',
      manifest: candidateManifest,
      sourceShardMapText,
      sourceUrl: SOURCE_URL,
      sourceBytes: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_BYTES,
      sourceSha256: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256,
      mapSha256: EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256,
      repository: EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
      revision: EXPERIMENTAL_KEV_UPSTREAM_REVISION,
      sourcePath: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH,
      parameterRegionStart: EXPERIMENTAL_KEV_UPSTREAM_PARAMETER_REGION_START,
      tensorCount: EXPERIMENTAL_KEV_UPSTREAM_TENSOR_COUNT,
      progressChunkBytes: 16 * 1024 * 1024,
      loadBundledAssets: () => import('./bundled-assets.js'),
      cacheBase: EXPERIMENTAL_CACHE_BASE
    },
    overrides,
    validateExperimentalLoadOptions
  );
}

export function createKev06StableReconstructedSource(overrides: SourceOverrides = {}): KevReconstructedSource {
  return createKevReconstructedSourceFor(
    {
      displayName: 'stable Kev',
      bundledAssetExport: 'stableKevBundledUrls',
      manifest: stableManifest,
      sourceShardMapText,
      sourceUrl: SOURCE_URL,
      sourceBytes: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_BYTES,
      sourceSha256: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256,
      mapSha256: EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256,
      repository: EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
      revision: EXPERIMENTAL_KEV_UPSTREAM_REVISION,
      sourcePath: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH,
      parameterRegionStart: EXPERIMENTAL_KEV_UPSTREAM_PARAMETER_REGION_START,
      tensorCount: EXPERIMENTAL_KEV_UPSTREAM_TENSOR_COUNT,
      progressChunkBytes: 16 * 1024 * 1024,
      loadBundledAssets: () => import('./stable-bundled-assets.js'),
      cacheBase: STABLE_CACHE_BASE
    },
    overrides,
    validateStableLoadOptions
  );
}

export async function readExperimentalKevManifest(
  response: Response,
  signal?: AbortSignal
): Promise<PinnedKevManifest> {
  const value: unknown = await readPinnedManifest(
    response,
    signal,
    EXPERIMENTAL_KEV_UPSTREAM_MANIFEST_SHA256,
    'experimental Kev'
  );
  if (!isSourceManifest(value) || !isRecord(value.checkpoint))
    throw invalidAsset('The experimental Kev manifest does not match its candidate WASM pin.');
  if (
    !value.experimental ||
    value.checkpoint.weightSha256 !== EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256 ||
    !isAssetMetadata(value.files['model.wasm']) ||
    value.files['model.wasm'].sha256 !== EXPERIMENTAL_KEV_UPSTREAM_WASM_SHA256
  )
    throw invalidAsset('The experimental Kev manifest does not match its candidate WASM pin.');
  return value;
}

export function setExperimentalKevProgress(
  source: KevReconstructedSource,
  callback: (progress: ModelProgress) => void,
  loadedBytes = 0
): void {
  source.setProgress(callback, loadedBytes);
}
