import candidateManifest from '../../assets/experimental-kev-upstream/manifest.json' with {type: 'json'};
import sourceShardMapText from '../../assets/experimental-kev-upstream/source-shard-map.json?raw';
import {cancelBody, MAX_ASSET_BYTES, ResponseSizeError} from '../bounded-response.js';
import {assetUrl, openAssetCache, sha256, waitForAbort} from '../assets.js';
import {requestF16Adapter} from '../gpu-preflight.js';
import {WebDecisionError} from '../kev.js';
import {readPinnedManifest} from '../pinned-manifest.js';
import {validatePrefetchBytes} from '../shard-prefetch.js';
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

export const EXPERIMENTAL_KEV_MODEL = EXPERIMENTAL_KEV_UPSTREAM_MODEL;

const MAX_PREFETCH_BYTES = 0;
const CACHE_BASE = new URL('https://kev-assets.web-decision.invalid/experimental-kev-upstream/');
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

const sourceLoads = new Map();
const mappingPromises = new Map();

function invalidAsset(message, cause) {
  return new WebDecisionError('INVALID_ASSET', message, cause ? {cause} : undefined);
}

function requireBundledDataUrls(value) {
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    Object.entries(value).some(
      ([key, url]) => typeof key !== 'string' || typeof url !== 'string' || !url.startsWith('data:')
    )
  )
    throw invalidAsset('The experimental Kev bundle is missing pinned inline assets.');
  return Object.freeze({...value});
}

function requireExperimentalWebGpu() {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu)
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'
    );
  return gpu;
}

function responseFromDataUrl(value, signal) {
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

function validateLoadOptions(options) {
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
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw new WebDecisionError('INVALID_INPUT', 'Experimental Kev load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !allowed.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new WebDecisionError('INVALID_INPUT', 'Unsupported experimental Kev load option.');
  }
  if (
    options.model !== EXPERIMENTAL_KEV_MODEL ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache === false) ||
    (options.signal != null &&
      (typeof options.signal.throwIfAborted !== 'function' ||
        typeof options.signal.addEventListener !== 'function' ||
        typeof options.signal.removeEventListener !== 'function')) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    validatePrefetchBytes(options.maxPrefetchBytes) !== MAX_PREFETCH_BYTES
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid experimental Kev load option values.');
}

async function validatedMapping(signal, profile) {
  let mappingPromise = mappingPromises.get(profile.mapSha256);
  if (!mappingPromise) {
    mappingPromise = (async () => {
      if ((await sha256(new TextEncoder().encode(profile.sourceShardMapText))) !== profile.mapSha256)
        throw invalidAsset('The experimental Kev source map does not match its pinned digest.');
      const mapping = JSON.parse(profile.sourceShardMapText);
      if (
        mapping.source?.repository !== profile.repository ||
        mapping.source?.revision !== profile.revision ||
        mapping.source?.path !== profile.sourcePath ||
        mapping.source?.bytes !== profile.sourceBytes ||
        mapping.source?.sha256 !== profile.sourceSha256 ||
        mapping.source?.parameterRegionStart !== profile.parameterRegionStart ||
        !Array.isArray(mapping.records) ||
        mapping.records.length !== profile.tensorCount ||
        !Array.isArray(mapping.target?.shards) ||
        !mapping.target.shards.length
      )
        throw invalidAsset('The experimental Kev source map metadata is invalid.');

      const sourceRecords = [...mapping.records].sort((left, right) => left.sourceOffset - right.sourceOffset);
      let sourceCursor = profile.parameterRegionStart;
      for (const record of sourceRecords) {
        if (
          !Number.isSafeInteger(record.sourceOffset) ||
          !Number.isSafeInteger(record.targetOffset) ||
          !Number.isSafeInteger(record.bytes) ||
          record.bytes < 1 ||
          record.sourceOffset !== sourceCursor ||
          record.sourceOffset + record.bytes > profile.sourceBytes ||
          !Number.isSafeInteger(record.shardIndex) ||
          record.shardIndex < 0 ||
          record.shardIndex >= mapping.target.shards.length
        )
          throw invalidAsset('The experimental Kev source map has a source gap or overlap.');
        sourceCursor += record.bytes;
      }
      if (sourceCursor !== profile.sourceBytes)
        throw invalidAsset('The experimental Kev source map has incomplete coverage.');

      for (let shardIndex = 0; shardIndex < mapping.target.shards.length; shardIndex++) {
        const shard = mapping.target.shards[shardIndex];
        const metadata = profile.manifest.files[shard.path];
        if (!metadata || metadata.bytes !== shard.bytes || metadata.sha256 !== shard.sha256)
          throw invalidAsset(`The source map does not match pinned target metadata for ${shard.path}.`);
        const records = mapping.records
          .filter((record) => record.shardIndex === shardIndex)
          .sort((left, right) => left.targetOffset - right.targetOffset);
        let targetCursor = 0;
        for (const record of records) {
          if (record.targetOffset !== targetCursor)
            throw invalidAsset(`The source map has a target gap or overlap for ${shard.path}.`);
          targetCursor += record.bytes;
        }
        if (targetCursor !== shard.bytes)
          throw invalidAsset(`The source map has incomplete coverage for ${shard.path}.`);
      }
      return mapping;
    })();
    mappingPromises.set(profile.mapSha256, mappingPromise);
    mappingPromise.catch(() => {
      if (mappingPromises.get(profile.mapSha256) === mappingPromise) mappingPromises.delete(profile.mapSha256);
    });
  }
  return waitForAbort(() => mappingPromise, signal);
}

async function readSourceBytes(response, signal, onProgress, profile) {
  const declaredLength = response.headers.get('content-length');
  if (declaredLength !== null && (!/^\d+$/u.test(declaredLength) || Number(declaredLength) !== profile.sourceBytes)) {
    cancelBody(response.body);
    throw invalidAsset('The pinned ONNX external-data response has an unexpected Content-Length.');
  }
  const body = response.body;
  if (!body) throw invalidAsset('The pinned ONNX external-data response has no body.');
  const reader = body.getReader();
  const buffer = new Uint8Array(profile.sourceBytes);
  let length = 0;
  let lastReported = 0;
  let done = false;
  try {
    for (;;) {
      const part = await waitForAbort(() => reader.read(), signal);
      if (part.done) {
        done = true;
        break;
      }
      if (!ArrayBuffer.isView(part.value) || part.value.BYTES_PER_ELEMENT !== 1)
        throw invalidAsset('The pinned ONNX external-data stream did not return bytes.');
      if (part.value.byteLength > profile.sourceBytes - length)
        throw new ResponseSizeError('ONNX external-data body exceeds its pinned size.');
      buffer.set(part.value, length);
      length += part.value.byteLength;
      if (length === profile.sourceBytes || length - lastReported >= profile.progressChunkBytes) {
        lastReported = length;
        onProgress?.(length);
      }
    }
    signal?.throwIfAborted();
    if (length !== profile.sourceBytes)
      throw new ResponseSizeError('ONNX external-data body is shorter than its pinned size.');
    const bytes = buffer.buffer;
    const digest = await waitForAbort(() => sha256(bytes), signal);
    signal?.throwIfAborted();
    if (digest !== profile.sourceSha256)
      throw invalidAsset('The pinned ONNX external-data source failed SHA-256 verification.');
    return bytes;
  } finally {
    if (!done) cancelBody(reader);
    try {
      reader.releaseLock();
    } catch {}
  }
}

async function fetchAndVerifySource(signal, onProgress, profile) {
  let response;
  try {
    response = await waitForAbort(
      () => profile.fetchSource(profile.sourceUrl, {signal, credentials: 'omit', cache: 'no-store'}),
      signal,
      (lateResponse) => cancelBody(lateResponse?.body)
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
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw invalidAsset('Could not verify the pinned Kev source weights.', cause);
  }
}

function acquireSourceLease(profile) {
  const sourceKey = `${profile.sourceUrl}\n${profile.sourceSha256}`;
  let entry = sourceLoads.get(sourceKey);
  if (!entry) {
    const controller = new AbortController();
    entry = {controller, references: 0, settled: false, promise: undefined, progressCallbacks: new Set()};
    entry.promise = fetchAndVerifySource(
      controller.signal,
      (loadedBytes) => {
        entry.loadedBytes = loadedBytes;
        for (const callback of entry.progressCallbacks) callback(loadedBytes);
      },
      profile
    ).finally(() => {
      entry.settled = true;
      if (entry.references === 0 && sourceLoads.get(sourceKey) === entry) sourceLoads.delete(sourceKey);
    });
    entry.promise.catch(() => {});
    sourceLoads.set(sourceKey, entry);
  }
  entry.references++;
  let released = false;
  return {
    read(signal, onProgress) {
      let callbackFailure;
      const sourceProgress = onProgress
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

export function createKevReconstructedSource(overrides = {}) {
  const {bundledUrls: providedBundledUrls, ...sourceOverrides} = overrides;
  const profile = {
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
    bundledUrls: providedBundledUrls === undefined ? undefined : requireBundledDataUrls(providedBundledUrls),
    loadBundledAssets: () => import('./bundled-assets.js'),
    cacheBase: CACHE_BASE,
    fetchSource: (url, init) => fetch(url, init),
    preflight: (signal, gpu) => requestF16Adapter(gpu, signal),
    openCache: (enabled, signal) => openAssetCache(enabled, signal),
    ...sourceOverrides
  };
  let signal;
  let sourceLease;
  let mapping;
  let onProgress;
  let bundledUrlsPromise;

  async function getBundledUrls(requestSignal) {
    if (profile.bundledUrls) return profile.bundledUrls;
    if (!bundledUrlsPromise) {
      const pending = Promise.resolve()
        .then(() => profile.loadBundledAssets())
        .then((module) => requireBundledDataUrls(module.experimentalKevBundledUrls));
      bundledUrlsPromise = pending;
      pending.catch(() => {
        if (bundledUrlsPromise === pending) bundledUrlsPromise = undefined;
      });
    }
    return waitForAbort(() => bundledUrlsPromise, requestSignal);
  }

  async function prepare(options) {
    validateLoadOptions(options);
    signal = options.signal;
    signal?.throwIfAborted();
    const gpu = requireExperimentalWebGpu();
    if (!globalThis.crypto?.subtle)
      throw new WebDecisionError(
        'UNSUPPORTED_ENVIRONMENT',
        'Web Crypto is required to verify the experimental Kev source.'
      );
    mapping = await validatedMapping(signal, profile);
    await profile.preflight(signal, gpu);
    const cache = await profile.openCache(options.cache !== false, signal);
    signal?.throwIfAborted();
    const manifestUrl = assetUrl(profile.cacheBase, 'manifest.json');
    const response = new Response(JSON.stringify(profile.manifest), {
      headers: {'content-type': 'application/json'}
    });
    return {base: profile.cacheBase, cache, signal, manifestUrl, response, maxPrefetchBytes: MAX_PREFETCH_BYTES};
  }

  async function fetchAsset(url, init = {}) {
    const requestSignal = init.signal ?? signal;
    const parsed = new URL(url);
    const relative = parsed.pathname.slice(profile.cacheBase.pathname.length);
    if (BUNDLED_ASSETS.has(relative)) {
      const bundledUrls = await getBundledUrls(requestSignal);
      const bundledUrl = bundledUrls[relative];
      if (!bundledUrl) throw invalidAsset(`Missing experimental Kev asset: ${relative}`);
      if (bundledUrl.startsWith('data:')) return responseFromDataUrl(bundledUrl, requestSignal);
      return fetch(bundledUrl, {signal: requestSignal, credentials: 'omit'});
    }
    const match = /^weights\/params_shard_(\d+)\.bin$/u.exec(relative);
    if (!match) throw invalidAsset(`Unknown experimental Kev asset: ${relative}`);
    const shardIndex = Number(match[1]);
    const shard = mapping.target.shards[shardIndex];
    if (!shard || shard.path !== relative) throw invalidAsset(`Unknown experimental Kev shard: ${relative}`);
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
    const records = mapping.records
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
    setProgress(callback) {
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

export async function readExperimentalKevManifest(response, signal) {
  const manifest = await readPinnedManifest(
    response,
    signal,
    EXPERIMENTAL_KEV_UPSTREAM_MANIFEST_SHA256,
    'experimental Kev'
  );
  if (
    !manifest.experimental ||
    manifest.checkpoint?.weightSha256 !== EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256 ||
    manifest.files['model.wasm']?.sha256 !== EXPERIMENTAL_KEV_UPSTREAM_WASM_SHA256
  )
    throw invalidAsset('The experimental Kev manifest does not match its candidate WASM pin.');
  return manifest;
}

export function setExperimentalKevProgress(source, callback, loadedBytes = 0) {
  source.setProgress(callback, loadedBytes);
}
