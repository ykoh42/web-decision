import {WebDecisionError} from './kev.js';
import {waitForAbort} from './abort.js';
import {
  cancelBody,
  cancelResponse,
  MAX_ASSET_BYTES,
  MAX_MANIFEST_BYTES,
  MAX_VERIFIED_ASSET_BYTES,
  readBoundedResponse,
  ResponseSizeError
} from './bounded-response.js';

export {waitForAbort} from './abort.js';

export const CACHE_NAME = 'web-decision-assets-v1';
const error = (message) => new WebDecisionError('INVALID_ASSET', message);

export function assetUrl(base, path) {
  if (typeof path !== 'string' || !path || path.startsWith('/') || /[\\?#]/.test(path))
    throw error('Invalid model asset path.');
  // A bare ? or # is absent from search/hash but remains in href. Remove only
  // those empty delimiters on a copy before checking the directory prefix.
  const directory = new URL(base.href);
  if (!directory.search) directory.search = '';
  if (!directory.hash) directory.hash = '';
  const url = new URL(path, directory);
  if (url.origin !== directory.origin || !url.href.startsWith(directory.href) || url.username || url.password)
    throw error('Model asset path leaves the configured directory.');
  return url.href;
}

export function validateManifest(manifest) {
  if (
    !manifest ||
    manifest.formatVersion !== 1 ||
    manifest.model !== 'kev-0.6b-q4f16' ||
    manifest.backend !== 'tvm-webgpu' ||
    Object.hasOwn(manifest, 'maxInputTokens') ||
    manifest.maxStateTokens !== 8192 ||
    manifest.maxBranchTokens !== 8192 ||
    manifest.parameterCount !== 394 ||
    manifest.checkpoint?.revision !== '5a827091b3d58fe4fdfc02a5a5b2ce364047d678'
  ) {
    throw error('The asset manifest does not match the supported Kev Q4 model.');
  }
  if (!manifest.files || typeof manifest.files !== 'object') throw error('The asset manifest has no files.');
  for (const name of [
    'model.wasm',
    'tokenizer.json',
    'tokenizer_config.json',
    'model_config.json',
    'weights/tensor-cache.json'
  ]) {
    if (!manifest.files[name]) throw error(`Missing asset: ${name}`);
  }
  for (const [name, entry] of Object.entries(manifest.files)) {
    if (!entry || !/^[0-9a-f]{64}$/.test(entry.sha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0) {
      throw error(`Invalid asset metadata: ${name}`);
    }
  }
  return manifest;
}

/** Parse the manifest without turning a cancelled response body into a format error. */
export async function readManifestResponse(response, signal) {
  try {
    const value = parseJson(await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal}), 'manifest.json');
    signal?.throwIfAborted();
    return validateManifest(value);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Invalid model manifest.', {cause});
  }
}

export async function sha256(bytes) {
  if (!globalThis.crypto?.subtle)
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'Web Crypto is required to verify model files. Use HTTPS or localhost.'
    );
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

export async function openAssetCache(enabled, signal) {
  signal?.throwIfAborted();
  if (!enabled || !globalThis.caches) return null;
  try {
    return await waitForAbort(() => caches.open(CACHE_NAME), signal);
  } catch {
    signal?.throwIfAborted();
    return null;
  }
}

/** Serialize writes to one manifest while allowing a cancelled caller to stop waiting. */
export async function commitCachedManifest(cache, url, manifest, signal, locks = globalThis.navigator?.locks) {
  signal?.throwIfAborted();
  const write = () =>
    cache.put(url, new Response(JSON.stringify(manifest), {headers: {'content-type': 'application/json'}}));
  if (!locks || typeof locks.request !== 'function') {
    // Without a cross-context lock, wait for the write even after cancellation:
    // otherwise a later, serialized load could be overwritten by this write.
    await write();
    signal?.throwIfAborted();
    return;
  }
  const options = signal ? {mode: 'exclusive', signal} : {mode: 'exclusive'};
  return waitForAbort(
    () =>
      locks.request(`web-decision-manifest-v1:${url}`, options, async () => {
        signal?.throwIfAborted();
        // Keep the browser lock until the underlying Cache API operation settles,
        // even if the initiating caller has stopped waiting for it.
        await write();
      }),
    signal
  );
}

async function writeAssetCache(cache, key, buffer, signal) {
  if (!cache) return false;
  try {
    await waitForAbort(() => cache.put(key, new Response(buffer)), signal);
    return true;
  } catch {
    signal?.throwIfAborted();
    return false;
  }
}

async function checkedCacheEntry(cache, key, metadata, signal) {
  try {
    const cached = await waitForAbort(() => cache.match(key), signal, cancelResponse);
    if (!cached) return null;
    const buffer = await readBoundedResponse(cached, metadata.bytes, {exact: true, signal});
    return (await waitForAbort(() => sha256(buffer), signal)) === metadata.sha256 ? buffer : null;
  } catch {
    signal?.throwIfAborted();
    return null;
  }
}

async function findCachedAsset(cache, url, sharedKey, metadata, signal) {
  if (!cache) return null;
  const previousKey = new URL(url);
  previousKey.searchParams.set('__web_decision_sha256', metadata.sha256);
  // Do not delete bad or legacy entries: another load may have repaired one,
  // and older manifests can still depend on them.
  for (const [key, shared] of [
    [sharedKey, true],
    [previousKey.href, false],
    [url, false]
  ]) {
    const buffer = await checkedCacheEntry(cache, key, metadata, signal);
    if (buffer) return {buffer, cachedByHash: shared || (await writeAssetCache(cache, sharedKey, buffer, signal))};
  }
  return null;
}

async function fetchVerifiedAsset(url, name, metadata, signal) {
  let response;
  try {
    response = await waitForAbort(() => fetch(url, {signal}), signal, cancelResponse);
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not fetch ${name}.`, {cause});
  }
  if (!response.ok) {
    cancelBody(response.body);
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not fetch ${name}: HTTP ${response.status}.`);
  }
  let buffer;
  try {
    buffer = await readBoundedResponse(response, metadata.bytes, {exact: true, signal});
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof ResponseSizeError) throw error(`Integrity check failed for ${name}.`);
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not read ${name}.`, {cause});
  }
  let digest;
  try {
    digest = await waitForAbort(() => sha256(buffer), signal);
  } catch (cause) {
    signal?.throwIfAborted();
    throw cause;
  }
  signal?.throwIfAborted();
  if (digest !== metadata.sha256) throw error(`Integrity check failed for ${name}.`);
  return buffer;
}

function reportAssetRead(result, {signal, onBytes, onCacheStatus}) {
  try {
    onCacheStatus?.(result.cachedByHash);
    signal?.throwIfAborted();
    onBytes?.(result.buffer.byteLength);
    signal?.throwIfAborted();
  } catch (cause) {
    signal?.throwIfAborted();
    throw cause;
  }
}

/** Verify assets and share identical content between model directories on one asset origin. */
export async function readVerifiedAsset(
  base,
  manifest,
  name,
  {cache, signal, onBytes, onCacheStatus, offline = false, maxAssetBytes = MAX_ASSET_BYTES} = {}
) {
  const metadata = manifest.files[name];
  if (!metadata) throw error(`Asset not listed in manifest: ${name}`);
  if (
    !Number.isSafeInteger(maxAssetBytes) ||
    maxAssetBytes < 0 ||
    maxAssetBytes > MAX_VERIFIED_ASSET_BYTES ||
    !Number.isSafeInteger(metadata.bytes) ||
    metadata.bytes < 0 ||
    metadata.bytes > maxAssetBytes
  )
    throw error(`Invalid model asset size: ${name}.`);
  const url = assetUrl(base, name);
  // This key is never fetched. Its origin prevents sharing between hosts.
  const cacheKey = new URL('/.web-decision-cache/sha256/', url);
  cacheKey.searchParams.set('digest', metadata.sha256);
  signal?.throwIfAborted();
  let result = await findCachedAsset(cache, url, cacheKey.href, metadata, signal);
  signal?.throwIfAborted();
  if (!result) {
    if (offline) throw new WebDecisionError('ASSET_LOAD_FAILED', `Verified asset is not available offline: ${name}.`);
    const buffer = await fetchVerifiedAsset(url, name, metadata, signal);
    result = {buffer, cachedByHash: await writeAssetCache(cache, cacheKey.href, buffer, signal)};
  }
  signal?.throwIfAborted();
  reportAssetRead(result, {signal, onBytes, onCacheStatus});
  return result.buffer;
}

export function parseJson(bytes, name) {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', `Invalid JSON asset: ${name}.`, {cause});
  }
}
