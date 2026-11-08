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

export interface AssetFileMetadata {
  readonly bytes: number;
  readonly sha256: string;
}

export interface AssetFileManifest {
  readonly files: Readonly<Record<string, AssetFileMetadata>>;
}

export interface KevAssetManifest extends AssetFileManifest {
  readonly formatVersion: number;
  readonly model: string;
  readonly backend: string;
  readonly experimental?: unknown;
  readonly maxStateTokens: number;
  readonly maxBranchTokens: number;
  readonly parameterCount: number;
  readonly checkpoint: {readonly revision: string};
}

export interface AssetCache {
  match(key: string): Promise<Response | null | undefined>;
  put(key: string, response: Response): Promise<void>;
}

type NarrowAssetFetch = (url: string, init?: {readonly signal?: AbortSignal | undefined}) => Promise<Response>;

export interface ReadVerifiedAssetOptions {
  readonly cache?: AssetCache | null | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onBytes?: ((count: number) => void) | undefined;
  readonly onCacheStatus?: ((cached: boolean) => void) | undefined;
  readonly offline?: boolean | undefined;
  readonly maxAssetBytes?: number | undefined;
  readonly fetchImpl?: typeof fetch | NarrowAssetFetch | undefined;
}

export const CACHE_NAME = 'web-decision-assets-v1';
const error = (message: string) => new WebDecisionError('INVALID_ASSET', message);

function isAbortError(cause: unknown): boolean {
  return (
    ((typeof cause === 'object' && cause !== null) || typeof cause === 'function') &&
    (cause as {readonly name?: unknown}).name === 'AbortError'
  );
}

export function assetUrl(base: URL, path: string): string {
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

export function validateManifest(manifest: unknown): KevAssetManifest {
  const candidate = manifest as Record<string, unknown> | null | undefined;
  if (
    !manifest ||
    candidate?.formatVersion !== 1 ||
    candidate.model !== 'kev-0.6b-q4f16' ||
    candidate.backend !== 'tvm-webgpu' ||
    Object.hasOwn(manifest as object, 'maxInputTokens') ||
    candidate.maxStateTokens !== 8192 ||
    candidate.maxBranchTokens !== 8192 ||
    candidate.parameterCount !== 394 ||
    (candidate.checkpoint as {revision?: unknown} | null | undefined)?.revision !==
      '5a827091b3d58fe4fdfc02a5a5b2ce364047d678'
  ) {
    throw error('The asset manifest does not match the supported Kev Q4 model.');
  }
  const files = candidate.files as Record<string, unknown> | null | undefined;
  if (!files || typeof files !== 'object') throw error('The asset manifest has no files.');
  for (const name of [
    'model.wasm',
    'tokenizer.json',
    'tokenizer_config.json',
    'model_config.json',
    'weights/tensor-cache.json'
  ]) {
    if (!files[name]) throw error(`Missing asset: ${name}`);
  }
  for (const [name, entry] of Object.entries(files)) {
    const metadata = entry as AssetFileMetadata | null | undefined;
    if (
      !metadata ||
      !/^[0-9a-f]{64}$/.test(metadata.sha256) ||
      !Number.isSafeInteger(metadata.bytes) ||
      metadata.bytes < 0
    ) {
      throw error(`Invalid asset metadata: ${name}`);
    }
  }
  return manifest as KevAssetManifest;
}

/** Parse the manifest without turning a cancelled response body into a format error. */
export async function readManifestResponse(
  response: Pick<Response, 'body'>,
  signal?: AbortSignal | null
): Promise<KevAssetManifest> {
  try {
    const value = parseJson(
      await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal} as NonNullable<
        Parameters<typeof readBoundedResponse>[2]
      >),
      'manifest.json'
    );
    signal?.throwIfAborted();
    return validateManifest(value);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || isAbortError(cause)) throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Invalid model manifest.', {cause});
  }
}

export async function sha256(bytes: ArrayBuffer | ArrayBufferView<ArrayBufferLike>): Promise<string> {
  if (!globalThis.crypto?.subtle)
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'Web Crypto is required to verify model files. Use HTTPS or localhost.'
    );
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource)))
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

export async function openAssetCache(enabled: boolean | undefined, signal?: AbortSignal | null): Promise<Cache | null> {
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
export async function commitCachedManifest(
  cache: AssetCache | null | undefined,
  url: string,
  manifest: unknown,
  signal?: AbortSignal | null,
  locks: LockManager | null | undefined = globalThis.navigator?.locks
): Promise<void> {
  signal?.throwIfAborted();
  const write = () =>
    cache!.put(url, new Response(JSON.stringify(manifest) as string, {headers: {'content-type': 'application/json'}}));
  if (!locks || typeof locks.request !== 'function') {
    // Without a cross-context lock, wait for the write even after cancellation:
    // otherwise a later, serialized load could be overwritten by this write.
    await write();
    signal?.throwIfAborted();
    return;
  }
  const options: LockOptions = signal ? {mode: 'exclusive', signal} : {mode: 'exclusive'};
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

async function writeAssetCache(
  cache: AssetCache | null | undefined,
  key: string,
  buffer: ArrayBuffer,
  signal?: AbortSignal | null
): Promise<boolean> {
  if (!cache) return false;
  try {
    await waitForAbort(() => cache.put(key, new Response(buffer)), signal);
    return true;
  } catch {
    signal?.throwIfAborted();
    return false;
  }
}

async function checkedCacheEntry(
  cache: AssetCache,
  key: string,
  metadata: AssetFileMetadata,
  signal?: AbortSignal | null
): Promise<ArrayBuffer | null> {
  try {
    const cached = await waitForAbort(() => cache.match(key), signal, cancelResponse);
    if (!cached) return null;
    const buffer = await readBoundedResponse(cached, metadata.bytes, {exact: true, signal} as NonNullable<
      Parameters<typeof readBoundedResponse>[2]
    >);
    return (await waitForAbort(() => sha256(buffer), signal)) === metadata.sha256 ? buffer : null;
  } catch {
    signal?.throwIfAborted();
    return null;
  }
}

interface CachedAsset {
  readonly buffer: ArrayBuffer;
  readonly cachedByHash: boolean;
}

async function findCachedAsset(
  cache: AssetCache | null | undefined,
  url: string,
  sharedKey: string,
  metadata: AssetFileMetadata,
  signal?: AbortSignal | null
): Promise<CachedAsset | null> {
  if (!cache) return null;
  const previousKey = new URL(url);
  previousKey.searchParams.set('__web_decision_sha256', metadata.sha256);
  // Do not delete bad or legacy entries: another load may have repaired one,
  // and older manifests can still depend on them.
  const cacheKeys: ReadonlyArray<readonly [string, boolean]> = [
    [sharedKey, true],
    [previousKey.href, false],
    [url, false]
  ];
  for (const [key, shared] of cacheKeys) {
    const buffer = await checkedCacheEntry(cache, key, metadata, signal);
    if (buffer) return {buffer, cachedByHash: shared || (await writeAssetCache(cache, sharedKey, buffer, signal))};
  }
  return null;
}

async function fetchVerifiedAsset(
  url: string,
  name: string,
  metadata: AssetFileMetadata,
  signal: AbortSignal | null | undefined,
  fetchImpl: NonNullable<ReadVerifiedAssetOptions['fetchImpl']> = globalThis.fetch
): Promise<ArrayBuffer> {
  let response;
  try {
    const request = fetchImpl as (url: string, init: RequestInit) => Promise<Response>;
    response = await waitForAbort(() => request(url, {signal} as RequestInit), signal, cancelResponse);
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
    buffer = await readBoundedResponse(response, metadata.bytes, {exact: true, signal} as NonNullable<
      Parameters<typeof readBoundedResponse>[2]
    >);
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

function reportAssetRead(
  result: CachedAsset,
  options: Pick<ReadVerifiedAssetOptions, 'signal' | 'onBytes' | 'onCacheStatus'>
): void {
  const {signal, onBytes, onCacheStatus} = options;
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
  base: URL,
  manifest: AssetFileManifest,
  name: string,
  {
    cache,
    signal,
    onBytes,
    onCacheStatus,
    offline = false,
    maxAssetBytes = MAX_ASSET_BYTES,
    fetchImpl
  }: ReadVerifiedAssetOptions = {}
): Promise<ArrayBuffer> {
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
    const buffer = await fetchVerifiedAsset(url, name, metadata, signal, fetchImpl);
    result = {buffer, cachedByHash: await writeAssetCache(cache, cacheKey.href, buffer, signal)};
  }
  signal?.throwIfAborted();
  reportAssetRead(result, {signal, onBytes, onCacheStatus});
  return result.buffer;
}

export function parseJson(bytes: ArrayBuffer | ArrayBufferView<ArrayBufferLike>, name: string): unknown {
  try {
    return JSON.parse(new TextDecoder().decode(bytes as BufferSource));
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', `Invalid JSON asset: ${name}.`, {cause});
  }
}
