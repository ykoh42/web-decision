import {cancelBody, cancelResponse} from '../bounded-response.js';
import {assetUrl, openAssetCache, waitForAbort} from '../assets.js';
import {requestF16Adapter} from '../gpu-preflight.js';
import {WebDecisionError} from '../kev.js';
import {validatePrefetchBytes} from '../shard-prefetch.js';

type DirectSourceOptions = {
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal | undefined;
  readonly maxPrefetchBytes?: number;
};

type DirectPreparedSource = {
  readonly base: URL;
  readonly cache: Cache | null;
  readonly signal?: AbortSignal | undefined;
  readonly manifestUrl: string;
  readonly response: Response;
  readonly maxPrefetchBytes: number;
};

export function resolveDirectKevAssetUrl(root: string, name: string): string {
  let directory: URL;
  try {
    directory = new URL(root);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'The direct Kev artifact root is invalid.', {cause});
  }
  const segments = name.split('/');
  if (
    directory.protocol !== 'https:' ||
    !directory.pathname.endsWith('/') ||
    directory.search ||
    directory.hash ||
    directory.username ||
    directory.password ||
    !name ||
    name.startsWith('/') ||
    /[\\?#%]/u.test(name) ||
    segments.some((segment) => !segment || segment === '.' || segment === '..')
  )
    throw new WebDecisionError('INVALID_ASSET', 'The direct Kev artifact path is invalid.');

  const relativePath = segments.map(encodeURIComponent).join('/');
  const url = new URL(relativePath, directory);
  if (url.origin !== directory.origin || !url.pathname.startsWith(directory.pathname))
    throw new WebDecisionError('INVALID_ASSET', 'The direct Kev artifact path leaves its pinned directory.');
  return url.href;
}

function packageCacheBase(manifestSha256: string): URL {
  const origin = globalThis.location?.origin;
  if (!origin)
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'The direct Kev package requires a browser page origin.');
  return new URL(`/.web-decision-package-cache/kev-0.6b-q4f16/${manifestSha256}/`, origin);
}

async function readPackageManifest(
  url: string,
  offline: boolean,
  cache: Cache | null,
  manifestUrl: string,
  signal: AbortSignal | undefined
): Promise<Response> {
  if (offline) {
    if (!cache) throw new WebDecisionError('ASSET_LOAD_FAILED', 'The model cache is unavailable for offline loading.');
    let response;
    try {
      response = await waitForAbort(() => cache.match(manifestUrl), signal, cancelResponse);
    } catch (cause) {
      signal?.throwIfAborted();
      throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not read the cached model manifest.', {cause});
    }
    if (!response) {
      signal?.throwIfAborted();
      throw new WebDecisionError(
        'ASSET_LOAD_FAILED',
        'The model manifest is not available offline. Load the model online first.'
      );
    }
    return response;
  }

  let response;
  try {
    response = await waitForAbort(
      () => fetch(url, {...(signal ? {signal} : {}), cache: 'no-cache'}),
      signal,
      cancelResponse
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not fetch the direct Kev package manifest.', {cause});
  }
  if (!response.ok) {
    cancelBody(response.body);
    signal?.throwIfAborted();
    throw new WebDecisionError(
      'ASSET_LOAD_FAILED',
      `Could not fetch the direct Kev package manifest: HTTP ${response.status}.`
    );
  }
  return response;
}

/** Fetch direct Kev artifacts only from the curated model's pinned HTTPS directory. */
export function createDirectKevPackageSource(manifestSha256: string, directArtifactRoot: string) {
  let cacheBase: URL | undefined;
  const prepare = async (options: DirectSourceOptions): Promise<DirectPreparedSource> => {
    if (options.offline !== undefined && typeof options.offline !== 'boolean')
      throw new WebDecisionError('INVALID_INPUT', 'offline must be a boolean.');
    if (options.offline && options.cache === false)
      throw new WebDecisionError('INVALID_INPUT', 'Offline loading requires the model cache.');
    const maxPrefetchBytes = validatePrefetchBytes(options.maxPrefetchBytes);
    const signal = options.signal;
    signal?.throwIfAborted();
    if (typeof navigator === 'undefined' || !navigator.gpu)
      throw new WebDecisionError(
        'UNSUPPORTED_ENVIRONMENT',
        'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'
      );
    if (!globalThis.crypto?.subtle)
      throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Web Crypto is unavailable. Use HTTPS or localhost.');
    await requestF16Adapter(navigator.gpu, signal);
    const cache = await openAssetCache(options.cache !== false, signal);
    signal?.throwIfAborted();
    cacheBase = packageCacheBase(manifestSha256);
    const manifestUrl = assetUrl(cacheBase, 'manifest.json');
    const response = await readPackageManifest(
      resolveDirectKevAssetUrl(directArtifactRoot, 'manifest.json'),
      options.offline === true,
      cache,
      manifestUrl,
      signal
    );
    return {
      base: cacheBase,
      cache,
      ...(signal ? {signal} : {}),
      manifestUrl,
      response,
      maxPrefetchBytes
    };
  };

  const fetchAsset = async (url: string, init?: {readonly signal?: AbortSignal | undefined}): Promise<Response> => {
    if (!cacheBase) throw new WebDecisionError('INVALID_ASSET', 'The direct Kev package source is not prepared.');
    const basePath = cacheBase.pathname;
    const requestUrl = new URL(url);
    if (requestUrl.origin !== cacheBase.origin || !requestUrl.pathname.startsWith(basePath))
      throw new WebDecisionError('INVALID_ASSET', 'The direct Kev asset URL leaves its package cache namespace.');
    const name = decodeURIComponent(requestUrl.pathname.slice(basePath.length));
    const artifactUrl = resolveDirectKevAssetUrl(directArtifactRoot, name);
    return fetch(artifactUrl, init?.signal ? {signal: init.signal} : {});
  };

  return {
    prepare,
    fetch: fetchAsset,
    bundledProgressPhase: 'bundled-assets' as const,
    assetProgressPhase: 'asset-verification' as const
  };
}
