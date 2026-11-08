import {assetUrl, openAssetCache, waitForAbort} from './assets.js';
import {cancelBody, cancelResponse} from './bounded-response.js';
import {requestF16Adapter} from './gpu-preflight.js';
import {WebDecisionError} from './kev.js';
import {validatePrefetchBytes} from './shard-prefetch.js';

interface ModelSourceOptions {
  readonly assetsUrl?: unknown;
  readonly offline?: unknown;
  readonly cache?: unknown;
  readonly maxPrefetchBytes?: unknown;
  readonly signal?: AbortSignal | undefined;
}

interface ModelSourceDependencies<Options extends ModelSourceOptions> {
  readonly validateModelOptions?: ((options: Options) => unknown) | undefined;
  readonly offlineMissingMessage?: string | undefined;
}

interface PreparedModelSource {
  readonly base: URL;
  readonly cache: Cache | null;
  readonly signal: AbortSignal | undefined;
  readonly manifestUrl: string;
  readonly response: Response;
  readonly maxPrefetchBytes: number;
  readonly modelOptions: unknown;
}

function validateSource<Options extends ModelSourceOptions>(
  options: Options,
  validateModelOptions?: (options: Options) => unknown
) {
  if (!options || typeof options.assetsUrl !== 'string' || !options.assetsUrl)
    throw new WebDecisionError('INVALID_INPUT', 'assetsUrl is required.');
  if (options.offline !== undefined && typeof options.offline !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'offline must be a boolean.');
  if (options.offline && options.cache === false)
    throw new WebDecisionError('INVALID_INPUT', 'Offline loading requires the model cache.');
  const maxPrefetchBytes = validatePrefetchBytes(options.maxPrefetchBytes);
  const modelOptions = validateModelOptions?.(options);
  if (typeof navigator === 'undefined' || !navigator.gpu)
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'
    );
  if (!globalThis.crypto?.subtle)
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Web Crypto is unavailable. Use HTTPS or localhost.');
  let base;
  try {
    base = new URL(options.assetsUrl as string, globalThis.location?.href);
  } catch (cause) {
    throw new WebDecisionError('INVALID_INPUT', 'Invalid assetsUrl.', {cause});
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.search || base.hash || base.username || base.password) {
    throw new WebDecisionError(
      'INVALID_INPUT',
      'assetsUrl must be an HTTP or HTTPS directory URL without credentials, query, or fragment.'
    );
  }
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {base, maxPrefetchBytes, modelOptions};
}

async function getManifestResponse(
  options: ModelSourceOptions,
  cache: Cache | null,
  manifestUrl: string,
  signal: AbortSignal | null | undefined,
  offlineMissingMessage: string | undefined
): Promise<Response> {
  if (options.offline) {
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
      throw new WebDecisionError('ASSET_LOAD_FAILED', offlineMissingMessage!);
    }
    return response;
  }
  let response;
  try {
    response = await waitForAbort(
      () => fetch(manifestUrl, {signal, cache: 'no-cache'} as RequestInit),
      signal,
      cancelResponse
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not fetch manifest.json.', {cause});
  }
  if (!response.ok) {
    cancelBody(response.body);
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not fetch manifest.json: HTTP ${response.status}.`);
  }
  return response;
}

/** Validate source options and obtain the matching manifest without creating a GPU device. */
export async function prepareModelSource<Options extends ModelSourceOptions>(
  options: Options,
  {validateModelOptions, offlineMissingMessage}: ModelSourceDependencies<Options>
): Promise<PreparedModelSource> {
  const sourceOptions = options;
  const {base, maxPrefetchBytes, modelOptions} = validateSource(sourceOptions, validateModelOptions);
  const signal = sourceOptions.signal;
  signal?.throwIfAborted();
  await requestF16Adapter(navigator.gpu, signal);
  const cache = await openAssetCache(sourceOptions.cache !== false, signal);
  signal?.throwIfAborted();
  const manifestUrl = assetUrl(base, 'manifest.json');
  const response = await getManifestResponse(sourceOptions, cache, manifestUrl, signal, offlineMissingMessage);
  return {
    base,
    cache,
    signal,
    manifestUrl,
    response,
    maxPrefetchBytes,
    modelOptions
  };
}
