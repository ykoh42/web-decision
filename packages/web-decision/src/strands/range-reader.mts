import {assetUrl, sha256, waitForAbort, commitCachedManifest} from '../assets.js';
import type {AssetCache} from '../assets.js';
import {cancelBody, readBoundedResponse, ResponseSizeError} from '../bounded-response.js';
import {WebDecisionError} from '../kev.js';

export {commitCachedManifest};
interface VerifiedRangeSpec {
  readonly path: string;
  readonly totalBytes: number;
  readonly offset: number;
  readonly length: number;
  readonly sha256: string;
}

type NetworkRange = Pick<VerifiedRangeSpec, 'path' | 'totalBytes' | 'offset' | 'length'>;

interface ReadVerifiedRangeOptions {
  readonly cache?: AssetCache | false | null | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly offline?: boolean | undefined;
}

interface ReadVerifiedRangeResult {
  readonly bytes: ArrayBuffer;
  readonly cachedByHash: boolean;
}

const fail = (message: string): Error => new WebDecisionError('INVALID_ASSET', message);
const MAX = 8 * 1024 * 1024;
// Keep CDN stream resets bounded while allowing one short recovery window.
const MAX_TRANSPORT_ATTEMPTS = 3;
const RETRY_BACKOFF_MS = 100;

function isTransientTransportFailure(cause: unknown): boolean {
  return (
    (cause instanceof TypeError && cause.message !== 'Response stream did not return bytes.') ||
    (typeof DOMException !== 'undefined' && cause instanceof DOMException && cause.name === 'NetworkError')
  );
}

function waitBeforeRetry(attempt: number, signal?: AbortSignal | null | undefined): Promise<void> {
  const delay = RETRY_BACKOFF_MS * 2 ** (attempt - 1);
  if (!signal) return new Promise((resolve) => setTimeout(resolve, delay));
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const finish = () => {
      signal.removeEventListener('abort', abort);
      resolve();
    };
    const abort = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', abort);
      reject(signal.reason);
    };
    const timer = setTimeout(finish, delay);
    signal.addEventListener('abort', abort, {once: true});
    if (signal.aborted) abort();
  });
}

async function readCachedRange(
  cache: AssetCache,
  key: URL,
  length: number,
  digest: string,
  signal?: AbortSignal | null | undefined
): Promise<ArrayBuffer | null> {
  try {
    const hit = await waitForAbort(
      () => cache.match(key.href),
      signal,
      (value) => cancelBody(value?.body)
    );
    if (hit) {
      const bytes = await readBoundedResponse(hit, length, {exact: true, signal} as Parameters<
        typeof readBoundedResponse
      >[2]);
      const hash = await waitForAbort(() => sha256(bytes), signal);
      if (bytes.byteLength === length && hash === digest) {
        signal?.throwIfAborted();
        return bytes;
      }
    }
  } catch {
    signal?.throwIfAborted();
  }
  return null;
}

function assertRangeResponse(
  response: Response,
  {path, totalBytes, offset, length}: NetworkRange,
  signal?: AbortSignal | null | undefined
): void {
  if (signal?.aborted) {
    cancelBody(response?.body);
    signal.throwIfAborted();
  }
  const declaredLength = response.headers.get('content-length');
  const actualContentRange = response.headers.get('content-range');
  const validLength =
    declaredLength === null ||
    (/^[0-9]+$/.test(declaredLength) &&
      Number.isSafeInteger(Number(declaredLength)) &&
      Number(declaredLength) === length);
  if (
    response.status !== 206 ||
    actualContentRange !== `bytes ${offset}-${offset + length - 1}/${totalBytes}` ||
    !validLength
  ) {
    cancelBody(response.body);
    signal?.throwIfAborted();
    throw fail(
      `Invalid or inaccessible Content-Range or Content-Length response: assetPath=${JSON.stringify(path)}, requestedOffset=${offset}, requestedLength=${length}, responseStatus=${response.status}, actualContentRange=${JSON.stringify(actualContentRange)}, contentLength=${JSON.stringify(declaredLength)}.`
    );
  }
}

async function readNetworkRange(
  url: string,
  range: NetworkRange,
  signal?: AbortSignal | null | undefined
): Promise<ArrayBuffer> {
  const {offset, length} = range;
  for (let attempt = 1; attempt <= MAX_TRANSPORT_ATTEMPTS; attempt += 1) {
    let response: Response;
    try {
      response = await waitForAbort(
        () => fetch(url, {signal, headers: {Range: `bytes=${offset}-${offset + length - 1}`}} as RequestInit),
        signal,
        (value) => cancelBody(value?.body)
      );
    } catch (cause) {
      signal?.throwIfAborted();
      if (!isTransientTransportFailure(cause) || attempt === MAX_TRANSPORT_ATTEMPTS)
        throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not fetch range.', {cause});
      await waitBeforeRetry(attempt, signal);
      continue;
    }

    assertRangeResponse(response, range, signal);
    try {
      return await readBoundedResponse(response, length, {exact: true, signal} as Parameters<
        typeof readBoundedResponse
      >[2]);
    } catch (cause) {
      signal?.throwIfAborted();
      cancelBody(response.body);
      if (
        cause instanceof ResponseSizeError ||
        !isTransientTransportFailure(cause) ||
        attempt === MAX_TRANSPORT_ATTEMPTS
      )
        throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not read range.', {cause});
      await waitBeforeRetry(attempt, signal);
    }
  }
  throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not fetch range.');
}

// The caller must pin the file and hashes in its validated manifest, and only
// commit that manifest after all required reads report cachedByHash=true.
export async function readVerifiedRange(
  base: URL,
  {path, totalBytes, offset, length, sha256: digest}: VerifiedRangeSpec,
  {cache, signal, offline = false}: ReadVerifiedRangeOptions = {}
): Promise<ReadVerifiedRangeResult> {
  if (
    !Number.isSafeInteger(totalBytes) ||
    totalBytes <= 0 ||
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    !Number.isSafeInteger(length) ||
    length < 1 ||
    length > MAX ||
    offset > totalBytes - length ||
    typeof digest !== 'string' ||
    !/^[0-9a-f]{64}$/.test(digest)
  )
    throw fail('Invalid range metadata.');
  const url = assetUrl(base, path);
  const key = new URL('/.web-decision-cache/sha256/', url);
  key.searchParams.set('digest', digest);
  signal?.throwIfAborted();
  if (cache) {
    const cached = await readCachedRange(cache, key, length, digest, signal);
    if (cached) {
      signal?.throwIfAborted();
      return {bytes: cached, cachedByHash: true};
    }
  }
  signal?.throwIfAborted();
  if (offline) throw new WebDecisionError('ASSET_LOAD_FAILED', 'Verified range is not available offline.');
  const bytes = await readNetworkRange(url, {path, totalBytes, offset, length}, signal);
  const actualDigest = await waitForAbort(() => sha256(bytes), signal);
  signal?.throwIfAborted();
  if (actualDigest !== digest) throw fail('Range integrity check failed.');
  let cachedByHash = false;
  if (cache) {
    try {
      await waitForAbort(() => cache.put(key.href, new Response(bytes)), signal);
      cachedByHash = true;
    } catch {
      signal?.throwIfAborted();
    }
  }
  signal?.throwIfAborted();
  return {bytes, cachedByHash};
}
