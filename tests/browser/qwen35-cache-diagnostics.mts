export interface CachePutFailure {
  readonly digest: string | null;
  readonly errorName: string;
  readonly errorMessage: string;
  readonly quota?: number;
  readonly requested?: number;
}

export function describeCachePutError(cause: unknown): Omit<CachePutFailure, 'digest'> {
  const candidate = cause as
    | {
        readonly name?: unknown;
        readonly message?: unknown;
        readonly quota?: unknown;
        readonly requested?: unknown;
      }
    | null
    | undefined;
  const errorName = typeof candidate?.name === 'string' ? candidate.name : 'UnknownError';
  const message = typeof candidate?.message === 'string' ? candidate.message : String(cause);
  return {
    errorName,
    errorMessage: message.replace(/https?:\/\/[^\s"'<>]+/g, '[URL redacted]'),
    ...(typeof candidate?.quota === 'number' && Number.isFinite(candidate.quota) ? {quota: candidate.quota} : {}),
    ...(typeof candidate?.requested === 'number' && Number.isFinite(candidate.requested)
      ? {requested: candidate.requested}
      : {})
  };
}

export interface CachePutTracker {
  readonly recordSuccess: () => void;
  readonly recordFailure: (failure: CachePutFailure) => void;
  readonly snapshot: () => Readonly<{
    successfulPuts: number;
    failedPuts: number;
    failuresByErrorName: Readonly<Record<string, number>>;
    failedDigests: readonly string[];
    firstFailure: CachePutFailure | null;
    lastFailure: CachePutFailure | null;
    firstFailedRange: Readonly<{digest: string; expectedBytes: number}> | null;
  }>;
}

export interface CachePutPrototype {
  put(this: unknown, request: RequestInfo | URL, response: Response): Promise<void>;
}

function rangeDigest(request: RequestInfo | URL, baseUrl: string): string | null {
  try {
    const url =
      typeof request === 'string' ? new URL(request, baseUrl) : request instanceof URL ? request : new URL(request.url);
    if (url.pathname !== '/.web-decision-cache/sha256/') return null;
    const digest = url.searchParams.get('digest');
    return digest && /^[0-9a-f]{64}$/.test(digest) ? digest : null;
  } catch {
    return null;
  }
}

/** Observe only verified range writes while preserving Cache.put results and failures. */
export function observeRangeCachePuts(
  prototype: CachePutPrototype,
  tracker: CachePutTracker,
  baseUrl: string
): () => void {
  const originalPut = prototype.put;
  const observedPut: CachePutPrototype['put'] = async function (this: unknown, request, response) {
    const digest = rangeDigest(request, baseUrl);
    if (!digest) return originalPut.call(this, request, response);
    try {
      await originalPut.call(this, request, response);
      tracker.recordSuccess();
    } catch (cause) {
      tracker.recordFailure({digest, ...describeCachePutError(cause)});
      throw cause;
    }
  };
  prototype.put = observedPut;
  return () => {
    if (prototype.put === observedPut) prototype.put = originalPut;
  };
}

export function summarizeCacheStorageUrls(
  urls: readonly string[],
  sourceMarkerUrl: string,
  tokenizerMarkerUrl: string
): Readonly<{
  entryCount: number;
  rangeCacheEntryCount: number;
  sourceRangeCacheEntryCount: number;
  sourceRangeDigestCount: number;
  sourceMarkerPresent: boolean;
  tokenizerMarkerPresent: boolean;
}> {
  const sourceOrigin = new URL(sourceMarkerUrl).origin;
  const normalizedUrls = urls.map((url) => new URL(url));
  const rangeEntries = normalizedUrls.filter(({pathname}) => pathname === '/.web-decision-cache/sha256/');
  const sourceRangeEntries = rangeEntries.filter(({origin}) => origin === sourceOrigin);
  const rangeDigests = new Set(
    sourceRangeEntries.map(({searchParams}) => searchParams.get('digest')).filter((digest) => digest !== null)
  );
  const normalizedSourceMarkerUrl = new URL(sourceMarkerUrl).href;
  const normalizedTokenizerMarkerUrl = new URL(tokenizerMarkerUrl).href;
  return Object.freeze({
    entryCount: normalizedUrls.length,
    rangeCacheEntryCount: rangeEntries.length,
    sourceRangeCacheEntryCount: sourceRangeEntries.length,
    sourceRangeDigestCount: rangeDigests.size,
    sourceMarkerPresent: normalizedUrls.some(({href}) => href === normalizedSourceMarkerUrl),
    tokenizerMarkerPresent: normalizedUrls.some(({href}) => href === normalizedTokenizerMarkerUrl)
  });
}

export function createCachePutTracker(
  expectedRangeBytesByDigest: ReadonlyMap<string, number> = new Map()
): CachePutTracker {
  let successfulPuts = 0;
  let failedPuts = 0;
  let firstFailure: CachePutFailure | null = null;
  let lastFailure: CachePutFailure | null = null;
  let firstFailedRange: Readonly<{digest: string; expectedBytes: number}> | null = null;
  const failuresByErrorName = new Map<string, number>();
  const failedDigests: string[] = [];
  const sampledDigests = new Set<string>();

  return Object.freeze({
    recordSuccess() {
      successfulPuts += 1;
    },
    recordFailure(failure: CachePutFailure) {
      failedPuts += 1;
      failuresByErrorName.set(failure.errorName, (failuresByErrorName.get(failure.errorName) ?? 0) + 1);
      if (failure.digest && !sampledDigests.has(failure.digest) && failedDigests.length < 3) {
        sampledDigests.add(failure.digest);
        failedDigests.push(failure.digest);
      }
      firstFailure ??= failure;
      lastFailure = failure;
      if (!firstFailedRange && failure.digest) {
        const expectedBytes = expectedRangeBytesByDigest.get(failure.digest);
        if (expectedBytes !== undefined) firstFailedRange = Object.freeze({digest: failure.digest, expectedBytes});
      }
    },
    snapshot() {
      return Object.freeze({
        successfulPuts,
        failedPuts,
        failuresByErrorName: Object.freeze(
          Object.fromEntries([...failuresByErrorName].sort(([left], [right]) => left.localeCompare(right)))
        ),
        failedDigests: Object.freeze([...failedDigests]),
        firstFailure,
        lastFailure,
        firstFailedRange
      });
    }
  });
}
