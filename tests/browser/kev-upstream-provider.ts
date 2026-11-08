import {
  createKevReconstructedSource,
  EXPERIMENTAL_KEV_MODEL
} from '../../packages/web-decision/src/kev06/upstream-source.mjs';
import {createModelAssetReader as createModelAssetReaderFromSource} from '../../packages/web-decision/src/model-assets.js';
import {
  readVerifiedAsset as readVerifiedAssetFromSource,
  sha256 as sha256FromSource
} from '../../packages/web-decision/src/assets.js';
import {loadDecisionModel} from 'web-decision';

declare global {
  interface Window {
    lastCachedStatus?: boolean;
  }
}

interface AssetFile {
  bytes: number;
  sha256: string;
}

interface AssetManifest {
  files: Record<string, AssetFile>;
}

interface ProviderProgress {
  readonly phase?: string;
  readonly loadedBytes?: number;
}

interface KevProvider {
  setProgress(callback: (event: ProviderProgress) => void): void;
  prepare(options: {model: string; cache: boolean; signal?: AbortSignal}): Promise<{base: URL}>;
  fetch: typeof fetch;
  dispose(): Promise<void>;
}

interface KevProfile {
  manifest: AssetManifest;
  sourceShardMapText: string;
  sourceUrl: string;
  sourceBytes: number;
  sourceSha256: string;
  mapSha256: string;
  repository: string;
  revision: string;
  sourcePath: string;
  parameterRegionStart: number;
  tensorCount: number;
  progressChunkBytes: number;
  bundledUrls: Record<string, string>;
  cacheBase: URL;
  preflight: (_signal?: AbortSignal, _gpu?: unknown) => Promise<void>;
  openCache: (_enabled: boolean, _signal?: AbortSignal) => Promise<null>;
  fetchSource: (url: string, init: RequestInit) => Promise<Response>;
}

interface MemoryCache {
  entries: Map<string, Response>;
  match(key: string): Promise<Response | undefined>;
  put(key: string, response: Response): Promise<void>;
}

interface FixtureOptions {
  sourceBytes?: Uint8Array<ArrayBuffer> | null;
  sourceResponse?: Response;
  responseHeaders?: HeadersInit;
  expectedSource?: Uint8Array<ArrayBuffer>;
  sourceUrl?: string;
}

interface Fixture {
  manifest: AssetManifest;
  profile: KevProfile;
  createProvider(): KevProvider;
  fetchCount(): number;
  sourceSignal(): AbortSignal | undefined;
  resolve(response: Response): void;
  reject(cause: unknown): void;
}

interface ModelAssetReader {
  read(name: string): Promise<ArrayBuffer>;
  isCacheComplete(): boolean;
  setProgressPhase(phase: string): void;
}

const bytes = new Uint8Array([3, 1, 4, 1, 5, 9, 2, 6]);
const shardPath = 'weights/params_shard_0.bin';
const sourceUrl = 'https://hf-test.invalid/pinned/data.bin';
const cacheBase = new URL('https://cache-test.invalid/experimental-kev/');

function label(value: unknown): void {
  const result = document.querySelector<HTMLElement>('#result');
  if (!result) throw new Error('The browser test result element is missing.');
  result.textContent = JSON.stringify(value, null, 2) ?? 'null';
}

function digest(value: ArrayBuffer | Uint8Array<ArrayBuffer>): Promise<string> {
  return sha256FromSource(value) as Promise<string>;
}

const readVerifiedAsset = readVerifiedAssetFromSource as (
  base: URL,
  manifest: AssetManifest,
  name: string,
  options: {
    cache?: MemoryCache;
    signal?: AbortSignal | undefined;
    fetchImpl?: typeof fetch;
    onCacheStatus?: (cached: boolean) => void;
  }
) => Promise<ArrayBuffer>;

function memoryCache({failPut = false}: {failPut?: boolean} = {}): MemoryCache {
  const entries = new Map<string, Response>();
  return {
    entries,
    async match(key) {
      return entries.get(String(key))?.clone();
    },
    async put(key, response) {
      if (failPut) throw new DOMException('quota exceeded', 'QuotaExceededError');
      entries.set(String(key), response.clone());
    }
  };
}

async function fixture({
  sourceBytes = bytes,
  sourceResponse,
  responseHeaders = {},
  expectedSource = bytes,
  sourceUrl: overrideUrl = sourceUrl
}: FixtureOptions = {}): Promise<Fixture> {
  const sourceSha = await digest(expectedSource);
  const shardSha = await digest(expectedSource);
  const manifest: AssetManifest = {
    files: {[shardPath]: {bytes: expectedSource.byteLength, sha256: shardSha}}
  };
  const map = {
    schemaVersion: 1,
    source: {
      repository: 'test/model',
      revision: 'deadbeef',
      path: 'onnx/data.bin',
      bytes: expectedSource.byteLength,
      sha256: sourceSha,
      parameterRegionStart: 0
    },
    target: {shards: [{path: shardPath, bytes: expectedSource.byteLength, sha256: shardSha}]},
    records: [
      {sourceOffset: 0, targetOffset: 0, bytes: 4, shardIndex: 0},
      {sourceOffset: 4, targetOffset: 4, bytes: 4, shardIndex: 0}
    ]
  };
  const mapText = JSON.stringify(map);
  if (mapText === undefined) throw new Error('Could not serialize the source shard map.');
  const mapSha = await digest(new TextEncoder().encode(mapText));
  const headers = new Headers(responseHeaders);
  let fetches = 0;
  let sourceSignal: AbortSignal | undefined;
  let resolveSource: ((response: Response) => void) | undefined;
  let rejectSource: ((cause: unknown) => void) | undefined;
  let sourcePromise: Promise<Response> | undefined;
  const profile: KevProfile = {
    manifest,
    sourceShardMapText: mapText,
    sourceUrl: overrideUrl,
    sourceBytes: expectedSource.byteLength,
    sourceSha256: sourceSha,
    mapSha256: mapSha,
    repository: 'test/model',
    revision: 'deadbeef',
    sourcePath: 'onnx/data.bin',
    parameterRegionStart: 0,
    tensorCount: 2,
    progressChunkBytes: 2,
    bundledUrls: {},
    cacheBase,
    preflight: async () => {},
    openCache: async () => null,
    fetchSource: (_url, init) => {
      fetches++;
      sourceSignal = init.signal ?? undefined;
      if (!sourcePromise) {
        sourcePromise = new Promise<Response>((resolve, reject) => {
          resolveSource = resolve;
          rejectSource = reject;
        });
      }
      if (init.signal) init.signal.addEventListener('abort', () => rejectSource?.(init.signal?.reason), {once: true});
      if (sourceResponse) queueMicrotask(() => resolveSource?.(sourceResponse));
      else if (sourceBytes) queueMicrotask(() => resolveSource?.(new Response(sourceBytes, {headers})));
      return sourcePromise;
    }
  };
  return {
    manifest,
    profile,
    createProvider: () => createKevReconstructedSource(profile) as KevProvider,
    fetchCount: () => fetches,
    sourceSignal: () => sourceSignal,
    resolve: (response) => resolveSource?.(response),
    reject: (cause) => rejectSource?.(cause)
  };
}

async function cacheShard(
  provider: KevProvider,
  manifest: AssetManifest,
  cache: MemoryCache,
  signal?: AbortSignal
): Promise<ArrayBuffer> {
  const options = signal
    ? {model: EXPERIMENTAL_KEV_MODEL as string, cache: true, signal}
    : {model: EXPERIMENTAL_KEV_MODEL as string, cache: true};
  const prepared = await provider.prepare(options);
  return readVerifiedAsset(prepared.base, manifest, shardPath, {
    cache,
    signal,
    fetchImpl: provider.fetch,
    onCacheStatus: (cached) => {
      window.lastCachedStatus = cached;
    }
  });
}

async function waitFor(predicate: () => boolean, message: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(message);
}

function errorCode(value: unknown): string | undefined {
  if (typeof value !== 'object' || value === null || !('code' in value)) return undefined;
  return typeof value.code === 'string' ? value.code : undefined;
}

async function expectUnsupportedWebGpu(navigatorValue: unknown, scenario: string): Promise<void> {
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  if (originalNavigator && !originalNavigator.configurable)
    throw new Error('The browser cannot replace navigator for this test.');
  const originalFetch = globalThis.fetch;
  let fetchCalls = 0;
  globalThis.fetch = (...args) => {
    fetchCalls++;
    return originalFetch(...args);
  };
  let result: unknown;
  try {
    Object.defineProperty(globalThis, 'navigator', {configurable: true, value: navigatorValue});
    try {
      await loadDecisionModel({modelId: 'kev-0.6b-q4f16', cache: true});
      throw new Error(`${scenario}: prepare unexpectedly succeeded.`);
    } catch (cause) {
      result = cause;
    }
    const code = errorCode(result);
    if (code !== 'UNSUPPORTED_ENVIRONMENT')
      throw new Error(`${scenario}: expected UNSUPPORTED_ENVIRONMENT, received ${code ?? String(result)}.`);
    if (fetchCalls !== 0) throw new Error(`${scenario}: fetch began before environment rejection.`);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator);
    else Reflect.deleteProperty(globalThis, 'navigator');
  }
}

async function verifyEnvironmentRejection(): Promise<void> {
  await expectUnsupportedWebGpu(undefined, 'Missing navigator');
  await expectUnsupportedWebGpu({}, 'Missing navigator.gpu');
}

async function verifyExactUnboundedReadAndProgress(): Promise<void> {
  const success = await fixture({responseHeaders: {}});
  const provider = success.createProvider();
  const cache = memoryCache();
  const progress: ProviderProgress[] = [];
  provider.setProgress((event) => progress.push(event));
  const value = await cacheShard(provider, success.manifest, cache);
  if ((await digest(value)) !== (await digest(bytes)))
    throw new Error('No-Content-Length exact-size verification failed.');
  if (success.fetchCount() !== 1 || cache.entries.size !== 1)
    throw new Error('Verified shard was not cached exactly once.');
  if (!progress.some((event) => event.phase === 'upstream-source' && event.loadedBytes === bytes.byteLength))
    throw new Error('Coarse upstream-source progress was not reported.');
  await provider.dispose();
}

function dataUrl(value: Uint8Array<ArrayBuffer>): string {
  return `data:application/json;base64,${btoa(String.fromCharCode(...value))}`;
}

async function verifyBundledDataUrl(): Promise<void> {
  const original = await fixture({responseHeaders: {}});
  const bundledJson = new TextEncoder().encode('{"source":"bundled"}');
  const manifest: AssetManifest = {
    files: {
      ...original.manifest.files,
      'model_config.json': {bytes: bundledJson.byteLength, sha256: await digest(bundledJson)}
    }
  };
  const provider = createKevReconstructedSource({
    ...original.profile,
    manifest,
    bundledUrls: {'model_config.json': dataUrl(bundledJson)}
  }) as KevProvider;
  const prepared = await provider.prepare({model: EXPERIMENTAL_KEV_MODEL as string, cache: true});
  const cache = memoryCache();
  const result = await readVerifiedAsset(prepared.base, manifest, 'model_config.json', {
    cache,
    fetchImpl: provider.fetch
  });
  if (new TextDecoder().decode(result) !== '{"source":"bundled"}' || cache.entries.size !== 1)
    throw new Error('Bundled data URL was not decoded and SHA-verified without fetch().');
  await provider.dispose();
}

async function expectRejectedSource(options: FixtureOptions, message: string): Promise<void> {
  const failed = await fixture(options);
  const provider = failed.createProvider();
  const cache = memoryCache();
  let rejected = false;
  try {
    await cacheShard(provider, failed.manifest, cache);
  } catch {
    rejected = true;
  }
  if (!rejected || cache.entries.size !== 0) throw new Error(message);
  await provider.dispose();
}

async function verifyRejectedSources(): Promise<void> {
  await expectRejectedSource(
    {sourceBytes: new Uint8Array([...bytes, 0]), responseHeaders: {}},
    'Oversized source was accepted or cached.'
  );
  await expectRejectedSource(
    {sourceBytes: new Uint8Array([9, 9, 9, 9, 5, 9, 2, 6]), responseHeaders: {}},
    'Bad source SHA was accepted or cached.'
  );
  await expectRejectedSource(
    {responseHeaders: {'content-length': '7'}},
    'Wrong declared source length was accepted or cached.'
  );
}

async function verifyCacheQuotaFailure(): Promise<void> {
  const quota = await fixture({responseHeaders: {}});
  const provider = quota.createProvider();
  const cache = memoryCache({failPut: true});
  const progress: ProviderProgress[] = [];
  const prepared = await provider.prepare({model: EXPERIMENTAL_KEV_MODEL as string, cache: true});
  const reader = createModelAssetReaderFromSource({
    base: prepared.base,
    manifest: quota.manifest,
    cache,
    signal: undefined,
    options: {offline: false, onProgress: (event: ProviderProgress) => progress.push(event)},
    fetchImpl: provider.fetch,
    progressPhase: 'asset-verification'
  }) as ModelAssetReader;
  await reader.read(shardPath);
  if (reader.isCacheComplete()) throw new Error('Cache quota failure was treated as a complete verified pack.');
  if (!progress.some((event) => event.phase === 'asset-verification'))
    throw new Error('Asset-verification progress phase was not reported.');
  await provider.dispose();
}

async function verifyProgressPhases(): Promise<void> {
  const phaseFixture = await fixture({responseHeaders: {}});
  const packagedConfig = new TextEncoder().encode('{}');
  const manifest: AssetManifest = {
    files: {
      ...phaseFixture.manifest.files,
      'model_config.json': {bytes: packagedConfig.byteLength, sha256: await digest(packagedConfig)}
    }
  };
  const provider = createKevReconstructedSource({
    ...phaseFixture.profile,
    manifest,
    bundledUrls: {'model_config.json': dataUrl(packagedConfig)}
  }) as KevProvider;
  const prepared = await provider.prepare({model: EXPERIMENTAL_KEV_MODEL as string, cache: true});
  const progress: ProviderProgress[] = [];
  provider.setProgress((event) => progress.push(event));
  const cache = memoryCache({failPut: true});
  const reader = createModelAssetReaderFromSource({
    base: prepared.base,
    manifest,
    cache,
    signal: undefined,
    options: {offline: false, onProgress: (event: ProviderProgress) => progress.push(event)},
    fetchImpl: provider.fetch,
    progressPhase: 'bundled-assets'
  }) as ModelAssetReader;
  await reader.read('model_config.json');
  reader.setProgressPhase('asset-verification');
  await reader.read(shardPath);
  const firstVerification = progress.findIndex((event) => event.phase === 'asset-verification');
  if (firstVerification < 0) throw new Error('Phase test did not reach shard verification.');
  await reader.read(shardPath);
  const phases = progress
    .map((event) => event.phase)
    .filter((phase): phase is string => Boolean(phase))
    .filter((phase, index, sequence) => index === 0 || phase !== sequence[index - 1]);
  if (JSON.stringify(phases) !== JSON.stringify(['bundled-assets', 'upstream-source', 'asset-verification']))
    throw new Error(`Progress phases did not advance monotonically: ${JSON.stringify(phases)}`);
  if (progress.slice(firstVerification + 1).some((event) => event.phase === 'upstream-source'))
    throw new Error('Source completion was replayed after shard verification began.');
  await provider.dispose();
}

async function verifyCancellationDoesNotWaitForBodyCleanup(): Promise<void> {
  let cancelStarted = false;
  const neverSettlingBody = new ReadableStream<Uint8Array>({
    pull() {},
    cancel() {
      cancelStarted = true;
      return new Promise<void>(() => {});
    }
  });
  const cancellation = await fixture({sourceBytes: null, sourceResponse: new Response(neverSettlingBody)});
  const provider = cancellation.createProvider();
  const controller = new AbortController();
  const prepared = await provider.prepare({
    model: EXPERIMENTAL_KEV_MODEL as string,
    cache: true,
    signal: controller.signal
  });
  const pending = readVerifiedAsset(prepared.base, cancellation.manifest, shardPath, {
    cache: memoryCache(),
    signal: controller.signal,
    fetchImpl: provider.fetch
  });
  await waitFor(() => cancellation.fetchCount() === 1, 'Cancellation test did not start source GET.');
  controller.abort(new DOMException('stop source read', 'AbortError'));
  const outcome = await Promise.race([
    pending.then(
      () => 'unexpected-success' as const,
      () => 'aborted' as const
    ),
    new Promise<'timed-out'>((resolve) => setTimeout(() => resolve('timed-out'), 1000))
  ]);
  await provider.dispose();
  if (outcome !== 'aborted' || !cancelStarted)
    throw new Error('Abort waited for a non-settling body cancel operation.');
}

async function verifyConcurrentSubscriberCancellation(): Promise<{fetchCount: number; secondAlive: boolean}> {
  const concurrent = await fixture({sourceBytes: null, responseHeaders: {}});
  const providerA = concurrent.createProvider();
  const providerB = concurrent.createProvider();
  const controllerA = new AbortController();
  const cacheA = memoryCache();
  const cacheB = memoryCache();
  const preparedA = await providerA.prepare({
    model: EXPERIMENTAL_KEV_MODEL as string,
    cache: true,
    signal: controllerA.signal
  });
  const preparedB = await providerB.prepare({model: EXPERIMENTAL_KEV_MODEL as string, cache: true});
  let providerACalls = 0;
  let providerBCalls = 0;
  const fetchA = providerA.fetch;
  const fetchB = providerB.fetch;
  providerA.fetch = (...args) => {
    providerACalls++;
    return fetchA(...args);
  };
  providerB.fetch = (...args) => {
    providerBCalls++;
    return fetchB(...args);
  };
  const loadA = readVerifiedAsset(preparedA.base, concurrent.manifest, shardPath, {
    cache: cacheA,
    signal: controllerA.signal,
    fetchImpl: providerA.fetch
  });
  const loadB = readVerifiedAsset(preparedB.base, concurrent.manifest, shardPath, {
    cache: cacheB,
    fetchImpl: providerB.fetch
  });
  await waitFor(
    () => providerACalls === 1 && providerBCalls === 1 && concurrent.fetchCount() === 1,
    'Concurrent providers did not share one source GET.'
  );
  controllerA.abort(new DOMException('cancel only first load', 'AbortError'));
  let firstAborted = false;
  try {
    await loadA;
  } catch {
    firstAborted = true;
  }
  await providerA.dispose();
  const sourceSignal = concurrent.sourceSignal();
  if (!firstAborted || !sourceSignal || sourceSignal.aborted)
    throw new Error('Aborting one subscriber canceled the shared source read.');
  concurrent.resolve(new Response(bytes, {headers: {'content-length': String(bytes.byteLength)}}));
  const survivingBytes = await loadB;
  if ((await digest(survivingBytes)) !== (await digest(bytes)))
    throw new Error('Second concurrent provider did not finish after first abort.');
  await providerB.dispose();
  return {fetchCount: concurrent.fetchCount(), secondAlive: !sourceSignal.aborted};
}

async function run(): Promise<void> {
  await verifyEnvironmentRejection();
  await verifyExactUnboundedReadAndProgress();
  await verifyBundledDataUrl();
  await verifyRejectedSources();
  await verifyCacheQuotaFailure();
  await verifyProgressPhases();
  await verifyCancellationDoesNotWaitForBodyCleanup();
  const concurrent = await verifyConcurrentSubscriberCancellation();
  label({
    ok: true,
    checks: {
      missingContentLengthAcceptedWithExactBound: true,
      bundledDataUrlDecodedWithoutConnectFetch: true,
      missingNavigatorAndNavigatorGpuReturnUnsupportedEnvironment: true,
      oversizedSourceRejectedWithoutCacheWrite: true,
      sourceHashMismatchRejectedWithoutCacheWrite: true,
      wrongContentLengthRejectedWithoutCacheWrite: true,
      cacheQuotaLeavesReaderIncomplete: true,
      progressPhasesAdvanceOnceAcrossBundledSourceAndShards: true,
      nonSettlingBodyCancelDoesNotBlockAbort: true,
      coarseProgressHasDistinctPhases: true,
      concurrentLoadsSharedOneSourceGet: concurrent.fetchCount === 1,
      firstAbortLeftSecondSourceReadAlive: concurrent.secondAlive,
      secondConcurrentLoadCompleted: true
    }
  });
}

void run().catch((cause: unknown) =>
  label({
    ok: false,
    error: String(cause),
    ...(cause instanceof Error && cause.stack ? {stack: cause.stack} : {})
  })
);
