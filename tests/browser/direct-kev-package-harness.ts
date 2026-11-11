import {loadDecisionModel} from 'web-decision';
import {assetUrl, CACHE_NAME, openAssetCache} from '../../packages/web-decision/src/assets.js';
import {MODEL_CATALOG_BY_ID} from '../../packages/web-decision/src/catalog/index.generated.ts';
import type {KevTokenizer} from '../../packages/web-decision/src/kev.js';
import {loadCuratedKevModel} from '../../packages/web-decision/src/kev-loader.js';
import type {RepresentativeDecisionCase} from './kev06-expanded-inference.js';

const MODEL_ID = 'kev-0.6b-q4f16';
const DIAGNOSTIC_STORAGE_KEY = 'direct-kev-r6-webgpu-diagnostics';
const PROGRESS_PREFIX = '__DIRECT_KEV_PROGRESS__';
const DIRECT_ARTIFACT_ROOT = 'https://direct-kev-package-test.invalid/';
const NATIVE_KEV_INPUT_LIMITS = {
  sourceRepository: 'jaredpalmer/kev',
  sourceRevision: '37cb2f6155afec1054de2cc78dfd7c7dd24a956d',
  sourceFile: 'kev/serve.py',
  maxStateTokens: 8192,
  maxBranchTokens: 8192,
  maxPositionEmbeddings: 32768,
  aggregateLimit: null
} as const;

type ProgressEvent = Readonly<{stage: string; timestamp: string; details?: unknown}>;

function recordProgress(stage: string, details?: unknown): void {
  const event: ProgressEvent = {stage, timestamp: new Date().toISOString(), ...(details ? {details} : {})};
  let events: ProgressEvent[] = [];
  try {
    const previous = JSON.parse(localStorage.getItem(DIAGNOSTIC_STORAGE_KEY) ?? '{}') as {events?: unknown};
    if (Array.isArray(previous.events)) events = previous.events as ProgressEvent[];
  } catch {
    // Console checkpoints still reach the Playwright process if localStorage is unavailable.
  }
  events.push(event);
  if (events.length > 256) events = events.slice(-256);
  const snapshot = JSON.stringify({last: event, events});
  try {
    localStorage.setItem(DIAGNOSTIC_STORAGE_KEY, snapshot);
  } catch {
    // The test runner also captures each console checkpoint for timeout diagnostics.
  }
  console.info(`${PROGRESS_PREFIX}${JSON.stringify(event)}`);
}

type GateDecision = Readonly<{
  id: string;
  tokenCount: number;
  selected: string;
  probabilities: readonly Readonly<{option: string; logit: number; probability: number}>[];
}>;

type GateOutput = Readonly<{
  adapter: AdapterDiagnostics;
  nativeInputLimits: typeof NATIVE_KEV_INPUT_LIMITS;
  publicOptionsRejected: boolean;
  onlineFetchPaths: readonly string[];
  offlineFetchPaths: readonly string[];
  qualityIssues: readonly string[];
  online: readonly GateDecision[];
  cacheDiagnostics: DirectCacheDiagnostics;
  boundaryProbe?: Readonly<{
    requestedTokenCount: number;
    acceptedTokenCount?: number;
    outputFinite?: boolean;
    probabilityTotal?: number;
    atLimitError?: string;
    overLimitErrorCode?: string;
    overLimitMessage?: string;
  }>;
  boundaryProbeError?: string;
  offlineBaseline: GateDecision;
}>;

type DefaultSourceSmokeOutput = Readonly<{
  onlineBaseline: GateDecision;
  offlineBaseline: GateDecision;
  cacheDiagnostics: DirectCacheDiagnostics;
}>;

type CpuBaseline = Readonly<{
  logits: readonly number[];
  probabilities: readonly number[];
  selected: string;
}>;

type AdapterDiagnostics = Readonly<{
  info: Readonly<{vendor: string; architecture: string; device: string; description: string}>;
  isFallbackAdapter: boolean | null;
  limits: Readonly<{maxBufferSize: number; maxStorageBufferBindingSize: number}>;
}>;

type DirectCacheDiagnostics = Readonly<{
  cacheNames: readonly string[];
  namespace: string;
  cacheKeyCount: number;
  cacheKeyUrls: readonly string[];
  cachedAssetNames: readonly string[];
  missingAssetNames: readonly string[];
  cachePutFailures: readonly string[];
  storageEstimate: Readonly<{usage: number | null; quota: number | null}>;
}>;

async function runDecisions(
  model: Awaited<ReturnType<typeof loadCuratedKevModel>>,
  cases: readonly RepresentativeDecisionCase[],
  stagePrefix: string
): Promise<GateDecision[]> {
  const decisions = [];
  for (const item of cases) {
    recordProgress(`${stagePrefix}-case-started`, {id: item.id, tokenCount: item.tokenCount});
    const result = await model.decide(item.request);
    const answer = result.answers[0];
    if (!answer) throw new Error(`The direct Kev model omitted answer ${item.id}.`);
    const decision = {
      id: item.id,
      tokenCount: result.tokenCount,
      selected: answer.selected,
      probabilities: answer.probabilities.map(({option, logit, probability}) => ({option, logit, probability}))
    };
    decisions.push(decision);
    recordProgress(`${stagePrefix}-case-complete`, decision);
  }
  return decisions;
}

function tokenCount(tokenizer: KevTokenizer, text: string): number {
  return tokenizer.encode(text, {add_special_tokens: false}).ids.length;
}

function maxDifference(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) return Number.POSITIVE_INFINITY;
  return Math.max(...left.map((value, index) => Math.abs(value - right[index]!)));
}

function describeFailure(value: unknown): string {
  const parts: string[] = [];
  let current: unknown = value;
  for (let depth = 0; depth < 4 && current; depth += 1) {
    if (typeof current !== 'object') {
      parts.push(String(current));
      break;
    }
    const error = current as {name?: unknown; code?: unknown; message?: unknown; cause?: unknown};
    const description = [error.name, error.code, error.message]
      .filter((part) => typeof part === 'string' && part.length > 0)
      .join(': ');
    if (description) parts.push(description);
    current = error.cause;
  }
  return parts.join(' <- ') || String(value);
}

function instrumentCache(cache: Cache, failures: string[]): Cache {
  return new Proxy(cache, {
    get(target, property) {
      if (property === 'put') {
        return async (request: RequestInfo, response: Response) => {
          try {
            await target.put(request, response);
          } catch (cause) {
            failures.push(`${typeof request === 'string' ? request : request.url}: ${describeFailure(cause)}`);
            throw cause;
          }
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    }
  });
}

async function readAdapterDiagnostics(): Promise<AdapterDiagnostics> {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in Chromium.');
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw new Error('Chromium did not provide a high-performance WebGPU adapter.');
  const info = adapter.info ?? {};
  const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
  const infoWithFallback = info as GPUAdapterInfo & {readonly isFallbackAdapter?: boolean};
  return {
    info: {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? ''
    },
    isFallbackAdapter: adapterWithFallback.isFallbackAdapter ?? infoWithFallback.isFallbackAdapter ?? null,
    limits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
    }
  };
}

function exactTokenText(tokenizer: KevTokenizer, target: number): string {
  for (const piece of [' x', ' a', ' z', ' q', '0 ', ' test ']) {
    let low = 0;
    let high = target * 4;
    while (low <= high) {
      const repetitions = Math.floor((low + high) / 2);
      const text = piece.repeat(repetitions);
      const count = tokenCount(tokenizer, text);
      if (count === target) return text;
      if (count < target) low = repetitions + 1;
      else high = repetitions - 1;
    }
  }
  throw new Error(`Could not compose exactly ${target} tokens with the shipped Kev tokenizer.`);
}

function outputValidityIssues(actual: GateDecision): string[] {
  const issues: string[] = [];
  const logits = actual.probabilities.map(({logit}) => logit);
  const probabilities = actual.probabilities.map(({probability}) => probability);
  if (![...logits, ...probabilities].every(Number.isFinite))
    issues.push(`The direct Kev GPU output is non-finite for ${actual.id}.`);
  const probabilityTotal = probabilities.reduce((total, value) => total + value, 0);
  if (!Number.isFinite(probabilityTotal) || Math.abs(probabilityTotal - 1) > 0.000001)
    issues.push(`The direct Kev GPU probabilities are not normalized for ${actual.id}: ${probabilityTotal}.`);
  return issues;
}

function pinnedCaseIssues(actual: GateDecision, expected: RepresentativeDecisionCase): string[] {
  const issues = outputValidityIssues(actual);
  if (actual.id !== expected.id || actual.tokenCount !== expected.tokenCount)
    issues.push(`The direct Kev GPU input differs from the pinned case ${actual.id}.`);
  if (actual.selected !== expected.reference.selected)
    issues.push(`The direct Kev GPU selected a different Q4 option for ${actual.id}.`);
  const logitDelta = maxDifference(
    actual.probabilities.map(({logit}) => logit),
    expected.reference.logits
  );
  if (logitDelta > 0.15) issues.push(`The direct Kev GPU logits exceed Q4 tolerance for ${actual.id}: ${logitDelta}.`);
  return issues;
}

function cpuParityIssues(actual: GateDecision | undefined, cpuBaseline: CpuBaseline): string[] {
  if (!actual) return ['The direct Kev GPU baseline is missing.'];
  const issues = outputValidityIssues(actual);
  if (actual.selected !== cpuBaseline.selected)
    issues.push('The direct Kev GPU baseline selection differs from the retained CPU baseline.');
  const logitDelta = maxDifference(
    actual.probabilities.map(({logit}) => logit),
    cpuBaseline.logits
  );
  const probabilityDelta = maxDifference(
    actual.probabilities.map(({probability}) => probability),
    cpuBaseline.probabilities
  );
  if (logitDelta > 0.15) issues.push(`The direct Kev GPU baseline logits differ from CPU: ${logitDelta}.`);
  if (probabilityDelta > 0.001)
    issues.push(`The direct Kev GPU baseline probabilities differ from CPU: ${probabilityDelta}.`);
  return issues;
}

function pinnedGpuMatrixIssues(
  decisions: readonly GateDecision[],
  cases: readonly RepresentativeDecisionCase[],
  cpuBaseline: CpuBaseline
): string[] {
  const issues: string[] = [];
  if (decisions.length !== cases.length) issues.push('The direct Kev GPU matrix is incomplete.');
  for (const [index, actual] of decisions.entries()) {
    const expected = cases[index];
    if (expected) issues.push(...pinnedCaseIssues(actual, expected));
    else issues.push(`The pinned Kev case ${index} is missing.`);
  }
  issues.push(...cpuParityIssues(decisions[0], cpuBaseline));
  return issues;
}

async function runTokenLimitProbe(
  model: Awaited<ReturnType<typeof loadCuratedKevModel>>
): Promise<NonNullable<GateOutput['boundaryProbe']>> {
  const tokenizer = (model as unknown as {tokenizer: KevTokenizer}).tokenizer;
  const option = 'x';
  const optionTokens = tokenCount(tokenizer, option);
  const instructionTokens = 8192 - 5 - optionTokens;
  const instruction = exactTokenText(tokenizer, instructionTokens);
  const result: {
    requestedTokenCount: number;
    acceptedTokenCount?: number;
    outputFinite?: boolean;
    probabilityTotal?: number;
    atLimitError?: string;
    overLimitErrorCode?: string;
    overLimitMessage?: string;
  } = {requestedTokenCount: 8192};
  recordProgress('boundary-8193-rejection-started', {requestedTokenCount: 8193});
  const overLimitInstruction = exactTokenText(tokenizer, instructionTokens + 1);
  let overLimitError: {code?: unknown; message?: unknown} | undefined;
  try {
    await model.decide({
      state: '',
      questions: [{instructions: overLimitInstruction, options: [option]}]
    });
  } catch (cause) {
    overLimitError = cause as {code?: unknown; message?: unknown};
  }
  if (overLimitError) {
    if (typeof overLimitError.code === 'string') result.overLimitErrorCode = overLimitError.code;
    if (typeof overLimitError.message === 'string') result.overLimitMessage = overLimitError.message;
  }
  recordProgress('boundary-8193-rejection-complete', result);

  recordProgress('boundary-8192-inference-started', {requestedTokenCount: 8192});
  try {
    const atLimit = await model.decide({state: '', questions: [{instructions: instruction, options: [option]}]});
    const answer = atLimit.answers[0];
    result.acceptedTokenCount = atLimit.tokenCount;
    if (answer) {
      result.outputFinite = answer.probabilities.every(
        ({logit, probability}) => Number.isFinite(logit) && Number.isFinite(probability)
      );
      result.probabilityTotal = answer.probabilities.reduce((total, item) => total + item.probability, 0);
    }
  } catch (cause) {
    result.atLimitError = describeFailure(cause);
  }
  recordProgress('boundary-8192-inference-complete', result);
  return result;
}

function createTestSource(manifestSha256: string) {
  const base = new URL(`/.web-decision-direct-test/kev-0.6b-q4f16/${manifestSha256}/`, location.origin);
  const fetchPaths: string[] = [];
  const cachePutFailures: string[] = [];
  let manifestFiles: Readonly<Record<string, {sha256: string}>> = {};
  let activeCache: Cache | null = null;
  const source = {
    async prepare(options: {cache?: boolean; offline?: boolean; signal?: AbortSignal; maxPrefetchBytes?: number}) {
      const openedCache = await openAssetCache(options.cache !== false, options.signal);
      const cache = openedCache ? instrumentCache(openedCache, cachePutFailures) : null;
      activeCache = cache;
      const manifestUrl = assetUrl(base, 'manifest.json');
      let response: Response | undefined;
      if (options.offline) {
        response = (await cache?.match(manifestUrl)) ?? undefined;
        if (!response) throw new Error('The direct package manifest was not cached online.');
      } else {
        fetchPaths.push('manifest.json');
        response = await fetch(new URL('manifest.json', DIRECT_ARTIFACT_ROOT), {
          ...(options.signal ? {signal: options.signal} : {}),
          cache: 'no-cache'
        });
      }
      if (response) {
        const manifest = (await response.clone().json()) as {files?: unknown};
        if (manifest.files && typeof manifest.files === 'object' && !Array.isArray(manifest.files))
          manifestFiles = manifest.files as Readonly<Record<string, {sha256: string}>>;
      }
      return {
        base,
        cache,
        ...(options.signal ? {signal: options.signal} : {}),
        manifestUrl,
        response,
        maxPrefetchBytes: options.maxPrefetchBytes ?? 0
      };
    },
    async fetch(url: string, init?: {signal?: AbortSignal}) {
      const requestUrl = new URL(url);
      if (requestUrl.origin !== base.origin || !requestUrl.pathname.startsWith(base.pathname))
        throw new Error('A direct Kev test asset escaped its cache namespace.');
      const path = decodeURIComponent(requestUrl.pathname.slice(base.pathname.length));
      if (!path || path.split('/').includes('..')) throw new Error('Invalid direct Kev test asset path.');
      fetchPaths.push(path);
      return fetch(new URL(path, DIRECT_ARTIFACT_ROOT), init);
    }
  } as NonNullable<Parameters<typeof loadCuratedKevModel>[2]>;
  return {
    source,
    fetchPaths,
    async readCacheDiagnostics(): Promise<DirectCacheDiagnostics> {
      const cache = activeCache;
      const cacheNames = await caches.keys();
      const keys = cache ? await cache.keys() : [];
      const cacheKeyUrls = keys.map((request) => request.url);
      const cachedDigests = new Set(
        cacheKeyUrls
          .map((url) => new URL(url).searchParams.get('digest'))
          .filter((digest): digest is string => digest !== null)
      );
      const assetNames = Object.keys(manifestFiles).sort();
      const cachedAssetNames = assetNames.filter((name) => cachedDigests.has(manifestFiles[name]!.sha256));
      const estimate = await navigator.storage?.estimate();
      return {
        cacheNames,
        namespace: base.href,
        cacheKeyCount: keys.length,
        cacheKeyUrls,
        cachedAssetNames,
        missingAssetNames: assetNames.filter((name) => !cachedAssetNames.includes(name)),
        cachePutFailures: [...cachePutFailures],
        storageEstimate: {usage: estimate?.usage ?? null, quota: estimate?.quota ?? null}
      };
    }
  };
}

function createPackageDescription(manifestSha256: string): Parameters<typeof loadCuratedKevModel>[1] {
  const {adapter: _adapter, ...described} = MODEL_CATALOG_BY_ID[MODEL_ID]!.description;
  return {
    ...described,
    modelId: MODEL_ID,
    model: MODEL_ID,
    checkpoint: {
      repository: 'jaredpalmer/kev-0.6b',
      revision: 'abe7110dedfd68a185839550079bcf7c26f561ac'
    },
    assets: {
      ...described.assets,
      manifestSha256,
      directArtifactRoot: DIRECT_ARTIFACT_ROOT
    }
  } as Parameters<typeof loadCuratedKevModel>[1];
}

async function readDefaultSourceCacheDiagnostics(
  manifestSha256: string,
  manifestFiles: Readonly<Record<string, {sha256: string}>>
): Promise<DirectCacheDiagnostics> {
  const base = new URL(`/.web-decision-package-cache/${MODEL_ID}/${manifestSha256}/`, location.origin);
  const cacheNames = await caches.keys();
  const cache = await caches.open(CACHE_NAME);
  const keys = await cache.keys();
  const cacheKeyUrls = keys
    .map((request) => request.url)
    .filter((url) => new URL(url).pathname.startsWith(base.pathname));
  const cachedDigests = new Set(
    cacheKeyUrls
      .map((url) => new URL(url).searchParams.get('digest'))
      .filter((digest): digest is string => digest !== null)
  );
  const assetNames = Object.keys(manifestFiles).sort();
  const cachedAssetNames = assetNames.filter((name) => cachedDigests.has(manifestFiles[name]!.sha256));
  const estimate = await navigator.storage?.estimate();
  return {
    cacheNames,
    namespace: base.href,
    cacheKeyCount: cacheKeyUrls.length,
    cacheKeyUrls,
    cachedAssetNames,
    missingAssetNames: assetNames.filter((name) => !cachedAssetNames.includes(name)),
    cachePutFailures: [],
    storageEstimate: {usage: estimate?.usage ?? null, quota: estimate?.quota ?? null}
  };
}

export async function runDirectKevDefaultSourceSmoke(
  manifestSha256: string,
  cases: readonly RepresentativeDecisionCase[],
  manifestFiles: Readonly<Record<string, {sha256: string}>>
): Promise<DefaultSourceSmokeOutput> {
  const [baselineCase] = cases;
  if (!baselineCase) throw new Error('The pinned direct Kev baseline case is missing.');
  const description = createPackageDescription(manifestSha256);
  recordProgress('default-source-online-load-started', {providerInjected: false});
  const onlineModel = await loadCuratedKevModel(
    {model: MODEL_ID, cache: true, offline: false, maxPrefetchBytes: 0},
    description
  );
  recordProgress('default-source-online-loaded', {providerInjected: false});
  let onlineBaseline: GateDecision;
  try {
    const [online] = await runDecisions(onlineModel, [baselineCase], 'default-source-online-baseline');
    if (!online) throw new Error('The default-source direct Kev baseline was not returned.');
    onlineBaseline = online;
  } finally {
    await onlineModel.dispose();
  }
  const cacheDiagnostics = await readDefaultSourceCacheDiagnostics(manifestSha256, manifestFiles);
  recordProgress('default-source-cache-diagnostics-complete', cacheDiagnostics);

  recordProgress('default-source-offline-load-started', {providerInjected: false});
  const offlineModel = await loadCuratedKevModel(
    {model: MODEL_ID, cache: true, offline: true, maxPrefetchBytes: 0},
    description
  );
  recordProgress('default-source-offline-loaded', {providerInjected: false});
  let offlineBaseline: GateDecision;
  try {
    const [offline] = await runDecisions(offlineModel, [baselineCase], 'default-source-offline-baseline');
    if (!offline) throw new Error('The default-source offline direct Kev baseline was not returned.');
    offlineBaseline = offline;
  } finally {
    await offlineModel.dispose();
  }
  recordProgress('default-source-smoke-complete', {onlineBaseline, offlineBaseline});
  return {onlineBaseline, offlineBaseline, cacheDiagnostics};
}

export async function runDirectKevPackageGate(
  manifestSha256: string,
  cases: readonly RepresentativeDecisionCase[],
  cpuBaseline: CpuBaseline
): Promise<GateOutput> {
  recordProgress('native-input-limits-verified', NATIVE_KEV_INPUT_LIMITS);
  recordProgress('adapter-request-started');
  const adapter = await readAdapterDiagnostics();
  recordProgress('adapter-ready', adapter);
  await caches.delete(CACHE_NAME);
  let publicOptionsRejected = false;
  try {
    await (loadDecisionModel as (input: unknown) => Promise<unknown>)({
      modelId: MODEL_ID,
      assetsUrl: DIRECT_ARTIFACT_ROOT
    });
  } catch (cause) {
    publicOptionsRejected = (cause as {code?: unknown} | null)?.code === 'INVALID_INPUT';
  }
  if (!publicOptionsRejected) throw new Error('The public loader accepted unsupported direct package options.');
  recordProgress('public-options-checked', {publicOptionsRejected});

  const description = createPackageDescription(manifestSha256);
  const testSource = createTestSource(manifestSha256);
  recordProgress('online-model-load-started');
  const onlineModel = await loadCuratedKevModel(
    {model: MODEL_ID, cache: true, offline: false, maxPrefetchBytes: 0},
    description,
    testSource.source
  );
  recordProgress('online-model-loaded', {fetchedPaths: [...testSource.fetchPaths]});
  if (!testSource.fetchPaths.includes('manifest.json')) {
    throw new Error('The direct Kev test provider was not used to fetch manifest.json.');
  }
  let online: GateDecision[];
  let boundaryProbe: NonNullable<GateOutput['boundaryProbe']> | undefined;
  let boundaryProbeError: string | undefined;
  let qualityIssues: string[];
  let cacheDiagnostics: DirectCacheDiagnostics;
  try {
    online = await runDecisions(onlineModel, cases, 'online-matrix');
    qualityIssues = pinnedGpuMatrixIssues(online, cases, cpuBaseline);
    recordProgress('online-matrix-complete', {decisions: online, qualityIssues});
    cacheDiagnostics = await testSource.readCacheDiagnostics();
    recordProgress('online-cache-diagnostics-complete', cacheDiagnostics);
    recordProgress('online-phase-complete', {
      adapter,
      qualityIssues,
      online,
      cacheDiagnostics
    });
  } finally {
    await onlineModel.dispose();
  }
  const onlineFetchPaths = [...testSource.fetchPaths];
  testSource.fetchPaths.length = 0;

  recordProgress('offline-model-load-started');
  const offlineModel = await loadCuratedKevModel(
    {model: MODEL_ID, cache: true, offline: true, maxPrefetchBytes: 0},
    description,
    testSource.source
  );
  recordProgress('offline-model-loaded');
  let offlineBaseline: GateDecision;
  try {
    const [baseline] = await runDecisions(offlineModel, cases.slice(0, 1), 'offline-baseline');
    if (!baseline) throw new Error('The offline direct Kev baseline was not returned.');
    offlineBaseline = baseline;
    recordProgress('offline-baseline-complete', {offlineBaseline, offlineFetchPaths: [...testSource.fetchPaths]});
    try {
      boundaryProbe = await runTokenLimitProbe(offlineModel);
    } catch (cause) {
      boundaryProbeError = describeFailure(cause);
    }
    recordProgress('offline-phase-complete', {
      offlineBaseline,
      offlineFetchPaths: [...testSource.fetchPaths],
      boundaryProbe: boundaryProbe ?? null,
      boundaryProbeError: boundaryProbeError ?? null
    });
  } finally {
    await offlineModel.dispose();
  }

  recordProgress('direct-package-gate-complete', {qualityIssues});
  return {
    adapter,
    nativeInputLimits: NATIVE_KEV_INPUT_LIMITS,
    publicOptionsRejected,
    onlineFetchPaths,
    offlineFetchPaths: [...testSource.fetchPaths],
    qualityIssues,
    cacheDiagnostics,
    ...(boundaryProbe ? {boundaryProbe} : {}),
    ...(boundaryProbeError ? {boundaryProbeError} : {}),
    online,
    offlineBaseline
  };
}

declare global {
  interface Window {
    runDirectKevPackageGate: typeof runDirectKevPackageGate;
    runDirectKevDefaultSourceSmoke: typeof runDirectKevDefaultSourceSmoke;
  }
}

window.runDirectKevPackageGate = runDirectKevPackageGate;
window.runDirectKevDefaultSourceSmoke = runDirectKevDefaultSourceSmoke;
