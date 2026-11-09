import {waitForAbort} from '../assets.js';
import {initializeH2oRuntime} from '../h2o/owner.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {FLASH_LIMITS, FLASH_MODEL} from './contract.mjs';
import {FlashTvmRunner} from './runner.mjs';
import {createFlashSession} from './session.mjs';
import {openFlashSource} from './source.mjs';
import {createFlashTokenizer} from './tokenizer.mjs';
import {uploadFlashParameters} from './upload-parameters.mjs';
import type {ErrorCode} from '../curated/types.js';
import type {
  FlashDecisionConfig,
  FlashManifestEntry,
  FlashRunnerArgs,
  FlashSource,
  FlashTensor,
  FlashTokenizer,
  FlashTvmRuntime
} from './types.js';

export {FLASH_LIMITS, FLASH_MODEL};
const KEYS = new Set([
  'model',
  'assetsUrl',
  'signal',
  'cache',
  'offline',
  'maxPrefetchBytes',
  'onProgress',
  'debertaReadback'
]);
const PROTOTYPE_KEYS = new Set<string>(Object.getOwnPropertyNames(Object.prototype));
const fail = (code: ErrorCode, message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

interface FlashGpuDevice {
  readonly lost: PromiseLike<unknown>;
  destroy(): void;
}

interface FlashGpuAdapter {
  readonly features: {has(feature: string): boolean};
  readonly limits?: Record<string, number>;
  readonly info?: {isFallbackAdapter?: unknown};
  readonly isFallbackAdapter?: unknown;
  requestDevice(descriptor: {
    requiredFeatures: string[];
    requiredLimits: typeof FLASH_LIMITS;
  }): PromiseLike<FlashGpuDevice>;
}

interface FlashGpuAdapterCandidate {
  features?: {has: (feature: string) => boolean} | null;
  limits?: Record<string, number> | null;
  info?: {isFallbackAdapter?: unknown} | null;
  isFallbackAdapter?: unknown;
  requestDevice?: unknown;
}

interface FlashGpuApi {
  requestAdapter(): PromiseLike<FlashGpuAdapter | null>;
}

interface FlashOwner {
  runtime: FlashTvmRuntime;
  device: unknown;
  gpu: FlashGpuDevice;
  fn: (...args: unknown[]) => FlashTensor;
  lifecycle: {loss: unknown; disposed: boolean};
  disposalGuard: {blocked: boolean; retain(): void};
  uncaptured: unknown[];
  teardownObserver: unknown;
  teardownTimeoutMs: number;
}

interface FlashTokenizerBundle {
  tokenizer: FlashTokenizer;
  config: FlashDecisionConfig;
}

interface FlashSessionRunner {
  run(packed: unknown): Promise<Float32Array[]>;
  dispose(): unknown | PromiseLike<unknown>;
  releaseCachedMemory(): unknown | PromiseLike<unknown>;
}

interface FlashOrchestrationDeps {
  gpu?: FlashGpuApi | undefined;
  beforeDevice?: (() => unknown | PromiseLike<unknown>) | undefined;
  retainResources?: ((runtime: unknown, gpu: unknown) => void) | undefined;
  openSource(base: URL, options?: {signal?: AbortSignal | undefined}): Promise<FlashSource>;
  prepareTokenizer(data: unknown, config: unknown): FlashTokenizerBundle;
  instantiate(wasm: Uint8Array): FlashTvmRuntime | PromiseLike<FlashTvmRuntime>;
  initialize(runtime: FlashTvmRuntime, gpu: FlashGpuDevice): FlashOwner;
  upload(owner: FlashOwner, source: FlashSource, signal: AbortSignal): Promise<unknown[]>;
  runner(args: FlashRunnerArgs): FlashSessionRunner;
  session(args: FlashTokenizerBundle & {runner: FlashSessionRunner}): unknown;
}

interface ValidatedFlashOptions {
  base: URL;
  signal: AbortSignal | undefined;
}

interface FlashCandidateDependencies {
  Tokenizer?: unknown;
  instantiate?: (wasm: Uint8Array, wasi: unknown) => FlashTvmRuntime | PromiseLike<FlashTvmRuntime>;
  createPolyfillWASI?: () => unknown;
  gpu?: FlashGpuApi;
  beforeDevice?: () => unknown | PromiseLike<unknown>;
}

function hasAbortErrorName(value: unknown): boolean {
  return (value as {name?: unknown} | null | undefined)?.name === 'AbortError';
}

export function validateFlashOptions(options: unknown): ValidatedFlashOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Perplexity options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null ||
      Reflect.ownKeys(prototype).some((key) => typeof key !== 'string' || !PROTOTYPE_KEYS.has(key)))
  )
    throw fail('INVALID_INPUT', 'Unsupported Perplexity option prototype.');
  const snapshot: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string') throw fail('INVALID_INPUT', 'Unsupported Perplexity load option.');
    if (!KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Perplexity load option.');
    snapshot[key] = descriptor.value;
  }
  if (
    snapshot.model !== FLASH_MODEL ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw fail('INVALID_INPUT', 'Unsupported Perplexity load option.');
  let base: URL;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Perplexity asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Perplexity asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal as AbortSignal | undefined};
}

function validateFlashAdapter(value: unknown): asserts value is FlashGpuAdapter {
  const adapter = value as FlashGpuAdapterCandidate | null | undefined;
  if (
    typeof adapter?.features?.has !== 'function' ||
    !adapter.features.has('shader-f16') ||
    typeof adapter.requestDevice !== 'function' ||
    adapter.info?.isFallbackAdapter ||
    adapter.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Perplexity requires a hardware WebGPU adapter with shader-f16.');
  for (const [name, minimum] of Object.entries(FLASH_LIMITS))
    if (!Number.isSafeInteger(adapter.limits?.[name]) || (adapter.limits?.[name] ?? -Infinity) < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `Perplexity requires ${name} of at least ${minimum}.`);
}

async function upload(owner: FlashOwner, source: FlashSource, signal: AbortSignal): Promise<unknown[]> {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function') throw new Error('Pinned TVM upload API unavailable.');
  return uploadFlashParameters(source.manifest, {
    signal,
    createTensor(entry: FlashManifestEntry): FlashTensor {
      assertUsable(owner);
      if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.uncaptured.length)
        throw new Error('GPU unavailable.');
      return owner.runtime.empty(entry.shape, entry.dtype, owner.device);
    },
    readRange: (entry, index, part) => source.readRange(entry, index, part, signal),
    writeRange: (tensor, offset, bytes) =>
      withStrandsScope(owner, 'Perplexity upload failed.', async (lease: {syncWithoutReadback(): Promise<void>}) => {
        if (typeof tensor.byteOffset !== 'number' || tensor.byteOffset % 4) throw new Error('Invalid tensor offset.');
        context.copyRawBytesToBuffer!(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
        await lease.syncWithoutReadback();
      })
  });
}

function cleanupFailedLoad(
  owner: FlashOwner | undefined,
  runtime: FlashTvmRuntime | undefined,
  gpu: FlashGpuDevice | undefined,
  initializationStarted: boolean,
  retain: (runtime: unknown, gpu: unknown) => void
): void {
  if (owner) {
    try {
      owner.disposalGuard.retain();
    } catch {
      retain(runtime, gpu);
    }
  } else if (runtime && initializationStarted) retain(runtime, gpu);
  else {
    try {
      runtime?.dispose();
    } catch {
      retain(runtime, gpu);
    }
    try {
      gpu?.destroy();
    } catch {
      retain(undefined, gpu);
    }
  }
}

async function orchestrate({base, signal}: ValidatedFlashOptions, deps: FlashOrchestrationDeps): Promise<unknown> {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  const retain = deps.retainResources ?? retainStrandsResources;
  let gpu: FlashGpuDevice | undefined,
    runtime: FlashTvmRuntime | undefined,
    owner: FlashOwner | undefined,
    watch: ReturnType<typeof watchLoadDeviceLoss> | undefined,
    initializationStarted = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => deps.gpu!.requestAdapter(), signal);
    validateFlashAdapter(adapter);
    const source = await deps.openSource(base, {signal});
    signal?.throwIfAborted();
    const tokenizer = deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig);
    signal?.throwIfAborted();
    if (deps.beforeDevice) await deps.beforeDevice();
    signal?.throwIfAborted();
    const requestedDevice = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: FLASH_LIMITS}),
      signal,
      (late: FlashGpuDevice | null | undefined) => {
        try {
          late?.destroy();
        } catch {
          retain(undefined, late);
        }
      }
    );
    gpu = requestedDevice;
    signal?.throwIfAborted();
    watch = watchLoadDeviceLoss(requestedDevice, signal);
    const loadedRuntime = await waitForAbort(
      () => deps.instantiate(source.wasm),
      watch.signal,
      (late: FlashTvmRuntime | null | undefined) => {
        try {
          late?.dispose();
        } catch {
          retain(late, gpu);
        }
      }
    );
    runtime = loadedRuntime;
    watch.signal.throwIfAborted();
    initializationStarted = true;
    owner = deps.initialize(loadedRuntime, requestedDevice);
    const weights = await deps.upload(owner, source, watch.signal);
    watch.signal.throwIfAborted();
    if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.disposalGuard.blocked || owner.uncaptured.length)
      throw new Error('Perplexity runtime unavailable.');
    const runner = deps.runner({...owner, weights});
    const session = deps.session({...tokenizer, runner});
    watch.signal.throwIfAborted();
    watch.close();
    watch = undefined;
    return session;
  } catch (cause) {
    watch?.close();
    cleanupFailedLoad(owner, runtime, gpu, initializationStarted, retain);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || hasAbortErrorName(cause)) throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Perplexity candidate.', cause);
  }
}

export function loadFlashCandidate(
  options: unknown,
  {
    Tokenizer,
    instantiate,
    createPolyfillWASI,
    gpu = globalThis.navigator?.gpu as unknown as FlashGpuApi | undefined,
    beforeDevice
  }: FlashCandidateDependencies = {}
): Promise<unknown> {
  const checked = validateFlashOptions(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Perplexity runtime bindings are required.');
  if (beforeDevice !== undefined && typeof beforeDevice !== 'function')
    throw new TypeError('Invalid pre-device guard.');
  const runtimeInstantiate = instantiate;
  const createWasi = createPolyfillWASI;
  return orchestrate(checked, {
    gpu,
    beforeDevice,
    openSource: openFlashSource,
    prepareTokenizer: (data: unknown, config: unknown) => createFlashTokenizer(Tokenizer, data, config),
    instantiate: (wasm: Uint8Array) => runtimeInstantiate(wasm, createWasi()),
    initialize: initializeH2oRuntime,
    upload,
    runner: (args: FlashRunnerArgs) => new FlashTvmRunner(args),
    session: createFlashSession
  });
}

export function loadFakeFlashCandidate(
  options: unknown,
  deps: FlashOrchestrationDeps & {fake?: unknown}
): Promise<unknown> {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies required.');
  return orchestrate(validateFlashOptions(options), deps);
}
