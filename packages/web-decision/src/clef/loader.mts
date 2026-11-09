import {waitForAbort} from '../assets.js';
import {initializeH2oRuntime} from '../h2o/owner.mjs';
import type {H2oRuntime} from '../h2o/owner.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {CLEF_LIMITS, CLEF_MODEL} from './contract.mjs';
import {ClefTvmRunner} from './runner.mjs';
import type {ClefRunnerArgs, TvmRuntime, TvmTensor} from './runner.mjs';
import {createClefSession} from './session.mjs';
import {openClefSource} from './source.mjs';
import {createClefTokenizer} from './tokenizer.mjs';
import {uploadClefParameters} from './upload.mjs';
import type {ErrorCode} from '../curated/types.js';
import type {ClefTokenizer, ClefSource, ClefTensorEntry} from './types.js';

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

interface ClefGpuDevice {
  readonly lost: PromiseLike<unknown>;
  destroy(): void;
}

interface ClefGpuAdapter {
  readonly features: {has(feature: string): boolean};
  readonly limits?: Record<string, number>;
  readonly info?: {isFallbackAdapter?: boolean};
  readonly isFallbackAdapter?: boolean;
  requestDevice(descriptor: {
    requiredFeatures: string[];
    requiredLimits: typeof CLEF_LIMITS;
  }): PromiseLike<ClefGpuDevice>;
}

interface ClefGpuApi {
  requestAdapter(): PromiseLike<ClefGpuAdapter | null>;
}

type ClefLoadRuntime = TvmRuntime & H2oRuntime;

interface ClefRuntimeOwner {
  runtime: ClefLoadRuntime;
  device: unknown;
  gpu: ClefGpuDevice;
  fn: (...args: unknown[]) => TvmTensor;
  teardownObserver: unknown;
  teardownTimeoutMs: number;
  lifecycle: {loss: unknown; disposed: boolean};
  disposalGuard: {blocked: boolean; retain(): void};
  uncaptured: unknown[];
}

interface ClefTokenizerBundle {
  tokenizer: ClefTokenizer;
  controlIds: number[];
}

interface ClefSessionRunner {
  run(packed: unknown, options?: {signal?: AbortSignal | undefined}): Promise<Float32Array[]>;
  dispose(): unknown | PromiseLike<unknown>;
  releaseCachedMemory(): unknown | PromiseLike<unknown>;
}

interface ClefOrchestrationDeps {
  gpu?: ClefGpuApi | undefined;
  beforeDevice?: (() => unknown | PromiseLike<unknown>) | undefined;
  openSource(base: URL, options?: {signal?: AbortSignal | undefined}): Promise<ClefSource>;
  prepareTokenizer(data: unknown, config: unknown): ClefTokenizerBundle | PromiseLike<ClefTokenizerBundle>;
  instantiate(wasm: Uint8Array): ClefLoadRuntime | PromiseLike<ClefLoadRuntime>;
  initialize(runtime: ClefLoadRuntime, gpu: ClefGpuDevice): ClefRuntimeOwner;
  upload(owner: ClefRuntimeOwner, source: ClefSource, signal: AbortSignal): Promise<unknown[]>;
  runner(args: ClefRunnerArgs): ClefSessionRunner;
  session(args: ClefTokenizerBundle & {runner: ClefSessionRunner}): unknown;
}

interface ValidatedClefOptions {
  base: URL;
  signal: AbortSignal | undefined;
}

interface ClefCandidateDependencies {
  Tokenizer?: unknown;
  instantiate?: (wasm: Uint8Array, wasi: unknown) => ClefLoadRuntime | PromiseLike<ClefLoadRuntime>;
  createPolyfillWASI?: () => unknown;
  gpu?: ClefGpuApi;
  beforeDevice?: () => unknown | PromiseLike<unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasAbortErrorName(value: unknown): boolean {
  return (
    ((typeof value === 'object' && value !== null) || typeof value === 'function') &&
    'name' in value &&
    value.name === 'AbortError'
  );
}

export function validateClefOptions(options: unknown): ValidatedClefOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Clef options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null ||
      Reflect.ownKeys(prototype).some((key) => typeof key !== 'string' || !PROTOTYPE_KEYS.has(key)))
  )
    throw fail('INVALID_INPUT', 'Unsupported option prototype.');
  const snapshot: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string') throw fail('INVALID_INPUT', 'Unsupported Clef load option.');
    if (!KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Clef load option.');
    snapshot[key] = descriptor.value;
  }
  if (
    snapshot.model !== CLEF_MODEL ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw fail('INVALID_INPUT', 'Unsupported Clef load option.');
  let base: URL;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Clef asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Clef asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal as AbortSignal | undefined};
}

export function validateClefAdapter(value: unknown): asserts value is ClefGpuAdapter {
  const adapter = isRecord(value) ? value : null;
  const features = isRecord(adapter?.features) ? adapter.features : null;
  if (
    !adapter ||
    !features ||
    typeof features.has !== 'function' ||
    !(features.has as (feature: string) => boolean)('shader-f16') ||
    typeof adapter.requestDevice !== 'function' ||
    !isRecord(adapter.info) ||
    adapter.info.isFallbackAdapter !== false ||
    adapter.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Clef requires a hardware WebGPU adapter with shader-f16.');
  const checked = adapter as unknown as ClefGpuAdapter;
  for (const [name, minimum] of Object.entries(CLEF_LIMITS))
    if (!Number.isSafeInteger(checked.limits?.[name]) || (checked.limits?.[name] ?? -Infinity) < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `Clef requires ${name} of at least ${minimum}.`);
}

async function upload(owner: ClefRuntimeOwner, source: ClefSource, signal: AbortSignal): Promise<unknown[]> {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof context?.gpuBufferFromPtr !== 'function')
    throw new Error('Pinned TVM upload API unavailable.');
  return uploadClefParameters(source.entries, {
    signal,
    createTensor(entry: ClefTensorEntry) {
      assertUsable(owner);
      if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.uncaptured.length)
        throw new Error('GPU unavailable.');
      const tensor = owner.runtime.empty(entry.shape, entry.dtype, owner.device);
      if (
        typeof tensor?.getDataPtr !== 'function' ||
        !Number.isSafeInteger(tensor.byteOffset) ||
        tensor.byteOffset < 0 ||
        tensor.byteOffset % 4
      )
        throw new Error('Invalid Clef tensor offset.');
      const buffer = context.gpuBufferFromPtr!(tensor.getDataPtr());
      if (
        !buffer ||
        !Number.isSafeInteger(buffer.size) ||
        buffer.size < tensor.byteOffset + Math.ceil(entry.bytes / 4) * 4
      )
        throw new Error('Clef tensor allocation too small.');
      return tensor;
    },
    readRange: (entry, part, readSignal) => source.readRange(entry, part, readSignal),
    writeRange: (tensor, offset, bytes) =>
      withStrandsScope(owner, 'Clef upload failed.', async (lease: {syncWithoutReadback(): Promise<void>}) => {
        if (!isUploadTensor(tensor)) throw new TypeError('Invalid Clef upload tensor.');
        context.copyRawBytesToBuffer!(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
        await lease.syncWithoutReadback();
      })
  });
}

interface UploadTensor {
  byteOffset: number;
  getDataPtr(): number;
}

function isUploadTensor(value: unknown): value is UploadTensor {
  return isRecord(value) && typeof value.byteOffset === 'number' && typeof value.getDataPtr === 'function';
}

function cleanup(
  owner: ClefRuntimeOwner | undefined,
  runtime: TvmRuntime | undefined,
  gpu: ClefGpuDevice | undefined,
  initialized: boolean
): void {
  if (owner) {
    try {
      owner.disposalGuard.retain();
    } catch {
      retainStrandsResources(runtime, gpu);
    }
  } else if (runtime && initialized) retainStrandsResources(runtime, gpu);
  else {
    try {
      runtime?.dispose();
    } catch {
      retainStrandsResources(runtime, gpu);
    }
    try {
      gpu?.destroy();
    } catch {}
  }
}

async function orchestrate({base, signal}: ValidatedClefOptions, deps: ClefOrchestrationDeps): Promise<unknown> {
  const gpuApi = deps.gpu;
  if (typeof gpuApi?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let gpu: ClefGpuDevice | undefined,
    runtime: ClefLoadRuntime | undefined,
    owner: ClefRuntimeOwner | undefined,
    watch: ReturnType<typeof watchLoadDeviceLoss> | undefined,
    initialized = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => gpuApi.requestAdapter(), signal);
    if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU adapter unavailable.');
    validateClefAdapter(adapter);
    const source = await deps.openSource(base, {signal});
    signal?.throwIfAborted();
    const tokenizer = await waitForAbort(() => deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig), signal);
    if (deps.beforeDevice) await deps.beforeDevice();
    signal?.throwIfAborted();
    gpu = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: CLEF_LIMITS}),
      signal,
      (late: ClefGpuDevice | null | undefined) => {
        try {
          late?.destroy();
        } catch {}
      }
    );
    signal?.throwIfAborted();
    watch = watchLoadDeviceLoss(gpu, signal);
    runtime = await waitForAbort(
      () => deps.instantiate(source.wasm),
      watch.signal,
      (late: TvmRuntime | null | undefined) => {
        try {
          late?.dispose();
        } catch {
          retainStrandsResources(late, gpu);
        }
      }
    );
    watch.signal.throwIfAborted();
    initialized = true;
    owner = deps.initialize(runtime, gpu);
    const weights = await deps.upload(owner, source, watch.signal);
    watch.signal.throwIfAborted();
    if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.disposalGuard.blocked || owner.uncaptured.length)
      throw new Error('Clef runtime unavailable.');
    const runner = deps.runner({...owner, weights, source});
    const session = deps.session({...tokenizer, runner});
    watch.signal.throwIfAborted();
    watch.close();
    watch = undefined;
    return session;
  } catch (cause) {
    let failure = cause;
    try {
      watch?.close();
    } catch (closeError) {
      failure = new AggregateError([cause, closeError], 'Clef watcher cleanup failed.');
    }
    cleanup(owner, runtime, gpu, initialized);
    signal?.throwIfAborted();
    if (failure instanceof WebDecisionError || hasAbortErrorName(failure)) throw failure;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Clef candidate.', failure);
  }
}

export function loadClefCandidate(
  options: unknown,
  {
    Tokenizer,
    instantiate,
    createPolyfillWASI,
    gpu = globalThis.navigator?.gpu as unknown as ClefGpuApi | undefined,
    beforeDevice
  }: ClefCandidateDependencies = {}
): Promise<unknown> {
  const checked = validateClefOptions(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Clef runtime bindings are required.');
  if (beforeDevice !== undefined && typeof beforeDevice !== 'function')
    throw new TypeError('Invalid pre-device guard.');
  const runtimeInstantiate = instantiate;
  const createWasi = createPolyfillWASI;
  return orchestrate(checked, {
    gpu,
    beforeDevice,
    openSource: openClefSource,
    prepareTokenizer: (data: unknown, config: unknown) => createClefTokenizer(Tokenizer, data, config),
    instantiate: (wasm: Uint8Array) => runtimeInstantiate(wasm, createWasi()),
    initialize: initializeH2oRuntime as ClefOrchestrationDeps['initialize'],
    upload,
    runner: (args: ClefRunnerArgs) => new ClefTvmRunner(args),
    session: createClefSession
  });
}
export function loadFakeClefCandidate(
  options: unknown,
  deps: ClefOrchestrationDeps & {fake?: unknown}
): Promise<unknown> {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies required.');
  return orchestrate(validateClefOptions(options), deps);
}
