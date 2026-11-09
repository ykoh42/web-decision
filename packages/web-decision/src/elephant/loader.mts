import {waitForAbort} from '../assets.js';
import {initializeH2oRuntime} from '../h2o/owner.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {uploadParameterEntries} from '../parameter-upload.mjs';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {elephantVariant, ELEPHANT_LIMITS, ELEPHANT_MODEL} from './contract.mjs';
import {ElephantTvmRunner} from './runner.mjs';
import {createElephantSession} from './session.mjs';
import {openElephantSource} from './source.mjs';
import {createElephantTokenizer} from './tokenizer.mjs';
import type {ErrorCode} from '../curated/types.js';

export {ELEPHANT_LIMITS, ELEPHANT_MODEL};
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
const PROTOTYPE_KEYS = new Set<PropertyKey>(Object.getOwnPropertyNames(Object.prototype));
const fail = (code: ErrorCode, message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

interface ValidatedElephantOptions {
  base: URL;
  signal: AbortSignal | undefined;
  model: string;
}

interface ElephantGpuDevice {
  readonly lost: PromiseLike<unknown>;
  destroy(): void;
}

interface ElephantGpuAdapterCandidate {
  features?: {has?: unknown} | null;
  requestDevice?: unknown;
  info?: {isFallbackAdapter?: unknown} | null;
  isFallbackAdapter?: unknown;
  limits?: Record<string, unknown> | null;
}

interface ElephantGpuAdapter extends ElephantGpuAdapterCandidate {
  features: {has(feature: string): boolean};
  requestDevice(descriptor: {
    requiredFeatures: string[];
    requiredLimits: typeof ELEPHANT_LIMITS;
  }): PromiseLike<ElephantGpuDevice>;
}

interface ElephantGpuApi {
  requestAdapter?: unknown;
}

interface ElephantTensor {
  readonly byteOffset: number;
  readonly dtype: string;
  readonly shape: number[];
  copyFrom(data: unknown): unknown;
  getDataPtr(): number;
  toArray(): unknown;
}

interface ElephantTvmRuntime {
  readonly lib?: {
    readonly webGPUContext?: {
      copyRawBytesToBuffer?: (bytes: Uint8Array, pointer: number, offset: number, length: number) => void;
    };
  };
  empty(shape: number[], dtype: string, device: unknown): ElephantTensor;
  cpu(): unknown;
  dispose(): unknown;
}

interface ElephantRuntimeOwner {
  runtime: ElephantTvmRuntime;
  gpu: ElephantGpuDevice;
  device: unknown;
  fn: (...args: unknown[]) => ElephantTensor | null | undefined;
  lifecycle: {loss: unknown; disposed: boolean};
  disposalGuard: {blocked: boolean; retain(): void};
  uncaptured: unknown[];
  teardownObserver: unknown;
  teardownTimeoutMs: number;
  [key: string]: unknown;
}

interface ElephantUploadEntry {
  readonly shape: number[];
  readonly dtype: string;
}

interface ElephantUploadTensor {
  readonly byteOffset: number;
  getDataPtr(): number;
}

interface ElephantReadbackLease {
  syncWithoutReadback(): Promise<void>;
}

interface ElephantOrchestrationDeps {
  gpu?: ElephantGpuApi | null | undefined;
  beforeDevice?: unknown;
  openSource: typeof openElephantSource;
  prepareTokenizer(data: unknown, config: unknown): ReturnType<typeof createElephantTokenizer>;
  instantiate(wasm: Uint8Array): ElephantTvmRuntime | PromiseLike<ElephantTvmRuntime>;
  initialize(runtime: ElephantTvmRuntime, gpu: ElephantGpuDevice): ElephantRuntimeOwner;
  upload(
    owner: ElephantRuntimeOwner,
    source: Awaited<ReturnType<typeof openElephantSource>>,
    signal: AbortSignal
  ): Promise<unknown[]>;
  runner(args: ConstructorParameters<typeof ElephantTvmRunner>[0]): InstanceType<typeof ElephantTvmRunner>;
  session(args: Parameters<typeof createElephantSession>[0]): ReturnType<typeof createElephantSession>;
  fake?: unknown;
}

interface ElephantRuntimeBindings {
  Tokenizer?: unknown;
  instantiate?: unknown;
  createPolyfillWASI?: unknown;
  gpu?: ElephantGpuApi | null | undefined;
  beforeDevice?: unknown;
}

export function validateElephantOptions(options: unknown): ValidatedElephantOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Elephant 9B options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null || Reflect.ownKeys(prototype).some((key) => !PROTOTYPE_KEYS.has(key)))
  )
    throw fail('INVALID_INPUT', 'Unsupported option prototype.');
  const snapshot: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Elephant 9B load option.');
    snapshot[key] = descriptor.value;
  }
  if (
    !elephantVariant(snapshot.model) ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw fail('INVALID_INPUT', 'Unsupported Elephant 9B load option.');
  let base: URL;
  try {
    base = new URL(snapshot.assetsUrl as string, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Elephant 9B asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Elephant 9B asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal as AbortSignal | undefined, model: snapshot.model as string};
}

export function validateElephantAdapter(
  adapter: ElephantGpuAdapterCandidate | null | undefined
): asserts adapter is ElephantGpuAdapter {
  if (
    typeof adapter?.features?.has !== 'function' ||
    !(adapter.features.has as (feature: string) => boolean)('shader-f16') ||
    typeof adapter.requestDevice !== 'function' ||
    adapter.info?.isFallbackAdapter ||
    adapter.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Elephant 9B requires a hardware WebGPU adapter with shader-f16.');
  for (const [name, minimum] of Object.entries(ELEPHANT_LIMITS))
    if (
      !Number.isSafeInteger(adapter.limits?.[name]) ||
      ((adapter.limits as Record<string, unknown>)[name] as number) < minimum
    )
      throw fail('UNSUPPORTED_ENVIRONMENT', `Elephant 9B requires ${name} of at least ${minimum}.`);
}

async function upload(
  owner: ElephantRuntimeOwner,
  source: Awaited<ReturnType<typeof openElephantSource>>,
  signal: AbortSignal
): Promise<unknown[]> {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function') throw new Error('Pinned TVM upload API unavailable.');
  return uploadParameterEntries(source.entries, {
    signal,
    expectedCount: 684,
    createTensor(entry: ElephantUploadEntry) {
      assertUsable(owner);
      if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.uncaptured.length)
        throw new Error('GPU unavailable.');
      return owner.runtime.empty(entry.shape, entry.dtype, owner.device);
    },
    readRange: (entry, index, part) => source.readRange(entry, index, part, signal),
    writeRange: (tensor: ElephantUploadTensor, offset: number, bytes: Uint8Array) =>
      withStrandsScope(owner, 'Elephant 9B upload failed.', async (lease: ElephantReadbackLease) => {
        if (typeof tensor.byteOffset !== 'number' || tensor.byteOffset % 4) throw new Error('Invalid tensor offset.');
        context.copyRawBytesToBuffer!(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
        await lease.syncWithoutReadback();
      })
  });
}

function cleanup(
  owner: ElephantRuntimeOwner | undefined,
  runtime: ElephantTvmRuntime | undefined,
  gpu: ElephantGpuDevice | undefined,
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

async function orchestrate(
  {base, signal, model}: ValidatedElephantOptions,
  deps: ElephantOrchestrationDeps
): Promise<unknown> {
  const {maxBranchTokens} = elephantVariant(model)!;
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let gpu: ElephantGpuDevice | undefined,
    runtime: ElephantTvmRuntime | undefined,
    owner: ElephantRuntimeOwner | undefined,
    watch: ReturnType<typeof watchLoadDeviceLoss> | null | undefined,
    initialized = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(
      () => (deps.gpu!.requestAdapter as () => PromiseLike<ElephantGpuAdapterCandidate | null | undefined>)(),
      signal
    );
    validateElephantAdapter(adapter);
    const source = await deps.openSource(base, {signal, model} as Parameters<typeof openElephantSource>[1]);
    signal?.throwIfAborted();
    const tokenizer = await waitForAbort(() => deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig), signal);
    if (deps.beforeDevice) await (deps.beforeDevice as () => unknown | PromiseLike<unknown>)();
    signal?.throwIfAborted();
    gpu = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: ELEPHANT_LIMITS}),
      signal,
      (late: ElephantGpuDevice | null | undefined) => {
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
      (late: ElephantTvmRuntime | null | undefined) => {
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
      throw new Error('Elephant 9B runtime unavailable.');
    const runner = deps.runner({...owner, weights, maxBranchTokens});
    const session = deps.session({...tokenizer, runner, maxBranchTokens});
    watch.signal.throwIfAborted();
    watch.close();
    watch = null;
    return session;
  } catch (cause) {
    watch?.close();
    cleanup(owner, runtime, gpu, initialized);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Elephant 9B candidate.', cause);
  }
}

export function loadElephantCandidate(
  options: unknown,
  {
    Tokenizer,
    instantiate,
    createPolyfillWASI,
    gpu = globalThis.navigator?.gpu as unknown as ElephantGpuApi | undefined,
    beforeDevice
  }: ElephantRuntimeBindings = {}
): Promise<unknown> {
  const checked = validateElephantOptions(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Elephant 9B runtime bindings are required.');
  if (beforeDevice !== undefined && typeof beforeDevice !== 'function')
    throw new TypeError('Invalid pre-device guard.');
  return orchestrate(checked, {
    gpu,
    beforeDevice,
    openSource: openElephantSource,
    prepareTokenizer: (data, config) => createElephantTokenizer(Tokenizer, data, config),
    instantiate: (wasm) =>
      (instantiate as (bytes: Uint8Array, wasi: unknown) => ElephantTvmRuntime | PromiseLike<ElephantTvmRuntime>)(
        wasm,
        (createPolyfillWASI as () => unknown)()
      ),
    initialize: initializeH2oRuntime,
    upload,
    runner: (args) => new ElephantTvmRunner(args),
    session: createElephantSession
  });
}

export function loadFakeElephantCandidate(
  options: unknown,
  deps: ElephantOrchestrationDeps | (ElephantOrchestrationDeps & {fake?: unknown})
): Promise<unknown> {
  if ((deps as (ElephantOrchestrationDeps & {fake?: unknown}) | null | undefined)?.fake !== true)
    throw new TypeError('Fake dependencies required.');
  return orchestrate(validateElephantOptions(options), deps);
}
