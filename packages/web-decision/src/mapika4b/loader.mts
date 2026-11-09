import {waitForAbort} from '../assets.js';
import {initializeH2oRuntime} from '../h2o/owner.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import type {ErrorCode} from '../curated/types.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {uploadParameterEntries} from '../parameter-upload.mjs';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {MAPIKA4_DLIGHT_ALL_MODEL, MAPIKA4_DLIGHT_MODEL, MAPIKA4_LIMITS, MAPIKA4_MODEL} from './contract.mjs';
import {Mapika4TvmRunner} from './runner.mjs';
import {createMapika4Session} from './session.mjs';
import {openMapika4Source} from './source.mjs';
import type {Mapika4FileEntry, Mapika4ParameterEntry, Mapika4RangePart, Mapika4Source} from './source.mjs';
import {createMapika4Tokenizer} from './tokenizer.mjs';

export {MAPIKA4_LIMITS, MAPIKA4_MODEL};

type Mapika4Model = typeof MAPIKA4_MODEL | typeof MAPIKA4_DLIGHT_MODEL | typeof MAPIKA4_DLIGHT_ALL_MODEL;

interface Mapika4OptionsSnapshot {
  readonly base: URL;
  readonly signal: AbortSignal | undefined;
  readonly model: Mapika4Model;
}

interface Mapika4Tensor {
  readonly byteOffset?: unknown;
  getDataPtr(): unknown;
}

interface Mapika4WebGpuContext {
  readonly copyRawBytesToBuffer?: unknown;
}

interface Mapika4Runtime {
  readonly lib?: {readonly webGPUContext?: Mapika4WebGpuContext | null | undefined} | null | undefined;
  empty(shape: number[], dtype: string, device: unknown): Mapika4Tensor;
  dispose(): unknown;
}

interface Mapika4GpuDevice {
  readonly lost: PromiseLike<unknown>;
  destroy(): unknown;
}

interface Mapika4GpuAdapter {
  readonly features: {has(feature: string): boolean};
  readonly requestDevice: Mapika4DeviceFactory;
  readonly info?: {readonly isFallbackAdapter?: unknown} | null | undefined;
  readonly isFallbackAdapter?: unknown;
  readonly limits?: Readonly<Record<string, unknown>> | null | undefined;
}

interface Mapika4GpuProvider {
  readonly requestAdapter?: unknown;
}

interface Mapika4Lifecycle {
  readonly loss: unknown;
  readonly disposed: boolean;
}

interface Mapika4DisposalGuard {
  readonly blocked: boolean;
  retain(): void;
}

interface Mapika4Owner {
  readonly runtime: Mapika4Runtime;
  readonly gpu: Mapika4GpuDevice;
  readonly device: unknown;
  readonly lifecycle: Mapika4Lifecycle;
  readonly disposalGuard: Mapika4DisposalGuard;
  readonly uncaptured: readonly unknown[];
  readonly fn: (...args: unknown[]) => unknown;
  readonly teardownObserver: {
    beforeDispose(): unknown;
    afterDispose(...args: unknown[]): unknown;
  };
  readonly teardownTimeoutMs: number;
}

interface Mapika4TokenizerResult {
  readonly tokenizer: {encode(...args: unknown[]): unknown};
  readonly labels: {
    readonly names: readonly string[];
    readonly ids: readonly number[];
    readonly open: readonly number[];
  };
  readonly limits: {
    readonly controlIds: readonly number[];
    readonly maxStateTokens: number;
    readonly maxInputTokens: number;
    readonly temperature: number;
  };
}

interface Mapika4Runner {
  run(packed: {
    readonly ids: readonly number[];
    readonly groups: readonly {readonly slot: number; readonly options: readonly string[]}[];
  }): readonly Float32Array[] | PromiseLike<readonly Float32Array[]>;
  releaseCachedMemory(): void | PromiseLike<void>;
  dispose(): unknown | PromiseLike<unknown>;
}

interface Mapika4RunnerArgs extends Mapika4Owner {
  readonly weights: unknown[];
}

interface Mapika4LoadDependencies {
  readonly fake?: boolean | undefined;
  readonly gpu: Mapika4GpuProvider | null | undefined;
  readonly beforeDevice?: (() => unknown) | undefined;
  readonly openSource: (
    base: URL,
    options: {readonly signal: AbortSignal | undefined; readonly model: Mapika4Model}
  ) => Mapika4Source | PromiseLike<Mapika4Source>;
  readonly prepareTokenizer: (
    data: unknown,
    config: unknown
  ) => Mapika4TokenizerResult | PromiseLike<Mapika4TokenizerResult>;
  readonly instantiate: (wasm: Uint8Array) => Mapika4Runtime | PromiseLike<Mapika4Runtime>;
  readonly initialize: (runtime: Mapika4Runtime, gpu: Mapika4GpuDevice) => Mapika4Owner;
  readonly upload: (
    owner: Mapika4Owner,
    source: Mapika4Source,
    signal: AbortSignal
  ) => unknown[] | PromiseLike<unknown[]>;
  readonly runner: (args: Mapika4RunnerArgs) => Mapika4Runner;
  readonly session: (args: Mapika4TokenizerResult & {readonly runner: Mapika4Runner}) => unknown;
}

interface Mapika4Bindings {
  readonly Tokenizer?: unknown;
  readonly instantiate?: unknown;
  readonly createPolyfillWASI?: unknown;
  readonly gpu?: Mapika4GpuProvider | null | undefined;
  readonly beforeDevice?: (() => unknown) | undefined;
}

interface Mapika4LossWatch {
  readonly signal: AbortSignal;
  close(): void;
}

type Mapika4RuntimeFactory = (wasm: Uint8Array, wasi: unknown) => Mapika4Runtime | PromiseLike<Mapika4Runtime>;
type Mapika4WasiFactory = () => unknown;
type Mapika4DeviceFactory = (descriptor: {
  readonly requiredFeatures: string[];
  readonly requiredLimits: typeof MAPIKA4_LIMITS;
}) => Mapika4GpuDevice | PromiseLike<Mapika4GpuDevice>;
type Mapika4CopyRawBytes = (bytes: Uint8Array, pointer: unknown, byteOffset: number, byteLength: number) => unknown;

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
const PROTOTYPE_KEYS = new Set(Object.getOwnPropertyNames(Object.prototype));
const fail = (code: ErrorCode, message: string, cause?: unknown) =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateMapika4Options(options: unknown): Mapika4OptionsSnapshot {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Mapika 4B options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null ||
      Reflect.ownKeys(prototype).some((key) => !PROTOTYPE_KEYS.has(key as string)))
  )
    throw fail('INVALID_INPUT', 'Unsupported option prototype.');
  const snapshot: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Mapika 4B load option.');
    snapshot[key] = descriptor.value;
  }
  if (
    ![MAPIKA4_MODEL, MAPIKA4_DLIGHT_MODEL, MAPIKA4_DLIGHT_ALL_MODEL].includes(snapshot.model as string) ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw fail('INVALID_INPUT', 'Unsupported Mapika 4B load option.');
  let base: URL;
  try {
    base = new URL(snapshot.assetsUrl as string, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Mapika 4B asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Mapika 4B asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {
    base,
    signal: snapshot.signal as AbortSignal | undefined,
    model: snapshot.model as Mapika4Model
  };
}

function validateMapika4Adapter(adapter: unknown): asserts adapter is Mapika4GpuAdapter {
  const candidate = adapter as Mapika4GpuAdapter | null | undefined;
  if (
    typeof candidate?.features?.has !== 'function' ||
    !candidate.features.has('shader-f16') ||
    typeof candidate.requestDevice !== 'function' ||
    candidate.info?.isFallbackAdapter ||
    candidate.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Mapika 4B requires a hardware WebGPU adapter with shader-f16.');
  for (const [name, minimum] of Object.entries(MAPIKA4_LIMITS))
    if (!Number.isSafeInteger(candidate.limits?.[name]) || (candidate.limits?.[name] as number) < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `Mapika 4B requires ${name} of at least ${minimum}.`);
}

async function upload(owner: Mapika4Owner, source: Mapika4Source, signal: AbortSignal): Promise<unknown[]> {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function') throw new Error('Pinned TVM upload API unavailable.');
  return (
    uploadParameterEntries as unknown as (
      entries: readonly Mapika4ParameterEntry[],
      options: {
        readonly signal?: AbortSignal | null | undefined;
        readonly createTensor: (entry: Mapika4ParameterEntry) => Mapika4Tensor;
        readonly readRange: (
          entry: Mapika4FileEntry,
          index: number,
          part: Mapika4RangePart
        ) => ArrayBuffer | PromiseLike<ArrayBuffer>;
        readonly writeRange: (
          tensor: Mapika4Tensor,
          offset: number,
          bytes: Uint8Array
        ) => unknown | PromiseLike<unknown>;
      }
    ) => Promise<unknown[]>
  )(source.entries, {
    signal,
    createTensor(entry: Mapika4ParameterEntry) {
      assertUsable(owner);
      if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.uncaptured.length)
        throw new Error('GPU unavailable.');
      return owner.runtime.empty(entry.shape, entry.dtype, owner.device);
    },
    readRange: (entry: Mapika4FileEntry, index: number, part: Mapika4RangePart) =>
      source.readRange(entry, index, part, signal),
    writeRange: (tensor: Mapika4Tensor, offset: number, bytes: Uint8Array) =>
      withStrandsScope(owner, 'Mapika 4B upload failed.', async (lease: {syncWithoutReadback(): PromiseLike<void>}) => {
        if (typeof tensor.byteOffset !== 'number' || tensor.byteOffset % 4) throw new Error('Invalid tensor offset.');
        (context.copyRawBytesToBuffer as Mapika4CopyRawBytes)(
          bytes,
          tensor.getDataPtr(),
          tensor.byteOffset + offset,
          bytes.byteLength
        );
        await lease.syncWithoutReadback();
      })
  });
}

function cleanup(
  owner: Mapika4Owner | undefined,
  runtime: Mapika4Runtime | undefined,
  gpu: Mapika4GpuDevice | undefined,
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
  {base, signal, model}: Mapika4OptionsSnapshot,
  deps: Mapika4LoadDependencies
): Promise<unknown> {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let gpu: Mapika4GpuDevice | undefined,
    runtime: Mapika4Runtime | undefined,
    owner: Mapika4Owner | undefined,
    watch: Mapika4LossWatch | null = null,
    initialized = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => (deps.gpu!.requestAdapter as () => unknown)(), signal);
    validateMapika4Adapter(adapter);
    const source = await deps.openSource(base, {signal, model});
    signal?.throwIfAborted();
    const tokenizer = await waitForAbort(() => deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig), signal);
    if (deps.beforeDevice) await deps.beforeDevice();
    signal?.throwIfAborted();
    gpu = await waitForAbort(
      () =>
        adapter.requestDevice!({
          requiredFeatures: ['shader-f16'],
          requiredLimits: MAPIKA4_LIMITS
        }),
      signal,
      (late) => {
        try {
          late?.destroy();
        } catch {}
      }
    );
    signal?.throwIfAborted();
    watch = watchLoadDeviceLoss(gpu, signal) as Mapika4LossWatch;
    runtime = await waitForAbort(
      () => deps.instantiate(source.wasm),
      watch.signal,
      (late) => {
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
      throw new Error('Mapika 4B runtime unavailable.');
    const runner = deps.runner({...owner, weights});
    const session = deps.session({...tokenizer, runner});
    watch.signal.throwIfAborted();
    watch.close();
    watch = null;
    return session;
  } catch (cause: unknown) {
    watch?.close();
    cleanup(owner, runtime, gpu, initialized);
    signal?.throwIfAborted();
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Mapika 4B candidate.', cause);
  }
}

export function loadMapika4Candidate(
  options: unknown,
  {
    Tokenizer,
    instantiate,
    createPolyfillWASI,
    gpu = (globalThis.navigator as Navigator & {readonly gpu?: Mapika4GpuProvider}).gpu,
    beforeDevice
  }: Mapika4Bindings = {}
): Promise<unknown> {
  const checked = validateMapika4Options(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Mapika 4B runtime bindings are required.');
  if (beforeDevice !== undefined && typeof beforeDevice !== 'function')
    throw new TypeError('Invalid pre-device guard.');
  return orchestrate(checked, {
    gpu,
    beforeDevice,
    openSource: openMapika4Source,
    prepareTokenizer: (data, config) => createMapika4Tokenizer(Tokenizer, data, config),
    instantiate: (wasm) => (instantiate as Mapika4RuntimeFactory)(wasm, (createPolyfillWASI as Mapika4WasiFactory)()),
    initialize: initializeH2oRuntime,
    upload,
    runner: (args) => new Mapika4TvmRunner(args),
    session: createMapika4Session
  });
}

export function loadFakeMapika4Candidate(options: unknown, deps: unknown): Promise<unknown> {
  if ((deps as {readonly fake?: unknown} | null | undefined)?.fake !== true)
    throw new TypeError('Fake dependencies required.');
  return orchestrate(validateMapika4Options(options), deps as Mapika4LoadDependencies);
}
