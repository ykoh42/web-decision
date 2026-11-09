import {WebDecisionError} from '../kev.js';
import {waitForAbort} from '../assets.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {Hash} from '../h2o/sha256.mjs';
import {validateNariOptions} from './options.mjs';
import {openNariSource} from './source.mjs';
import {createNariTokenizer} from './tokenizer.mjs';
import {createNariSession} from './session.mjs';
import {
  runtimeOwnership,
  uploadFromSource,
  type NariGpuDevice,
  type NariRuntimeOwner,
  type NariRuntimeOwnership,
  type NariTvmRuntime,
  type NariUploadContext,
  type NariUploadSource
} from './runtime.mjs';
import {NariToqRunner} from './runner.mjs';
import {NARI_FULL, NARI_MIXED} from './variants.mjs';

type NariErrorCode = ConstructorParameters<typeof WebDecisionError>[0];
type NariSource = Awaited<ReturnType<typeof openNariSource>> & NariUploadSource;
type NariVariant = typeof NARI_FULL | typeof NARI_MIXED;
type NariCheckedOptions = ReturnType<typeof validateNariOptions>;
type NariUploadedWeights = Awaited<ReturnType<typeof uploadFromSource>>;

interface NariPlanSizeEntry {
  readonly source: string;
  readonly file?: {readonly bytes: number} | undefined;
}

interface NariPlanSizeView {
  readonly ordered: readonly NariPlanSizeEntry[];
  readonly letterFile: {readonly bytes: number};
}

type NariAdapter = GPUAdapter;
type NariGpuProvider = GPU;

interface NariTokenizerResult {
  readonly tokenizer: unknown;
  readonly controlIds: number[];
}

interface NariRunner {
  run(input: unknown): Promise<unknown>;
  releaseCachedMemory(): unknown;
  dispose(): unknown;
}

interface NariRunnerOwner extends NariRuntimeOwner {
  readonly bindings: NariUploadSource['plan']['ordered'];
  readonly parameters: NariUploadedWeights['parameters'];
  readonly letterRows: NariUploadedWeights['letterRows'];
  readonly expectedBindings: number;
}

interface NariUploadResult {
  readonly parameters: NariUploadedWeights['parameters'];
  readonly letterRows: NariUploadedWeights['letterRows'];
}

interface NariLoaderDependencies {
  readonly gpu: NariGpuProvider | null | undefined;
  readonly ownership: () => NariRuntimeOwnership;
  readonly openSource: (
    base: URL,
    options: {readonly signal?: AbortSignal | undefined; readonly variant: NariVariant}
  ) => Promise<NariSource>;
  readonly tokenizer: (tokens: Uint8Array, config: Uint8Array) => NariTokenizerResult;
  readonly instantiate: (wasm: Uint8Array) => PromiseLike<NariTvmRuntime>;
  readonly upload: (
    owner: NariRuntimeOwner,
    source: NariSource,
    options: NariUploadContext & {
      readonly guard: (phase: string) => PromiseLike<unknown> | unknown;
      readonly expectedBindings: number;
    }
  ) => Promise<NariUploadResult>;
  readonly runner: (value: NariRunnerOwner) => NariRunner;
}

interface NariBrowserBindings {
  readonly Tokenizer?: unknown;
  readonly instantiate?: unknown;
  readonly createPolyfillWASI?: unknown;
  readonly gpu?: NariGpuProvider | undefined;
  readonly expectedModel?: unknown;
}

type NariWasmInstantiator = (wasm: Uint8Array, wasi: unknown) => PromiseLike<NariTvmRuntime>;
type NariWasiFactory = () => unknown;

const failure = (code: NariErrorCode, message: string, cause?: unknown) =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);
export const NARI_LIMITS = Object.freeze({
  maxStorageBuffersPerShaderStage: 6,
  maxStorageBufferBindingSize: 134184960,
  maxBufferSize: 134184960
});

function checkedNariLimits(
  adapter: NariAdapter | null | undefined,
  plan?: NariPlanSizeView,
  variant: unknown = NARI_FULL
): typeof NARI_LIMITS {
  if (variant !== NARI_FULL && variant !== NARI_MIXED) throw failure('INVALID_INPUT', 'Unknown Nari variant.');
  if (!adapter?.features?.has('shader-f16') || typeof adapter.requestDevice !== 'function')
    throw failure('UNSUPPORTED_ENVIRONMENT', 'Nari requires WebGPU shader-f16.');
  for (const [name, minimum] of Object.entries(NARI_LIMITS))
    if (
      !Number.isSafeInteger(
        (adapter.limits as unknown as Readonly<Record<string, number | undefined>> | null | undefined)?.[name]
      ) ||
      (adapter!.limits as unknown as Readonly<Record<string, number | undefined>>)[name]! < minimum
    )
      throw failure('UNSUPPORTED_ENVIRONMENT', `Nari requires ${name} of at least ${minimum}.`);
  if (plan !== undefined) {
    const files = [
      ...plan.ordered.filter((item) => item.source === 'planned-asset').map((item) => item.file!.bytes),
      plan.letterFile.bytes
    ];
    if (
      files.length !== (variant as NariVariant).bindings - 1 ||
      files.some((size) => !Number.isSafeInteger(size) || size < 1) ||
      Math.max(...files.map((value) => Math.ceil(value / 4) * 4)) !== NARI_LIMITS.maxBufferSize
    )
      throw failure('INVALID_ASSET', 'Invalid Nari weight sizes.');
  }
  return NARI_LIMITS;
}

async function uploadCandidateWeights(
  deps: NariLoaderDependencies,
  owner: NariRuntimeOwner,
  source: NariSource,
  options: NariUploadContext & {
    readonly guard: (phase: string) => PromiseLike<unknown> | unknown;
    readonly expectedBindings: number;
  }
): Promise<NariUploadResult> {
  try {
    return await deps.upload(owner, source, options);
  } catch (cause) {
    options.signal?.throwIfAborted();
    if (cause instanceof WebDecisionError && cause.code === 'INFERENCE_ERROR')
      throw failure('MODEL_LOAD_FAILED', 'Nari candidate weight upload failed.', cause);
    throw cause;
  }
}

async function orchestrate({base, signal, variant}: NariCheckedOptions, deps: NariLoaderDependencies) {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw failure('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let source: NariSource | null | undefined,
    gpu: NariGpuDevice | undefined,
    runtime: NariTvmRuntime | undefined,
    owner: NariRuntimeOwner | undefined,
    watch: ReturnType<typeof watchLoadDeviceLoss> | null | undefined;
  const ownership = deps.ownership();
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => deps.gpu!.requestAdapter(), signal);
    checkedNariLimits(adapter);
    source = await waitForAbort(
      () => deps.openSource(base, {signal, variant}),
      signal,
      (late) => late?.close()
    );
    const limits = checkedNariLimits(adapter, source!.plan as NariPlanSizeView, variant);
    const [wasm, tokenizerBytes, configBytes] = await waitForAbort(
      () =>
        Promise.all([
          source!.readAsset('model.wasm', {signal}),
          source!.readAsset('tokenizer.json', {signal}),
          source!.readAsset('tokenizer_config.json', {signal}),
          source!.readAsset('config.json', {signal})
        ]),
      signal
    );
    signal?.throwIfAborted();
    const {tokenizer, controlIds} = deps.tokenizer(tokenizerBytes, configBytes);
    gpu = await waitForAbort(
      () => adapter!.requestDevice!({requiredFeatures: ['shader-f16'], requiredLimits: limits}),
      signal,
      (late) => {
        try {
          late?.destroy();
        } catch {
          retainStrandsResources(undefined, late);
        }
      }
    );
    signal?.throwIfAborted();
    watch = watchLoadDeviceLoss(gpu, signal);
    runtime = await waitForAbort(
      () => deps.instantiate(wasm),
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
    owner = ownership.initialize(runtime!, gpu!);
    const uploaded = await uploadCandidateWeights(deps, owner, source!, {
      signal: watch.signal,
      guard: async () => {},
      expectedBindings: variant.bindings
    });
    watch.signal.throwIfAborted();
    if (owner.blocked || owner.lost || owner.phase !== 'prepared' || owner.uncaptured.length)
      throw Error('Nari runtime unavailable.');
    const runner = deps.runner({
      ...owner,
      bindings: source!.plan.ordered,
      parameters: uploaded.parameters,
      letterRows: uploaded.letterRows,
      expectedBindings: variant.bindings
    });
    const session = createNariSession({tokenizer, controlIds, runner});
    source!.close();
    source = null;
    watch.signal.throwIfAborted();
    watch.close();
    watch = null;
    return session;
  } catch (cause) {
    watch?.close();
    try {
      source?.close();
    } catch {}
    ownership.fail(runtime, gpu);
    if (owner)
      try {
        owner.retain();
      } catch {
        retainStrandsResources(runtime, gpu);
      }
    signal?.throwIfAborted();
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    throw failure('MODEL_LOAD_FAILED', 'Could not load Nari candidate.', cause);
  }
}

export function loadNariCandidate(
  options: unknown,
  {Tokenizer, instantiate, createPolyfillWASI, gpu = globalThis.navigator?.gpu, expectedModel}: NariBrowserBindings = {}
) {
  const checked = validateNariOptions(options);
  if (expectedModel !== undefined && checked.variant.model !== expectedModel)
    throw failure('INVALID_INPUT', 'Nari model registration changed during loading.');
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Nari browser runtime bindings are required.');
  return orchestrate(checked, {
    gpu,
    ownership: runtimeOwnership,
    openSource: openNariSource as unknown as NariLoaderDependencies['openSource'],
    tokenizer: (tokens, config) => createNariTokenizer(Tokenizer, tokens, config),
    instantiate: (wasm) => (instantiate as NariWasmInstantiator)(wasm, (createPolyfillWASI as NariWasiFactory)()),
    upload: (owner, source, context) =>
      uploadFromSource(owner, source, {...context, HashClass: Hash as NariUploadContext['HashClass']}),
    runner: (value) => new NariToqRunner(value as ConstructorParameters<typeof NariToqRunner>[0])
  });
}

export function loadFakeNariCandidate(options: unknown, deps: unknown) {
  if ((deps as {readonly fake?: unknown} | null | undefined)?.fake !== true)
    throw new TypeError('Fake dependencies required.');
  try {
    return orchestrate(validateNariOptions(options), deps as NariLoaderDependencies);
  } catch (error) {
    return Promise.reject(error);
  }
}
