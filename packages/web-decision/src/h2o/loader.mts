// Experimental H2O loader for a separately hosted, pinned asset directory.
import {WebDecisionError} from '../kev.js';
import {waitForAbort} from '../assets.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {
  preparePublicCandidatePlan,
  verifyWasm,
  h2oGraphIdentity,
  H2O_BASELINE_MODEL_ID,
  H2O_DLIGHT_MODEL_ID,
  H2O_DUAL_MODEL_ID
} from './plan.mjs';
import {prepareCheckedH2oTokenizer} from './tokenizer-adapter.mjs';
import {createH2oSession} from './session.mjs';
import {initializeH2oRuntime, attachH2oRunner} from './owner.mjs';
import {uploadH2oParameters} from './upload.mjs';
import {openH2oAssetSource} from './source-entry.mjs';
import {createIncrementalSha256} from './hasher.mjs';
import {Hash} from './sha256.mjs';
import {validateH2oSignal} from './signal.mjs';
import type {ErrorCode} from '../curated/types.js';
import type {H2oGpu, H2oRuntime, H2oRuntimeOwner, H2oRunner, H2oTensor} from './owner.mjs';
import type {H2oHashConstructor, H2oOpenedSource} from './source.mjs';
import type {H2oSession} from './session.mjs';
import type {H2oTokenizerConstructor} from './tokenizer.mjs';

type H2oLoadErrorCode = ErrorCode;
type H2oCheckedOptions = {base: URL; signal: AbortSignal | undefined; model: string};
type H2oFileParameter = {
  kind: 'file';
  path: string;
  dtype: string;
  shape: number[];
  sha256: string;
  bytes: number;
  target: string;
};
type H2oRotaryParameter = {
  kind: 'rotary';
  target: string;
  dtype: string;
  shape: number[];
  values: number[];
};
type H2oPreparedPlan = {
  parameters: Array<H2oFileParameter | H2oRotaryParameter>;
  labels: string[];
  labelIds: number[];
};
type H2oGpuDevice = H2oGpu & {destroy(): unknown};
type H2oAdapter = {
  features: {has(name: string): unknown};
  limits?: Readonly<Record<string, unknown>> | null | undefined;
  requestDevice(options: {
    requiredFeatures: string[];
    requiredLimits: typeof PRELIMINARY_LIMITS;
  }): H2oGpuDevice | PromiseLike<H2oGpuDevice>;
};
type H2oGpuProvider = {
  requestAdapter?: (() => unknown | PromiseLike<unknown>) | undefined;
};
type H2oLoadWatch = {signal: AbortSignal; close(): void};
type H2oTokenizerResult = Awaited<ReturnType<typeof prepareCheckedH2oTokenizer>>;
type H2oRuntimeBindings = {
  Tokenizer?: H2oTokenizerConstructor | undefined;
  instantiate?: ((wasm: Uint8Array, wasi: unknown) => H2oRuntime | PromiseLike<H2oRuntime>) | undefined;
  createPolyfillWASI?: (() => unknown) | undefined;
};
type H2oLoadDependencies = {
  gpu?: H2oGpuProvider | null | undefined;
  openSource(options: {base: URL; signal: AbortSignal | undefined}): H2oOpenedSource | PromiseLike<H2oOpenedSource>;
  preparePlan(
    stage: Uint8Array,
    manifest: Uint8Array,
    candidate: Uint8Array | undefined,
    signal: AbortSignal | undefined
  ): H2oPreparedPlan | PromiseLike<H2oPreparedPlan>;
  verifyWasm(bytes: Uint8Array, signal: AbortSignal | undefined): unknown | PromiseLike<unknown>;
  prepareTokenizer(
    tokens: Uint8Array,
    config: Uint8Array,
    contract: Uint8Array,
    signal: AbortSignal | undefined
  ): H2oTokenizerResult | PromiseLike<H2oTokenizerResult>;
  beforeDevice?: (() => unknown | PromiseLike<unknown>) | undefined;
  createRuntime(wasm: Uint8Array): H2oRuntime | PromiseLike<H2oRuntime>;
  initialize(runtime: H2oRuntime, gpu: H2oGpuDevice): H2oRuntimeOwner;
  upload: typeof uploadH2oParameters;
  attach(owner: H2oRuntimeOwner, weights: readonly H2oTensor[]): H2oRunner;
  createSession(options: {
    tokenizer: H2oTokenizerResult['tokenizer'];
    addedTokens: H2oTokenizerResult['addedTokens'];
    plan: H2oPreparedPlan;
    runner: H2oRunner;
  }): H2oSession | PromiseLike<H2oSession>;
  makeHasher: () => ReturnType<typeof createIncrementalSha256>;
};
type H2oLoadContext = {
  beforeDevice?: (() => unknown | PromiseLike<unknown>) | undefined;
  runtimeBindings?: H2oRuntimeBindings | undefined;
  expectedModel?: string | undefined;
};

// Necessary preliminary constraints from the selected embedding shard and TIR
// inventory; generated WGSL and complete adapter requirements are unverified.
export const PRELIMINARY_LIMITS = Object.freeze({
  maxBufferSize: 134216960,
  maxStorageBufferBindingSize: 134216960,
  maxStorageBuffersPerShaderStage: 8
});
export const H2O_CANDIDATE_MODEL_ID = H2O_BASELINE_MODEL_ID;

const retained = new Set<unknown>();
const failure = (code: H2oLoadErrorCode, message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);
const objectPrototypeKeys = new Set<PropertyKey>([
  'constructor',
  '__defineGetter__',
  '__defineSetter__',
  'hasOwnProperty',
  '__lookupGetter__',
  '__lookupSetter__',
  'isPrototypeOf',
  'propertyIsEnumerable',
  'toString',
  'valueOf',
  '__proto__',
  'toLocaleString'
]);
const snapshot = (source: unknown, name: string, limit: number): Uint8Array => {
  const bytes = (source as Record<string, unknown> | null | undefined)?.[name];
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < 1 || bytes.byteLength > limit)
    throw failure('INVALID_ASSET', `Invalid ${name} bytes.`);
  return Uint8Array.from(bytes);
};

export function validateOptions(options: unknown, expectedModel: string = H2O_BASELINE_MODEL_ID): H2oCheckedOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw failure('INVALID_INPUT', 'Unsupported H2O load options.');
  const prototype = Object.getPrototypeOf(options);
  // An ordinary object's immediate prototype has a null parent, including in
  // another realm. Reject added/inherited fields without relying on realm-local
  // Object.prototype identity. This is an input contract, not proxy protection.
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null ||
      Reflect.ownKeys(prototype).some((key) => !objectPrototypeKeys.has(key)))
  )
    throw failure('INVALID_INPUT', 'Options inherit unsupported prototype properties.');
  const allowed = new Set([
    'model',
    'assetsUrl',
    'signal',
    'cache',
    'offline',
    'maxPrefetchBytes',
    'onProgress',
    'debertaReadback'
  ]);
  const snapshot: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !allowed.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw failure('INVALID_INPUT', 'Unsupported H2O load options.');
    snapshot[key] = descriptor.value;
  }
  if (
    ![H2O_BASELINE_MODEL_ID, H2O_DLIGHT_MODEL_ID, H2O_DUAL_MODEL_ID].includes(expectedModel) ||
    (snapshot.model === undefined ? H2O_BASELINE_MODEL_ID : snapshot.model) !== expectedModel ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw failure('INVALID_INPUT', 'Unsupported H2O load options.');
  let base: URL;
  try {
    base = new URL(snapshot.assetsUrl as string, globalThis.location?.href);
  } catch {
    throw failure('INVALID_INPUT', 'Invalid asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw failure('INVALID_INPUT', 'Invalid asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal as AbortSignal | undefined, model: expectedModel};
}

function validateAdapter(adapter: unknown): asserts adapter is H2oAdapter {
  if (
    !adapter ||
    typeof (adapter as H2oAdapter).features?.has !== 'function' ||
    !(adapter as H2oAdapter).features.has('shader-f16')
  )
    throw failure('UNSUPPORTED_ENVIRONMENT', 'H2O requires WebGPU shader-f16.');
  for (const [name, minimum] of Object.entries(PRELIMINARY_LIMITS)) {
    const actual = (adapter as H2oAdapter).limits?.[name];
    if (typeof actual !== 'number' || !Number.isFinite(actual) || actual < minimum)
      throw failure('UNSUPPORTED_ENVIRONMENT', `H2O requires ${name} of at least ${minimum}.`);
  }
  if (typeof (adapter as H2oAdapter).requestDevice !== 'function')
    throw failure('UNSUPPORTED_ENVIRONMENT', 'WebGPU device request unavailable.');
}

function disposeUninitialized(runtime: H2oRuntime | null | undefined, gpu: H2oGpuDevice | null | undefined): void {
  if (runtime) {
    try {
      runtime.dispose();
    } catch {
      retainStrandsResources(runtime, gpu);
    }
  }
  try {
    gpu?.destroy();
  } catch {}
}

async function cleanupFailedH2oLoad({
  watch,
  source,
  owner,
  runner,
  session,
  runtime,
  gpu,
  initializationStarted
}: {
  watch?: H2oLoadWatch | null | undefined;
  source?: H2oOpenedSource | null | undefined;
  owner?: H2oRuntimeOwner | null | undefined;
  runner?: H2oRunner | null | undefined;
  session?: H2oSession | null | undefined;
  runtime?: H2oRuntime | null | undefined;
  gpu?: H2oGpuDevice | null | undefined;
  initializationStarted: boolean;
}): Promise<void> {
  watch?.close();
  if (source)
    try {
      await source.close();
    } catch {}
  if (owner) {
    // Retain objects if GPU work may have been submitted.
    try {
      owner.disposalGuard?.retain();
    } catch {
      retainStrandsResources(runtime, gpu);
    }
    retained.add({owner, runner, session});
  } else if (runtime && initializationStarted) {
    retainStrandsResources(runtime, gpu);
    retained.add({runtime, gpu});
  } else {
    disposeUninitialized(runtime, gpu);
  }
}

async function orchestrate({base, signal, model}: H2oCheckedOptions, deps: H2oLoadDependencies): Promise<H2oSession> {
  signal?.throwIfAborted();
  if (typeof deps.gpu?.requestAdapter !== 'function') throw failure('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let adapter: unknown,
    source: H2oOpenedSource | null | undefined,
    gpu: H2oGpuDevice | undefined,
    runtime: H2oRuntime | undefined,
    owner: H2oRuntimeOwner | undefined,
    runner: H2oRunner | undefined,
    session: H2oSession | undefined,
    watch: H2oLoadWatch | null | undefined;
  let attached = false,
    initializationStarted = false;
  try {
    adapter = await waitForAbort(() => deps.gpu!.requestAdapter!(), signal);
    signal?.throwIfAborted();
    validateAdapter(adapter);
    source = await waitForAbort(
      () => deps.openSource({base, signal}),
      signal,
      (late) => {
        try {
          Promise.resolve(late?.close()).catch(() => {});
        } catch {}
      }
    );
    signal?.throwIfAborted();
    const rangeMethod = source?.readRange;
    if (typeof rangeMethod !== 'function' || typeof source?.close !== 'function')
      throw failure('INVALID_ASSET', 'Invalid H2O asset source.');
    const readRange = rangeMethod.bind(source);
    // Snapshot mutable inputs before the first asynchronous hash or parser.
    const bytes = {
      stage: snapshot(source, 'stage', 1_000_000),
      manifest: snapshot(source, 'manifest', 1_000_000),
      wasm: snapshot(source, 'wasm', 6_000_000),
      tokenizer: snapshot(source, 'tokenizer', 24_000_000),
      tokenizerConfig: snapshot(source, 'tokenizerConfig', 1_000_000),
      tokenizerContract: snapshot(source, 'tokenizerContract', 1_000_000),
      ...(model !== H2O_BASELINE_MODEL_ID ? {candidateMetadata: snapshot(source, 'candidateMetadata', 1_000_000)} : {})
    };
    const plan = await waitForAbort(
      () => deps.preparePlan(bytes.stage, bytes.manifest, bytes.candidateMetadata, signal),
      signal
    );
    signal?.throwIfAborted();
    const verifiedWasm = await waitForAbort(() => deps.verifyWasm(bytes.wasm, signal), signal);
    signal?.throwIfAborted();
    if (!(verifiedWasm instanceof Uint8Array))
      throw failure('INVALID_ASSET', 'WASM verification did not return bytes.');
    const tokenizer = await waitForAbort(
      () => deps.prepareTokenizer(bytes.tokenizer, bytes.tokenizerConfig, bytes.tokenizerContract, signal),
      signal
    );
    signal?.throwIfAborted();
    if (deps.beforeDevice) await deps.beforeDevice();
    signal?.throwIfAborted();
    // requestDevice is not cancellable: await it so a late device can be destroyed.
    gpu = await adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: PRELIMINARY_LIMITS});
    signal?.throwIfAborted();
    watch = watchLoadDeviceLoss(gpu, signal);
    runtime = await waitForAbort(
      () => deps.createRuntime(verifiedWasm),
      watch.signal,
      (late) => {
        Promise.resolve()
          .then(() => late?.dispose())
          .catch(() => retainStrandsResources(late, gpu));
      }
    );
    watch.signal.throwIfAborted();
    initializationStarted = true;
    owner = deps.initialize(runtime, gpu);
    watch.signal.throwIfAborted();
    const weights = await uploadCandidateWeights(deps, {
      owner,
      plan,
      inventory: source.inventory,
      base,
      readRange,
      makeHasher: deps.makeHasher,
      signal: watch.signal
    });
    watch.signal.throwIfAborted();
    runner = deps.attach(owner, weights);
    attached = true;
    session = await deps.createSession({
      tokenizer: tokenizer.tokenizer,
      addedTokens: tokenizer.addedTokens,
      plan,
      runner
    });
    watch.signal.throwIfAborted();
    await source.close();
    source = null;
    watch.signal.throwIfAborted();
    watch.close();
    watch = null;
    return session;
  } catch (cause) {
    await cleanupFailedH2oLoad({watch, source, owner, runner, session, runtime, gpu, initializationStarted});
    signal?.throwIfAborted();
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    throw failure(
      'MODEL_LOAD_FAILED',
      attached ? 'Could not finish H2O session setup.' : 'Could not load H2O candidate.',
      cause
    );
  }
}

type H2oUploadInvocation = NonNullable<Parameters<typeof uploadH2oParameters>[0]>;

async function uploadCandidateWeights(
  deps: Pick<H2oLoadDependencies, 'upload'>,
  options: H2oUploadInvocation
): Promise<readonly H2oTensor[]> {
  try {
    return await deps.upload(options);
  } catch (cause) {
    options.signal?.throwIfAborted();
    if (cause instanceof WebDecisionError && cause.code === 'INFERENCE_ERROR')
      throw failure('MODEL_LOAD_FAILED', 'H2O candidate weight upload failed.', cause);
    throw cause;
  }
}

/** Load using the entry's runtime bindings; no browser system-memory guard is implied. */
export async function loadH2oCandidate(options: unknown, context?: unknown): Promise<H2oSession> {
  const {beforeDevice, runtimeBindings, expectedModel = H2O_BASELINE_MODEL_ID} = (context ?? {}) as H2oLoadContext;
  if (beforeDevice !== undefined && typeof beforeDevice !== 'function')
    throw failure('MODEL_LOAD_FAILED', 'Invalid pre-device callback.');
  const checked = validateOptions(options, expectedModel);
  const factories = createH2oBrowserFactories(runtimeBindings, checked.model);
  return orchestrate(checked, {
    ...factories,
    ...pathFreeSourceDependencies(checked.model),
    beforeDevice,
    gpu: globalThis.navigator?.gpu
  });
}

/** Internal asset dependencies shared by the loader and integration checks. */
export function pathFreeSourceDependencies(
  model: string = H2O_BASELINE_MODEL_ID
): Pick<H2oLoadDependencies, 'openSource' | 'makeHasher'> {
  h2oGraphIdentity(model);
  return {
    openSource: ({base, signal}) => openH2oAssetSource({assetsUrl: base, signal, model}),
    makeHasher: () => createIncrementalSha256(Hash as unknown as H2oHashConstructor)
  };
}

/** Dependency injection is for fixtures; fake is not an authorization boundary. */
export function loadFakeH2oCandidate(options: unknown, deps: unknown): Promise<H2oSession> {
  if ((deps as {readonly fake?: unknown} | null | undefined)?.fake !== true)
    throw new TypeError('Fake dependencies required');
  try {
    const checked = validateOptions(
      options,
      (deps as H2oLoadDependencies & {readonly expectedModel?: string | null | undefined}).expectedModel ??
        H2O_BASELINE_MODEL_ID
    );
    return orchestrate(checked, deps as H2oLoadDependencies);
  } catch (error) {
    return Promise.reject(error);
  }
}

/** Bind the same runtime imports already used by the package entry. */
export function createH2oBrowserFactories(
  {Tokenizer, instantiate, createPolyfillWASI}: H2oRuntimeBindings = {},
  model: string = H2O_BASELINE_MODEL_ID
): Omit<H2oLoadDependencies, 'gpu' | 'openSource' | 'makeHasher' | 'beforeDevice'> {
  h2oGraphIdentity(model);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('H2O browser runtime bindings are required');
  return {
    preparePlan: (stage, manifest, candidate, signal) =>
      preparePublicCandidatePlan(stage, manifest, model, candidate, signal),
    verifyWasm: (bytes, signal) => verifyWasm(bytes, model, signal),
    prepareTokenizer: (tokens, config, contract, signal) =>
      prepareCheckedH2oTokenizer(Tokenizer, tokens, config, contract, signal),
    createRuntime: (wasm) => instantiate(wasm, createPolyfillWASI()),
    initialize: initializeH2oRuntime,
    upload: uploadH2oParameters,
    attach: attachH2oRunner,
    createSession: createH2oSession
  };
}
