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

// Necessary preliminary constraints from the selected embedding shard and TIR
// inventory; generated WGSL and complete adapter requirements are unverified.
export const PRELIMINARY_LIMITS = Object.freeze({
  maxBufferSize: 134216960,
  maxStorageBufferBindingSize: 134216960,
  maxStorageBuffersPerShaderStage: 8
});
export const H2O_CANDIDATE_MODEL_ID = H2O_BASELINE_MODEL_ID;

const retained = new Set();
const failure = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);
const objectPrototypeKeys = new Set([
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
const snapshot = (source, name, limit) => {
  const bytes = source?.[name];
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < 1 || bytes.byteLength > limit)
    throw failure('INVALID_ASSET', `Invalid ${name} bytes.`);
  return Uint8Array.from(bytes);
};

export function validateOptions(options, expectedModel = H2O_BASELINE_MODEL_ID) {
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
  const snapshot = Object.create(null);
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
  let base;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw failure('INVALID_INPUT', 'Invalid asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw failure('INVALID_INPUT', 'Invalid asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal, model: expectedModel};
}

function validateAdapter(adapter) {
  if (!adapter || typeof adapter.features?.has !== 'function' || !adapter.features.has('shader-f16'))
    throw failure('UNSUPPORTED_ENVIRONMENT', 'H2O requires WebGPU shader-f16.');
  for (const [name, minimum] of Object.entries(PRELIMINARY_LIMITS)) {
    const actual = adapter.limits?.[name];
    if (typeof actual !== 'number' || !Number.isFinite(actual) || actual < minimum)
      throw failure('UNSUPPORTED_ENVIRONMENT', `H2O requires ${name} of at least ${minimum}.`);
  }
  if (typeof adapter.requestDevice !== 'function')
    throw failure('UNSUPPORTED_ENVIRONMENT', 'WebGPU device request unavailable.');
}

function disposeUninitialized(runtime, gpu) {
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

async function cleanupFailedH2oLoad({watch, source, owner, runner, session, runtime, gpu, initializationStarted}) {
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

async function orchestrate({base, signal, model}, deps) {
  signal?.throwIfAborted();
  if (typeof deps.gpu?.requestAdapter !== 'function') throw failure('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let adapter, source, gpu, runtime, owner, runner, session, watch;
  let attached = false,
    initializationStarted = false;
  try {
    adapter = await waitForAbort(() => deps.gpu.requestAdapter(), signal);
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
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw failure(
      'MODEL_LOAD_FAILED',
      attached ? 'Could not finish H2O session setup.' : 'Could not load H2O candidate.',
      cause
    );
  }
}

async function uploadCandidateWeights(deps, options) {
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
export async function loadH2oCandidate(options, context) {
  const {beforeDevice, runtimeBindings, expectedModel = H2O_BASELINE_MODEL_ID} = context ?? {};
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
export function pathFreeSourceDependencies(model = H2O_BASELINE_MODEL_ID) {
  h2oGraphIdentity(model);
  return {
    openSource: ({base, signal}) => openH2oAssetSource({assetsUrl: base, signal, model}),
    makeHasher: () => createIncrementalSha256(Hash)
  };
}

/** Dependency injection is for fixtures; fake is not an authorization boundary. */
export function loadFakeH2oCandidate(options, deps) {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies required');
  try {
    const checked = validateOptions(options, deps.expectedModel ?? H2O_BASELINE_MODEL_ID);
    return orchestrate(checked, deps);
  } catch (error) {
    return Promise.reject(error);
  }
}

/** Bind the same runtime imports already used by the package entry. */
export function createH2oBrowserFactories(
  {Tokenizer, instantiate, createPolyfillWASI} = {},
  model = H2O_BASELINE_MODEL_ID
) {
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
