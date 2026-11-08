import {waitForAbort} from '../assets.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {Hash} from '../h2o/sha256.mjs';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {runtimeOwnership} from '../nari/runtime.mjs';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {IMAJEV_MODEL} from './contract.mjs';
import {createImajevResultProcessor} from './result-processor.mjs';
import {ImajevSession} from './session.mjs';
import {openImajevSource} from './source.mjs';
import {uploadImajevFromSource} from './upload.mjs';

const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);
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
export const IMAJEV_LIMITS = Object.freeze({
  maxStorageBuffersPerShaderStage: 6,
  maxStorageBufferBindingSize: 134184960,
  maxBufferSize: 134184960
});

export function validateImajevOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Imajev options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null || Reflect.ownKeys(prototype).some((key) => !PROTOTYPE_KEYS.has(key)))
  )
    throw fail('INVALID_INPUT', 'Unsupported Imajev option prototype.');
  const snapshot = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const field = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !KEYS.has(key) || !field || !Object.hasOwn(field, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Imajev load option.');
    snapshot[key] = field.value;
  }
  if (
    snapshot.model !== IMAJEV_MODEL ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw fail('INVALID_INPUT', 'Unsupported Imajev load option.');
  let base;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Imajev asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Imajev asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal};
}

function checkedImajevLimits(adapter) {
  if (!adapter?.features?.has('shader-f16') || typeof adapter.requestDevice !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Imajev requires WebGPU shader-f16.');
  for (const [name, minimum] of Object.entries(IMAJEV_LIMITS))
    if (!Number.isSafeInteger(adapter.limits?.[name]) || adapter.limits[name] < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `Imajev requires ${name} of at least ${minimum}.`);
  return IMAJEV_LIMITS;
}

function processorFromAssets(Tokenizer, assets, maxInputTokens) {
  const json = (bytes) => parseStrictJson(bytes, {maxBytes: 16 * 1024 * 1024});
  const tokenizer = new Tokenizer(json(assets[1]), json(assets[2]));
  return createImajevResultProcessor({
    tokenizer,
    manifest: json(assets[3]),
    calibration: json(assets[4]),
    maxInputTokens
  });
}

async function orchestrate({base, signal}, deps) {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let source, gpu, runtime, owner, watch;
  const ownership = deps.ownership();
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => deps.gpu.requestAdapter(), signal);
    const limits = checkedImajevLimits(adapter);
    source = await waitForAbort(
      () => deps.openSource(base, {signal}),
      signal,
      (late) => late?.close()
    );
    const assets = await waitForAbort(
      () =>
        Promise.all(
          ['model.wasm', 'tokenizer.json', 'tokenizer_config.json', 'decision_readout.json', 'calibration.json'].map(
            (name) => source.readAsset(name, {signal})
          )
        ),
      signal
    );
    signal?.throwIfAborted();
    const processor = deps.processor(assets, source.plan.maxInputTokens);
    gpu = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: limits}),
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
      () => deps.instantiate(assets[0]),
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
    owner = ownership.initialize(runtime, gpu);
    const weights = await deps.upload(owner, source, {signal: watch.signal});
    watch.signal.throwIfAborted();
    if (owner.blocked || owner.lost || owner.phase !== 'prepared' || owner.uncaptured.length)
      throw Error('Imajev runtime unavailable.');
    const session = deps.session({...owner, weights, processor, maxInputTokens: source.plan.maxInputTokens});
    source.close();
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
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Imajev candidate.', cause);
  }
}

/** Internal candidate only; the public model router does not yet expose it. */
export function loadImajevCandidate(
  options,
  {Tokenizer, instantiate, createPolyfillWASI, gpu = globalThis.navigator?.gpu} = {}
) {
  const checked = validateImajevOptions(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Imajev browser runtime bindings are required.');
  return orchestrate(checked, {
    gpu,
    ownership: runtimeOwnership,
    openSource: openImajevSource,
    processor: (assets, max) => processorFromAssets(Tokenizer, assets, max),
    instantiate: (wasm) => instantiate(wasm, createPolyfillWASI()),
    upload: (owner, source, context) => uploadImajevFromSource(owner, source, {...context, HashClass: Hash}),
    session: (value) => new ImajevSession(value)
  });
}

export function loadFakeImajevCandidate(options, deps) {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies required.');
  return orchestrate(validateImajevOptions(options), deps);
}
