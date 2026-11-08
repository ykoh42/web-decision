import {waitForAbort} from '../assets.js';
import {initializeH2oRuntime} from '../h2o/owner.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {uploadParameterEntries} from '../parameter-upload.mjs';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {MAPIKA4_DLIGHT_ALL_MODEL, MAPIKA4_DLIGHT_MODEL, MAPIKA4_LIMITS, MAPIKA4_MODEL} from './contract.mjs';
import {Mapika4TvmRunner} from './runner.mjs';
import {createMapika4Session} from './session.mjs';
import {openMapika4Source} from './source.mjs';
import {createMapika4Tokenizer} from './tokenizer.mjs';

export {MAPIKA4_LIMITS, MAPIKA4_MODEL};
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
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateMapika4Options(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Mapika 4B options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null || Reflect.ownKeys(prototype).some((key) => !PROTOTYPE_KEYS.has(key)))
  )
    throw fail('INVALID_INPUT', 'Unsupported option prototype.');
  const snapshot = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Mapika 4B load option.');
    snapshot[key] = descriptor.value;
  }
  if (
    ![MAPIKA4_MODEL, MAPIKA4_DLIGHT_MODEL, MAPIKA4_DLIGHT_ALL_MODEL].includes(snapshot.model) ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw fail('INVALID_INPUT', 'Unsupported Mapika 4B load option.');
  let base;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Mapika 4B asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Mapika 4B asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal, model: snapshot.model};
}

function validateMapika4Adapter(adapter) {
  if (
    typeof adapter?.features?.has !== 'function' ||
    !adapter.features.has('shader-f16') ||
    typeof adapter.requestDevice !== 'function' ||
    adapter.info?.isFallbackAdapter ||
    adapter.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Mapika 4B requires a hardware WebGPU adapter with shader-f16.');
  for (const [name, minimum] of Object.entries(MAPIKA4_LIMITS))
    if (!Number.isSafeInteger(adapter.limits?.[name]) || adapter.limits[name] < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `Mapika 4B requires ${name} of at least ${minimum}.`);
}

async function upload(owner, source, signal) {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function') throw new Error('Pinned TVM upload API unavailable.');
  return uploadParameterEntries(source.entries, {
    signal,
    createTensor(entry) {
      assertUsable(owner);
      if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.uncaptured.length)
        throw new Error('GPU unavailable.');
      return owner.runtime.empty(entry.shape, entry.dtype, owner.device);
    },
    readRange: (entry, index, part) => source.readRange(entry, index, part, signal),
    writeRange: (tensor, offset, bytes) =>
      withStrandsScope(owner, 'Mapika 4B upload failed.', async (lease) => {
        if (typeof tensor.byteOffset !== 'number' || tensor.byteOffset % 4) throw new Error('Invalid tensor offset.');
        context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
        await lease.syncWithoutReadback();
      })
  });
}

function cleanup(owner, runtime, gpu, initialized) {
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

async function orchestrate({base, signal, model}, deps) {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let gpu,
    runtime,
    owner,
    watch,
    initialized = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => deps.gpu.requestAdapter(), signal);
    validateMapika4Adapter(adapter);
    const source = await deps.openSource(base, {signal, model});
    signal?.throwIfAborted();
    const tokenizer = await waitForAbort(() => deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig), signal);
    if (deps.beforeDevice) await deps.beforeDevice();
    signal?.throwIfAborted();
    gpu = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: MAPIKA4_LIMITS}),
      signal,
      (late) => {
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
  } catch (cause) {
    watch?.close();
    cleanup(owner, runtime, gpu, initialized);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Mapika 4B candidate.', cause);
  }
}

export function loadMapika4Candidate(
  options,
  {Tokenizer, instantiate, createPolyfillWASI, gpu = globalThis.navigator?.gpu, beforeDevice} = {}
) {
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
    instantiate: (wasm) => instantiate(wasm, createPolyfillWASI()),
    initialize: initializeH2oRuntime,
    upload,
    runner: (args) => new Mapika4TvmRunner(args),
    session: createMapika4Session
  });
}

export function loadFakeMapika4Candidate(options, deps) {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies required.');
  return orchestrate(validateMapika4Options(options), deps);
}
