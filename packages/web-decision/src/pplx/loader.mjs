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
const PROTOTYPE_KEYS = new Set(Object.getOwnPropertyNames(Object.prototype));
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateFlashOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Perplexity options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null || Reflect.ownKeys(prototype).some((key) => !PROTOTYPE_KEYS.has(key)))
  )
    throw fail('INVALID_INPUT', 'Unsupported Perplexity option prototype.');
  const snapshot = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
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
  let base;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Perplexity asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Perplexity asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal};
}

function validateFlashAdapter(adapter) {
  if (
    typeof adapter?.features?.has !== 'function' ||
    !adapter.features.has('shader-f16') ||
    typeof adapter.requestDevice !== 'function' ||
    adapter.info?.isFallbackAdapter ||
    adapter.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Perplexity requires a hardware WebGPU adapter with shader-f16.');
  for (const [name, minimum] of Object.entries(FLASH_LIMITS))
    if (!Number.isSafeInteger(adapter.limits?.[name]) || adapter.limits[name] < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `Perplexity requires ${name} of at least ${minimum}.`);
}

async function upload(owner, source, signal) {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function') throw new Error('Pinned TVM upload API unavailable.');
  return uploadFlashParameters(source.manifest, {
    signal,
    createTensor(entry) {
      assertUsable(owner);
      if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.uncaptured.length)
        throw new Error('GPU unavailable.');
      return owner.runtime.empty(entry.shape, entry.dtype, owner.device);
    },
    readRange: (entry, index, part) => source.readRange(entry, index, part, signal),
    writeRange: (tensor, offset, bytes) =>
      withStrandsScope(owner, 'Perplexity upload failed.', async (lease) => {
        if (typeof tensor.byteOffset !== 'number' || tensor.byteOffset % 4) throw new Error('Invalid tensor offset.');
        context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
        await lease.syncWithoutReadback();
      })
  });
}

function cleanupFailedLoad(owner, runtime, gpu, initializationStarted, retain) {
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

async function orchestrate({base, signal}, deps) {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  const retain = deps.retainResources ?? retainStrandsResources;
  let gpu,
    runtime,
    owner,
    watch,
    initializationStarted = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => deps.gpu.requestAdapter(), signal);
    validateFlashAdapter(adapter);
    const source = await deps.openSource(base, {signal});
    signal?.throwIfAborted();
    const tokenizer = deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig);
    signal?.throwIfAborted();
    if (deps.beforeDevice) await deps.beforeDevice();
    signal?.throwIfAborted();
    gpu = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: FLASH_LIMITS}),
      signal,
      (late) => {
        try {
          late?.destroy();
        } catch {
          retain(undefined, late);
        }
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
          retain(late, gpu);
        }
      }
    );
    watch.signal.throwIfAborted();
    initializationStarted = true;
    owner = deps.initialize(runtime, gpu);
    const weights = await deps.upload(owner, source, watch.signal);
    watch.signal.throwIfAborted();
    if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.disposalGuard.blocked || owner.uncaptured.length)
      throw new Error('Perplexity runtime unavailable.');
    const runner = deps.runner({...owner, weights});
    const session = deps.session({...tokenizer, runner});
    watch.signal.throwIfAborted();
    watch.close();
    watch = null;
    return session;
  } catch (cause) {
    watch?.close();
    cleanupFailedLoad(owner, runtime, gpu, initializationStarted, retain);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Perplexity candidate.', cause);
  }
}

export function loadFlashCandidate(
  options,
  {Tokenizer, instantiate, createPolyfillWASI, gpu = globalThis.navigator?.gpu, beforeDevice} = {}
) {
  const checked = validateFlashOptions(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Perplexity runtime bindings are required.');
  if (beforeDevice !== undefined && typeof beforeDevice !== 'function')
    throw new TypeError('Invalid pre-device guard.');
  return orchestrate(checked, {
    gpu,
    beforeDevice,
    openSource: openFlashSource,
    prepareTokenizer: (data, config) => createFlashTokenizer(Tokenizer, data, config),
    instantiate: (wasm) => instantiate(wasm, createPolyfillWASI()),
    initialize: initializeH2oRuntime,
    upload,
    runner: (args) => new FlashTvmRunner(args),
    session: createFlashSession
  });
}

export function loadFakeFlashCandidate(options, deps) {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies required.');
  return orchestrate(validateFlashOptions(options), deps);
}
