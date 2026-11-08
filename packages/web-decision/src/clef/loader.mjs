import {waitForAbort} from '../assets.js';
import {initializeH2oRuntime} from '../h2o/owner.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {CLEF_LIMITS, CLEF_MODEL} from './contract.mjs';
import {ClefTvmRunner} from './runner.mjs';
import {createClefSession} from './session.mjs';
import {openClefSource} from './source.mjs';
import {createClefTokenizer} from './tokenizer.mjs';
import {uploadClefParameters} from './upload.mjs';

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

export function validateClefOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid Clef options.');
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
  let base;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid Clef asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid Clef asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal};
}

export function validateClefAdapter(adapter) {
  if (
    typeof adapter?.features?.has !== 'function' ||
    !adapter.features.has('shader-f16') ||
    typeof adapter.requestDevice !== 'function' ||
    adapter.info?.isFallbackAdapter !== false ||
    adapter.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Clef requires a hardware WebGPU adapter with shader-f16.');
  for (const [name, minimum] of Object.entries(CLEF_LIMITS))
    if (!Number.isSafeInteger(adapter.limits?.[name]) || adapter.limits[name] < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `Clef requires ${name} of at least ${minimum}.`);
}

async function upload(owner, source, signal) {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof context?.gpuBufferFromPtr !== 'function')
    throw new Error('Pinned TVM upload API unavailable.');
  return uploadClefParameters(source.entries, {
    signal,
    createTensor(entry) {
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
      const buffer = context.gpuBufferFromPtr(tensor.getDataPtr());
      if (!Number.isSafeInteger(buffer?.size) || buffer.size < tensor.byteOffset + Math.ceil(entry.bytes / 4) * 4)
        throw new Error('Clef tensor allocation too small.');
      return tensor;
    },
    readRange: (entry, part, readSignal) => source.readRange(entry, part, readSignal),
    writeRange: (tensor, offset, bytes) =>
      withStrandsScope(owner, 'Clef upload failed.', async (lease) => {
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

async function orchestrate({base, signal}, deps) {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU unavailable.');
  let gpu,
    runtime,
    owner,
    watch,
    initialized = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => deps.gpu.requestAdapter(), signal);
    validateClefAdapter(adapter);
    const source = await deps.openSource(base, {signal});
    signal?.throwIfAborted();
    const tokenizer = await waitForAbort(() => deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig), signal);
    if (deps.beforeDevice) await deps.beforeDevice();
    signal?.throwIfAborted();
    gpu = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: CLEF_LIMITS}),
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
      throw new Error('Clef runtime unavailable.');
    const runner = deps.runner({...owner, weights, source});
    const session = deps.session({...tokenizer, runner});
    watch.signal.throwIfAborted();
    watch.close();
    watch = null;
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
    if (failure instanceof WebDecisionError || failure?.name === 'AbortError') throw failure;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Clef candidate.', failure);
  }
}

export function loadClefCandidate(
  options,
  {Tokenizer, instantiate, createPolyfillWASI, gpu = globalThis.navigator?.gpu, beforeDevice} = {}
) {
  const checked = validateClefOptions(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('Clef runtime bindings are required.');
  if (beforeDevice !== undefined && typeof beforeDevice !== 'function')
    throw new TypeError('Invalid pre-device guard.');
  return orchestrate(checked, {
    gpu,
    beforeDevice,
    openSource: openClefSource,
    prepareTokenizer: (data, config) => createClefTokenizer(Tokenizer, data, config),
    instantiate: (wasm) => instantiate(wasm, createPolyfillWASI()),
    initialize: initializeH2oRuntime,
    upload,
    runner: (args) => new ClefTvmRunner(args),
    session: createClefSession
  });
}
export function loadFakeClefCandidate(options, deps) {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies required.');
  return orchestrate(validateClefOptions(options), deps);
}
