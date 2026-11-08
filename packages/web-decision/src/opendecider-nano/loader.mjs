import {waitForAbort} from '../assets.js';
import {initializeH2oRuntime} from '../h2o/owner.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {OPENDECIDER_NANO_LIMITS, OPENDECIDER_NANO_MODEL, OPENDECIDER_NANO_PARAMETER_COUNT} from './contract.mjs';
import {installOpenDeciderNanoGpuCompatibility} from './gpu-compat.mjs';
import {OpenDeciderNanoTvmRunner} from './runner.mjs';
import {createOpenDeciderNanoSession} from './session.mjs';
import {openOpenDeciderNanoSource} from './source.mjs';
import {createOpenDeciderNanoTokenizer} from './tokenizer.mjs';

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

function validateOpenDeciderNanoOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Invalid OpenDecider-nano options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null || Reflect.ownKeys(prototype).some((key) => !PROTOTYPE_KEYS.has(key)))
  )
    throw fail('INVALID_INPUT', 'Unsupported OpenDecider-nano option prototype.');
  const snapshot = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported OpenDecider-nano load option.');
    snapshot[key] = descriptor.value;
  }
  if (
    snapshot.model !== OPENDECIDER_NANO_MODEL ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    (snapshot.onProgress !== undefined && typeof snapshot.onProgress !== 'function') ||
    snapshot.debertaReadback !== undefined
  )
    throw fail('INVALID_INPUT', 'Unsupported OpenDecider-nano load option.');
  let base;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw fail('INVALID_INPUT', 'Invalid OpenDecider-nano asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Invalid OpenDecider-nano asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {base, signal: snapshot.signal, onProgress: snapshot.onProgress};
}

function validateOpenDeciderNanoAdapter(adapter) {
  if (
    typeof adapter?.features?.has !== 'function' ||
    !adapter.features.has('shader-f16') ||
    typeof adapter.requestDevice !== 'function' ||
    adapter.info?.isFallbackAdapter ||
    adapter.isFallbackAdapter
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'OpenDecider-nano requires a hardware WebGPU adapter with shader-f16.');
  for (const [name, minimum] of Object.entries(OPENDECIDER_NANO_LIMITS))
    if (!Number.isSafeInteger(adapter.limits?.[name]) || adapter.limits[name] < minimum)
      throw fail('UNSUPPORTED_ENVIRONMENT', `OpenDecider-nano requires ${name} of at least ${minimum}.`);
}

function alignedBytes(bytes) {
  if (bytes.byteLength % 4 === 0) return bytes;
  const aligned = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
  aligned.set(bytes);
  return aligned;
}

async function upload(owner, source, signal) {
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function') throw new Error('Pinned TVM upload API unavailable.');
  const weights = new Array(OPENDECIDER_NANO_PARAMETER_COUNT);
  for (const shard of source.tensorCache.records) {
    signal?.throwIfAborted();
    const bytes = await source.readShard(shard.dataPath, signal);
    const writes = [];
    for (const record of shard.records) {
      const index = Number(record.name.slice('param_'.length));
      const tensor = owner.runtime.empty(record.shape, record.dtype, owner.device);
      weights[index] = tensor;
      writes.push({tensor, bytes: alignedBytes(bytes.subarray(record.byteOffset, record.byteOffset + record.nbytes))});
    }
    await withStrandsScope(owner, 'OpenDecider-nano weight upload failed.', async (lease) => {
      for (const {tensor, bytes: raw} of writes) {
        assertUsable(owner);
        if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.uncaptured.length)
          throw new Error('GPU unavailable.');
        if (typeof tensor.byteOffset !== 'number' || tensor.byteOffset % 4 !== 0 || raw.byteLength % 4 !== 0)
          throw new Error('Invalid OpenDecider-nano tensor alignment.');
        context.copyRawBytesToBuffer(raw, tensor.getDataPtr(), tensor.byteOffset, raw.byteLength);
      }
      await lease.syncWithoutReadback();
    });
    signal?.throwIfAborted();
  }
  if (
    weights.length !== OPENDECIDER_NANO_PARAMETER_COUNT ||
    Array.from(
      {length: OPENDECIDER_NANO_PARAMETER_COUNT},
      (_, index) => !Object.hasOwn(weights, index) || !weights[index]
    ).some(Boolean)
  )
    throw fail('INVALID_ASSET', 'OpenDecider-nano weights are incomplete.');
  return weights;
}

function cleanupFailedLoad(owner, runtime, gpu, initialized, gpuCompatibility) {
  if (owner) {
    owner.gpuCompatibility = gpuCompatibility;
    try {
      owner.disposalGuard.retain();
    } catch {
      retainStrandsResources(runtime, gpu);
    }
  } else if (runtime && initialized) retainStrandsResources(runtime, gpu);
  else {
    gpuCompatibility?.restore();
    try {
      runtime?.dispose();
    } catch {
      retainStrandsResources(runtime, gpu);
    }
    try {
      gpu?.destroy();
    } catch {
      retainStrandsResources(undefined, gpu);
    }
  }
}

async function orchestrate({base, signal, onProgress}, deps) {
  if (typeof deps.gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'WebGPU is unavailable.');
  let gpu,
    runtime,
    owner,
    gpuCompatibility,
    watch,
    initialized = false;
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => deps.gpu.requestAdapter(), signal);
    validateOpenDeciderNanoAdapter(adapter);
    const source = await deps.openSource(base, {signal, onProgress});
    const tokenizer = deps.prepareTokenizer(source.tokenizer, source.tokenizerConfig);
    signal?.throwIfAborted();
    gpu = await waitForAbort(
      () => adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: OPENDECIDER_NANO_LIMITS}),
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
    owner = deps.initialize(runtime, gpu, {
      setupWebGPUContext(context) {
        gpuCompatibility = deps.installGpuCompatibility?.(context);
      }
    });
    owner.gpuCompatibility = gpuCompatibility;
    const weights = await deps.upload(owner, source, watch.signal);
    watch.signal.throwIfAborted();
    if (owner.lifecycle.loss || owner.lifecycle.disposed || owner.disposalGuard.blocked || owner.uncaptured.length)
      throw new Error('OpenDecider-nano runtime unavailable.');
    const runner = deps.runner({...owner, weights});
    const session = deps.session({tokenizer, runner});
    watch.signal.throwIfAborted();
    watch.close();
    watch = null;
    return session;
  } catch (cause) {
    watch?.close();
    cleanupFailedLoad(owner, runtime, gpu, initialized, gpuCompatibility);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load OpenDecider-nano.', cause);
  }
}

export function loadOpenDeciderNano(
  options,
  {Tokenizer, instantiate, createPolyfillWASI, gpu = globalThis.navigator?.gpu} = {}
) {
  const checked = validateOpenDeciderNanoOptions(options);
  if (typeof Tokenizer !== 'function' || typeof instantiate !== 'function' || typeof createPolyfillWASI !== 'function')
    throw new TypeError('OpenDecider-nano runtime bindings are required.');
  return orchestrate(checked, {
    gpu,
    openSource: openOpenDeciderNanoSource,
    prepareTokenizer: (data, config) => createOpenDeciderNanoTokenizer(Tokenizer, data, config),
    instantiate: (wasm) => instantiate(wasm, createPolyfillWASI()),
    initialize: initializeH2oRuntime,
    installGpuCompatibility: installOpenDeciderNanoGpuCompatibility,
    upload,
    runner: (owner) => new OpenDeciderNanoTvmRunner(owner),
    session: createOpenDeciderNanoSession
  });
}
