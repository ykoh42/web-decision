import {sha256, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime} from '../runtime-lifecycle.js';
import {uploadFusedRecord} from './fused-upload.mjs';
import {
  checkFusedAdapter,
  LARGEST_BINDING,
  parsePinnedManifest,
  TOKENIZER_CONFIG_SHA256,
  TOKENIZER_SHA256,
  uploadPinnedWeights,
  WASM_BYTES,
  WASM_SHA256
} from './manifest.mjs';
import {ExperimentalFusedRunner} from './runner.mjs';
import {ExperimentalStrandsSession} from './session.mjs';
import {requestStrandsAdapter} from './preflight.mjs';
import {guardStrandsDisposal, retainStrandsResources} from './disposal-guard.mjs';

const invalid = (message) => new WebDecisionError('INVALID_ASSET', message);

async function verifiedCopy(bytes, length, expected, signal, name) {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== length) throw invalid(`Invalid ${name} size.`);
  const copy = Uint8Array.from(bytes);
  const digest = await waitForAbort(() => sha256(copy), signal);
  signal?.throwIfAborted();
  if (digest !== expected) throw invalid(`${name} hash mismatch.`);
  return copy;
}

async function gpuScope(gpu, device, lifecycle, operation, disposalGuard) {
  const noFailure = Symbol('no failure');
  let count = 0,
    value,
    failure = noFailure,
    started = false,
    fallbackAttempted = false;
  async function fallbackSync() {
    if (!started || fallbackAttempted) return;
    fallbackAttempted = true;
    if (lifecycle.loss || lifecycle.disposed) {
      disposalGuard.retain();
      return;
    }
    // A failed sync does not prove that pending work settled. Keep the first error.
    try {
      await device.sync();
    } catch {
      disposalGuard.retain();
    }
  }
  try {
    gpu.pushErrorScope('out-of-memory');
    count++;
    gpu.pushErrorScope('validation');
    count++;
    started = true;
    value = await operation();
  } catch (cause) {
    failure = cause;
    await fallbackSync();
  }
  while (count--) {
    try {
      const error = await gpu.popErrorScope();
      if (failure === noFailure && error) failure = error;
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
    if (failure !== noFailure) await fallbackSync();
  }
  if (failure !== noFailure) throw failure;
  return value;
}

function protectRuntime(runtime, gpu) {
  try {
    const lifecycle = trackRuntime(runtime, gpu);
    return {lifecycle, disposalGuard: guardStrandsDisposal(runtime, gpu, lifecycle)};
  } catch (error) {
    retainStrandsResources(runtime, gpu);
    throw error;
  }
}

function initializeRuntime(runtime, gpu, disposalGuard) {
  try {
    runtime.initWebGPU(gpu);
    runtime.beginScope();
  } catch (error) {
    disposalGuard.retain();
    throw error;
  }
}

function cleanupStrandsLoad({loadWatch, runtime, scopeOpen, lifecycle, gpu, disposalGuard}) {
  if (runtime && (!disposalGuard || disposalGuard.blocked || lifecycle?.loss)) {
    loadWatch?.close();
    retainStrandsResources(runtime, gpu);
  } else {
    cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice: gpu});
  }
}

async function verifyPinnedBundle({manifestBytes, wasmBytes, tokenizerBytes, tokenizerConfigBytes}, signal) {
  if (!(manifestBytes instanceof Uint8Array) || manifestBytes.byteLength > 1024 * 1024)
    throw invalid('Invalid manifest bytes.');
  const manifestCopy = Uint8Array.from(manifestBytes);
  const [manifest, wasm, tokenizerJson, tokenizerConfig] = await Promise.all([
    parsePinnedManifest(manifestCopy, signal),
    verifiedCopy(wasmBytes, WASM_BYTES, WASM_SHA256, signal, 'Compiled WASM'),
    verifiedCopy(tokenizerBytes, 19989592, TOKENIZER_SHA256, signal, 'Tokenizer'),
    verifiedCopy(tokenizerConfigBytes, 1128, TOKENIZER_CONFIG_SHA256, signal, 'Tokenizer configuration')
  ]);
  return {manifest, wasm, tokenizerJson, tokenizerConfig};
}

/** Internal dependency seam for tests; never expose its second argument as user options. */
export async function loadStrandsFromBytes(
  {
    base,
    manifestBytes,
    wasmBytes,
    tokenizerBytes,
    tokenizerConfigBytes,
    createTokenizer,
    createRuntime,
    gpuApi,
    signal,
    onProgress,
    cache = false,
    offline = false
  } = {},
  {
    uploadTensor,
    readRange,
    finalize,
    preparedAdapter,
    verifyAssets = verifyPinnedBundle,
    uploadWeights = uploadPinnedWeights,
    retainLateDevice = (device) => retainStrandsResources(undefined, device)
  } = {}
) {
  if (cache !== false || offline !== false)
    throw new WebDecisionError('INVALID_INPUT', 'Experimental Strands loading does not yet support cache or offline.');
  if (
    typeof createTokenizer !== 'function' ||
    typeof createRuntime !== 'function' ||
    typeof gpuApi?.requestAdapter !== 'function'
  )
    throw new WebDecisionError('INVALID_INPUT', 'Tokenizer, runtime and WebGPU factories are required.');
  let assetBase;
  try {
    assetBase = new URL(base);
  } catch {
    throw new WebDecisionError('INVALID_INPUT', 'A valid asset base URL is required.');
  }
  if (
    !['https:', 'http:'].includes(assetBase.protocol) ||
    !assetBase.pathname.endsWith('/') ||
    assetBase.search ||
    assetBase.hash ||
    assetBase.username ||
    assetBase.password
  )
    throw new WebDecisionError('INVALID_INPUT', 'The asset base must be an HTTP(S) directory URL.');
  signal?.throwIfAborted();
  const {manifest, wasm, tokenizerJson, tokenizerConfig} = await verifyAssets(
    {manifestBytes, wasmBytes, tokenizerBytes, tokenizerConfigBytes},
    signal
  );
  signal?.throwIfAborted();
  let tokenizer;
  try {
    const decoder = new TextDecoder('utf-8', {fatal: true});
    tokenizer = createTokenizer(JSON.parse(decoder.decode(tokenizerJson)), JSON.parse(decoder.decode(tokenizerConfig)));
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Could not create the pinned tokenizer.', {cause});
  }
  const adapter = preparedAdapter === undefined ? await requestStrandsAdapter(gpuApi, signal) : preparedAdapter;
  checkFusedAdapter(adapter);
  signal?.throwIfAborted();
  let gpu,
    runtime,
    lifecycle,
    disposalGuard,
    loadWatch,
    scopeOpen = false,
    loading = true,
    forward;
  const uncaptured = [],
    loadAbort = new AbortController();
  try {
    gpu = await waitForAbort(
      () =>
        adapter.requestDevice({
          requiredFeatures: ['shader-f16'],
          requiredLimits: {
            maxBufferSize: LARGEST_BINDING,
            maxStorageBufferBindingSize: LARGEST_BINDING,
            maxStorageBuffersPerShaderStage: 8
          }
        }),
      signal,
      (late) => {
        try {
          late?.destroy();
        } catch {
          retainLateDevice(late);
        }
      }
    );
    signal?.throwIfAborted();
    loadWatch = watchLoadDeviceLoss(gpu, signal);
    forward = () => loadAbort.abort(loadWatch.signal.reason);
    loadWatch.signal.addEventListener('abort', forward, {once: true});
    if (loadWatch.signal.aborted) forward();
    gpu.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
      if (loading && !loadAbort.signal.aborted)
        loadAbort.abort(new WebDecisionError('MODEL_LOAD_FAILED', 'WebGPU reported an uncaptured loading error.'));
    });
    // Runtime creation does not use the GPU yet. Stop waiting on abort/loss,
    // but dispose a runtime that eventually resolves after the load failed.
    runtime = await waitForAbort(
      () => createRuntime(wasm),
      loadAbort.signal,
      (lateRuntime) => {
        try {
          lateRuntime?.dispose();
        } catch {}
      }
    );
    ({lifecycle, disposalGuard} = protectRuntime(runtime, gpu));
    loadAbort.signal.throwIfAborted();
    initializeRuntime(runtime, gpu, disposalGuard);
    scopeOpen = true;
    const device = runtime.webgpu();
    const check = () => {
      loadAbort.signal.throwIfAborted();
      assertLoadActive(lifecycle);
      if (uncaptured.length) throw new WebDecisionError('MODEL_LOAD_FAILED', uncaptured[0]);
    };
    const upload =
      uploadTensor ||
      ((record, tensor, uploadSignal) =>
        uploadFusedRecord(
          assetBase,
          record,
          runtime.lib.webGPUContext,
          tensor.getDataPtr(),
          tensor.byteOffset,
          () => device.sync(),
          check,
          (work) => gpuScope(gpu, device, lifecycle, work, disposalGuard),
          {signal: uploadSignal, readRange}
        ));
    const weights = await uploadWeights(manifest, {
      signal: loadAbort.signal,
      onProgress,
      allocate: (record) =>
        gpuScope(gpu, device, lifecycle, () => runtime.empty(record.shape, record.dtype, device), disposalGuard),
      upload
    });
    check();
    signal?.throwIfAborted();
    const fn = runtime.createVirtualMachine(device).getFunction('main');
    const runner = new ExperimentalFusedRunner({
      runtime,
      device,
      gpu,
      fn,
      weights,
      lifecycle,
      uncaptured,
      disposalGuard
    });
    const session = new ExperimentalStrandsSession({tokenizer, runner});
    if (finalize) await finalize(loadAbort.signal);
    check();
    signal?.throwIfAborted();
    loading = false;
    loadWatch.signal.removeEventListener('abort', forward);
    loadWatch.close();
    return session;
  } catch (cause) {
    loading = false;
    if (loadWatch && forward) loadWatch.signal.removeEventListener('abort', forward);
    cleanupStrandsLoad({loadWatch, runtime, scopeOpen, lifecycle, gpu, disposalGuard});
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load experimental Strands.', {cause});
  }
}
