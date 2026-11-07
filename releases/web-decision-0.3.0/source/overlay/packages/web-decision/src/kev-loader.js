import {Tokenizer} from '@huggingface/tokenizers';
import {instantiate} from '../../../.toolchains/apache-tvm-src/web/src/runtime.ts';
import {createPolyfillWASI} from '../../../.toolchains/apache-tvm-src/web/src/compact.ts';
import {detectKevGPUDevice as detectGPUDevice} from './kev-gpu.js';
import {commitCachedManifest, parseJson, waitForAbort} from './assets.js';
import {KEV_DELIMITERS, WebDecisionError} from './kev.js';
import {DecisionModel} from './model.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime} from './runtime-lifecycle.js';
import {watchLoadDeviceLoss} from './load-device-loss.js';
import {REFLEXES_ID, readReflexesManifestResponse, uploadReflexesHeads} from './kev-reflexes.js';
import {readAssetBatch} from './asset-batch.js';
import {prepareModelSource} from './model-source.js';
import {createModelAssetReader, uploadModelShards} from './model-assets.js';
import {readKev06ManifestResponse} from './kev06/contract.js';

function unsupported(message) {
  return new WebDecisionError('UNSUPPORTED_ENVIRONMENT', message);
}

function cacheRecordIndex(record) {
  const match = typeof record?.name === 'string' ? /^param_(\d+)$/.exec(record.name) : null;
  return match ? Number(match[1]) : -1;
}

function cacheRecordElementBytes(record) {
  return record?.dtype === 'uint32' ? 4 : record?.dtype === 'float16' ? 2 : 0;
}

function validCacheShard(shard, files) {
  if (!shard || typeof shard.dataPath !== 'string' || shard.dataPath.includes('/') || shard.dataPath.includes('\\'))
    return false;
  const file = files[`weights/${shard.dataPath}`];
  return Boolean(file) && shard.format === 'raw-shard' && shard.nbytes === file.bytes && Array.isArray(shard.records);
}

function validCacheRecordShape(record, bytesPerElement) {
  return (
    bytesPerElement > 0 &&
    Array.isArray(record.shape) &&
    !record.shape.some((value) => !Number.isSafeInteger(value) || value < 1) &&
    record.shape.reduce((a, b) => a * b, bytesPerElement) === record.nbytes
  );
}

function validCacheRecordRange(record, shard) {
  return (
    Number.isSafeInteger(record.byteOffset) &&
    record.byteOffset >= 0 &&
    record.byteOffset % 4 === 0 &&
    Number.isSafeInteger(record.nbytes) &&
    record.nbytes % 4 === 0 &&
    record.byteOffset + record.nbytes <= shard.nbytes
  );
}

function validCacheRecord(record, shard, seen) {
  if (!record || record.format !== 'raw') return false;
  const index = cacheRecordIndex(record);
  const bytesPerElement = cacheRecordElementBytes(record);
  return (
    index >= 0 &&
    index < 394 &&
    !seen.has(index) &&
    validCacheRecordShape(record, bytesPerElement) &&
    validCacheRecordRange(record, shard)
  );
}

function checkCacheManifest(manifest, cacheManifest) {
  if (cacheManifest?.metadata?.numParams !== 394 || !Array.isArray(cacheManifest.records))
    throw new WebDecisionError('INVALID_ASSET', 'Invalid TVM tensor cache metadata.');
  const seen = new Set();
  for (const shard of cacheManifest.records) {
    if (!validCacheShard(shard, manifest.files))
      throw new WebDecisionError('INVALID_ASSET', 'Invalid tensor shard metadata.');
    for (const record of shard.records) {
      if (!validCacheRecord(record, shard, seen))
        throw new WebDecisionError('INVALID_ASSET', 'Invalid model parameter metadata.');
      seen.add(cacheRecordIndex(record));
    }
  }
  if (seen.size !== 394) throw new WebDecisionError('INVALID_ASSET', 'Missing model parameters.');
}

function validateOptionalLimit(value, name) {
  if (value !== undefined && (!Number.isSafeInteger(value) || value < 0))
    throw new WebDecisionError('INVALID_INPUT', `${name} must be a non-negative safe integer.`);
}

async function loadKevModel(options, description, readManifest, uploadHeads) {
  validateOptionalLimit(options?.maxInputCharacters, 'maxInputCharacters');
  validateOptionalLimit(options?.maxInputTokens, 'maxInputTokens');
  const {
    base,
    cache,
    signal,
    manifestUrl,
    response: manifestResponse,
    maxPrefetchBytes
  } = await prepareModelSource(options, {
    offlineMissingMessage: 'The model manifest is not available offline. Load the model online first.'
  });
  const manifest = await readManifest(manifestResponse, signal);
  const assetReader = createModelAssetReader({base, manifest, cache, signal, options});
  const loadState = {
    runtime: undefined,
    gpuDevice: undefined,
    lifecycle: undefined,
    loadWatch: undefined,
    scopeOpen: false
  };
  try {
    const modelAssets = await readModelAssets({assetReader, manifest, signal});
    const modelRuntime = await uploadModelAssets({
      manifest,
      modelAssets,
      assetReader,
      maxPrefetchBytes,
      signal,
      uploadHeads,
      loadState
    });
    loadState.loadWatch.signal.throwIfAborted();
    signal?.throwIfAborted();
    assertLoadActive(loadState.lifecycle);
    await saveLoadedManifest({cache, manifestUrl, manifest, assetReader, options, signal, loadState});
    loadState.loadWatch.close();
    return createDecisionModel({options, description, manifest, modelAssets, modelRuntime, loadState});
  } catch (cause) {
    // Cleanup must not hide the original loading error, including when WebGPU
    // initialization failed before TVM took ownership of the device.
    cleanupFailedLoad(loadState);
    // AbortSignal can carry any caller-supplied reason, not just AbortError.
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load the decision model.', {cause});
  }
}

async function readModelAssets({assetReader, manifest, signal}) {
  const {read} = assetReader;
  const [tokenizerBytes, tokenizerConfigBytes, modelConfigBytes, wasmBytes, tensorCacheBytes] = await readAssetBatch(
    ['tokenizer.json', 'tokenizer_config.json', 'model_config.json', 'model.wasm', 'weights/tensor-cache.json'],
    read,
    signal
  );
  const modelConfig = parseJson(modelConfigBytes, 'model_config.json');
  for (const [name, id] of Object.entries(KEV_DELIMITERS)) {
    if (modelConfig.kev?.delimiter_ids?.[name] !== id)
      throw new WebDecisionError('INVALID_ASSET', 'Unexpected model delimiter IDs.');
  }
  const tokenizer = new Tokenizer(
    parseJson(tokenizerBytes, 'tokenizer.json'),
    parseJson(tokenizerConfigBytes, 'tokenizer_config.json')
  );
  const cacheManifest = parseJson(tensorCacheBytes, 'tensor-cache.json');
  checkCacheManifest(manifest, cacheManifest);
  signal?.throwIfAborted();
  return {tokenizer, cacheManifest, wasmBytes};
}

async function uploadModelAssets({
  manifest,
  modelAssets,
  assetReader,
  maxPrefetchBytes,
  signal,
  uploadHeads,
  loadState
}) {
  const {read} = assetReader;
  const modelRuntime = await initializeModelRuntime(modelAssets.wasmBytes, manifest, signal, loadState);
  await uploadModelShards({
    records: modelAssets.cacheManifest.records,
    read,
    runtime: loadState.runtime,
    device: modelRuntime.device,
    weights: modelRuntime.weights,
    signal: loadState.loadWatch.signal,
    maxPrefetchBytes,
    uncaptured: modelRuntime.uncaptured
  });
  if (uploadHeads)
    await uploadHeads({
      read,
      runtime: loadState.runtime,
      device: modelRuntime.device,
      weights: modelRuntime.weights,
      signal: loadState.loadWatch.signal,
      uncaptured: modelRuntime.uncaptured
    });
  return modelRuntime;
}

async function initializeModelRuntime(wasmBytes, manifest, signal, loadState) {
  let gpu;
  try {
    gpu = await detectGPUDevice(undefined, signal);
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not initialize a compatible WebGPU device.', {cause});
  }
  loadState.gpuDevice = gpu?.device;
  // The device helper disposes a device returned after cancellation.
  signal?.throwIfAborted();
  if (!gpu || !gpu.adapter.features.has('shader-f16')) {
    loadState.gpuDevice?.destroy();
    loadState.gpuDevice = undefined;
    throw unsupported('This model requires WebGPU shader-f16 support.');
  }
  const uncaptured = [];
  loadState.gpuDevice.addEventListener('uncapturederror', (event) => {
    if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
  });
  loadState.loadWatch = watchLoadDeviceLoss(loadState.gpuDevice, signal);
  loadState.runtime = await waitForAbort(
    () => instantiate(wasmBytes, createPolyfillWASI()),
    loadState.loadWatch.signal,
    (lateRuntime) => {
      try {
        lateRuntime?.dispose();
      } catch {}
    }
  );
  loadState.lifecycle = trackRuntime(loadState.runtime, loadState.gpuDevice);
  loadState.loadWatch.signal.throwIfAborted();
  await Promise.resolve();
  assertLoadActive(loadState.lifecycle);
  loadState.runtime.initWebGPU(loadState.gpuDevice);
  const device = loadState.runtime.webgpu();
  loadState.runtime.beginScope();
  loadState.scopeOpen = true;
  const vm = loadState.runtime.createVirtualMachine(device);
  return {
    device,
    fn: vm.getFunction('main'),
    weights: new Array(manifest.parameterCount),
    uncaptured
  };
}

async function saveLoadedManifest({
  cache,
  manifestUrl,
  manifest,
  assetReader,
  options,
  signal,
  loadState
}) {
  // Save only after a complete online load. Offline mode never requests the
  // network and still verifies every cached model file.
  if (!cache || !assetReader.isCacheComplete() || options.offline) return;
  try {
    await commitCachedManifest(cache, manifestUrl, manifest, loadState.loadWatch.signal);
  } catch {
    loadState.loadWatch.signal.throwIfAborted();
  }
  signal?.throwIfAborted();
  assertLoadActive(loadState.lifecycle);
}

function createDecisionModel({options, description, manifest, modelAssets, modelRuntime, loadState}) {
  const limits = {
    maxInputTokens: options?.maxInputTokens ?? manifest.maxInputTokens,
    maxStateTokens: manifest.maxStateTokens,
    maxBranchTokens: manifest.maxBranchTokens,
    maxQuestions: description.features.textChoice.limits.find(
      (limit) => limit.resource === 'questions' && limit.scope === 'request'
    )?.max,
    maxOptions: description.features.textChoice.limits.find(
      (limit) => limit.resource === 'options' && limit.scope === 'question'
    )?.max,
    maxInputCharacters: options?.maxInputCharacters
  };
  return new DecisionModel(
    loadState.runtime,
    modelRuntime.device,
    loadState.gpuDevice,
    modelRuntime.fn,
    modelRuntime.weights,
    modelAssets.tokenizer,
    limits,
    modelRuntime.uncaptured,
    loadState.lifecycle,
    true,
    description.model
  );
}

export function loadKev06DecisionModel(options, description) {
  return loadKevModel(options, description, readKev06ManifestResponse);
}

export function loadKevDecisionModel(options, description) {
  if (description.model !== REFLEXES_ID) return loadKev06DecisionModel(options, description);
  return loadKevModel(options, description, readReflexesManifestResponse, uploadReflexesHeads);
}
