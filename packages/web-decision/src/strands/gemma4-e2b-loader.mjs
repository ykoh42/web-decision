import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {cancelBody, readBoundedResponse} from '../bounded-response.js';
import {WebDecisionError} from '../kev.js';
import {trackRuntime, assertLoadActive, cleanupFailedLoad} from '../runtime-lifecycle.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {guardStrandsDisposal, retainStrandsResources} from './disposal-guard.mjs';
import {parseGemma4ShardPlan} from './gemma4-manifest.mjs';
import {uploadGemma4Shard} from './gemma4-upload.mjs';
import {createGemma4Progress} from './gemma4-progress.mjs';
import {
  STRANDS_GEMMA4_E2B_LIMITS,
  STRANDS_GEMMA4_E2B_MANIFEST_PATH,
  STRANDS_GEMMA4_E2B_MANIFEST_SHA256
} from './gemma4-e2b-contract.mjs';
import {createGemma4Session} from './gemma4-e2b-session.mjs';
import {StrandsGemma4Runner} from './gemma4-e2b-runner.mjs';

const MAX_MANIFEST_BYTES = 1024 * 1024;
const MAX_SHARD_BYTES = 128 * 1024 * 1024;
const invalid = (message, cause) => new WebDecisionError('INVALID_ASSET', message, cause ? {cause} : undefined);

function parseJson(bytes, name) {
  try {
    return JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes));
  } catch (cause) {
    throw invalid(`Invalid ${name}.`, cause);
  }
}

function checkedBase(value) {
  if (typeof value !== 'string' || value.trim().length === 0)
    throw new WebDecisionError('INVALID_INPUT', 'Strands Gemma 4 assetsUrl must be a non-empty URL string.');
  let base;
  try {
    base = new URL(value, globalThis.location?.href);
  } catch (cause) {
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 asset URL.', {cause});
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.search || base.hash || base.username || base.password)
    throw new WebDecisionError('INVALID_INPUT', 'The asset URL must be an HTTP(S) directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return base;
}

function rootFiles(manifest) {
  const expectedIds = ['graphWasm', 'inputSpecs', 'tokenizer', 'tokenizerConfig', 'cacheManifest'];
  if (
    !Array.isArray(manifest?.files) ||
    manifest.files.length !== expectedIds.length ||
    new Set(manifest.files.map((file) => file?.id)).size !== expectedIds.length
  )
    throw invalid('Unexpected Strands Gemma 4 runtime asset set.');
  const files = new Map();
  for (const file of manifest.files) {
    if (
      !expectedIds.includes(file.id) ||
      typeof file.path !== 'string' ||
      !Number.isSafeInteger(file.bytes) ||
      file.bytes < 1 ||
      !/^[0-9a-f]{64}$/u.test(file.sha256)
    )
      throw invalid('Invalid Strands Gemma 4 runtime asset metadata.');
    files.set(file.id, Object.freeze({...file}));
  }
  for (const id of expectedIds) if (!files.has(id)) throw invalid('Missing Strands Gemma 4 runtime asset.');
  for (const [id, path] of [
    ['tokenizer', 'checkpoint/tokenizer.json'],
    ['tokenizerConfig', 'checkpoint/tokenizer_config.json'],
    ['cacheManifest', 'tensor-cache-e2b-v26-fp16/tensor-cache-manifest.json']
  ])
    if (files.get(id).path !== path) throw invalid(`Unexpected pinned ${id} asset.`);
  return files;
}

function validateRuntimeManifest(manifest) {
  if (
    manifest?.schemaVersion !== 1 ||
    manifest.model !== 'StrandsAgents/strands-decider-E2B-gemma4-v1-2610' ||
    manifest.checkpointRevision !== 'd383ecf8534b13d49b3b567a1b66f88fca6d50ba' ||
    manifest.baseModel !== 'google/gemma-4-E2B-it' ||
    manifest.baseRevision !== '3e22461f65e89153144f8adb70e3b8c2cc9845a7' ||
    manifest.sourceBaseSha256 !== '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550' ||
    manifest.contextTokens !== 4096 ||
    manifest.choiceLimit !== 'bounded only by the tokenized context' ||
    manifest.sourceDtype !== 'bfloat16' ||
    manifest.computeDtype !== 'float16' ||
    manifest.weightConversion !== 'bfloat16-to-float16-lossy' ||
    manifest.derivedBufferConversion !== 'float32-to-float16-lossy' ||
    manifest.graphSha256 !== 'e749cce5d0c5ff793f54adac1e99f6416017e56a80178bc28b99c17973becbb5' ||
    manifest.inputSpecsSha256 !== '871af8982f633dcee0a2d22ecef3cfab650db26bf2c1b02bad07ccf028500a65' ||
    manifest.cacheDirectory !== 'tensor-cache-e2b-v26-fp16' ||
    manifest.constantTensors?.length !== 1 ||
    manifest.constantTensors[0]?.index !== 1207 ||
    manifest.constantTensors[0]?.target !== 'torso.embed_tokens.embed_scale' ||
    manifest.constantTensors[0]?.dtype !== 'float16' ||
    manifest.constantTensors[0]?.sourceValue !== 39.1875 ||
    manifest.constantTensors[0]?.shape?.length !== 0
  )
    throw invalid('The Strands Gemma 4 runtime manifest does not match the pinned E2B graph.');
  return rootFiles(manifest);
}

async function fetchPinnedFile(base, file, signal) {
  signal?.throwIfAborted();
  if (!Number.isSafeInteger(file.bytes) || file.bytes < 1 || file.bytes > MAX_SHARD_BYTES)
    throw invalid('A Strands Gemma 4 asset has an invalid size.');
  const url = assetUrl(base, file.path);
  let response;
  try {
    response = await waitForAbort(
      () => fetch(url, {signal}),
      signal,
      (late) => cancelBody(late?.body)
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not fetch ${file.path}.`, {cause});
  }
  if (signal?.aborted) {
    cancelBody(response?.body);
    signal.throwIfAborted();
  }
  const declaredLength = response.headers.get('content-length');
  if (
    response.status !== 200 ||
    (declaredLength !== null && (!/^[0-9]+$/u.test(declaredLength) || Number(declaredLength) !== file.bytes))
  ) {
    cancelBody(response.body);
    signal?.throwIfAborted();
    throw invalid(`Invalid or inaccessible ${file.path} response.`);
  }
  let bytes;
  try {
    bytes = new Uint8Array(await readBoundedResponse(response, file.bytes, {exact: true, signal}));
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not read ${file.path}.`, {cause});
  }
  const actual = await waitForAbort(() => sha256(bytes), signal);
  signal?.throwIfAborted();
  if (actual !== file.sha256) throw invalid(`Hash mismatch for ${file.path}.`);
  return bytes;
}

async function fetchRootManifest(base, signal) {
  const url = assetUrl(base, STRANDS_GEMMA4_E2B_MANIFEST_PATH);
  let response;
  try {
    response = await waitForAbort(
      () => fetch(url, {signal}),
      signal,
      (late) => cancelBody(late?.body)
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not fetch the Gemma 4 runtime manifest.', {cause});
  }
  if (signal?.aborted) {
    cancelBody(response?.body);
    signal.throwIfAborted();
  }
  const declaredLength = response.headers.get('content-length');
  if (
    response.status !== 200 ||
    (declaredLength !== null && (!/^[0-9]+$/u.test(declaredLength) || Number(declaredLength) > MAX_MANIFEST_BYTES))
  ) {
    cancelBody(response.body);
    throw invalid('Invalid or inaccessible Gemma 4 runtime manifest response.');
  }
  const bytes = new Uint8Array(await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal}));
  if (!bytes.byteLength || bytes.byteLength > MAX_MANIFEST_BYTES)
    throw invalid('Invalid Gemma 4 runtime manifest size.');
  const actual = await waitForAbort(() => sha256(bytes), signal);
  signal?.throwIfAborted();
  if (actual !== STRANDS_GEMMA4_E2B_MANIFEST_SHA256) throw invalid('Gemma 4 runtime manifest hash mismatch.');
  return bytes;
}

function verifiedSmallFile(files, id, expectedPath, expectedHash) {
  const file = files.get(id);
  if (!file || file.path !== expectedPath || file.sha256 !== expectedHash)
    throw invalid(`Unexpected pinned ${id} asset.`);
  return file;
}

function checkAdapter(adapter) {
  if (
    !adapter ||
    adapter.isFallbackAdapter ||
    adapter.info?.isFallbackAdapter ||
    typeof adapter.features?.has !== 'function' ||
    !adapter.features.has('shader-f16') ||
    typeof adapter.requestDevice !== 'function'
  )
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'Strands Gemma 4 needs a hardware WebGPU adapter with shader-f16.'
    );
  for (const [name, minimum] of Object.entries(STRANDS_GEMMA4_E2B_LIMITS))
    if (!Number.isSafeInteger(adapter.limits?.[name]) || adapter.limits[name] < minimum)
      throw new WebDecisionError(
        'UNSUPPORTED_ENVIRONMENT',
        `Strands Gemma 4 needs WebGPU ${name} of at least ${minimum}.`
      );
}

async function requestAdapter(gpuApi, signal) {
  if (typeof gpuApi?.requestAdapter !== 'function')
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'
    );
  if (typeof globalThis.crypto?.subtle?.digest !== 'function')
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Web Crypto is unavailable. Use HTTPS or localhost.');
  let adapter;
  try {
    adapter = await waitForAbort(() => gpuApi.requestAdapter({powerPreference: 'high-performance'}), signal);
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not request a WebGPU adapter.', {cause});
  }
  signal?.throwIfAborted();
  checkAdapter(adapter);
  return adapter;
}

async function syncAfterScopeFailure(started, lifecycle, device, disposalGuard) {
  if (!started || lifecycle.loss || lifecycle.disposed) return;
  try {
    await device.sync();
  } catch {
    disposalGuard.retain();
  }
}

async function closeGpuErrorScopes(gpu, count, failure, noFailure) {
  while (count-- > 0) {
    try {
      const error = await gpu.popErrorScope();
      if (failure === noFailure && error) failure = error;
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
  }
  return failure;
}

async function gpuScope(gpu, device, lifecycle, disposalGuard, operation) {
  const noFailure = Symbol('no failure');
  let count = 0,
    failure = noFailure,
    value,
    started = false;
  try {
    gpu.pushErrorScope('out-of-memory');
    count++;
    gpu.pushErrorScope('validation');
    count++;
    started = true;
    value = await operation();
  } catch (cause) {
    failure = cause;
    await syncAfterScopeFailure(started, lifecycle, device, disposalGuard);
  }
  failure = await closeGpuErrorScopes(gpu, count, failure, noFailure);
  if (failure !== noFailure) throw failure;
  return value;
}

async function verifyAndReadSource(base, signal, onProgress) {
  const rootBytes = await fetchRootManifest(base, signal);
  const manifest = parseJson(rootBytes, 'runtime manifest');
  const files = validateRuntimeManifest(manifest);
  const rootById = {
    graphWasm: verifiedSmallFile(
      files,
      'graphWasm',
      'graph-v26/model.wasm',
      'e749cce5d0c5ff793f54adac1e99f6416017e56a80178bc28b99c17973becbb5'
    ),
    inputSpecs: verifiedSmallFile(
      files,
      'inputSpecs',
      'graph-v26/input-specs.json',
      '871af8982f633dcee0a2d22ecef3cfab650db26bf2c1b02bad07ccf028500a65'
    ),
    tokenizer: files.get('tokenizer'),
    tokenizerConfig: files.get('tokenizerConfig'),
    cacheManifest: files.get('cacheManifest')
  };
  const fileBytes = Object.values(rootById).reduce((total, file) => total + file.bytes, rootBytes.byteLength);
  const assets = {};
  for (const [id, file] of Object.entries(rootById)) {
    assets[id] = await fetchPinnedFile(base, file, signal);
  }
  const cacheManifest = parseJson(assets.cacheManifest, 'tensor-cache manifest');
  if (
    cacheManifest.files?.length !== 84 ||
    cacheManifest.sourceWeightDtype !== 'bfloat16' ||
    cacheManifest.computeWeightDtype !== 'float16' ||
    cacheManifest.weightConversion !== 'bfloat16-to-float16-lossy' ||
    cacheManifest.derivedBufferConversion !== 'float32-to-float16-lossy'
  )
    throw invalid('Unexpected Gemma 4 tensor-cache manifest.');
  const cacheFiles = new Map(cacheManifest.files.map((file) => [file.path, file]));
  const cacheIndexFile = cacheFiles.get('tensor-cache.json');
  const sourceMapFile = cacheFiles.get('tensor-source-map.json');
  if (!cacheIndexFile || !sourceMapFile) throw invalid('The Gemma 4 tensor cache is incomplete.');
  const cacheBase = new URL(`${manifest.cacheDirectory}/`, base);
  const [cacheIndexBytes, sourceMapBytes] = await Promise.all([
    fetchPinnedFile(cacheBase, cacheIndexFile, signal),
    fetchPinnedFile(cacheBase, sourceMapFile, signal)
  ]);
  assets.cacheIndex = parseJson(cacheIndexBytes, 'tensor-cache index');
  assets.sourceMap = parseJson(sourceMapBytes, 'tensor-source map');
  const inputSpecs = parseJson(assets.inputSpecs, 'graph input specs');
  const expected = {
    model: 'StrandsAgents/strands-decider-E2B-gemma4-v1-2610',
    checkpointRevision: 'd383ecf8534b13d49b3b567a1b66f88fca6d50ba',
    baseModel: 'google/gemma-4-E2B-it',
    baseRevision: '3e22461f65e89153144f8adb70e3b8c2cc9845a7',
    sourceBaseSha256: '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550',
    graphSha256: 'e749cce5d0c5ff793f54adac1e99f6416017e56a80178bc28b99c17973becbb5',
    inputSpecsSha256: '871af8982f633dcee0a2d22ecef3cfab650db26bf2c1b02bad07ccf028500a65',
    cacheIndexSha256: cacheIndexFile.sha256,
    cacheIndexBytes: cacheIndexFile.bytes,
    sourceMapSha256: sourceMapFile.sha256,
    sourceMapBytes: sourceMapFile.bytes
  };
  let plan;
  try {
    plan = parseGemma4ShardPlan({
      inputSpecs,
      cacheManifest,
      cacheIndex: assets.cacheIndex,
      sourceMap: assets.sourceMap,
      expected
    });
  } catch (cause) {
    throw invalid('The Gemma 4 tensor-cache layout does not match the graph.', cause);
  }
  const shardBytes = plan.shards.reduce((total, shard) => total + shard.bytes, 0);
  const totalBytes = fileBytes + cacheIndexBytes.byteLength + sourceMapBytes.byteLength + shardBytes;
  const reportProgress = createGemma4Progress(onProgress, totalBytes);
  reportProgress(fileBytes + cacheIndexBytes.byteLength + sourceMapBytes.byteLength);
  return {manifest, files, assets, cacheBase, plan, reportProgress};
}

function retainLateDevice(device) {
  try {
    device?.destroy();
  } catch {
    retainStrandsResources(undefined, device);
  }
}

/** Internal runtime seam; application options cannot replace integrity or GPU controls. */
export async function loadStrandsGemma4FromSource(
  {
    assetsUrl,
    cache = false,
    offline = false,
    signal,
    onProgress,
    maxPrefetchBytes = 0,
    createTokenizer,
    createRuntime,
    gpuApi = globalThis.navigator?.gpu
  } = {},
  // Internal dependency seam for tests; do not expose these controls as load options.
  {readFile = fetchPinnedFile, preparedAdapter, verifySource = verifyAndReadSource} = {}
) {
  if (
    cache !== false ||
    offline !== false ||
    maxPrefetchBytes !== 0 ||
    typeof createTokenizer !== 'function' ||
    typeof createRuntime !== 'function'
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 loader options.');
  const base = checkedBase(assetsUrl);
  signal?.throwIfAborted();
  const adapter = preparedAdapter ?? (await requestAdapter(gpuApi, signal));
  checkAdapter(adapter);
  signal?.throwIfAborted();
  const source = await verifySource(base, signal, onProgress);
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
          requiredLimits: STRANDS_GEMMA4_E2B_LIMITS
        }),
      signal,
      retainLateDevice
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
    runtime = await waitForAbort(
      () => createRuntime(source.assets.graphWasm),
      loadAbort.signal,
      (lateRuntime) => {
        try {
          lateRuntime?.dispose();
        } catch {}
      }
    );
    lifecycle = trackRuntime(runtime, gpu);
    disposalGuard = guardStrandsDisposal(runtime, gpu, lifecycle);
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scopeOpen = true;
    const device = runtime.webgpu();
    const check = () => {
      loadAbort.signal.throwIfAborted();
      assertLoadActive(lifecycle);
      if (uncaptured.length) throw new WebDecisionError('MODEL_LOAD_FAILED', uncaptured[0]);
    };
    const context = runtime.lib?.webGPUContext;
    if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof context?.gpuBufferFromPtr !== 'function')
      throw new Error('The pinned TVM WebGPU upload API is unavailable.');
    const weights = new Array(source.plan.expectedParameters.length);
    for (const shard of source.plan.shards) {
      check();
      const file = {path: shard.path, bytes: shard.bytes, sha256: shard.sha256};
      const bytes = await readFile(source.cacheBase, file, loadAbort.signal);
      check();
      const uploaded = await gpuScope(gpu, device, lifecycle, disposalGuard, () =>
        uploadGemma4Shard({
          bytes,
          records: shard.records,
          expectedParameters: shard.expectedParameters,
          parameterCount: weights.length,
          allocate({shape, dtype}) {
            check();
            const tensor = runtime.empty(shape, dtype, device);
            if (
              typeof tensor?.getDataPtr !== 'function' ||
              !Number.isSafeInteger(tensor.byteOffset) ||
              tensor.byteOffset < 0 ||
              tensor.byteOffset % 4
            )
              throw new Error('Invalid Gemma 4 TVM tensor allocation.');
            const buffer = context.gpuBufferFromPtr(tensor.getDataPtr());
            if (!Number.isSafeInteger(buffer?.size) || buffer.size < tensor.byteOffset)
              throw new Error('Gemma 4 TVM tensor allocation has no verified capacity.');
            return {tensor, capacityBytes: buffer.size - tensor.byteOffset};
          },
          context,
          sync: () => device.sync(),
          check,
          signal: loadAbort.signal
        })
      );
      for (const [index, tensor] of uploaded) weights[index] = tensor;
      source.reportProgress(bytes.byteLength);
    }
    check();
    if (weights.length !== 1248 || weights.some((tensor) => !tensor)) throw invalid('Gemma 4 weights are incomplete.');
    const fn = runtime.createVirtualMachine(device).getFunction('main');
    const runner = new StrandsGemma4Runner({runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard});
    const tokenizerData = parseJson(source.assets.tokenizer, 'tokenizer');
    const tokenizerConfig = parseJson(source.assets.tokenizerConfig, 'tokenizer configuration');
    const tokenizer = createTokenizer(tokenizerData, tokenizerConfig);
    const session = createGemma4Session({tokenizer, runner});
    check();
    signal?.throwIfAborted();
    loading = false;
    loadWatch.signal.removeEventListener('abort', forward);
    loadWatch.close();
    return session;
  } catch (cause) {
    loading = false;
    if (loadWatch && forward) loadWatch.signal.removeEventListener('abort', forward);
    cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice: gpu});
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    if (cause?.name === 'GPUOutOfMemoryError')
      throw new WebDecisionError(
        'DEVICE_RESOURCE_LIMIT',
        'This WebGPU device ran out of resources while loading Gemma 4.',
        {cause}
      );
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load experimental Strands Gemma 4.', {cause});
  }
}

/** Source path for app-hosted, pinned model assets. */
export async function loadExperimentalStrandsGemma4FromSource(options) {
  return loadStrandsGemma4FromSource(options);
}
