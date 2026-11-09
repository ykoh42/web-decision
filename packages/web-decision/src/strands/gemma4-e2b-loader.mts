import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {cancelBody, readBoundedResponse} from '../bounded-response.js';
import {WebDecisionError} from '../kev.js';
import {trackRuntime, assertLoadActive, cleanupFailedLoad} from '../runtime-lifecycle.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {guardStrandsDisposal, retainStrandsResources} from './disposal-guard.mjs';
import {parseGemma4ShardPlan} from './gemma4-manifest.mjs';
import type {Gemma4ShardPlan} from './gemma4-manifest.mjs';
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
type Gemma4Record = Record<string, unknown>;
type Gemma4AssetId = 'graphWasm' | 'inputSpecs' | 'tokenizer' | 'tokenizerConfig' | 'cacheManifest';

interface Gemma4RuntimeAsset {
  readonly id: string;
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

interface Gemma4PinnedFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

interface Gemma4Adapter {
  readonly isFallbackAdapter?: boolean | undefined;
  readonly info?: {readonly isFallbackAdapter?: boolean | undefined} | undefined;
  readonly features?: {has(feature: string): boolean} | undefined;
  readonly limits?: Readonly<Record<string, number>> | undefined;
  requestDevice(descriptor: {
    readonly requiredFeatures: readonly string[];
    readonly requiredLimits: unknown;
  }): Promise<Gemma4GpuDevice>;
}

interface Gemma4GpuApi {
  requestAdapter(options: {readonly powerPreference: 'high-performance'}): Promise<Gemma4Adapter | null>;
}

interface Gemma4GpuDevice {
  readonly lost: PromiseLike<unknown>;
  addEventListener(type: 'uncapturederror', listener: (event: unknown) => void): void;
  pushErrorScope(filter: 'out-of-memory' | 'validation'): void;
  popErrorScope(): Promise<unknown>;
  destroy(): void;
}

interface Gemma4Device {
  sync(): Promise<unknown>;
}

interface Gemma4UploadContext {
  copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, offset: number, length: number): void;
  gpuBufferFromPtr(pointer: unknown): {readonly size: number} | undefined;
}

interface Gemma4Tensor {
  readonly byteOffset: number;
  readonly shape: number[];
  copyFrom(value: unknown): Gemma4Tensor;
  getDataPtr(): unknown;
  toArray(): unknown;
}

interface Gemma4Runtime {
  readonly lib?: {readonly webGPUContext?: Gemma4UploadContext | null | undefined} | undefined;
  initWebGPU(gpu: Gemma4GpuDevice): void;
  beginScope(): void;
  endScope(): void;
  webgpu(): Gemma4Device;
  empty(shape: number[], dtype: string, device: unknown): Gemma4Tensor;
  cpu(): unknown;
  getGlobalFunc(name: string): () => unknown;
  createVirtualMachine(device: Gemma4Device): {getFunction(name: string): (...args: unknown[]) => Gemma4Tensor};
  dispose(...args: unknown[]): unknown;
}

interface Gemma4Lifecycle {
  readonly loss: unknown;
  readonly disposed: boolean;
}

interface Gemma4DisposalGuard {
  retain(): void;
  readonly blocked?: boolean;
}

interface Gemma4LoadWatch {
  readonly signal: AbortSignal;
  close(): void;
}

interface Gemma4ProgressEvent {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

interface Gemma4VerifiedSource {
  readonly manifest: Gemma4Record;
  readonly files: Map<string, Gemma4RuntimeAsset>;
  readonly assets: Record<string, unknown>;
  readonly cacheBase: URL;
  readonly plan: Gemma4ShardPlan;
  readonly reportProgress: (bytes: number) => void;
}

interface Gemma4LoadOptions {
  readonly assetsUrl?: string | undefined;
  readonly cache?: boolean | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((event: Gemma4ProgressEvent) => void) | undefined;
  readonly maxPrefetchBytes?: number | undefined;
  readonly createTokenizer?: ((data: unknown, config: unknown) => {readonly encode: unknown}) | undefined;
  readonly createRuntime?: ((wasm: unknown) => unknown) | undefined;
  readonly gpuApi?: unknown;
}

interface Gemma4LoadDependencies {
  readonly readFile?: typeof fetchPinnedFile;
  readonly preparedAdapter?: unknown;
  readonly verifySource?: typeof verifyAndReadSource;
}

const invalid = (message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError('INVALID_ASSET', message, cause ? {cause} : undefined);

function parseJson(bytes: Uint8Array, name: string): unknown {
  try {
    return JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes));
  } catch (cause) {
    throw invalid(`Invalid ${name}.`, cause);
  }
}

function checkedBase(value: unknown): URL {
  if (typeof value !== 'string' || value.trim().length === 0)
    throw new WebDecisionError('INVALID_INPUT', 'Strands Gemma 4 assetsUrl must be a non-empty URL string.');
  let base: URL;
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

function rootFiles(manifest: unknown): Map<string, Gemma4RuntimeAsset> {
  const expectedIds: Gemma4AssetId[] = ['graphWasm', 'inputSpecs', 'tokenizer', 'tokenizerConfig', 'cacheManifest'];
  if (
    !Array.isArray((manifest as Gemma4Record | null | undefined)?.files) ||
    ((manifest as Gemma4Record).files as unknown[]).length !== expectedIds.length ||
    new Set(
      ((manifest as Gemma4Record).files as unknown[]).map(
        (file: unknown) => (file as Gemma4Record | null | undefined)?.id
      )
    ).size !== expectedIds.length
  )
    throw invalid('Unexpected Strands Gemma 4 runtime asset set.');
  const files = new Map<string, Gemma4RuntimeAsset>();
  for (const file of (manifest as Gemma4Record).files as unknown[]) {
    if (
      !expectedIds.includes((file as Gemma4Record).id as Gemma4AssetId) ||
      typeof (file as Gemma4Record).path !== 'string' ||
      !Number.isSafeInteger((file as Gemma4Record).bytes) ||
      ((file as Gemma4Record).bytes as number) < 1 ||
      !/^[0-9a-f]{64}$/u.test((file as Gemma4Record).sha256 as string)
    )
      throw invalid('Invalid Strands Gemma 4 runtime asset metadata.');
    files.set(
      (file as Gemma4Record).id as string,
      Object.freeze({...(file as Gemma4Record)}) as unknown as Gemma4RuntimeAsset
    );
  }
  for (const id of expectedIds) if (!files.has(id)) throw invalid('Missing Strands Gemma 4 runtime asset.');
  for (const [id, path] of [
    ['tokenizer', 'checkpoint/tokenizer.json'],
    ['tokenizerConfig', 'checkpoint/tokenizer_config.json'],
    ['cacheManifest', 'tensor-cache-e2b-v26-fp16/tensor-cache-manifest.json']
  ])
    if (files.get(id as string)!.path !== path) throw invalid(`Unexpected pinned ${id} asset.`);
  return files;
}

function validateRuntimeManifest(manifest: unknown): Map<string, Gemma4RuntimeAsset> {
  if (
    (manifest as Gemma4Record | null | undefined)?.schemaVersion !== 1 ||
    (manifest as Gemma4Record).model !== 'StrandsAgents/strands-decider-E2B-gemma4-v1-2610' ||
    (manifest as Gemma4Record).checkpointRevision !== 'd383ecf8534b13d49b3b567a1b66f88fca6d50ba' ||
    (manifest as Gemma4Record).baseModel !== 'google/gemma-4-E2B-it' ||
    (manifest as Gemma4Record).baseRevision !== '3e22461f65e89153144f8adb70e3b8c2cc9845a7' ||
    (manifest as Gemma4Record).sourceBaseSha256 !==
      '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550' ||
    (manifest as Gemma4Record).contextTokens !== 4096 ||
    (manifest as Gemma4Record).choiceLimit !== 'bounded only by the tokenized context' ||
    (manifest as Gemma4Record).sourceDtype !== 'bfloat16' ||
    (manifest as Gemma4Record).computeDtype !== 'float16' ||
    (manifest as Gemma4Record).weightConversion !== 'bfloat16-to-float16-lossy' ||
    (manifest as Gemma4Record).derivedBufferConversion !== 'float32-to-float16-lossy' ||
    (manifest as Gemma4Record).graphSha256 !== 'e749cce5d0c5ff793f54adac1e99f6416017e56a80178bc28b99c17973becbb5' ||
    (manifest as Gemma4Record).inputSpecsSha256 !==
      '871af8982f633dcee0a2d22ecef3cfab650db26bf2c1b02bad07ccf028500a65' ||
    (manifest as Gemma4Record).cacheDirectory !== 'tensor-cache-e2b-v26-fp16' ||
    ((manifest as Gemma4Record).constantTensors as {readonly length?: unknown} | null | undefined)?.length !== 1 ||
    (((manifest as Gemma4Record).constantTensors as unknown[])[0] as Gemma4Record | null | undefined)?.index !== 1207 ||
    (((manifest as Gemma4Record).constantTensors as unknown[])[0] as Gemma4Record | null | undefined)?.target !==
      'torso.embed_tokens.embed_scale' ||
    (((manifest as Gemma4Record).constantTensors as unknown[])[0] as Gemma4Record | null | undefined)?.dtype !==
      'float16' ||
    (((manifest as Gemma4Record).constantTensors as unknown[])[0] as Gemma4Record | null | undefined)?.sourceValue !==
      39.1875 ||
    (
      (((manifest as Gemma4Record).constantTensors as unknown[])[0] as Gemma4Record | null | undefined)?.shape as
        | {readonly length?: unknown}
        | null
        | undefined
    )?.length !== 0
  )
    throw invalid('The Strands Gemma 4 runtime manifest does not match the pinned E2B graph.');
  return rootFiles(manifest);
}

async function fetchPinnedFile(
  base: URL,
  file: Gemma4PinnedFile,
  signal: AbortSignal | null | undefined
): Promise<Uint8Array> {
  signal?.throwIfAborted();
  if (!Number.isSafeInteger(file.bytes) || file.bytes < 1 || file.bytes > MAX_SHARD_BYTES)
    throw invalid('A Strands Gemma 4 asset has an invalid size.');
  const url = assetUrl(base, file.path);
  let response;
  try {
    response = await waitForAbort(
      () => fetch(url, {signal} as RequestInit & {readonly signal?: AbortSignal | null | undefined}),
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
    bytes = new Uint8Array(
      await readBoundedResponse(response, file.bytes, {exact: true, signal} as unknown as Parameters<
        typeof readBoundedResponse
      >[2])
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not read ${file.path}.`, {cause});
  }
  const actual = await waitForAbort(() => sha256(bytes), signal);
  signal?.throwIfAborted();
  if (actual !== file.sha256) throw invalid(`Hash mismatch for ${file.path}.`);
  return bytes;
}

async function fetchRootManifest(base: URL, signal: AbortSignal | null | undefined): Promise<Uint8Array> {
  const url = assetUrl(base, STRANDS_GEMMA4_E2B_MANIFEST_PATH);
  let response;
  try {
    response = await waitForAbort(
      () => fetch(url, {signal} as RequestInit & {readonly signal?: AbortSignal | null | undefined}),
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
  const bytes = new Uint8Array(
    await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal} as unknown as Parameters<
      typeof readBoundedResponse
    >[2])
  );
  if (!bytes.byteLength || bytes.byteLength > MAX_MANIFEST_BYTES)
    throw invalid('Invalid Gemma 4 runtime manifest size.');
  const actual = await waitForAbort(() => sha256(bytes), signal);
  signal?.throwIfAborted();
  if (actual !== STRANDS_GEMMA4_E2B_MANIFEST_SHA256) throw invalid('Gemma 4 runtime manifest hash mismatch.');
  return bytes;
}

function verifiedSmallFile(
  files: Map<string, Gemma4RuntimeAsset>,
  id: Gemma4AssetId,
  expectedPath: string,
  expectedHash: string
): Gemma4RuntimeAsset {
  const file = files.get(id);
  if (!file || file.path !== expectedPath || file.sha256 !== expectedHash)
    throw invalid(`Unexpected pinned ${id} asset.`);
  return file;
}

function checkAdapter(adapter: unknown): asserts adapter is Gemma4Adapter {
  if (
    !adapter ||
    (adapter as Gemma4Adapter).isFallbackAdapter ||
    (adapter as Gemma4Adapter).info?.isFallbackAdapter ||
    typeof (adapter as Gemma4Adapter).features?.has !== 'function' ||
    !(adapter as Gemma4Adapter).features!.has('shader-f16') ||
    typeof (adapter as Gemma4Adapter).requestDevice !== 'function'
  )
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'Strands Gemma 4 needs a hardware WebGPU adapter with shader-f16.'
    );
  for (const [name, minimum] of Object.entries(STRANDS_GEMMA4_E2B_LIMITS))
    if (
      !Number.isSafeInteger((adapter as Gemma4Adapter).limits?.[name]) ||
      ((adapter as Gemma4Adapter).limits![name] as number) < minimum
    )
      throw new WebDecisionError(
        'UNSUPPORTED_ENVIRONMENT',
        `Strands Gemma 4 needs WebGPU ${name} of at least ${minimum}.`
      );
}

async function requestAdapter(gpuApi: unknown, signal: AbortSignal | null | undefined): Promise<Gemma4Adapter> {
  if (typeof (gpuApi as Gemma4GpuApi | null | undefined)?.requestAdapter !== 'function')
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'
    );
  if (typeof globalThis.crypto?.subtle?.digest !== 'function')
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Web Crypto is unavailable. Use HTTPS or localhost.');
  let adapter;
  try {
    adapter = await waitForAbort(
      () => (gpuApi as Gemma4GpuApi).requestAdapter({powerPreference: 'high-performance'}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not request a WebGPU adapter.', {cause});
  }
  signal?.throwIfAborted();
  checkAdapter(adapter);
  return adapter;
}

async function syncAfterScopeFailure(
  started: boolean,
  lifecycle: Gemma4Lifecycle,
  device: Gemma4Device,
  disposalGuard: Gemma4DisposalGuard
): Promise<void> {
  if (!started || lifecycle.loss || lifecycle.disposed) return;
  try {
    await device.sync();
  } catch {
    disposalGuard.retain();
  }
}

async function closeGpuErrorScopes(
  gpu: Gemma4GpuDevice,
  count: number,
  failure: unknown,
  noFailure: symbol
): Promise<unknown> {
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

async function gpuScope<T>(
  gpu: Gemma4GpuDevice,
  device: Gemma4Device,
  lifecycle: Gemma4Lifecycle,
  disposalGuard: Gemma4DisposalGuard,
  operation: () => Promise<T>
): Promise<T> {
  const noFailure = Symbol('no failure');
  let count = 0,
    failure: unknown = noFailure,
    value: T | undefined,
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
  return value as T;
}

async function verifyAndReadSource(
  base: URL,
  signal: AbortSignal | null | undefined,
  onProgress: ((event: Gemma4ProgressEvent) => void) | undefined
): Promise<Gemma4VerifiedSource> {
  const rootBytes = await fetchRootManifest(base, signal);
  const manifest = parseJson(rootBytes, 'runtime manifest') as Gemma4Record;
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
    tokenizer: files.get('tokenizer') as Gemma4RuntimeAsset,
    tokenizerConfig: files.get('tokenizerConfig') as Gemma4RuntimeAsset,
    cacheManifest: files.get('cacheManifest') as Gemma4RuntimeAsset
  };
  const fileBytes = Object.values(rootById).reduce((total, file) => total + file.bytes, rootBytes.byteLength);
  const assets: Record<string, unknown> = {};
  for (const [id, file] of Object.entries(rootById)) {
    assets[id] = await fetchPinnedFile(base, file, signal);
  }
  const cacheManifest = parseJson(assets.cacheManifest as Uint8Array, 'tensor-cache manifest') as Gemma4Record;
  if (
    (cacheManifest.files as {readonly length?: unknown} | null | undefined)?.length !== 84 ||
    cacheManifest.sourceWeightDtype !== 'bfloat16' ||
    cacheManifest.computeWeightDtype !== 'float16' ||
    cacheManifest.weightConversion !== 'bfloat16-to-float16-lossy' ||
    cacheManifest.derivedBufferConversion !== 'float32-to-float16-lossy'
  )
    throw invalid('Unexpected Gemma 4 tensor-cache manifest.');
  const cacheFiles = new Map<string, Gemma4PinnedFile>(
    (cacheManifest.files as Gemma4PinnedFile[]).map((file: Gemma4PinnedFile) => [file.path, file])
  );
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
  const inputSpecs = parseJson(assets.inputSpecs as Uint8Array, 'graph input specs');
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
  let plan: Gemma4ShardPlan;
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

function retainLateDevice(device: unknown): void {
  try {
    (device as {destroy(): void} | null | undefined)?.destroy();
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
  }: Gemma4LoadOptions = {},
  // Internal dependency seam for tests; do not expose these controls as load options.
  {readFile = fetchPinnedFile, preparedAdapter, verifySource = verifyAndReadSource}: Gemma4LoadDependencies = {}
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
  const source = await verifySource(base, signal, onProgress as ((event: Gemma4ProgressEvent) => void) | undefined);
  let gpu: Gemma4GpuDevice | undefined,
    runtime: Gemma4Runtime | undefined,
    lifecycle: Gemma4Lifecycle | undefined,
    disposalGuard: Gemma4DisposalGuard | undefined,
    loadWatch: Gemma4LoadWatch | undefined,
    scopeOpen = false,
    loading = true,
    forward: (() => void) | undefined;
  const uncaptured: string[] = [],
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
    forward = () => loadAbort.abort(loadWatch!.signal.reason);
    loadWatch.signal.addEventListener('abort', forward, {once: true});
    if (loadWatch.signal.aborted) forward();
    gpu.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10)
        uncaptured.push(
          (event as {readonly error?: {readonly message?: string} | undefined}).error?.message ||
            String((event as {readonly error?: unknown}).error)
        );
      if (loading && !loadAbort.signal.aborted)
        loadAbort.abort(new WebDecisionError('MODEL_LOAD_FAILED', 'WebGPU reported an uncaptured loading error.'));
    });
    runtime = (await waitForAbort(
      () => (createRuntime as (wasm: unknown) => unknown)(source.assets.graphWasm),
      loadAbort.signal,
      (lateRuntime) => {
        try {
          (lateRuntime as {dispose(): unknown} | null | undefined)?.dispose();
        } catch {}
      }
    )) as Gemma4Runtime;
    lifecycle = trackRuntime(runtime, gpu);
    disposalGuard = guardStrandsDisposal(runtime, gpu, lifecycle);
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scopeOpen = true;
    const device = runtime.webgpu();
    const check = () => {
      loadAbort.signal.throwIfAborted();
      assertLoadActive(lifecycle as Gemma4Lifecycle);
      if (uncaptured.length) throw new WebDecisionError('MODEL_LOAD_FAILED', uncaptured[0] as string);
    };
    const context = runtime.lib?.webGPUContext as Gemma4UploadContext;
    if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof context?.gpuBufferFromPtr !== 'function')
      throw new Error('The pinned TVM WebGPU upload API is unavailable.');
    // oxlint-disable-next-line unicorn/no-new-array -- Preserve sparse graph-weight slots until shard uploads write their entries.
    const weights: unknown[] = new Array(source.plan.expectedParameters.length);
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
          allocate({shape, dtype}: {readonly shape: number[]; readonly dtype: string}) {
            check();
            const tensor = runtime!.empty(shape, dtype, device);
            if (
              typeof tensor?.getDataPtr !== 'function' ||
              !Number.isSafeInteger(tensor.byteOffset) ||
              tensor.byteOffset < 0 ||
              tensor.byteOffset % 4
            )
              throw new Error('Invalid Gemma 4 TVM tensor allocation.');
            const buffer = context.gpuBufferFromPtr(tensor.getDataPtr());
            if (!Number.isSafeInteger(buffer?.size) || buffer!.size < tensor.byteOffset)
              throw new Error('Gemma 4 TVM tensor allocation has no verified capacity.');
            return {tensor, capacityBytes: buffer!.size - tensor.byteOffset};
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
    const tokenizerData = parseJson(source.assets.tokenizer as Uint8Array, 'tokenizer');
    const tokenizerConfig = parseJson(source.assets.tokenizerConfig as Uint8Array, 'tokenizer configuration');
    const tokenizer = (createTokenizer as (data: unknown, config: unknown) => {readonly encode: unknown})(
      tokenizerData,
      tokenizerConfig
    );
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
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    if ((cause as {readonly name?: unknown} | null | undefined)?.name === 'GPUOutOfMemoryError')
      throw new WebDecisionError(
        'DEVICE_RESOURCE_LIMIT',
        'This WebGPU device ran out of resources while loading Gemma 4.',
        {cause}
      );
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load experimental Strands Gemma 4.', {cause});
  }
}

/** Source path for app-hosted, pinned model assets. */
export async function loadExperimentalStrandsGemma4FromSource(options?: Gemma4LoadOptions) {
  return loadStrandsGemma4FromSource(options);
}
