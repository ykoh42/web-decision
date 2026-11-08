import {Tokenizer} from '@huggingface/tokenizers';
import {createPolyfillWASI, instantiate} from './.toolchains/runtime.js';
import type {DLDevice, Instance, PackedFunc, Tensor} from './.toolchains/runtime.js';
import {detectKevGPUDevice} from './kev-gpu.js';
import {
  commitCachedManifest as commitManifest,
  parseJson as parseAssetJson,
  waitForAbort as waitForAbortOperation
} from './assets.js';
import {KEV_DELIMITERS, WebDecisionError} from './kev.js';
import {DecisionModel as DecisionModelImplementation} from './model.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime} from './runtime-lifecycle.js';
import {watchLoadDeviceLoss} from './load-device-loss.js';
import {REFLEXES_ID, readReflexesManifestResponse, uploadReflexesHeads} from './kev-reflexes.js';
import {readAssetBatch} from './asset-batch.js';
import {prepareModelSource} from './model-source.js';
import {createModelAssetReader, uploadModelShards} from './model-assets.js';
import {readKev06ManifestResponse} from './kev06/contract.js';
import {
  createKev06StableReconstructedSource,
  createKevReconstructedSource,
  EXPERIMENTAL_KEV_MODEL,
  KEV06_STABLE_MODEL,
  readExperimentalKevManifest
} from './kev06/upstream-source.mjs';
import type {DecisionModel, ModelDescription, ModelProgress} from './curated/types.js';
import type {ModelId} from './catalog/types.generated.js';

const KEV_POSITION_EMBEDDINGS = 32_768;

type KevLoaderOptions = {
  readonly assetsUrl?: string;
  readonly model?: string;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal | undefined;
  readonly onProgress?: ((progress: ModelProgress) => void) | undefined;
  readonly maxPrefetchBytes?: number;
  readonly maxInputCharacters?: number;
  readonly maxInputTokens?: number;
};

type KevLoaderDescription = {
  readonly model: string;
  readonly features: {
    readonly textChoice: {
      readonly limits: readonly {
        readonly resource: string;
        readonly scope: string;
        readonly max?: number;
      }[];
    };
  };
};

type KevCuratedDescription = ModelDescription & {readonly model: ModelId};
type LoadedKevModel = Omit<DecisionModel, 'description'>;

type KevAssetMetadata = {readonly bytes: number; readonly sha256: string};
type KevAssetManifest = {
  readonly files: Readonly<Record<string, KevAssetMetadata>>;
  readonly parameterCount: number;
  readonly maxInputTokens?: number;
  readonly maxStateTokens: number;
  readonly maxBranchTokens: number;
};
type KevTensorRecord = {
  readonly name: string;
  readonly dtype: string;
  readonly format: string;
  readonly shape: number[];
  readonly byteOffset: number;
  readonly nbytes: number;
};
type KevTensorShard = {
  readonly dataPath: string;
  readonly format: string;
  readonly nbytes: number;
  readonly records: readonly KevTensorRecord[];
};
type KevTensorCacheManifest = {
  readonly metadata: {readonly numParams: number};
  readonly records: readonly KevTensorShard[];
};
type KevModelConfig = {
  readonly max_position_embeddings?: unknown;
  readonly kev?: {
    readonly delimiter_ids?: Readonly<Partial<Record<keyof typeof KEV_DELIMITERS, unknown>>>;
  };
};

type AssetFetch = (url: string, init?: {readonly signal?: AbortSignal | undefined}) => Promise<Response>;
type ModelSourcePreparation = {
  readonly offlineMissingMessage: string;
  readonly validateModelOptions: ((options: KevLoaderOptions) => unknown) | undefined;
};
type PreparedModelSource = {
  readonly base: URL;
  readonly cache: Cache | null;
  readonly signal?: AbortSignal | undefined;
  readonly manifestUrl: string;
  readonly response: Response;
  readonly maxPrefetchBytes: number;
};
type ModelSourceProvider = {
  readonly prepare: (options: KevLoaderOptions, config: ModelSourcePreparation) => Promise<PreparedModelSource>;
  readonly fetch?: AssetFetch;
  readonly setProgress?: (callback: KevLoaderOptions['onProgress']) => void;
  readonly bundledProgressPhase?: ModelProgress['phase'];
  readonly assetProgressPhase?: ModelProgress['phase'];
  readonly dispose?: () => Promise<void> | void;
};
type ModelAssetReader = {
  readonly read: (name: string, signal?: AbortSignal | undefined) => Promise<ArrayBuffer>;
  readonly isCacheComplete: () => boolean;
  readonly setProgressPhase: (phase: ModelProgress['phase']) => void;
};
type CreateAssetReader = (config: {
  readonly base: URL;
  readonly manifest: KevAssetManifest;
  readonly cache: Cache | null;
  readonly signal: AbortSignal | undefined;
  readonly options: KevLoaderOptions;
  readonly fetchImpl: AssetFetch | undefined;
  readonly progressPhase: ModelProgress['phase'];
}) => ModelAssetReader;
type InitialAssets = [ArrayBuffer, ArrayBuffer, ArrayBuffer, ArrayBuffer, ArrayBuffer];
type InitialAssetNames = readonly [string, string, string, string, string];
type ReadInitialAssets = (
  names: InitialAssetNames,
  read: ModelAssetReader['read'],
  signal: AbortSignal | undefined
) => Promise<InitialAssets>;
type UploadModelShards = (config: {
  readonly records: readonly KevTensorShard[];
  readonly read: ModelAssetReader['read'];
  readonly runtime: Instance;
  readonly device: DLDevice;
  readonly weights: Tensor[];
  readonly signal: AbortSignal | undefined;
  readonly maxPrefetchBytes: number;
  readonly uncaptured: string[];
}) => Promise<void>;
type UploadReflexesHeads = (config: {
  readonly read: ModelAssetReader['read'];
  readonly runtime: Instance;
  readonly device: DLDevice;
  readonly weights: Tensor[];
  readonly signal: AbortSignal | undefined;
  readonly uncaptured: string[];
}) => Promise<void>;
type UploadReflexes = UploadReflexesHeads;
type ReadManifest = (response: Response, signal?: AbortSignal | undefined) => Promise<KevAssetManifest>;
type RuntimeLifecycle = {readonly lossSignal: AbortSignal; readonly disposed: boolean; readonly loss: unknown};
type LoadDeviceWatch = {readonly signal: AbortSignal; close(): void};
type TrackRuntime = (runtime: Instance, device: GPUDevice) => RuntimeLifecycle;
type AssertLoadActive = (lifecycle: RuntimeLifecycle) => void;
type CleanupFailedLoad = (config: {
  readonly loadWatch: LoadDeviceWatch | undefined;
  readonly runtime: Instance | undefined;
  readonly scopeOpen: boolean;
  readonly lifecycle: RuntimeLifecycle | undefined;
  readonly gpuDevice: GPUDevice | undefined;
}) => void;
type WatchLoadDeviceLoss = (device: GPUDevice, callerSignal?: AbortSignal | undefined) => LoadDeviceWatch;
type KevGpuDevice = {
  readonly adapter: Pick<GPUAdapter, 'features'>;
  readonly device?: GPUDevice | undefined;
};
type DetectKevGpuDevice = (
  gpu?: GPU | undefined,
  signal?: AbortSignal | undefined
) => Promise<KevGpuDevice | undefined>;
type DecisionModelConstructor = new (
  runtime: Instance,
  device: DLDevice,
  gpu: GPUDevice,
  fn: PackedFunc,
  weights: Tensor[],
  tokenizer: Tokenizer,
  limits: {
    readonly maxInputTokens: number | undefined;
    readonly maxStateTokens: number;
    readonly maxBranchTokens: number;
    readonly maxQuestions: number | undefined;
    readonly maxOptions: number | undefined;
    readonly maxInputCharacters: number | undefined;
  },
  uncaptured: string[],
  lifecycle: RuntimeLifecycle,
  supportsSystemOne: boolean,
  modelId: string
) => LoadedKevModel;
type DecisionModelArguments = ConstructorParameters<DecisionModelConstructor>;
type WaitForAbort = <Value>(
  operation: () => Value | Promise<Value>,
  signal?: AbortSignal | undefined,
  onLateValue?: ((value: Value) => void) | undefined
) => Promise<Value>;
type CommitManifest = (
  cache: Cache,
  url: string,
  manifest: KevAssetManifest,
  signal?: AbortSignal | undefined
) => Promise<void>;
type ParseAssetJson = (bytes: ArrayBuffer, name: string) => unknown;

const detectGPUDevice: DetectKevGpuDevice = detectKevGPUDevice;
const createDecisionModel = (...args: DecisionModelArguments): LoadedKevModel =>
  new DecisionModelImplementation(...args) as LoadedKevModel;
const createAssetReader: CreateAssetReader = createModelAssetReader;
const readInitialAssets: ReadInitialAssets = async (names, read, signal) =>
  (await readAssetBatch(names, read, signal)) as InitialAssets;
const prepareExternalSource: ModelSourceProvider['prepare'] = prepareModelSource;
const createStableSource = (): ModelSourceProvider => createKev06StableReconstructedSource() as ModelSourceProvider;
const createExperimentalSource = (): ModelSourceProvider => createKevReconstructedSource() as ModelSourceProvider;
const waitForAbort: WaitForAbort = waitForAbortOperation;
const commitCachedManifest: CommitManifest = commitManifest;
const parseJson: ParseAssetJson = parseAssetJson;
const trackRuntimeForLoad: TrackRuntime = trackRuntime;
const assertLoadIsActive: AssertLoadActive = assertLoadActive;
const cleanupFailedKevLoad: CleanupFailedLoad = cleanupFailedLoad;
const watchKevDeviceLoss: WatchLoadDeviceLoss = watchLoadDeviceLoss;
const readStableManifest: ReadManifest = readKev06ManifestResponse;
const readReflexesManifest: ReadManifest = readReflexesManifestResponse;
const readExperimentalManifest: ReadManifest = readExperimentalKevManifest;
const uploadShards: UploadModelShards = uploadModelShards;
const uploadReflexes: UploadReflexes = uploadReflexesHeads;

function unsupported(message: string): WebDecisionError {
  return new WebDecisionError('UNSUPPORTED_ENVIRONMENT', message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isUnknownArray(value: unknown): value is unknown[] {
  return Array.isArray(value);
}

function isPositiveSafeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1;
}

function isAbortError(value: unknown): value is {readonly name: 'AbortError'} {
  return (
    (typeof value === 'object' || typeof value === 'function') &&
    value !== null &&
    'name' in value &&
    value.name === 'AbortError'
  );
}

function cacheRecordIndex(record: unknown): number {
  const name = isRecord(record) ? record.name : undefined;
  const match = typeof name === 'string' ? /^param_(\d+)$/.exec(name) : null;
  return match ? Number(match[1]) : -1;
}

function cacheRecordElementBytes(record: unknown): number {
  const dtype = isRecord(record) ? record.dtype : undefined;
  return dtype === 'uint32' ? 4 : dtype === 'float16' ? 2 : 0;
}

function validCacheShard(
  shard: unknown,
  files: Readonly<Record<string, KevAssetMetadata>>
): shard is Omit<KevTensorShard, 'records'> & {readonly records: unknown[]} {
  if (
    !isRecord(shard) ||
    typeof shard.dataPath !== 'string' ||
    shard.dataPath.includes('/') ||
    shard.dataPath.includes('\\')
  )
    return false;
  const file = files[`weights/${shard.dataPath}`];
  if (!file) return false;
  return (
    shard.format === 'raw-shard' &&
    shard.nbytes === file.bytes &&
    Number.isSafeInteger(shard.nbytes) &&
    isUnknownArray(shard.records)
  );
}

function validCacheRecordShape(record: Record<string, unknown>, bytesPerElement: number): boolean {
  const shape: unknown = record.shape;
  if (bytesPerElement < 1 || !isUnknownArray(shape) || !shape.every(isPositiveSafeInteger)) return false;
  return shape.reduce((a, b) => a * b, bytesPerElement) === record.nbytes;
}

function validCacheRecordRange(record: Record<string, unknown>, shard: {readonly nbytes: number}): boolean {
  return (
    typeof record.byteOffset === 'number' &&
    Number.isSafeInteger(record.byteOffset) &&
    record.byteOffset >= 0 &&
    record.byteOffset % 4 === 0 &&
    typeof record.nbytes === 'number' &&
    Number.isSafeInteger(record.nbytes) &&
    record.nbytes % 4 === 0 &&
    record.byteOffset + record.nbytes <= shard.nbytes
  );
}

function validCacheRecord(
  record: unknown,
  shard: {readonly nbytes: number},
  seen: ReadonlySet<number>
): record is KevTensorRecord {
  if (!isRecord(record)) return false;
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

function checkCacheManifest(manifest: KevAssetManifest, value: unknown): asserts value is KevTensorCacheManifest {
  if (
    !isRecord(value) ||
    !isRecord(value.metadata) ||
    value.metadata.numParams !== 394 ||
    !isUnknownArray(value.records)
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid TVM tensor cache metadata.');
  const seen = new Set<number>();
  for (const shard of value.records) {
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

function validateOptionalLimit(value: number | undefined, name: string): void {
  if (value !== undefined && (!Number.isSafeInteger(value) || value < 0))
    throw new WebDecisionError('INVALID_INPUT', `${name} must be a non-negative safe integer.`);
}

async function loadKevModel(
  options: KevLoaderOptions,
  description: KevLoaderDescription,
  readManifest: ReadManifest,
  uploadHeads?: UploadReflexes | undefined,
  sourceProvider?: ModelSourceProvider | undefined
): Promise<LoadedKevModel> {
  validateOptionalLimit(options?.maxInputCharacters, 'maxInputCharacters');
  validateOptionalLimit(options?.maxInputTokens, 'maxInputTokens');
  const prepareSource = sourceProvider?.prepare ?? prepareExternalSource;
  const {
    base,
    cache,
    signal,
    manifestUrl,
    response: manifestResponse,
    maxPrefetchBytes
  } = await prepareSource(options, {
    offlineMissingMessage: 'The model manifest is not available offline. Load the model online first.',
    validateModelOptions: undefined
  });
  const manifest = await readManifest(manifestResponse, signal);
  sourceProvider?.setProgress?.(options?.onProgress);
  const assetReader = createAssetReader({
    base,
    manifest,
    cache,
    signal,
    options,
    fetchImpl: sourceProvider?.fetch,
    progressPhase: sourceProvider?.bundledProgressPhase
  });
  const {read} = assetReader;
  let runtime: Instance | undefined,
    gpuDevice: GPUDevice | undefined,
    lifecycle: RuntimeLifecycle | undefined,
    loadWatch: LoadDeviceWatch | undefined,
    scopeOpen = false;
  try {
    const [tokenizerBytes, tokenizerConfigBytes, modelConfigBytes, wasmBytes, tensorCacheBytes] =
      await readInitialAssets(
        ['tokenizer.json', 'tokenizer_config.json', 'model_config.json', 'model.wasm', 'weights/tensor-cache.json'],
        read,
        signal
      );
    const modelConfig = parseJson(modelConfigBytes, 'model_config.json') as KevModelConfig;
    if (modelConfig.max_position_embeddings !== KEV_POSITION_EMBEDDINGS)
      throw new WebDecisionError('INVALID_ASSET', 'Unexpected Kev position embedding metadata.');
    for (const name of Object.keys(KEV_DELIMITERS) as (keyof typeof KEV_DELIMITERS)[]) {
      const id = KEV_DELIMITERS[name];
      if (modelConfig.kev?.delimiter_ids?.[name] !== id)
        throw new WebDecisionError('INVALID_ASSET', 'Unexpected model delimiter IDs.');
    }
    const tokenizer = new Tokenizer(
      parseJson(tokenizerBytes, 'tokenizer.json') as object,
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json') as object
    );
    const cacheManifest = parseJson(tensorCacheBytes, 'tensor-cache.json');
    checkCacheManifest(manifest, cacheManifest);
    signal?.throwIfAborted();
    if (sourceProvider?.assetProgressPhase) assetReader.setProgressPhase(sourceProvider.assetProgressPhase);
    let gpu;
    try {
      gpu = await detectGPUDevice(undefined, signal);
    } catch (cause) {
      signal?.throwIfAborted();
      throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not initialize a compatible WebGPU device.', {cause});
    }
    gpuDevice = gpu?.device;
    // The device helper disposes a device returned after cancellation.
    signal?.throwIfAborted();
    if (!gpu || !gpu.adapter.features.has('shader-f16')) {
      gpuDevice?.destroy();
      gpuDevice = undefined;
      throw unsupported('This model requires WebGPU shader-f16 support.');
    }
    const activeGpuDevice = gpuDevice as GPUDevice;
    const uncaptured: string[] = [];
    activeGpuDevice.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
    });
    loadWatch = watchKevDeviceLoss(activeGpuDevice, signal);
    runtime = await waitForAbort(
      () => instantiate(wasmBytes, createPolyfillWASI()),
      loadWatch.signal,
      (lateRuntime) => {
        try {
          lateRuntime?.dispose();
        } catch {}
      }
    );
    lifecycle = trackRuntimeForLoad(runtime, activeGpuDevice);
    loadWatch.signal.throwIfAborted();
    await Promise.resolve();
    assertLoadIsActive(lifecycle);
    runtime.initWebGPU(activeGpuDevice);
    const device = runtime.webgpu();
    runtime.beginScope();
    scopeOpen = true;
    const vm = runtime.createVirtualMachine(device),
      fn = vm.getFunction('main'),
      weights = Array<Tensor>(manifest.parameterCount);
    await uploadShards({
      records: cacheManifest.records,
      read,
      runtime,
      device,
      weights,
      signal: loadWatch.signal,
      maxPrefetchBytes,
      uncaptured
    });
    if (uploadHeads) await uploadHeads({read, runtime, device, weights, signal: loadWatch.signal, uncaptured});
    loadWatch.signal.throwIfAborted();
    signal?.throwIfAborted();
    assertLoadActive(lifecycle);
    // Save a manifest only after a complete online load. Offline mode never
    // makes network requests and still verifies every cached model file.
    if (cache && assetReader.isCacheComplete() && !options.offline) {
      try {
        await commitCachedManifest(cache, manifestUrl, manifest, loadWatch.signal);
      } catch {
        loadWatch.signal.throwIfAborted();
      }
      signal?.throwIfAborted();
      assertLoadIsActive(lifecycle);
    }
    loadWatch.close();
    return createDecisionModel(
      runtime,
      device,
      activeGpuDevice,
      fn,
      weights,
      tokenizer,
      {
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
      },
      uncaptured,
      lifecycle,
      true,
      description.model
    );
  } catch (cause) {
    // Cleanup must not hide the original loading error, including when WebGPU
    // initialization failed before TVM took ownership of the device.
    cleanupFailedKevLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice});
    // AbortSignal can carry any caller-supplied reason, not just AbortError.
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || isAbortError(cause)) throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load the decision model.', {cause});
  } finally {
    await sourceProvider?.dispose?.();
  }
}

export function loadKev06DecisionModel(
  options: KevLoaderOptions,
  description: KevLoaderDescription
): Promise<LoadedKevModel> {
  return loadKevModel(options, description, readStableManifest);
}

/** Load the curated stable artifact using its pinned upstream source. */
export function loadCuratedKevModel(options: object, description: KevCuratedDescription): Promise<LoadedKevModel> {
  return loadKevModel(options as KevLoaderOptions, description, readStableManifest, undefined, createStableSource());
}

export async function loadKevDecisionModel(
  options: KevLoaderOptions,
  description: KevLoaderDescription,
  stableModelId?: string | undefined
): Promise<LoadedKevModel> {
  if (description.model === EXPERIMENTAL_KEV_MODEL) {
    return loadKevModel(options, description, readExperimentalManifest, undefined, createExperimentalSource());
  }
  if (stableModelId === KEV06_STABLE_MODEL) {
    return loadKevModel(options, description, readStableManifest, undefined, createStableSource());
  }
  if (description.model !== REFLEXES_ID) return loadKev06DecisionModel(options, description);
  return loadKevModel(options, description, readReflexesManifest, uploadReflexes);
}
