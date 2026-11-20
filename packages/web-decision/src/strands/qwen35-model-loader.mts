import {destroyGpuDevice, requestGpuAdapter, requestGpuDevice} from '../abortable-gpu-device.js';
import {waitForAbort} from '../abort.js';
import type {AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {Qwen35GraphInputOptions} from './qwen35-graph-input.mts';
import type {Qwen35PinnedSource} from './qwen35-source.mts';
import type {Qwen35PinnedTokenizer} from './qwen35-tokenizer.mts';
import type {Qwen35VisionArtifacts} from './qwen35-vision-contract.mts';
import type {Qwen35VisionRuntimeRunner, Qwen35VisionTvmJs} from './qwen35-vision-runtime.mts';
import type {Qwen35R19Artifacts} from './qwen35-r19-artifacts.mts';
import type {Qwen35R19RuntimeHandle, Qwen35R19TvmJs, Qwen35R19GraphRunner} from './qwen35-r19-runtime.mts';
import type {Qwen35SourceEmbeddingLookup} from './qwen35-source-embedding-lookup.mts';
import type {Qwen35SessionOptions} from './qwen35-session.mts';
import type {Qwen35PreEmbedSourceResolver, Qwen35PreEmbedSourceResolverOptions} from './qwen35-pre-embed-sources.mts';
import {createQwen35Session} from './qwen35-session.mts';
import {createQwen35R19GraphRunner, instantiatePinnedQwen35R19Runtime} from './qwen35-r19-runtime.mts';
import {createQwen35PreEmbedSourceResolver} from './qwen35-pre-embed-sources.mts';
import {createQwen35VisionRuntimeRunner} from './qwen35-vision-runtime.mts';
import {qwen35VisionActivationPrecision} from './qwen35-vision-runner.mts';
import {openPinnedQwen35Source} from './qwen35-source.mts';
import {createQwen35SourceTokenEmbeddingLookup} from './qwen35-source-embedding-lookup.mts';

const MODEL_NAME = 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610';
const QWEN35_WORKGROUP_STORAGE_BYTES = 8 * 1024;
// The packaged TVM graphs bind their entire preallocated workspace buffers.
const QWEN35_WORKSPACE_BYTES = {text: 288 * 1024 ** 2, vision: 1024 ** 3} as const;
const COMPUTE_INVOCATION_CANDIDATES = [1024, 512, 256] as const;

export interface Qwen35LoadProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
  readonly phase: 'model-assets' | 'vision-weights' | 'upstream-source';
}

export interface Qwen35ModelLoadOptions {
  readonly cache?: AssetCache | null;
  readonly offline?: boolean;
  readonly signal?: AbortSignal | null;
  readonly gpuApi?: Pick<GPU, 'requestAdapter'>;
  readonly graphOptions?: Qwen35GraphInputOptions;
  readonly onProgress?: (progress: Qwen35LoadProgress) => void;
}

interface Qwen35GpuDevicePair {
  readonly languageModel: GPUDevice;
  readonly vision: GPUDevice;
}

interface Qwen35ModelSession {
  systemOne: ReturnType<typeof createQwen35Session>['systemOne'];
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

interface Qwen35LoadDependencies {
  readonly openSource: (options: {
    cache?: AssetCache | null | undefined;
    offline: boolean;
    signal?: AbortSignal | null | undefined;
  }) => Promise<Qwen35PinnedSource>;
  readonly openTokenizer: (options: {
    cache?: AssetCache | null | undefined;
    offline: boolean;
    signal?: AbortSignal | null | undefined;
  }) => Promise<Qwen35PinnedTokenizer>;
  readonly openR19Artifacts: (options: {
    cache?: AssetCache | null | undefined;
    offline: boolean;
    signal?: AbortSignal | null | undefined;
  }) => Promise<Qwen35R19Artifacts>;
  readonly openVisionArtifacts: (options: {
    cache?: AssetCache | null | undefined;
    offline: boolean;
    signal?: AbortSignal | null | undefined;
  }) => Promise<Qwen35VisionArtifacts>;
  readonly createEmbeddingLookup: (
    source: Qwen35PinnedSource,
    options: {signal?: AbortSignal | null | undefined}
  ) => Promise<Qwen35SourceEmbeddingLookup>;
  readonly requestDevices: (
    gpuApi: Pick<GPU, 'requestAdapter'> | undefined,
    signal?: AbortSignal | null
  ) => Promise<Qwen35GpuDevicePair>;
  readonly createVisionRunner: (options: {
    tvmjs: Qwen35VisionTvmJs;
    gpuDevice: GPUDevice;
    source: Qwen35PinnedSource;
    artifacts: Qwen35VisionArtifacts;
    signal?: AbortSignal | null | undefined;
    onProgress?: (bytes: number) => void;
  }) => Promise<Qwen35VisionRuntimeRunner>;
  readonly instantiateR19: (options: {
    tvmjs: Qwen35R19TvmJs;
    gpuDevice: GPUDevice;
    artifacts: Qwen35R19Artifacts;
    signal?: AbortSignal | null | undefined;
  }) => Promise<Qwen35R19RuntimeHandle>;
  readonly createR19Runner: (options: {
    handle: Qwen35R19RuntimeHandle;
    source: Qwen35PinnedSource;
    artifacts: Qwen35R19Artifacts;
    resolveInputSources: NonNullable<Parameters<typeof createQwen35R19GraphRunner>[0]>['resolveInputSources'];
    signal?: AbortSignal | null | undefined;
    onProgress?: (bytes: number) => void;
  }) => Promise<Qwen35R19GraphRunner>;
  readonly createSession: (options: Qwen35SessionOptions) => ReturnType<typeof createQwen35Session>;
  readonly tvmjs: Qwen35R19TvmJs & Qwen35VisionTvmJs;
  readonly resolveInputSources: (options: Qwen35PreEmbedSourceResolverOptions) => Qwen35PreEmbedSourceResolver;
}

function gpuDeviceDescriptor(adapter: GPUAdapter, graph: keyof typeof QWEN35_WORKSPACE_BYTES): GPUDeviceDescriptor {
  const limits = adapter.limits;
  if (limits.maxComputeWorkgroupStorageSize < QWEN35_WORKGROUP_STORAGE_BYTES)
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Qwen3.5 requires ${QWEN35_WORKGROUP_STORAGE_BYTES} bytes of workgroup storage. This adapter provides ${limits.maxComputeWorkgroupStorageSize}.`
    );
  const workspaceBytes = QWEN35_WORKSPACE_BYTES[graph];
  if (limits.maxBufferSize < workspaceBytes || limits.maxStorageBufferBindingSize < workspaceBytes)
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Qwen3.5 ${graph} requires maxBufferSize and maxStorageBufferBindingSize of at least ${workspaceBytes} bytes. This adapter provides ${limits.maxBufferSize} and ${limits.maxStorageBufferBindingSize}.`
    );

  const maxComputeInvocationsPerWorkgroup = COMPUTE_INVOCATION_CANDIDATES.find(
    (candidate) => candidate <= limits.maxComputeInvocationsPerWorkgroup
  );
  const requiredLimits: GPUDeviceDescriptor['requiredLimits'] = {
    maxBufferSize: limits.maxBufferSize,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: QWEN35_WORKGROUP_STORAGE_BYTES,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    ...(maxComputeInvocationsPerWorkgroup === undefined ? {} : {maxComputeInvocationsPerWorkgroup})
  };
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('shader-f16')) requiredFeatures.push('shader-f16');
  if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');
  return {requiredFeatures, requiredLimits};
}

/** Request one adapter per device: WebGPU adapters are single-use, and TVM instances destroy their devices. */
export async function requestQwen35GpuDevicePair(
  gpuApi: Pick<GPU, 'requestAdapter'> | undefined = typeof navigator === 'undefined' ? undefined : navigator.gpu,
  signal?: AbortSignal | null
): Promise<Qwen35GpuDevicePair> {
  if (!gpuApi) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Qwen3.5 requires WebGPU.');
  const languageAdapter = await requestGpuAdapter(gpuApi, signal ?? undefined);
  signal?.throwIfAborted();
  if (!languageAdapter) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'No WebGPU adapter is available.');
  const languageModel = await requestGpuDevice(
    languageAdapter,
    gpuDeviceDescriptor(languageAdapter, 'text'),
    signal ?? undefined
  );
  try {
    const visionAdapter = await requestGpuAdapter(gpuApi, signal ?? undefined);
    signal?.throwIfAborted();
    if (!visionAdapter) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'No second WebGPU adapter is available.');
    const vision = await requestGpuDevice(
      visionAdapter,
      gpuDeviceDescriptor(visionAdapter, 'vision'),
      signal ?? undefined
    );
    return Object.freeze({languageModel, vision});
  } catch (cause) {
    destroyGpuDevice(languageModel);
    throw cause;
  }
}

function weightByteCount(
  bindings: readonly {readonly shape: readonly number[]; readonly relaxShape?: readonly number[]}[]
): number {
  let bytes = 0;
  for (const binding of bindings) {
    const shape = binding.relaxShape ?? binding.shape;
    const elements = shape.reduce((product, dimension) => product * dimension, 1);
    if (!Number.isSafeInteger(elements) || elements < 0 || elements > (Number.MAX_SAFE_INTEGER - bytes) / 4)
      throw new WebDecisionError('INVALID_ASSET', 'Qwen3.5 weight size exceeds the safe integer range.');
    bytes += elements * 4;
  }
  return bytes;
}

function isAbortFailure(cause: unknown): boolean {
  return typeof cause === 'object' && cause !== null && 'name' in cause && cause.name === 'AbortError';
}

function preserveOnlineLoadFailure(cause: unknown, signal?: AbortSignal | null): void {
  signal?.throwIfAborted();
  if (cause instanceof WebDecisionError && cause.code === 'INVALID_ASSET') throw cause;
  if (isAbortFailure(cause)) throw cause;
}

function createProductionDependencies(): Promise<Qwen35LoadDependencies> {
  return Promise.all([
    import('../runtime/tvm.generated.mjs'),
    import('./qwen35-tokenizer.mts'),
    import('./qwen35-r19-artifacts.mts'),
    import('./qwen35-vision-artifacts.mts')
  ]).then(([runtime, tokenizer, r19Assets, visionAssets]) => {
    const tvmjs: Qwen35R19TvmJs & Qwen35VisionTvmJs = {
      instantiate: (wasm, imports, logger) => runtime.instantiate(wasm, imports, logger),
      createPolyfillWASI: () => runtime.createPolyfillWASI()
    };
    const openOptions = (options: {
      cache?: AssetCache | null | undefined;
      offline: boolean;
      signal?: AbortSignal | null | undefined;
    }) => ({cache: options.cache, offline: options.offline, signal: options.signal});
    return {
      openSource: openPinnedQwen35Source,
      openTokenizer: (options) => tokenizer.openPinnedQwen35Tokenizer(openOptions(options)) as never,
      openR19Artifacts: (options) => r19Assets.openPinnedQwen35R19Artifacts(openOptions(options)) as never,
      openVisionArtifacts: (options) => visionAssets.openPinnedQwen35VisionArtifacts(openOptions(options)) as never,
      createEmbeddingLookup: createQwen35SourceTokenEmbeddingLookup,
      requestDevices: requestQwen35GpuDevicePair,
      createVisionRunner: createQwen35VisionRuntimeRunner,
      instantiateR19: async ({tvmjs: runtimeBindings, gpuDevice, artifacts, signal}) =>
        waitForAbort(
          () =>
            instantiatePinnedQwen35R19Runtime({
              tvmjs: runtimeBindings,
              gpuDevice,
              fixtureBytes: artifacts.fixtureBytes,
              wasmBytes: artifacts.wasmBytes
            }),
          signal,
          (handle) => void handle.dispose().catch(() => {})
        ),
      createR19Runner: ({handle, source, artifacts, resolveInputSources, signal, onProgress}) =>
        createQwen35R19GraphRunner({
          handle,
          source,
          bindings: artifacts.bindings,
          expectedBindingHashes: {
            bindingsSha256: artifacts.manifest.abi.bindingsSha256,
            argumentNamesSha256: artifacts.manifest.abi.argumentNamesSha256
          },
          resolveInputSources,
          signal,
          onProgress
        }),
      createSession: createQwen35Session,
      tvmjs,
      resolveInputSources: createQwen35PreEmbedSourceResolver
    };
  });
}

async function disposeAfterLoadFailure(resources: {
  source?: Qwen35PinnedSource;
  tokenizer?: Qwen35PinnedTokenizer;
  embeddings?: Qwen35SourceEmbeddingLookup;
  vision?: Qwen35VisionRuntimeRunner;
  r19Runner?: Qwen35R19GraphRunner;
  r19Handle?: Qwen35R19RuntimeHandle;
  devices?: Set<GPUDevice>;
}): Promise<void> {
  await Promise.allSettled([
    Promise.resolve().then(() => resources.vision?.dispose()),
    Promise.resolve().then(() => resources.r19Runner?.dispose()),
    Promise.resolve().then(() => resources.r19Handle?.dispose()),
    Promise.resolve().then(() => resources.embeddings?.dispose()),
    Promise.resolve().then(() => resources.source?.dispose?.()),
    Promise.resolve().then(() => resources.tokenizer?.dispose())
  ]);
  for (const device of resources.devices ?? []) destroyGpuDevice(device);
}

function createOwnedRunner(
  graph: Qwen35R19GraphRunner,
  vision: Qwen35VisionRuntimeRunner,
  embeddings: Qwen35SourceEmbeddingLookup,
  source: Qwen35PinnedSource,
  tokenizer: Qwen35PinnedTokenizer
) {
  let disposal: Promise<void> | undefined;
  return Object.freeze({
    run: graph.run.bind(graph),
    releaseCachedMemory() {
      embeddings.releaseCachedMemory();
    },
    dispose() {
      if (disposal) return disposal;
      disposal = (async () => {
        let failure: unknown;
        for (const release of [
          () => vision.dispose(),
          () => graph.dispose(),
          () => embeddings.dispose(),
          () => source.dispose?.(),
          () => tokenizer.dispose()
        ]) {
          try {
            await release();
          } catch (cause) {
            if (failure === undefined) failure = cause;
          }
        }
        if (failure !== undefined) throw failure;
      })();
      return disposal;
    }
  });
}

async function openModelAssets(
  dependencies: Qwen35LoadDependencies,
  {cache, offline, signal}: Parameters<Qwen35LoadDependencies['openSource']>[0]
) {
  const opened = await Promise.allSettled([
    dependencies.openSource({cache, offline, signal}),
    dependencies.openTokenizer({cache, offline, signal}),
    dependencies.openR19Artifacts({cache, offline, signal}),
    dependencies.openVisionArtifacts({cache, offline, signal})
  ]);
  const [source, tokenizer, r19Artifacts, visionArtifacts] = opened;
  if (
    source.status === 'fulfilled' &&
    tokenizer.status === 'fulfilled' &&
    r19Artifacts.status === 'fulfilled' &&
    visionArtifacts.status === 'fulfilled'
  )
    return {
      source: source.value,
      tokenizer: tokenizer.value,
      r19Artifacts: r19Artifacts.value,
      visionArtifacts: visionArtifacts.value
    };

  const failure = opened.find((result): result is PromiseRejectedResult => result.status === 'rejected')!;
  if (source.status === 'fulfilled') source.value.dispose?.();
  if (tokenizer.status === 'fulfilled') tokenizer.value.dispose();
  throw failure.reason;
}

async function prefetchOnlineSource(
  source: Qwen35PinnedSource,
  signal: AbortSignal | null | undefined,
  onProgress: (bytes: number) => void
): Promise<void> {
  let progressFailed = false;
  let progressFailure: unknown;
  try {
    await source.prefetchRequiredRanges!({
      signal,
      onProgress(bytes) {
        try {
          onProgress(bytes);
        } catch (cause) {
          progressFailed = true;
          progressFailure = cause;
          throw cause;
        }
      }
    });
  } catch (cause) {
    if (progressFailed) throw progressFailure;
    preserveOnlineLoadFailure(cause, signal);
  }
}

/** Build the complete pinned text-and-image TVM model without a caller asset URL. */
export async function loadPinnedQwen35Model(
  {cache, offline = false, signal, gpuApi, graphOptions, onProgress}: Qwen35ModelLoadOptions = {},
  injectedDependencies?: Qwen35LoadDependencies
): Promise<Qwen35ModelSession> {
  if (typeof offline !== 'boolean') throw new WebDecisionError('INVALID_INPUT', 'offline must be a boolean.');
  signal?.throwIfAborted();
  const dependencies = injectedDependencies ?? (await createProductionDependencies());
  signal?.throwIfAborted();

  const {source, tokenizer, r19Artifacts, visionArtifacts} = await openModelAssets(dependencies, {
    cache,
    offline,
    signal
  });
  const resources: Parameters<typeof disposeAfterLoadFailure>[0] = {source, tokenizer};
  let visionRunner: Qwen35VisionRuntimeRunner | undefined;

  try {
    signal?.throwIfAborted();
    const weightBytes = weightByteCount(r19Artifacts.bindings) + weightByteCount(visionArtifacts.bindings);
    const prefetchBytes =
      cache && !offline && typeof source.prefetchRequiredRanges === 'function' ? (source.requiredRangeBytes ?? 0) : 0;
    const totalBytes = weightBytes + prefetchBytes;
    let loadedBytes = 0;
    const report = (bytes: number, phase: Qwen35LoadProgress['phase']) => {
      loadedBytes += bytes;
      onProgress?.({loadedBytes, totalBytes, fraction: Math.min(loadedBytes / totalBytes, 1), phase});
    };

    resources.embeddings = await dependencies.createEmbeddingLookup(source, {signal});
    const devices = await dependencies.requestDevices(gpuApi, signal);
    resources.devices = new Set([devices.languageModel, devices.vision]);

    const resolveInputSources = dependencies.resolveInputSources({
      embeddings: resources.embeddings,
      visionActivationPrecision: qwen35VisionActivationPrecision(visionArtifacts.artifact),
      vision: {
        run(input, runSignal) {
          if (!visionRunner) throw new WebDecisionError('MODEL_LOAD_FAILED', 'Qwen3.5 vision runtime is not ready.');
          return visionRunner.run(input, runSignal);
        }
      }
    });
    const languageDevice = devices.languageModel;
    resources.devices.delete(languageDevice);
    resources.r19Handle = await dependencies.instantiateR19({
      tvmjs: dependencies.tvmjs,
      gpuDevice: languageDevice,
      artifacts: r19Artifacts,
      signal
    });
    resources.r19Runner = await dependencies.createR19Runner({
      handle: resources.r19Handle,
      source,
      artifacts: r19Artifacts,
      resolveInputSources,
      signal,
      onProgress: (bytes) => report(bytes, 'model-assets')
    });
    delete resources.r19Handle;

    const visionDevice = devices.vision;
    resources.devices.delete(visionDevice);
    resources.vision = await dependencies.createVisionRunner({
      tvmjs: dependencies.tvmjs,
      gpuDevice: visionDevice,
      source,
      artifacts: visionArtifacts,
      signal,
      onProgress: (bytes) => report(bytes, 'vision-weights')
    });
    visionRunner = resources.vision;

    signal?.throwIfAborted();
    if (cache && !offline && typeof source.prefetchRequiredRanges === 'function')
      await prefetchOnlineSource(source, signal, (bytes) => report(bytes, 'upstream-source'));

    signal?.throwIfAborted();
    const cacheCommits = await Promise.allSettled([source.commit({signal}), tokenizer.commit({signal})]);
    signal?.throwIfAborted();
    if (cache && !offline) {
      for (const result of cacheCommits)
        if (result.status === 'rejected') preserveOnlineLoadFailure(result.reason, signal);
    }
    const model = dependencies.createSession({
      tokenizer: tokenizer.tokenizer,
      runner: createOwnedRunner(resources.r19Runner, resources.vision, resources.embeddings, source, tokenizer),
      modelName: MODEL_NAME,
      ...(graphOptions ? {graphOptions} : {})
    });
    signal?.throwIfAborted();
    source.finishLoading?.();
    delete resources.r19Runner;
    delete resources.vision;
    delete resources.embeddings;
    delete resources.source;
    delete resources.tokenizer;
    delete resources.devices;
    return model;
  } catch (cause) {
    await disposeAfterLoadFailure(resources);
    throw cause;
  }
}
