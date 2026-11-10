import {Tokenizer} from '@huggingface/tokenizers';
import {openAssetCache, sha256, waitForAbort} from '../../assets.js';
import {cancelResponse, readBoundedResponse, ResponseSizeError} from '../../bounded-response.js';
import {detectGPUDevice} from '../../kanana/tvm-webgpu.js';
import {instantiate, createPolyfillWASI} from '../../runtime/tvm.generated.mjs';
import {WebDecisionError} from '../errors.ts';
import type {KananaLoadOptions, ModelDescription, ModelProgress} from '../types.js';
import {KANANA_ARTIFACT_SOURCES} from '../../kanana/artifacts.js';
import {decodeKananaImageSource} from '../../kanana/browser-image.js';
import {createKananaDecisionSession} from '../../kanana/decision-session.js';
import {createKananaTvmImageEmbeddingEncoder} from '../../kanana/image-encoder.js';
import {createLazyKananaWasmLoader} from '../../kanana/lazy-wasm.js';
import {preprocessKananaImage} from '../../kanana/image-processing.js';
import {KANANA_ASSETS_URL, KANANA_MODEL_ID, KANANA_REPOSITORY, KANANA_REVISION} from '../../kanana/contract.mjs';
import {openKananaSource as openKananaSourceRuntime} from '../../kanana/source.mjs';
import {createKananaTokenizer} from '../../kanana/tokenizer-asset.mjs';
import {missingKananaWeightLimits} from '../../kanana/device-limits.js';
import {
  KANANA_SAFETENSORS_SPEC,
  openKananaVisionWeightSource,
  openKananaWeightSource
} from '../../kanana/weight-source.mjs';
import {convertFp16ToFp32} from '../../kanana/safetensors.mjs';
import {createKananaTvmInferenceSession, createKananaTvmVisionInferenceSession} from '../../kanana/tvm-session.js';
import {setupKananaTvmRuntime} from '../../kanana/tvm-runtime.js';
import {mapKananaTvmResourceError} from '../../kanana/tvm-resource-error.js';

type ArtifactFile = Readonly<{bytes: number; sha256: string}>;
type ArtifactManifest = Readonly<{
  format: 'web-decision-kanana-tvm-artifacts-v1';
  modelId: string;
  repository: string;
  revision: string;
  checkpointSha256: string;
  runtime: Readonly<{
    tvm: string;
    textGraph: Readonly<{mode: 'checkpoint_full'; weights: number; graphWeights: number}>;
    visionGraph: Readonly<{graphWeights: number; maxPatchCount: number}>;
  }>;
  files: Readonly<Record<string, ArtifactFile>>;
}>;
type ExternalWeight = Readonly<{
  name: string;
  parameterName?: string;
  shape: readonly number[];
  dtype: string;
  length: number;
}>;
type ExternalWeightManifest = Readonly<{
  format?: string;
  modelId?: string;
  modelRevision?: string;
  weights: readonly ExternalWeight[];
}>;
type RangeManifest = Readonly<Record<string, unknown>>;
type ImageGrid = readonly [number, number, number];
type TvmSetup = Awaited<ReturnType<typeof setupKananaTvmRuntime>>;
type TvmTextSession = Awaited<ReturnType<typeof createKananaTvmInferenceSession>>;
type TvmVisionSession = Awaited<ReturnType<typeof createKananaTvmVisionInferenceSession>>;
type KananaSession = ReturnType<typeof createKananaDecisionSession>;
type SourceProgress = Readonly<{loadedBytes: number; totalBytes: number; fraction: number}>;
type KananaAssetSource = Readonly<{read(name: string): Promise<ArrayBuffer>}>;
type OpenKananaSource = (options: {
  signal?: AbortSignal | undefined;
  cacheEnabled?: boolean | undefined;
  offline?: boolean | undefined;
  onProgress?: ((progress: SourceProgress) => void) | undefined;
}) => Promise<KananaAssetSource>;
type OpenKananaWeights = (options: {
  base: string;
  manifest: RangeManifest;
  requiredWeights: readonly ExternalWeight[];
  cache: Cache | null;
  offline: boolean;
  signal?: AbortSignal | undefined;
  onBytes?: ((count: number) => void) | undefined;
}) => Promise<{
  readTensor(name: string, options?: {signal?: AbortSignal | undefined}): Promise<{bytes: Uint8Array}>;
  commit(options?: {signal?: AbortSignal | undefined}): Promise<boolean>;
}>;
type ImageCacheEntry = Readonly<{
  decoded: Awaited<ReturnType<typeof decodeKananaImageSource>>;
  prepared: ReturnType<typeof preprocessKananaImage>;
}>;

const TEXT_WASM = 'kanana-text-model.wasm';
const TEXT_WEIGHTS = 'text-weight-manifest.json';
const VISION_WASM = 'kanana-vision-model.wasm';
const VISION_WEIGHTS = 'vision-weight-manifest.json';
const RANGE_MANIFEST = 'safetensors-range-manifest.json';
const openKananaSource = openKananaSourceRuntime as unknown as OpenKananaSource;
const openKananaWeights = openKananaWeightSource as unknown as OpenKananaWeights;
const openKananaVisionWeights = openKananaVisionWeightSource as unknown as OpenKananaWeights;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonical((value as Record<string, unknown>)[key])])
  );
}

function parseJson(source: string, label: string): unknown {
  try {
    return JSON.parse(source) as unknown;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', `Invalid packaged Kanana ${label}.`, {cause});
  }
}

async function verifyRawFile(name: string, source: string, expected: ArtifactFile): Promise<unknown> {
  const bytes = new TextEncoder().encode(source);
  if (bytes.byteLength !== expected.bytes || (await sha256(bytes)) !== expected.sha256)
    throw new WebDecisionError('INVALID_ASSET', `Packaged Kanana asset integrity check failed: ${name}.`);
  return parseJson(source, name);
}

async function readWasm(
  url: string,
  name: string,
  expected: ArtifactFile,
  signal: AbortSignal | undefined
): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await waitForAbort(() => fetch(new URL(url, import.meta.url), signal ? {signal} : undefined), signal);
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not read packaged Kanana asset ${name}.`, {cause});
  }
  if (!response.ok) {
    cancelResponse(response);
    throw new WebDecisionError(
      'ASSET_LOAD_FAILED',
      `Could not read packaged Kanana asset ${name}: HTTP ${response.status}.`
    );
  }
  let bytes: ArrayBuffer;
  try {
    bytes = await waitForAbort(
      () => readBoundedResponse(response, expected.bytes, {exact: true, ...(signal ? {signal} : {})}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof ResponseSizeError)
      throw new WebDecisionError('INVALID_ASSET', `Packaged Kanana asset integrity check failed: ${name}.`, {cause});
    throw cause;
  }
  if (bytes.byteLength !== expected.bytes || (await sha256(bytes)) !== expected.sha256)
    throw new WebDecisionError('INVALID_ASSET', `Packaged Kanana asset integrity check failed: ${name}.`);
  return bytes;
}

function artifactFiles(value: unknown): ArtifactManifest {
  if (!isRecord(value)) throw new WebDecisionError('INVALID_ASSET', 'Kanana package manifest must be an object.');
  const manifest = value as unknown as ArtifactManifest;
  if (
    manifest.format !== 'web-decision-kanana-tvm-artifacts-v1' ||
    manifest.modelId !== KANANA_MODEL_ID ||
    manifest.repository !== KANANA_REPOSITORY ||
    manifest.revision !== KANANA_REVISION ||
    manifest.checkpointSha256 !== KANANA_SAFETENSORS_SPEC.fileSha256 ||
    manifest.runtime?.textGraph?.mode !== 'checkpoint_full' ||
    manifest.runtime.textGraph.weights !== 3_180_636_672 ||
    manifest.runtime.textGraph.graphWeights !== 354 ||
    manifest.runtime.visionGraph?.graphWeights !== 333 ||
    !Number.isSafeInteger(manifest.runtime.visionGraph.maxPatchCount) ||
    !manifest.files
  )
    throw new WebDecisionError('INVALID_ASSET', 'Kanana package manifest does not match the pinned full TVM graphs.');
  for (const name of [TEXT_WASM, TEXT_WEIGHTS, VISION_WASM, VISION_WEIGHTS, RANGE_MANIFEST]) {
    const file = manifest.files[name];
    if (!file || !Number.isSafeInteger(file.bytes) || file.bytes < 1 || !/^[0-9a-f]{64}$/u.test(file.sha256))
      throw new WebDecisionError('INVALID_ASSET', `Kanana package manifest is missing valid metadata for ${name}.`);
  }
  return manifest;
}

async function readArtifacts(
  description: ModelDescription,
  signal?: AbortSignal,
  onProgress?: KananaLoadOptions['onProgress']
) {
  const artifact = artifactFiles(parseJson(KANANA_ARTIFACT_SOURCES.artifactManifest, 'package manifest'));
  const canonicalBytes = new TextEncoder().encode(`${JSON.stringify(canonical(artifact))}\n`);
  if ((await sha256(canonicalBytes)) !== description.assets.manifestSha256)
    throw new WebDecisionError('INVALID_ASSET', 'Kanana package manifest does not match the curated catalog pin.');

  const rawAssets = [
    [TEXT_WEIGHTS, KANANA_ARTIFACT_SOURCES.textWeights],
    [VISION_WEIGHTS, KANANA_ARTIFACT_SOURCES.visionWeights],
    [RANGE_MANIFEST, KANANA_ARTIFACT_SOURCES.rangeManifest]
  ] as const;
  let loadedBytes = 0;
  const totalBytes =
    rawAssets.reduce((total, [name]) => total + artifact.files[name]!.bytes, 0) + artifact.files[TEXT_WASM]!.bytes;
  const report = (count: number) => {
    loadedBytes += count;
    onProgress?.({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes, phase: 'bundled-assets'});
  };

  const parsed = new Map<string, unknown>();
  for (const [name, source] of rawAssets) {
    const value = await verifyRawFile(name, source, artifact.files[name]!);
    parsed.set(name, value);
    report(new TextEncoder().encode(source).byteLength);
  }
  const textWasm = await readWasm(KANANA_ARTIFACT_SOURCES.textWasmUrl, TEXT_WASM, artifact.files[TEXT_WASM]!, signal);
  report(textWasm.byteLength);

  const textManifest = parsed.get(TEXT_WEIGHTS) as ExternalWeightManifest;
  const visionManifest = parsed.get(VISION_WEIGHTS) as ExternalWeightManifest;
  if (
    textManifest?.modelId !== KANANA_REPOSITORY ||
    textManifest.modelRevision !== KANANA_REVISION ||
    !Array.isArray(textManifest.weights) ||
    textManifest.weights.length !== 354 ||
    !Array.isArray(visionManifest?.weights) ||
    visionManifest.weights.length !== 333
  )
    throw new WebDecisionError('INVALID_ASSET', 'Kanana graph weight manifests differ from the pinned TVM graphs.');

  return {
    artifact,
    textWasm,
    loadVisionWasm: createLazyKananaWasmLoader(async (visionSignal) => {
      const bytes = await readWasm(
        KANANA_ARTIFACT_SOURCES.visionWasmUrl,
        VISION_WASM,
        artifact.files[VISION_WASM]!,
        visionSignal
      );
      onProgress?.({
        loadedBytes: bytes.byteLength,
        totalBytes: bytes.byteLength,
        fraction: 1,
        phase: 'vision-assets'
      });
      return bytes;
    }),
    textManifest,
    visionManifest,
    rangeManifest: parsed.get(RANGE_MANIFEST) as RangeManifest
  };
}

function fp16VisionWeights(weights: readonly ExternalWeight[]): ExternalWeight[] {
  return weights.map(({name, shape}) => {
    const elementCount = shape.reduce((total, dimension) => total * dimension, 1);
    if (!Number.isSafeInteger(elementCount) || elementCount < 1)
      throw new WebDecisionError('INVALID_ASSET', `Invalid Kanana vision weight shape: ${name}.`);
    return {name, shape: [...shape], dtype: 'float16', length: elementCount * 2};
  });
}

function visionRuntimeWeights(weights: readonly ExternalWeight[]): ExternalWeight[] {
  return weights.map(({name, parameterName, shape}) => {
    const elementCount = shape.reduce((total, dimension) => total * dimension, 1);
    if (!Number.isSafeInteger(elementCount) || elementCount < 1)
      throw new WebDecisionError('INVALID_ASSET', `Invalid Kanana vision weight shape: ${name}.`);
    return {
      name,
      ...(parameterName === undefined ? {} : {parameterName}),
      shape,
      dtype: 'float32',
      length: elementCount * 4
    };
  });
}

function expectedWeightBytes(text: readonly ExternalWeight[], vision: readonly ExternalWeight[]): number {
  const count = [...text, ...vision].reduce((total, weight) => total + weight.length, 0);
  if (!Number.isSafeInteger(count))
    throw new WebDecisionError('INVALID_ASSET', 'Kanana weight size exceeds the safe range.');
  return count;
}

function expectedWeightReadBytes(weights: readonly ExternalWeight[], manifest: RangeManifest): number {
  const headerLength = manifest['headerLength'];
  if (!Number.isSafeInteger(headerLength) || (headerLength as number) < 2)
    throw new WebDecisionError('INVALID_ASSET', 'Kanana Safetensors header length is invalid.');
  const total = expectedWeightBytes(weights, []) + (headerLength as number) + 8;
  if (!Number.isSafeInteger(total))
    throw new WebDecisionError('INVALID_ASSET', 'Kanana weight read size exceeds the safe range.');
  return total;
}

function imageCache() {
  const entries = new Map<unknown, Promise<ImageCacheEntry>>();
  const preparedByPixels = new WeakMap<Uint8Array, ReturnType<typeof preprocessKananaImage>>();

  async function load(source: unknown, signal?: AbortSignal): Promise<ImageCacheEntry> {
    let pending = entries.get(source);
    if (!pending) {
      pending = (async () => {
        const decoded = await decodeKananaImageSource(source, signal ? {signal} : {});
        const prepared = preprocessKananaImage(decoded.rgb, decoded.width, decoded.height);
        preparedByPixels.set(decoded.rgb, prepared);
        return {decoded, prepared};
      })();
      entries.set(source, pending);
    }
    return pending;
  }

  return {
    load,
    preprocess(rgb: Uint8Array, width: number, height: number) {
      const prepared = preparedByPixels.get(rgb);
      if (!prepared || prepared.width !== width || prepared.height !== height)
        throw new WebDecisionError(
          'INFERENCE_ERROR',
          'Kanana image preprocessing cache does not match its decoded source.'
        );
      return prepared;
    },
    clear() {
      entries.clear();
    }
  };
}

function disposeBoth(text: TvmTextSession | undefined, vision: TvmVisionSession | undefined): Promise<void> {
  const sessions = [text, vision].filter(
    (session): session is TvmTextSession | TvmVisionSession => session !== undefined
  );
  return Promise.allSettled(sessions.map((session) => Promise.resolve().then(() => session.dispose()))).then(
    (results) => {
      const errors = results.flatMap((result) => (result.status === 'rejected' ? [result.reason] : []));
      if (errors.length === 1) throw errors[0];
      if (errors.length > 1)
        throw new AggregateError(errors, 'Kanana TVM sessions failed to dispose.', {cause: errors[0]});
    }
  );
}

/** Load Kanana text weights now and verified vision weights when an image is used. */
export async function loadModel(options: object, description: ModelDescription) {
  const loadOptions = options as KananaLoadOptions;
  const {cache = true, offline = false, signal, onProgress} = loadOptions;
  if (typeof cache !== 'boolean' || typeof offline !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'Kanana cache and offline options must be booleans.');
  if (onProgress !== undefined && typeof onProgress !== 'function')
    throw new WebDecisionError('INVALID_INPUT', 'Kanana onProgress must be a function.');
  signal?.throwIfAborted();

  let device: GPUDevice | undefined;
  let textSetup: TvmSetup | undefined;
  let visionSetup: TvmSetup | undefined;
  let textInference: TvmTextSession | undefined;
  let visionInference: TvmVisionSession | undefined;
  let visionLoading: Promise<TvmVisionSession> | undefined;
  let decision: KananaSession | undefined;
  let removeDeviceListener: (() => void) | undefined;
  let disposed = false;
  let deviceFailure: unknown;
  const gpuErrors: string[] = [];
  const modelAbort = new AbortController();

  try {
    const detected = await waitForAbort(
      () => detectGPUDevice(),
      signal,
      (result) => result?.device.destroy()
    );
    if (!detected)
      throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'This browser does not expose a compatible WebGPU device.');
    device = detected.device;
    const onUncapturedError = (event: Event) => {
      const message = (event as GPUUncapturedErrorEvent).error?.message;
      gpuErrors.push(message || 'Unspecified WebGPU error.');
    };
    device.addEventListener('uncapturederror', onUncapturedError);
    removeDeviceListener = () => device?.removeEventListener('uncapturederror', onUncapturedError);
    void device.lost.then((info) => {
      if (!disposed) deviceFailure = new Error(info.message || 'The GPU device was lost.');
    });

    const artifacts = await readArtifacts(description, signal, onProgress);
    const weightBufferBytes = Math.max(
      ...artifacts.textManifest.weights.map(({length}) => length),
      ...visionRuntimeWeights(artifacts.visionManifest.weights).map(({length}) => length)
    );
    const missingLimits = missingKananaWeightLimits(detected.adapter.limits, weightBufferBytes);
    if (missingLimits.length)
      throw new WebDecisionError(
        'DEVICE_RESOURCE_LIMIT',
        `Kanana needs ${weightBufferBytes} bytes for one TVM weight buffer; the WebGPU adapter cannot meet ${missingLimits.join(', ')}.`
      );
    const source = await openKananaSource({
      signal,
      cacheEnabled: cache,
      offline,
      onProgress(progress: Omit<ModelProgress, 'phase'>) {
        onProgress?.({...progress, phase: 'upstream-source'});
      }
    });
    const [tokenizerJsonBytes, tokenizerConfigBytes, modelConfigBytes] = await Promise.all([
      source.read('tokenizer.json'),
      source.read('tokenizer_config.json'),
      source.read('config.json')
    ]);
    const parseUpstreamJson = (bytes: ArrayBuffer, name: string) => {
      try {
        return JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)) as unknown;
      } catch (cause) {
        throw new WebDecisionError('INVALID_ASSET', `Invalid pinned Kanana ${name}.`, {cause});
      }
    };
    const parseUpstreamRecord = (bytes: ArrayBuffer, name: string): Record<string, unknown> => {
      const value = parseUpstreamJson(bytes, name);
      if (!isRecord(value)) throw new WebDecisionError('INVALID_ASSET', `Pinned Kanana ${name} must be an object.`);
      return value;
    };
    const tokenizer = createKananaTokenizer(
      parseUpstreamRecord(tokenizerJsonBytes, 'tokenizer'),
      parseUpstreamRecord(tokenizerConfigBytes, 'tokenizer configuration'),
      parseUpstreamRecord(modelConfigBytes, 'model configuration'),
      Tokenizer
    );
    const modelConfig = parseUpstreamRecord(modelConfigBytes, 'model configuration') as {
      vision_config?: {
        in_channels?: number;
        patch_size?: number;
        spatial_merge_size?: number;
        temporal_patch_size?: number;
        num_position_embeddings?: number;
      };
    };
    const visionConfig = modelConfig.vision_config;
    if (
      visionConfig?.in_channels !== 3 ||
      visionConfig.patch_size !== 16 ||
      visionConfig.temporal_patch_size !== 2 ||
      visionConfig.spatial_merge_size !== 2 ||
      visionConfig.num_position_embeddings !== 2304
    )
      throw new WebDecisionError('INVALID_ASSET', 'Kanana vision configuration differs from the pinned graph inputs.');

    const visionSourceWeights = fp16VisionWeights(artifacts.visionManifest.weights);
    const textWeightBytes = expectedWeightReadBytes(artifacts.textManifest.weights, artifacts.rangeManifest);
    let loadedTextWeightBytes = 0;
    const cacheStore = await openAssetCache(cache, signal);
    const textWeightSource = await openKananaWeights({
      base: KANANA_ASSETS_URL,
      manifest: artifacts.rangeManifest,
      requiredWeights: artifacts.textManifest.weights,
      cache: cacheStore,
      offline,
      signal,
      onBytes(bytes: number) {
        loadedTextWeightBytes += bytes;
        onProgress?.({
          loadedBytes: loadedTextWeightBytes,
          totalBytes: textWeightBytes,
          fraction: Math.min(1, loadedTextWeightBytes / textWeightBytes),
          phase: 'asset-verification'
        });
      }
    });

    textSetup = await waitForAbort(
      () =>
        setupKananaTvmRuntime({
          instantiate: instantiate as never,
          createPolyfillWASI,
          device: device!,
          bytes: artifacts.textWasm
        }),
      signal,
      (setup) => setup.dispose()
    );
    signal?.throwIfAborted();

    const assertDeviceUsable = () => {
      modelAbort.signal.throwIfAborted();
      if (deviceFailure)
        throw new WebDecisionError('INFERENCE_ERROR', 'The Kanana WebGPU device was lost.', {cause: deviceFailure});
      if (gpuErrors.length)
        throw new WebDecisionError('INFERENCE_ERROR', `Kanana WebGPU reported an error: ${gpuErrors[0]}`);
    };

    textInference = await createKananaTvmInferenceSession({
      setup: textSetup as never,
      weightSource: {
        manifest: {weights: artifacts.textManifest.weights},
        async readWeight(record) {
          signal?.throwIfAborted();
          const tensor = await textWeightSource.readTensor(record.name, {signal});
          signal?.throwIfAborted();
          return tensor.bytes;
        }
      }
    });
    textSetup = undefined;
    await textWeightSource.commit({signal});

    const loadVisionInference = (requestSignal?: AbortSignal): Promise<TvmVisionSession> => {
      requestSignal?.throwIfAborted();
      if (visionLoading) return visionLoading;
      if (visionInference) return Promise.resolve(visionInference);
      const visionWeightBytes = expectedWeightReadBytes(visionSourceWeights, artifacts.rangeManifest);
      let loadedVisionWeightBytes = 0;
      const pending = (async () => {
        let session: TvmVisionSession | undefined;
        const visionAbort = new AbortController();
        const abortVisionLoad = () => visionAbort.abort();
        modelAbort.signal.addEventListener('abort', abortVisionLoad, {once: true});
        requestSignal?.addEventListener('abort', abortVisionLoad, {once: true});
        if (modelAbort.signal.aborted || requestSignal?.aborted) abortVisionLoad();
        const visionSignal = visionAbort.signal;
        try {
          visionSignal.throwIfAborted();
          assertDeviceUsable();
          const visionWasm = await artifacts.loadVisionWasm(visionSignal);
          visionSignal.throwIfAborted();
          const visionWeightSource = await openKananaVisionWeights({
            base: KANANA_ASSETS_URL,
            manifest: artifacts.rangeManifest,
            requiredWeights: visionSourceWeights,
            cache: cacheStore,
            offline,
            signal: visionSignal,
            onBytes(bytes: number) {
              loadedVisionWeightBytes += bytes;
              onProgress?.({
                loadedBytes: loadedVisionWeightBytes,
                totalBytes: visionWeightBytes,
                fraction: Math.min(1, loadedVisionWeightBytes / visionWeightBytes),
                phase: 'vision-weights'
              });
            }
          });
          visionSetup = await waitForAbort(
            () =>
              setupKananaTvmRuntime({
                instantiate: instantiate as never,
                createPolyfillWASI,
                device: device!,
                bytes: visionWasm
              }),
            visionSignal,
            (setup) => setup.dispose()
          );
          session = await createKananaTvmVisionInferenceSession({
            setup: visionSetup as never,
            weightSource: {
              manifest: {weights: visionRuntimeWeights(artifacts.visionManifest.weights)},
              async readWeight(record) {
                visionSignal.throwIfAborted();
                const tensor = await visionWeightSource.readTensor(record.name, {signal: visionSignal});
                visionSignal.throwIfAborted();
                return convertFp16ToFp32(tensor.bytes);
              }
            },
            patchValues: visionConfig.in_channels! * visionConfig.temporal_patch_size! * visionConfig.patch_size! ** 2,
            numPositionEmbeddings: visionConfig.num_position_embeddings!,
            spatialMergeSize: visionConfig.spatial_merge_size!,
            maxPatchCount: artifacts.artifact.runtime.visionGraph.maxPatchCount
          });
          visionSetup = undefined;
          visionInference = session;
          await visionWeightSource.commit({signal: visionSignal});
          assertDeviceUsable();
          return session;
        } catch (cause) {
          const setup = visionSetup;
          visionSetup = undefined;
          visionInference = undefined;
          const cleanup = await Promise.allSettled([
            Promise.resolve().then(() => session?.dispose()),
            Promise.resolve().then(() => setup?.dispose())
          ]);
          const cleanupErrors = cleanup.flatMap((result) => (result.status === 'rejected' ? [result.reason] : []));
          const failure = cleanupErrors.length
            ? new AggregateError([cause, ...cleanupErrors], 'Kanana vision loading and cleanup both failed.', {
                cause
              })
            : cause;
          throw mapKananaTvmResourceError(failure);
        } finally {
          modelAbort.signal.removeEventListener('abort', abortVisionLoad);
          requestSignal?.removeEventListener('abort', abortVisionLoad);
        }
      })();
      visionLoading = pending;
      void pending.then(
        () => {
          if (visionLoading === pending) visionLoading = undefined;
        },
        () => {
          if (visionLoading === pending) visionLoading = undefined;
        }
      );
      return pending;
    };

    const images = imageCache();
    const prepareImageEmbeddings = createKananaTvmImageEmbeddingEncoder({
      vision: {
        async infer(pixelValues, visionGridThw, requestSignal) {
          assertDeviceUsable();
          requestSignal?.throwIfAborted();
          const inference = await loadVisionInference(requestSignal);
          requestSignal?.throwIfAborted();
          assertDeviceUsable();
          const result = await inference.infer(pixelValues, visionGridThw);
          requestSignal?.throwIfAborted();
          assertDeviceUsable();
          return result;
        }
      },
      async decode(image, imageSignal) {
        return (await images.load(image, imageSignal)).decoded;
      },
      preprocess: images.preprocess
    });
    decision = createKananaDecisionSession({
      tokenizer,
      imageTokenThw: async (image, _index, requestSignal) => {
        try {
          const {prepared} = await images.load(image, requestSignal);
          return prepared.imageTokenThw as ImageGrid;
        } catch (cause) {
          images.clear();
          throw cause;
        }
      },
      prepareImageEmbeddings: async (imageSources, grids, imageSignal) => {
        try {
          return await prepareImageEmbeddings(imageSources, grids, imageSignal);
        } finally {
          images.clear();
        }
      },
      infer: async ({inputIds, attentionMask, imageEmbeddings, imagePositions, positionIds}) => {
        assertDeviceUsable();
        const result = await textInference!.infer(inputIds, attentionMask, {
          imageEmbeddings,
          imagePositions,
          positionIds
        });
        assertDeviceUsable();
        return result;
      },
      dispose: async () => {
        await visionLoading?.catch(() => undefined);
        await disposeBoth(textInference, visionInference);
      }
    });

    const session = decision;
    const runDecision: KananaSession['systemOne'] = (...args) =>
      session.systemOne(...args).catch((cause) => {
        throw mapKananaTvmResourceError(cause);
      });
    let disposal: Promise<void> | undefined;
    return {
      modelId: KANANA_MODEL_ID,
      decide: runDecision,
      systemOne: runDecision,
      dispose() {
        if (!disposal) {
          disposed = true;
          modelAbort.abort();
          disposal = session.dispose().finally(() => {
            removeDeviceListener?.();
            device?.destroy();
          });
        }
        return disposal;
      }
    };
  } catch (cause) {
    disposed = true;
    modelAbort.abort();
    removeDeviceListener?.();
    const cleanup = await Promise.allSettled([
      Promise.resolve().then(() => decision?.dispose()),
      Promise.resolve().then(() => textInference?.dispose()),
      Promise.resolve().then(() => visionInference?.dispose()),
      Promise.resolve().then(() => textSetup?.dispose()),
      Promise.resolve().then(() => visionSetup?.dispose())
    ]);
    device?.destroy();
    const cleanupErrors = cleanup.flatMap((result) => (result.status === 'rejected' ? [result.reason] : []));
    const failure = cleanupErrors.length
      ? new AggregateError([cause, ...cleanupErrors], 'Kanana model loading and cleanup both failed.', {cause})
      : cause;
    throw mapKananaTvmResourceError(failure);
  }
}
