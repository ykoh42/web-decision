import {Tokenizer} from '@huggingface/tokenizers';
import {requestGpuAdapter, requestGpuDevice} from '../abortable-gpu-device.js';
import {parseJson, readVerifiedAsset, sha256, waitForAbort, type AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime} from '../runtime-lifecycle.js';
import {createPolyfillWASI, instantiate} from '../runtime/tvm.generated.mjs';
import type {Instance} from '../runtime/tvm-types.js';
import {guardStrandsDisposal} from '../strands/disposal-guard.mjs';
import type {ModelProgress} from '../curated/types.js';
import {
  MAPIKA2B_REPOSITORY,
  MAPIKA2B_REVISION,
  MAPIKA2B_TOKENIZER_CONFIG_SHA256,
  MAPIKA2B_TOKENIZER_SHA256,
  type Mapika2bLimits,
  type Mapika2bTemperatureOptions
} from './contract.mjs';
import {createMapika2bLabels, resolveMapika2bLimits, resolveMapika2bTemperatures} from './protocol.mjs';
import {assembleMapika2bTextRow, createMapika2bSession} from './session.mjs';
import {openPinnedMapika2bSource, type Mapika2bSource, type Mapika2bSourceOpenOptions} from './source.mjs';
import {runMapika2bTextRow, uploadMapika2bTextParameters, type Mapika2bTextBinding} from './text-runtime.mjs';
import {cacheMapika2bVisionAssets, openMapika2bVision} from './vision-runtime.mjs';

const base = new URL(`https://huggingface.co/${MAPIKA2B_REPOSITORY}/resolve/${MAPIKA2B_REVISION}/`);
const tokenizerAsset = {
  files: {
    'tokenizer.json': {bytes: 19_989_325, sha256: MAPIKA2B_TOKENIZER_SHA256},
    'tokenizer_config.json': {bytes: 1_127, sha256: MAPIKA2B_TOKENIZER_CONFIG_SHA256}
  }
} as const;

interface Mapika2bCandidateOptions extends Mapika2bSourceOpenOptions, Mapika2bTemperatureOptions {
  readonly onProgress?: ((progress: ModelProgress) => void) | undefined;
  readonly limits?: Partial<Mapika2bLimits> | undefined;
  readonly artifacts: {
    readonly textWasm: Uint8Array;
    readonly textWasmSha256: string;
    readonly textBindings: readonly Mapika2bTextBinding[];
  };
}

async function readCandidate(options: Mapika2bCandidateOptions) {
  options.signal?.throwIfAborted();
  const {textWasm, textWasmSha256, textBindings} = options.artifacts;
  const wasm = Uint8Array.from(textWasm);
  const bindings = textBindings.map(({name, shape}) => ({name, shape: [...shape]}));
  const [tokenizerBytes, tokenizerConfig, digest] = await Promise.all([
    readVerifiedAsset(base, tokenizerAsset, 'tokenizer.json', options),
    readVerifiedAsset(base, tokenizerAsset, 'tokenizer_config.json', options),
    waitForAbort(() => sha256(wasm), options.signal)
  ]);
  if (digest !== textWasmSha256) throw new WebDecisionError('INVALID_ASSET', 'Mapika 2B v12 text WASM hash differs.');
  return {
    wasm: wasm.buffer,
    bindings,
    tokenizer: new Tokenizer(
      parseJson(tokenizerBytes, 'tokenizer.json') as object,
      parseJson(tokenizerConfig, 'tokenizer_config.json') as object
    )
  };
}

async function requestTextDevice(signal?: AbortSignal | null): Promise<GPUDevice> {
  const api = globalThis.navigator?.gpu;
  if (!api) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Mapika 2B v12 requires WebGPU.');
  const adapter = await requestGpuAdapter(api, signal ?? undefined);
  if (!adapter) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'No WebGPU adapter is available.');
  const {maxBufferSize, maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize} = adapter.limits;
  return requestGpuDevice(
    adapter,
    {requiredLimits: {maxBufferSize, maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize}},
    signal ?? undefined
  );
}

async function cacheModelSource(
  source: Mapika2bSource,
  cache: AssetCache,
  signal: AbortSignal,
  onProgress?: (progress: ModelProgress) => void
): Promise<void> {
  await cacheMapika2bVisionAssets({cache, signal});
  let loadedBytes = 0;
  await source.prefetchRequiredRanges({
    signal,
    onProgress(bytes) {
      loadedBytes += bytes;
      onProgress?.({
        loadedBytes,
        totalBytes: source.requiredRangeBytes,
        fraction: loadedBytes / source.requiredRangeBytes,
        phase: 'upstream-source'
      });
    }
  });
  await source.commit({signal});
}

async function initText(
  gpu: GPUDevice,
  source: Mapika2bSource,
  candidate: Awaited<ReturnType<typeof readCandidate>>,
  options: Mapika2bCandidateOptions
) {
  const watch = watchLoadDeviceLoss(gpu, options.signal);
  let runtime: Instance | undefined;
  let lifecycle: ReturnType<typeof trackRuntime> | undefined;
  let scopeOpen = false;
  const dispose = () => cleanupFailedLoad({loadWatch: watch, gpuDevice: gpu, runtime, lifecycle, scopeOpen});
  try {
    runtime = await waitForAbort(
      () => instantiate(candidate.wasm, createPolyfillWASI()),
      watch.signal,
      (lateRuntime) => lateRuntime.dispose()
    );
    lifecycle = trackRuntime(runtime, gpu);
    guardStrandsDisposal(runtime, gpu, lifecycle);
    watch.signal.throwIfAborted();
    const uncaptured: string[] = [];
    gpu.addEventListener('uncapturederror', ({error}) => {
      if (uncaptured.length < 10) uncaptured.push(error.message);
    });
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scopeOpen = true;
    const device = runtime.webgpu();
    const main = runtime.createVirtualMachine(device).getFunction('main');
    const {ids} = createMapika2bLabels(candidate.tokenizer);
    const answerRows = await source.readEmbeddingRows(ids, {signal: watch.signal});
    const totalBytes =
      answerRows.byteLength +
      candidate.bindings.reduce((sum, {shape}) => sum + shape.reduce((size, dimension) => size * dimension, 4), 0);
    let loadedBytes = 0;
    const parameters = await uploadMapika2bTextParameters(
      runtime,
      device,
      candidate.bindings,
      answerRows,
      (binding, signal) => source.readParameter(binding, {signal}),
      watch.signal,
      (bytes) => {
        loadedBytes += bytes;
        options.onProgress?.({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes, phase: 'model-assets'});
      }
    );
    if (options.cache && !options.offline)
      await cacheModelSource(source, options.cache, watch.signal, options.onProgress);
    watch.signal.throwIfAborted();
    assertLoadActive(lifecycle);
    if (uncaptured.length) throw new Error(uncaptured[0]);
    watch.close();
    return {runtime, gpu, device, main, parameters, lifecycle, uncaptured, dispose};
  } catch (cause) {
    dispose();
    throw cause;
  }
}

/** Internal QA entry point for the compiled Mapika v12 candidate. */
export async function loadMapika2bCandidate(options: Mapika2bCandidateOptions) {
  let source: Mapika2bSource | undefined;
  let text: Awaited<ReturnType<typeof initText>> | undefined;
  try {
    const temperatures = resolveMapika2bTemperatures(options);
    const limits = resolveMapika2bLimits(options.limits);
    const candidate = await readCandidate(options);
    const weights = (source = await openPinnedMapika2bSource(options));
    const gpu = await requestTextDevice(options.signal);
    const textRunner = (text = await initText(gpu, weights, candidate, options));
    weights.finishLoading();
    const {cache, offline} = options;
    let vision: ReturnType<typeof openMapika2bVision> | undefined;
    return createMapika2bSession({
      tokenizer: candidate.tokenizer,
      temperatures,
      limits,
      async encodeImage(asset) {
        vision ??= openMapika2bVision({source: weights, cache, offline}).catch((cause: unknown) => {
          vision = undefined;
          throw cause;
        });
        return (await vision).encodeImage(asset);
      },
      async run(row) {
        const packed = await weights.readEmbeddingRows(row.inputIds);
        return runMapika2bTextRow(textRunner, assembleMapika2bTextRow(row, packed));
      },
      async dispose() {
        try {
          await (await vision)?.dispose();
        } finally {
          textRunner.dispose();
          weights.dispose();
        }
      }
    });
  } catch (cause) {
    text?.dispose();
    source?.dispose();
    options.signal?.throwIfAborted();
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load the Mapika 2B v12 candidate.', {cause});
  }
}
