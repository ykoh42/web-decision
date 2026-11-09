/** Loader for the pinned experimental DeBERTa preview. */
import {Tokenizer} from '@huggingface/tokenizers';
import {instantiate, createPolyfillWASI} from '../runtime/tvm.generated.mjs';
import {detectDebertaGPUDevice as detectGPUDevice} from './device.mjs';
import {commitCachedManifest, parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime} from '../runtime-lifecycle.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {ExperimentalDebertaSession} from './model.mjs';
import {runWithPackedReadback} from './packed-readback.mjs';
import {validateDebertaReadback} from './readback-mode.mjs';
import {readDebertaManifestResponse, validateDebertaSplit, validateDebertaTensorCache} from './contract.mjs';
import {readAssetBatch} from '../asset-batch.js';
import {prepareModelSource} from '../model-source.js';
import {createModelAssetReader, uploadModelShards} from '../model-assets.js';
import type {Tensor} from '../runtime/tvm-types.js';

type LoaderOptions = Parameters<typeof prepareModelSource>[0] &
  Parameters<typeof createModelAssetReader>[0]['options'] & {debertaReadback?: unknown};
type DebertaConfig = {
  vocab_size?: unknown;
  open_jev?: {max_len?: unknown; max_state_tokens?: unknown; temperature?: unknown} | null;
};

const unsupported = (message: string) => new WebDecisionError('UNSUPPORTED_ENVIRONMENT', message);
const markers = Object.freeze({'[CLS]': 1, '[SEP]': 2, '[STATE]': 128001, '[Q]': 128002, '[OPT]': 128003});

/** Load only the pinned, experimental split DeBERTa Q4F16 build. */
export async function loadExperimentalDeberta(options: LoaderOptions) {
  const {
    base,
    cache,
    signal,
    manifestUrl,
    response,
    maxPrefetchBytes,
    modelOptions: readback
  } = await prepareModelSource(options, {
    validateModelOptions: (value) => validateDebertaReadback(value.debertaReadback),
    offlineMissingMessage: 'The model manifest is not available offline. Load it online first.'
  });
  const manifest = await readDebertaManifestResponse(response, signal);
  const assetReader = createModelAssetReader({base, manifest, cache, signal, options});
  const {read} = assetReader;
  let runtime,
    gpuDevice,
    lifecycle,
    loadWatch,
    scopeOpen = false;
  try {
    const [
      tokenizerBytes,
      tokenizerConfigBytes,
      modelConfigBytes,
      wasmBytes,
      tensorCacheBytes,
      splitBytes,
      mappingBytes
    ] = await readAssetBatch(
      [
        'tokenizer.json',
        'tokenizer_config.json',
        'model_config.json',
        'model.wasm',
        'weights/tensor-cache.json',
        'split.json',
        'mapping.json'
      ],
      read,
      signal
    );
    const config = parseJson(modelConfigBytes!, 'model_config.json') as DebertaConfig;
    if (
      config.vocab_size !== 128100 ||
      config.open_jev?.max_len !== 512 ||
      config.open_jev?.max_state_tokens !== 256 ||
      config.open_jev?.temperature !== 1.05
    ) {
      throw new WebDecisionError('INVALID_ASSET', 'Unexpected DeBERTa model configuration.');
    }
    const tokenizer = new Tokenizer(
      parseJson(tokenizerBytes!, 'tokenizer.json') as object,
      parseJson(tokenizerConfigBytes!, 'tokenizer_config.json') as object
    );
    for (const [text, id] of Object.entries(markers)) {
      const ids = tokenizer.encode(text, {add_special_tokens: false}).ids;
      if (ids.length !== 1 || ids[0] !== id)
        throw new WebDecisionError('INVALID_ASSET', 'Unexpected DeBERTa tokenizer markers.');
    }
    const tensorCache = parseJson(tensorCacheBytes!, 'tensor-cache.json') as {
      records: Parameters<typeof uploadModelShards>[0]['records'];
    };
    validateDebertaTensorCache(manifest, tensorCache);
    const split = validateDebertaSplit(parseJson(splitBytes!, 'split.json'), parseJson(mappingBytes!, 'mapping.json'));
    signal?.throwIfAborted();
    let gpu;
    try {
      gpu = await detectGPUDevice(undefined, signal);
    } catch (cause) {
      signal?.throwIfAborted();
      throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not initialize a compatible WebGPU device.', {cause});
    }
    gpuDevice = gpu?.device;
    signal?.throwIfAborted();
    if (!gpu || !gpu.adapter.features.has('shader-f16')) {
      gpuDevice?.destroy();
      gpuDevice = undefined;
      throw unsupported('This experimental model requires WebGPU shader-f16 support.');
    }
    const uncaptured: string[] = [];
    gpuDevice!.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
    });
    loadWatch = watchLoadDeviceLoss(gpuDevice!, signal);
    runtime = await waitForAbort(
      () => instantiate(wasmBytes!, createPolyfillWASI()),
      loadWatch.signal,
      (lateRuntime) => {
        try {
          lateRuntime?.dispose();
        } catch {}
      }
    );
    lifecycle = trackRuntime(runtime, gpuDevice!);
    loadWatch.signal.throwIfAborted();
    await Promise.resolve();
    assertLoadActive(lifecycle);
    runtime.initWebGPU(gpuDevice!);
    const device = runtime.webgpu();
    runtime.beginScope();
    scopeOpen = true;
    const vm = runtime.createVirtualMachine(device),
      encoder = vm.getFunction('encode'),
      head = vm.getFunction('score_head');
    // Preserve sparse allocation until validated shards are uploaded.
    // oxlint-disable-next-line unicorn/no-new-array
    const weights = new Array<Tensor>(343);
    await uploadModelShards({
      records: tensorCache.records,
      read,
      runtime,
      device,
      weights,
      signal: loadWatch.signal,
      maxPrefetchBytes,
      uncaptured
    });
    loadWatch.signal.throwIfAborted();
    signal?.throwIfAborted();
    assertLoadActive(lifecycle);
    if (cache && assetReader.isCacheComplete() && !options.offline) {
      try {
        await commitCachedManifest(cache, manifestUrl, manifest, loadWatch.signal);
      } catch {
        loadWatch.signal.throwIfAborted();
      }
      signal?.throwIfAborted();
      assertLoadActive(lifecycle);
    }
    loadWatch.close();
    const session = new ExperimentalDebertaSession({
      runtime,
      device,
      gpu: gpuDevice!,
      encoder,
      head,
      encoderWeights: split.encoderIndices.map((index) => weights[index]!),
      headWeights: split.headIndices.map((index) => weights[index]!),
      tokenizer,
      uncaptured,
      lifecycle
    });
    if (readback === 'packed') session.run = runWithPackedReadback;
    return session;
  } catch (cause) {
    cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice});
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load the experimental DeBERTa model.', {cause});
  }
}
