import {Tokenizer} from '@huggingface/tokenizers';
import {instantiate} from '../../../.toolchains/apache-tvm-src/web/src/runtime.ts';
import {createPolyfillWASI} from '../../../.toolchains/apache-tvm-src/web/src/compact.ts';
import {commitCachedManifest, parseJson, waitForAbort} from './assets.js';
import {readAssetBatch} from './asset-batch.js';
import {WebDecisionError} from './kev.js';
import {watchLoadDeviceLoss} from './load-device-loss.js';
import {createModelAssetReader} from './model-assets.js';
import {prepareModelSource} from './model-source.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime, trySyncBeforeCleanup} from './runtime-lifecycle.js';

const noFailure = Symbol('no failure');

async function uploadChecked(gpu, device, lifecycle, operation) {
  let failure = noFailure,
    scopeCount = 0,
    operationStarted = false;
  try {
    gpu.pushErrorScope('out-of-memory');
    scopeCount++;
    gpu.pushErrorScope('validation');
    scopeCount++;
    operationStarted = true;
    await operation();
  } catch (cause) {
    failure = cause;
    // A failed upload may have submitted GPU work before reaching its sync.
    // Try before closing error scopes; device loss can still dispose the runtime.
    if (operationStarted) await trySyncBeforeCleanup(device, lifecycle);
  } finally {
    while (scopeCount > 0) {
      try {
        const error = await gpu.popErrorScope();
        if (error && failure === noFailure) failure = error;
      } catch (cause) {
        if (failure === noFailure) failure = cause;
      }
      scopeCount--;
    }
  }
  if (failure !== noFailure) throw failure;
}

/** Load one pinned text graph, model-specific contract and session. */
export async function loadPinnedTextModel(options, model) {
  const {base, cache, signal, manifestUrl, response, maxPrefetchBytes} = await prepareModelSource(options, {
    offlineMissingMessage: model.offlineMissingMessage
  });
  const manifest = await model.readManifest(response, signal);
  const assetReader = createModelAssetReader({base, manifest, cache, signal, options});
  const {read} = assetReader;
  let runtime,
    gpu,
    lifecycle,
    loadWatch,
    scopeOpen = false;
  try {
    const [tokenizerBytes, configBytes, wasmBytes, indexBytes] = await readAssetBatch(
      ['tokenizer.json', 'tokenizer_config.json', 'model.wasm', 'weights/index.json'],
      read,
      signal
    );
    const data = parseJson(tokenizerBytes, 'tokenizer.json');
    const tokenizer = new Tokenizer(data, parseJson(configBytes, 'tokenizer_config.json'));
    const index = model.validateIndex(manifest, parseJson(indexBytes, 'weights/index.json'));
    const extras = model.prepare(data, tokenizer, manifest);
    gpu = await model.requestDevice(navigator.gpu, signal);
    signal?.throwIfAborted();
    const uncaptured = [];
    gpu.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
    });
    loadWatch = watchLoadDeviceLoss(gpu, signal);
    runtime = await waitForAbort(
      () => instantiate(wasmBytes, createPolyfillWASI()),
      loadWatch.signal,
      (lateRuntime) => {
        try {
          lateRuntime?.dispose();
        } catch {}
      }
    );
    lifecycle = trackRuntime(runtime, gpu);
    loadWatch.signal.throwIfAborted();
    await Promise.resolve();
    assertLoadActive(lifecycle);
    runtime.initWebGPU(gpu);
    const device = runtime.webgpu();
    runtime.beginScope();
    scopeOpen = true;
    const vm = runtime.createVirtualMachine(device),
      fn = vm.getFunction('main'),
      weights = new Array(manifest.parameterCount);
    await uploadChecked(gpu, device, lifecycle, () =>
      model.upload({
        index,
        read,
        runtime,
        device,
        weights,
        extras,
        signal: loadWatch.signal,
        maxPrefetchBytes,
        lifecycle,
        uncaptured
      })
    );
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
    return model.createSession({runtime, device, gpu, fn, weights, tokenizer, uncaptured, lifecycle, ...extras});
  } catch (cause) {
    cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice: gpu});
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', `Could not load the ${model.name} model.`, {cause});
  }
}
