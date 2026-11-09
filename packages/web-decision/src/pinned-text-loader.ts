import {Tokenizer} from '@huggingface/tokenizers';
import {createPolyfillWASI, instantiate} from './runtime/tvm.generated.mjs';
import {commitCachedManifest, parseJson, waitForAbort} from './assets.js';
import {readAssetBatch} from './asset-batch.js';
import {WebDecisionError} from './kev.js';
import {watchLoadDeviceLoss} from './load-device-loss.js';
import {createModelAssetReader} from './model-assets.js';
import {prepareModelSource} from './model-source.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime, trySyncBeforeCleanup} from './runtime-lifecycle.js';
import type {AssetFileManifest} from './assets.js';
import type {DLDevice, Instance, Tensor} from './runtime/tvm-types.js';

const noFailure = Symbol('no failure');

interface PinnedTextModelContract<Manifest, Index, Extras, UploadArgs, SessionArgs, Session> {
  readonly name: string;
  readonly offlineMissingMessage: string;
  readManifest(response: Response, signal?: AbortSignal | undefined): Promise<Manifest>;
  validateIndex(manifest: Manifest, index: unknown): Index;
  requestDevice(gpu: GPU, signal?: AbortSignal | undefined): Promise<GPUDevice>;
  prepare(data: unknown, tokenizer: InstanceType<typeof Tokenizer>, manifest: Manifest): Extras;
  upload(args: UploadArgs): Promise<unknown>;
  createSession(args: SessionArgs): Session;
}

type RuntimeLifecycle = ReturnType<typeof trackRuntime>;
type InitialAssetBytes = [ArrayBuffer, ArrayBuffer, ArrayBuffer, ArrayBuffer];

async function uploadChecked(
  gpu: GPUDevice,
  device: DLDevice,
  lifecycle: RuntimeLifecycle,
  operation: () => unknown | Promise<unknown>
): Promise<void> {
  let failure: unknown = noFailure,
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
export async function loadPinnedTextModel<Manifest, Index, Extras, UploadArgs, SessionArgs, Session>(
  options: unknown,
  model: PinnedTextModelContract<Manifest, Index, Extras, UploadArgs, SessionArgs, Session>
): Promise<Session> {
  const {base, cache, signal, manifestUrl, response, maxPrefetchBytes} = await prepareModelSource(
    options as Parameters<typeof prepareModelSource>[0],
    {
      offlineMissingMessage: model.offlineMissingMessage
    }
  );
  const manifest = await model.readManifest(response, signal);
  const assetReader = createModelAssetReader({
    base,
    manifest: manifest as unknown as AssetFileManifest,
    cache,
    signal,
    options: options as Parameters<typeof createModelAssetReader>[0]['options']
  });
  const {read} = assetReader;
  let runtime: Instance | undefined,
    gpu: GPUDevice | undefined,
    lifecycle: RuntimeLifecycle | undefined,
    loadWatch: ReturnType<typeof watchLoadDeviceLoss> | undefined,
    scopeOpen = false;
  try {
    const [tokenizerBytes, configBytes, wasmBytes, indexBytes] = (await readAssetBatch(
      ['tokenizer.json', 'tokenizer_config.json', 'model.wasm', 'weights/index.json'],
      read,
      signal
    )) as InitialAssetBytes;
    const data = parseJson(tokenizerBytes, 'tokenizer.json');
    const tokenizer = new Tokenizer(data as object, parseJson(configBytes, 'tokenizer_config.json') as object);
    const index = model.validateIndex(manifest, parseJson(indexBytes, 'weights/index.json'));
    const extras = model.prepare(data, tokenizer, manifest);
    gpu = await model.requestDevice(navigator.gpu, signal);
    signal?.throwIfAborted();
    const uncaptured: string[] = [];
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
      weights = new Array<Tensor | undefined>((manifest as {parameterCount: number}).parameterCount);
    await uploadChecked(gpu, device, lifecycle, () =>
      model.upload({
        index,
        read,
        runtime: runtime!,
        device,
        weights,
        extras,
        signal: loadWatch!.signal,
        maxPrefetchBytes,
        lifecycle,
        uncaptured
      } as UploadArgs)
    );
    loadWatch!.signal.throwIfAborted();
    signal?.throwIfAborted();
    assertLoadActive(lifecycle);
    if (cache && assetReader.isCacheComplete() && !(options as {offline?: unknown}).offline) {
      try {
        await commitCachedManifest(cache, manifestUrl, manifest, loadWatch!.signal);
      } catch {
        loadWatch!.signal.throwIfAborted();
      }
      signal?.throwIfAborted();
      assertLoadActive(lifecycle);
    }
    loadWatch!.close();
    return model.createSession({
      runtime,
      device,
      gpu,
      fn,
      weights,
      tokenizer,
      uncaptured,
      lifecycle,
      ...extras
    } as SessionArgs);
  } catch (cause) {
    cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice: gpu});
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', `Could not load the ${model.name} model.`, {cause});
  }
}
