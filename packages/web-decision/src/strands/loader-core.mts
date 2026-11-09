import {sha256, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {watchLoadDeviceLoss} from '../load-device-loss.js';
import {assertLoadActive, cleanupFailedLoad, trackRuntime} from '../runtime-lifecycle.js';
import {uploadFusedRecord} from './fused-upload.mjs';
import {
  checkFusedAdapter,
  LARGEST_BINDING,
  parsePinnedManifest,
  TOKENIZER_CONFIG_SHA256,
  TOKENIZER_SHA256,
  uploadPinnedWeights,
  WASM_BYTES,
  WASM_SHA256
} from './manifest.mjs';
import {ExperimentalFusedRunner} from './runner.mjs';
import {ExperimentalStrandsSession} from './session.mjs';
import {requestStrandsAdapter} from './preflight.mjs';
import {guardStrandsDisposal, retainStrandsResources} from './disposal-guard.mjs';

interface StrandsInputBytes {
  readonly manifestBytes?: unknown;
  readonly wasmBytes?: unknown;
  readonly tokenizerBytes?: unknown;
  readonly tokenizerConfigBytes?: unknown;
}

interface GpuDeviceLike {
  readonly lost: PromiseLike<unknown>;
  addEventListener(
    type: string,
    listener: (event: UncapturedGpuErrorEvent) => void,
    options?: AddEventListenerOptions
  ): void;
  removeEventListener(type: string, listener: (event: UncapturedGpuErrorEvent) => void): void;
  destroy(): void;
  pushErrorScope(filter: 'out-of-memory' | 'validation'): void;
  popErrorScope(): PromiseLike<unknown>;
}

interface UncapturedGpuErrorEvent {
  readonly error?: {readonly message?: string | undefined} | null | undefined;
}

interface RuntimeWebGpuDevice {
  sync(): unknown | PromiseLike<unknown>;
}

interface RuntimeTensor {
  readonly byteOffset: number;
  copyFrom(value: unknown): RuntimeTensor;
  getDataPtr(): unknown;
  toArray(): unknown;
}

interface FusedUploadContext {
  copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, byteOffset: number, byteLength: number): void;
}

interface StrandsRuntime {
  dispose(...args: unknown[]): unknown;
  endScope(): unknown;
  initWebGPU(gpu: GpuDeviceLike): unknown;
  beginScope(): unknown;
  webgpu(): RuntimeWebGpuDevice;
  cpu(): unknown;
  getGlobalFunc(name: string): () => unknown;
  readonly lib: {readonly webGPUContext: FusedUploadContext};
  empty(shape: number[], dtype: string, device: unknown): RuntimeTensor;
  createVirtualMachine(device: RuntimeWebGpuDevice): {getFunction(name: string): (...args: unknown[]) => unknown};
}

interface StrandsAdapter {
  requestDevice(descriptor: {
    readonly requiredFeatures: string[];
    readonly requiredLimits: {
      readonly maxBufferSize: number;
      readonly maxStorageBufferBindingSize: number;
      readonly maxStorageBuffersPerShaderStage: number;
    };
  }): PromiseLike<GpuDeviceLike>;
}

interface LoadOptions extends StrandsInputBytes {
  readonly base?: unknown;
  readonly createTokenizer?: unknown;
  readonly createRuntime?: unknown;
  readonly gpuApi?: unknown;
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: UploadOptions['onProgress'];
  readonly cache?: unknown;
  readonly offline?: unknown;
}

type UploadOptions = NonNullable<Parameters<typeof uploadPinnedWeights>[1]>;
type UploadTensor = NonNullable<UploadOptions['upload']>;

type FusedReadRange = NonNullable<NonNullable<Parameters<typeof uploadFusedRecord>[8]>['readRange']>;
type BivariantFusedReadRange = {
  bivarianceHack(
    base: Parameters<FusedReadRange>[0],
    spec: Parameters<FusedReadRange>[1],
    options: Parameters<FusedReadRange>[2]
  ): ReturnType<FusedReadRange>;
}['bivarianceHack'];

interface LoadDependencies {
  readonly uploadTensor?: UploadTensor | undefined;
  readonly readRange?: BivariantFusedReadRange | undefined;
  readonly finalize?: ((signal: AbortSignal) => unknown | PromiseLike<unknown>) | undefined;
  readonly preparedAdapter?: unknown;
  readonly verifyAssets?: typeof verifyPinnedBundle | undefined;
  readonly uploadWeights?: typeof uploadPinnedWeights | undefined;
  readonly retainLateDevice?: ((device: unknown) => void) | undefined;
}

type RuntimeLifecycle = ReturnType<typeof trackRuntime>;
type StrandsDisposalGuard = ReturnType<typeof guardStrandsDisposal>;

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_ASSET', message);

async function verifiedCopy(
  bytes: unknown,
  length: number,
  expected: string,
  signal: AbortSignal | null | undefined,
  name: string
): Promise<Uint8Array> {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== length) throw invalid(`Invalid ${name} size.`);
  const copy = Uint8Array.from(bytes);
  const digest = await waitForAbort(() => sha256(copy), signal);
  signal?.throwIfAborted();
  if (digest !== expected) throw invalid(`${name} hash mismatch.`);
  return copy;
}

async function gpuScope<Value>(
  gpu: GpuDeviceLike,
  device: RuntimeWebGpuDevice,
  lifecycle: RuntimeLifecycle,
  operation: () => Value | PromiseLike<Value>,
  disposalGuard: StrandsDisposalGuard
): Promise<Value> {
  const noFailure = Symbol('no failure');
  let count = 0,
    value!: Value,
    failure: unknown | typeof noFailure = noFailure,
    started = false,
    fallbackAttempted = false;
  async function fallbackSync() {
    if (!started || fallbackAttempted) return;
    fallbackAttempted = true;
    if (lifecycle.loss || lifecycle.disposed) {
      disposalGuard.retain();
      return;
    }
    // A failed sync does not prove that pending work settled. Keep the first error.
    try {
      await device.sync();
    } catch {
      disposalGuard.retain();
    }
  }
  try {
    gpu.pushErrorScope('out-of-memory');
    count++;
    gpu.pushErrorScope('validation');
    count++;
    started = true;
    value = await operation();
  } catch (cause) {
    failure = cause;
    await fallbackSync();
  }
  while (count--) {
    try {
      const error = await gpu.popErrorScope();
      if (failure === noFailure && error) failure = error;
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
    if (failure !== noFailure) await fallbackSync();
  }
  if (failure !== noFailure) throw failure;
  return value;
}

function protectRuntime(
  runtime: StrandsRuntime,
  gpu: GpuDeviceLike
): {
  readonly lifecycle: RuntimeLifecycle;
  readonly disposalGuard: StrandsDisposalGuard;
} {
  try {
    const lifecycle = trackRuntime(runtime, gpu);
    return {lifecycle, disposalGuard: guardStrandsDisposal(runtime, gpu, lifecycle)};
  } catch (error) {
    retainStrandsResources(runtime, gpu);
    throw error;
  }
}

function initializeRuntime(runtime: StrandsRuntime, gpu: GpuDeviceLike, disposalGuard: StrandsDisposalGuard): void {
  try {
    runtime.initWebGPU(gpu);
    runtime.beginScope();
  } catch (error) {
    disposalGuard.retain();
    throw error;
  }
}

function cleanupStrandsLoad({
  loadWatch,
  runtime,
  scopeOpen,
  lifecycle,
  gpu,
  disposalGuard
}: {
  readonly loadWatch?: ReturnType<typeof watchLoadDeviceLoss> | undefined;
  readonly runtime?: StrandsRuntime | undefined;
  readonly scopeOpen?: boolean | undefined;
  readonly lifecycle?: RuntimeLifecycle | undefined;
  readonly gpu?: GpuDeviceLike | undefined;
  readonly disposalGuard?: StrandsDisposalGuard | undefined;
}): void {
  if (runtime && (!disposalGuard || disposalGuard.blocked || lifecycle?.loss)) {
    loadWatch?.close();
    retainStrandsResources(runtime, gpu);
  } else {
    cleanupFailedLoad({loadWatch, runtime, scopeOpen, lifecycle, gpuDevice: gpu});
  }
}

async function verifyPinnedBundle(
  {manifestBytes, wasmBytes, tokenizerBytes, tokenizerConfigBytes}: StrandsInputBytes,
  signal: AbortSignal | null | undefined
) {
  if (!(manifestBytes instanceof Uint8Array) || manifestBytes.byteLength > 1024 * 1024)
    throw invalid('Invalid manifest bytes.');
  const manifestCopy = Uint8Array.from(manifestBytes);
  const [manifest, wasm, tokenizerJson, tokenizerConfig] = await Promise.all([
    parsePinnedManifest(manifestCopy, signal),
    verifiedCopy(wasmBytes, WASM_BYTES, WASM_SHA256, signal, 'Compiled WASM'),
    verifiedCopy(tokenizerBytes, 19989592, TOKENIZER_SHA256, signal, 'Tokenizer'),
    verifiedCopy(tokenizerConfigBytes, 1128, TOKENIZER_CONFIG_SHA256, signal, 'Tokenizer configuration')
  ]);
  return {manifest, wasm, tokenizerJson, tokenizerConfig};
}

/** Internal dependency seam for tests; never expose its second argument as user options. */
export async function loadStrandsFromBytes(
  {
    base,
    manifestBytes,
    wasmBytes,
    tokenizerBytes,
    tokenizerConfigBytes,
    createTokenizer,
    createRuntime,
    gpuApi,
    signal,
    onProgress,
    cache = false,
    offline = false
  }: LoadOptions = {},
  {
    uploadTensor,
    readRange,
    finalize,
    preparedAdapter,
    verifyAssets = verifyPinnedBundle,
    uploadWeights = uploadPinnedWeights,
    retainLateDevice = (device) => retainStrandsResources(undefined, device)
  }: LoadDependencies = {}
): Promise<ExperimentalStrandsSession> {
  if (cache !== false || offline !== false)
    throw new WebDecisionError('INVALID_INPUT', 'Experimental Strands loading does not yet support cache or offline.');
  if (
    typeof createTokenizer !== 'function' ||
    typeof createRuntime !== 'function' ||
    typeof (gpuApi as {readonly requestAdapter?: unknown} | null | undefined)?.requestAdapter !== 'function'
  )
    throw new WebDecisionError('INVALID_INPUT', 'Tokenizer, runtime and WebGPU factories are required.');
  let assetBase;
  try {
    assetBase = new URL(base as string);
  } catch {
    throw new WebDecisionError('INVALID_INPUT', 'A valid asset base URL is required.');
  }
  if (
    !['https:', 'http:'].includes(assetBase.protocol) ||
    !assetBase.pathname.endsWith('/') ||
    assetBase.search ||
    assetBase.hash ||
    assetBase.username ||
    assetBase.password
  )
    throw new WebDecisionError('INVALID_INPUT', 'The asset base must be an HTTP(S) directory URL.');
  signal?.throwIfAborted();
  const {manifest, wasm, tokenizerJson, tokenizerConfig} = await verifyAssets(
    {manifestBytes, wasmBytes, tokenizerBytes, tokenizerConfigBytes},
    signal
  );
  signal?.throwIfAborted();
  let tokenizer;
  try {
    const decoder = new TextDecoder('utf-8', {fatal: true});
    tokenizer = (createTokenizer as (tokenizerJson: unknown, tokenizerConfig: unknown) => unknown)(
      JSON.parse(decoder.decode(tokenizerJson)),
      JSON.parse(decoder.decode(tokenizerConfig))
    );
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Could not create the pinned tokenizer.', {cause});
  }
  const adapter: unknown =
    preparedAdapter === undefined ? await requestStrandsAdapter(gpuApi, signal) : preparedAdapter;
  checkFusedAdapter(adapter);
  signal?.throwIfAborted();
  let gpu: GpuDeviceLike | undefined,
    runtime: StrandsRuntime | undefined,
    lifecycle: RuntimeLifecycle | undefined,
    disposalGuard: StrandsDisposalGuard | undefined,
    loadWatch: ReturnType<typeof watchLoadDeviceLoss> | undefined,
    scopeOpen = false,
    loading = true,
    forward: (() => void) | undefined;
  const uncaptured: string[] = [],
    loadAbort = new AbortController();
  try {
    gpu = await waitForAbort(
      () =>
        (adapter as StrandsAdapter).requestDevice({
          requiredFeatures: ['shader-f16'],
          requiredLimits: {
            maxBufferSize: LARGEST_BINDING,
            maxStorageBufferBindingSize: LARGEST_BINDING,
            maxStorageBuffersPerShaderStage: 8
          }
        }),
      signal,
      (late) => {
        try {
          (late as GpuDeviceLike | null | undefined)?.destroy();
        } catch {
          retainLateDevice(late);
        }
      }
    );
    signal?.throwIfAborted();
    loadWatch = watchLoadDeviceLoss(gpu, signal);
    forward = () => loadAbort.abort(loadWatch!.signal.reason);
    loadWatch.signal.addEventListener('abort', forward, {once: true});
    if (loadWatch.signal.aborted) forward();
    gpu!.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
      if (loading && !loadAbort.signal.aborted)
        loadAbort.abort(new WebDecisionError('MODEL_LOAD_FAILED', 'WebGPU reported an uncaptured loading error.'));
    });
    // Runtime creation does not use the GPU yet. Stop waiting on abort/loss,
    // but dispose a runtime that eventually resolves after the load failed.
    runtime = await waitForAbort(
      () => (createRuntime as (wasm: Uint8Array) => StrandsRuntime | PromiseLike<StrandsRuntime>)(wasm),
      loadAbort.signal,
      (lateRuntime) => {
        try {
          (lateRuntime as StrandsRuntime | null | undefined)?.dispose();
        } catch {}
      }
    );
    ({lifecycle, disposalGuard} = protectRuntime(runtime, gpu));
    loadAbort.signal.throwIfAborted();
    initializeRuntime(runtime!, gpu!, disposalGuard!);
    scopeOpen = true;
    const device = runtime!.webgpu();
    const check = () => {
      loadAbort.signal.throwIfAborted();
      assertLoadActive(lifecycle!);
      if (uncaptured.length) throw new WebDecisionError('MODEL_LOAD_FAILED', uncaptured[0]!);
    };
    const upload: UploadTensor =
      uploadTensor ||
      ((record, tensor, uploadSignal) =>
        uploadFusedRecord(
          assetBase,
          record,
          runtime!.lib.webGPUContext,
          (tensor as RuntimeTensor).getDataPtr(),
          (tensor as RuntimeTensor).byteOffset,
          () => device.sync(),
          check,
          (work) => gpuScope(gpu!, device, lifecycle!, work, disposalGuard!),
          {signal: uploadSignal, readRange}
        ));
    const weights = await uploadWeights(manifest, {
      signal: loadAbort.signal,
      onProgress,
      allocate: (record) =>
        gpuScope(gpu!, device, lifecycle!, () => runtime!.empty(record.shape, record.dtype, device), disposalGuard!),
      upload
    });
    check();
    signal?.throwIfAborted();
    const fn = runtime!.createVirtualMachine(device).getFunction('main');
    const runner = new ExperimentalFusedRunner({
      runtime: runtime!,
      device,
      gpu: gpu!,
      fn,
      weights,
      lifecycle: lifecycle!,
      uncaptured,
      disposalGuard
    });
    const session = new ExperimentalStrandsSession({tokenizer, runner});
    if (finalize) await finalize(loadAbort.signal);
    check();
    signal?.throwIfAborted();
    loading = false;
    loadWatch.signal.removeEventListener('abort', forward);
    loadWatch.close();
    return session;
  } catch (cause) {
    loading = false;
    if (loadWatch && forward) loadWatch.signal.removeEventListener('abort', forward);
    cleanupStrandsLoad({loadWatch, runtime, scopeOpen, lifecycle, gpu, disposalGuard});
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not load experimental Strands.', {cause});
  }
}
