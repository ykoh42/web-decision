import {WebDecisionError} from '../kev.js';
import type {Qwen35PinnedSource} from './qwen35-source.mts';
import type {DLDevice, Tensor, TVMImportObject, TVMLibraryProvider, VirtualMachine} from '../runtime/tvm-types.js';
import {verifyPinnedQwen35VisionArtifacts} from './qwen35-vision-contract.mts';
import type {Qwen35VisionArtifacts} from './qwen35-vision-contract.mts';
import {createQwen35VisionFeatureRunner} from './qwen35-vision-runner.mts';
import type {Qwen35VisionFeatureRunner, Qwen35VisionImageRunner} from './qwen35-vision-runner.mts';
import {
  loadQwen35VisionParameterTensors,
  loadQwen35VisionParameterTensorsFromSource,
  type Qwen35VisionSourceWeightLoadOptions,
  type Qwen35VisionWeightLoadOptions
} from './qwen35-vision-weights.mts';

export interface Qwen35VisionRuntimeRunner extends Qwen35VisionFeatureRunner {
  /** Release the TVM instance scope and its resident vision weights. */
  dispose(): Promise<void>;
}

type Qwen35VisionImageRuntimeRunner = Qwen35VisionImageRunner & Pick<Qwen35VisionRuntimeRunner, 'dispose'>;

type VisionRuntimeRunnerOptions = Readonly<{
  readonly tvmjs: Qwen35VisionTvmJs;
  readonly gpuDevice: GPUDevice;
  readonly artifacts: Qwen35VisionArtifacts;
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((bytes: number) => void) | undefined;
  readonly logger?: ((message: string) => void) | undefined;
}>;

export type CreateQwen35VisionRuntimeRunnerOptions = VisionRuntimeRunnerOptions &
  Readonly<{source: Qwen35PinnedSource}>;

export interface Qwen35VisionRuntime {
  initWebGPU(device: GPUDevice): unknown;
  beginScope(): unknown;
  endScope(): unknown;
  dispose(): unknown;
  webgpu(): DLDevice;
  cpu(): DLDevice;
  empty(shape: number[], dtype: string, device: DLDevice): Tensor;
  createVirtualMachine(device: DLDevice): VirtualMachine;
}

export interface Qwen35VisionTvmJs {
  instantiate(
    wasm: ArrayBuffer,
    imports: TVMImportObject,
    logger?: (message: string) => void
  ): PromiseLike<Qwen35VisionRuntime>;
  createPolyfillWASI(): TVMLibraryProvider;
}

export function createQwen35VisionRuntimeRunner({
  source,
  ...options
}: CreateQwen35VisionRuntimeRunnerOptions): Promise<Qwen35VisionRuntimeRunner> {
  return openQwen35VisionRuntime(options, (weights) => loadQwen35VisionParameterTensors({...weights, source}));
}

export function createQwen35VisionRuntimeRunnerFromSource({
  readWeight,
  readAhead,
  ...options
}: VisionRuntimeRunnerOptions &
  Pick<Qwen35VisionSourceWeightLoadOptions, 'readWeight' | 'readAhead'>): Promise<Qwen35VisionImageRuntimeRunner> {
  return openQwen35VisionRuntime(options, (weights) =>
    loadQwen35VisionParameterTensorsFromSource({...weights, readWeight, readAhead})
  );
}

/** Instantiate the pinned vision graph and take ownership of its dedicated GPU device. */
async function openQwen35VisionRuntime(
  {tvmjs, gpuDevice, artifacts, signal, onProgress, logger = () => {}}: VisionRuntimeRunnerOptions,
  loadParameters: (options: Omit<Qwen35VisionWeightLoadOptions, 'source'>) => Promise<ReadonlyMap<string, Tensor>>
): Promise<Qwen35VisionImageRuntimeRunner> {
  let runtime: Qwen35VisionRuntime | undefined;
  let device: DLDevice | undefined;
  let ownerScopeOpen = false;
  try {
    await verifyPinnedQwen35VisionArtifacts(artifacts);
    signal?.throwIfAborted();
    const wasm = new ArrayBuffer(artifacts.wasmBytes.byteLength);
    new Uint8Array(wasm).set(artifacts.wasmBytes);
    runtime = await tvmjs.instantiate(wasm, tvmjs.createPolyfillWASI(), logger);
    runtime.initWebGPU(gpuDevice);
    runtime.beginScope();
    ownerScopeOpen = true;
    device = runtime.webgpu();

    const parameterTensors = await loadParameters({
      runtime,
      device,
      artifact: artifacts.artifact,
      bindings: artifacts.bindings,
      signal,
      onProgress
    });
    signal?.throwIfAborted();
    const featureRunner = createQwen35VisionFeatureRunner({
      runtime,
      artifact: artifacts.artifact,
      parameterTensors
    });

    let disposed = false;
    let queue: Promise<unknown> = Promise.resolve();
    let disposal: Promise<void> | undefined;
    function run(input: Parameters<Qwen35VisionImageRunner['run']>[0], runSignal?: AbortSignal | null) {
      if (disposed)
        return Promise.reject(new WebDecisionError('SESSION_CLOSED', 'Qwen3.5 vision runtime is disposed.'));
      const task = queue.then(() => featureRunner.run(input, runSignal));
      queue = task.catch(() => {});
      return task;
    }

    function dispose(): Promise<void> {
      if (disposal) return disposal;
      disposed = true;
      disposal = queue.then(async () => {
        let failure: unknown;
        try {
          await device!.sync();
        } catch (cause) {
          failure = cause;
        }
        if (ownerScopeOpen) {
          ownerScopeOpen = false;
          try {
            runtime!.endScope();
          } catch (cause) {
            if (failure === undefined) failure = cause;
          }
        }
        try {
          runtime!.dispose();
        } catch (cause) {
          if (failure === undefined) failure = cause;
        }
        try {
          gpuDevice.destroy();
        } catch (cause) {
          if (failure === undefined) failure = cause;
        }
        if (failure !== undefined) throw failure;
      });
      return disposal;
    }

    return Object.freeze({run, dispose});
  } catch (cause) {
    if (runtime) {
      if (ownerScopeOpen) {
        try {
          runtime.endScope();
        } catch {}
      }
      try {
        runtime.dispose();
      } catch {}
    }
    try {
      gpuDevice.destroy();
    } catch {}
    throw cause;
  }
}
