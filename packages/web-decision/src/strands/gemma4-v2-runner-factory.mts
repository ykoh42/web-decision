import {readVerifiedAsset, type AssetCache, type AssetFileMetadata} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {createPolyfillWASI, instantiate} from '../runtime/tvm.generated.mjs';
import type {DLDevice, Instance, Tensor, VirtualMachine} from '../runtime/tvm-types.js';
import {createGemma4V2Runner, type Gemma4V2SplitExecution} from './gemma4-v2-runner.mjs';
import {
  disposeGemma4V2Runtime,
  guardGemma4V2Runner,
  watchGemma4V2Device,
  withGemma4V2GpuErrors
} from './gemma4-v2-gpu-errors.mjs';
import type {Gemma4V2RunnerFactory} from './gemma4-v2-loader.mjs';
import type {Gemma4V2RunnerInput} from './gemma4-v2-session.mjs';

export type Gemma4V2GraphAsset = Readonly<{base: URL; name: string; metadata: AssetFileMetadata}>;
export interface Gemma4V2GraphAssets {
  readonly wasm: Gemma4V2GraphAsset;
  readonly localFrequency: Gemma4V2GraphAsset;
  readonly globalFrequency: Gemma4V2GraphAsset;
  readonly hostReadout?: true;
}
export interface Gemma4V2RunnerFactoryOptions {
  readonly assets: Gemma4V2GraphAssets;
  /** Internal: supplied only when assets.wasm contains the matching verified graphs. */
  readonly splitExecution?: Gemma4V2SplitExecution;
  readonly createImageInserter: (
    input: Readonly<{
      runtime: Instance;
      device: DLDevice;
      vm: VirtualMachine;
      source: Parameters<Gemma4V2RunnerFactory>[0]['source'];
      signal?: AbortSignal | null;
    }>
  ) => Promise<
    ((word: Tensor, input: Gemma4V2RunnerInput, signal?: AbortSignal | null) => Promise<Tensor>) & {
      releaseCachedMemory?: () => void;
    }
  >;
}

async function read(
  asset: Gemma4V2GraphAsset,
  cache: AssetCache | null | undefined,
  offline: boolean,
  signal?: AbortSignal | null
) {
  return readVerifiedAsset(asset.base, {files: {[asset.name]: asset.metadata}}, asset.name, {cache, offline, signal});
}

/** Connect verified graph artifacts and the pinned checkpoint to the concrete TVM runner. */
export function createGemma4V2RunnerFactory(options: Gemma4V2RunnerFactoryOptions): Gemma4V2RunnerFactory {
  return async ({source, readout, cache, offline, signal}) => {
    signal?.throwIfAborted();
    const gpuApi = typeof navigator === 'undefined' ? undefined : navigator.gpu;
    if (!gpuApi) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'WebGPU is unavailable.');
    const [wasm, local, global] = await Promise.all([
      read(options.assets.wasm, cache, offline, signal),
      read(options.assets.localFrequency, cache, offline, signal),
      read(options.assets.globalFrequency, cache, offline, signal)
    ]);
    signal?.throwIfAborted();
    const adapter = await gpuApi.requestAdapter({powerPreference: 'high-performance'});
    signal?.throwIfAborted();
    if (!adapter) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'No WebGPU adapter is available.');
    const limits = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
        maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup
      }
    });
    const lifecycle = watchGemma4V2Device(device);
    const assertDeviceActive = () => {
      if (lifecycle.lost)
        throw new WebDecisionError('MODEL_LOAD_FAILED', 'The GPU device was lost while loading.', {
          cause: lifecycle.cause
        });
    };
    let runtime: Awaited<ReturnType<typeof instantiate>> | undefined;
    let scoped = false;
    let gpuInitialized = false;
    try {
      signal?.throwIfAborted();
      runtime = await instantiate(wasm, createPolyfillWASI());
      signal?.throwIfAborted();
      assertDeviceActive();
      runtime.initWebGPU(device);
      gpuInitialized = true;
      runtime.beginScope();
      scoped = true;
      const tvmDevice = runtime.webgpu();
      const vm = runtime.createVirtualMachine(tvmDevice);
      const insertImages = await withGemma4V2GpuErrors(device, () =>
        options.createImageInserter({
          runtime: runtime!,
          device: tvmDevice,
          vm,
          source,
          ...(signal === undefined ? {} : {signal})
        })
      );
      signal?.throwIfAborted();
      assertDeviceActive();
      const owned = runtime;
      const runner = await createGemma4V2Runner({
        runtime: owned as Parameters<typeof createGemma4V2Runner>[0]['runtime'],
        device: tvmDevice,
        vm,
        source,
        readout,
        ...(options.assets.hostReadout ? {hostReadout: true} : {}),
        localFrequency: new Float32Array(local),
        globalFrequency: new Float32Array(global),
        ...(options.splitExecution ? {splitExecution: options.splitExecution} : {}),
        insertImages,
        dispose() {
          lifecycle.close();
          disposeGemma4V2Runtime(owned, device, true, lifecycle.lost);
        }
      });
      signal?.throwIfAborted();
      assertDeviceActive();
      return guardGemma4V2Runner(device, lifecycle, runner);
    } catch (error) {
      lifecycle.close();
      try {
        const alreadyDisposed = gpuInitialized && lifecycle.lost;
        disposeGemma4V2Runtime(runtime, device, scoped, alreadyDisposed);
      } catch {
        // Preserve the original load failure if a lost device also breaks cleanup.
      }
      throw error;
    }
  };
}
