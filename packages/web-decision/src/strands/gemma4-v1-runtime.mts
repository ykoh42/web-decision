import {openAssetCache, readVerifiedAsset, type AssetFileManifest} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {ModelProgress} from '../curated/types.js';
import {createPolyfillWASI, instantiate} from '../runtime/tvm.generated.mjs';
import type {DLDevice, Instance, Tensor, VirtualMachine} from '../runtime/tvm-types.js';
import {withGemma4V2Cleanup} from './gemma4-v2-cleanup.mjs';
import {withGemma4V2GpuErrors, watchGemma4V2Device} from './gemma4-v2-gpu-errors.mjs';
import {
  createGemma4V1Decoder,
  type Gemma4V1Graph,
  type Gemma4V1HostTensor,
  type Gemma4V1Stages
} from './gemma4-v1-decoder.mjs';
import {STRANDS_GEMMA4_MAX_LENGTH} from './gemma4-protocol.mjs';
import {
  createGemma4V1Session,
  resolveGemma4V1ServingOptions,
  type Gemma4V1ServingOptions
} from './gemma4-v1-session.mjs';
import {createGemma4V1Source, type Gemma4V1Inventory} from './gemma4-v1-source.mjs';
import {Tokenizer} from '@huggingface/tokenizers';

export interface Options extends Gemma4V1ServingOptions {
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: ModelProgress) => void;
}

function outputs(result: unknown) {
  if (result && typeof result === 'object' && 'get' in result && 'size' in result) {
    const values = result as {size(): number; get(index: number): Tensor};
    return Array.from({length: values.size()}, (_, index) => values.get(index));
  }
  return [result as Tensor];
}

export function createGemma4V1TvmStages(
  runtime: Instance,
  device: DLDevice,
  source: Pick<ReturnType<typeof createGemma4V1Source>, 'tensor'>,
  gpu?: GPUDevice
): Gemma4V1Stages {
  const lifecycle = gpu ? watchGemma4V2Device(gpu) : undefined;
  let vm: VirtualMachine | undefined;
  function getVm() {
    if (vm) return vm;
    runtime.beginScope();
    try {
      vm = runtime.createVirtualMachine(device);
    } catch (error) {
      runtime.endScope();
      throw error;
    }
    return vm;
  }
  async function closeScopes(release: () => void) {
    if (lifecycle?.lost) return;
    try {
      await device.sync();
    } finally {
      if (!lifecycle?.lost) release();
    }
  }

  async function releaseCachedMemory() {
    if (vm && !lifecycle?.lost) {
      await device.sync();
      runtime.endScope();
    }
    vm = undefined;
  }

  return {
    async withWeights(weights, caller, operation) {
      const signal = lifecycle ? AbortSignal.any([lifecycle.signal, caller]) : caller;
      signal.throwIfAborted();
      const machine = getVm();
      const execute = async () => {
        const resident = new Map<Gemma4V1HostTensor, Tensor>();
        let residentScope = false;
        const releaseResident = () => {
          if (!residentScope) return;
          residentScope = false;
          resident.clear();
          runtime.endScope();
        };

        runtime.beginScope();
        return withGemma4V2Cleanup(
          async () => {
            const params = new Map<string, Tensor[]>();
            for (const [name, bindings] of Object.entries(weights)) {
              const data = [];
              for (const binding of bindings) {
                const values = await source.tensor(binding, signal);
                signal.throwIfAborted();
                data.push(runtime.empty([...binding.shape], binding.dtype, device).copyFrom(values));
              }
              params.set(name, data);
            }
            const upload = ({shape, data}: Gemma4V1HostTensor) =>
              runtime.empty(shape, 'float32', device).copyFrom(data);
            return operation(async (name, inputs, releaseReusedInputs) => {
              signal.throwIfAborted();
              // Open one K/V scope inside the weight scope. Each tile's own
              // allocations live in its nested call scope.
              for (const input of inputs) {
                if (!input.reuseWithinLayer || resident.has(input)) continue;
                if (!residentScope) {
                  runtime.beginScope();
                  residentScope = true;
                }
                resident.set(input, upload(input));
              }
              runtime.beginScope();
              const output = await withGemma4V2Cleanup(
                async () => {
                  const active = inputs.map((input) => resident.get(input) ?? upload(input));
                  const values = outputs(machine.getFunction(name)(...active, ...(params.get(name) ?? [])));
                  const host: Gemma4V1HostTensor[] = [];
                  for (const tensor of values) {
                    if (tensor.dtype !== 'float32') throw new Error('Gemma 4 v1 graph returned a non-float32 tensor.');
                    const cpu = runtime.empty(tensor.shape, 'float32', runtime.cpu()).copyFrom(tensor);
                    await device.sync();
                    signal.throwIfAborted();
                    host.push({shape: tensor.shape, data: (cpu.toArray() as Float32Array).slice()});
                  }
                  return host;
                },
                () => closeScopes(() => runtime.endScope())
              );
              if (releaseReusedInputs && !lifecycle?.lost) releaseResident();
              return output;
            });
          },
          () =>
            closeScopes(() => {
              try {
                releaseResident();
              } finally {
                runtime.endScope();
              }
            })
        );
      };
      return gpu ? withGemma4V2GpuErrors(gpu, execute) : execute();
    },
    releaseCachedMemory,
    async dispose() {
      try {
        if (!lifecycle?.lost) {
          try {
            await releaseCachedMemory();
          } finally {
            runtime.dispose();
          }
        }
      } finally {
        lifecycle?.close();
        gpu?.destroy();
      }
    }
  };
}

function gemma4V1AllocationBytes(tokens: number): number {
  // Static projection [12288, 1536], one K/V head [tokens, 512], and
  // a score tile [8, 64, tokens] bound the largest FP32 allocation.
  const elements = Math.max(12288 * 1536, tokens * 512);
  return Math.ceil((elements * Float32Array.BYTES_PER_ELEMENT) / 4096) * 4096;
}

export function checkGemma4V1Buffer(
  limits: Pick<GPUSupportedLimits, 'maxBufferSize' | 'maxStorageBufferBindingSize'>,
  tokens: number
): void {
  const bytes = gemma4V1AllocationBytes(tokens);
  if (bytes > 0x7fffffff)
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Gemma 4 v1 requires a ${bytes}-byte allocation for ${tokens} tokens. The pinned TVM WebGPU runtime uses signed 32-bit allocation sizes.`
    );
  const {maxBufferSize, maxStorageBufferBindingSize} = limits;
  if (bytes > Math.min(maxBufferSize, maxStorageBufferBindingSize))
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Gemma 4 v1 requires a ${bytes}-byte WebGPU buffer and storage binding for ${tokens} tokens; this device provides ${maxBufferSize} and ${maxStorageBufferBindingSize} bytes.`
    );
}

async function requestGpu(maxLength: number) {
  const adapter = await globalThis.navigator?.gpu?.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw new WebDecisionError('DEVICE_RESOURCE_LIMIT', 'A WebGPU adapter is unavailable.');
  checkGemma4V1Buffer(adapter.limits, STRANDS_GEMMA4_MAX_LENGTH);
  const bytes = gemma4V1AllocationBytes(maxLength);
  try {
    return await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: Math.min(bytes, adapter.limits.maxBufferSize),
        maxStorageBufferBindingSize: Math.min(bytes, adapter.limits.maxStorageBufferBindingSize)
      }
    });
  } catch (cause) {
    throw new WebDecisionError('DEVICE_RESOURCE_LIMIT', 'Could not acquire the required WebGPU device.', {cause});
  }
}

const runtimeFiles = [
  'graph.wasm',
  'graph.json',
  'source-inventory.json',
  'tokenizer.json',
  'tokenizer_config.json'
] as const;
export type RuntimeAssetUrls = Readonly<Record<(typeof runtimeFiles)[number], URL>>;

export async function loadGemma4V1(urls: RuntimeAssetUrls, pins: AssetFileManifest, options: Options = {}) {
  const {signal, offline = false} = options;
  const policy = resolveGemma4V1ServingOptions(options);
  signal?.throwIfAborted();
  if (offline && options.cache === false)
    throw new WebDecisionError('ASSET_LOAD_FAILED', 'Offline Gemma 4 v1 requires a cache.');
  const cache = await openAssetCache(options.cache, signal);
  const totalBytes = runtimeFiles.reduce((total, name) => total + pins.files[name]!.bytes, 0);
  let loadedBytes = 0;
  const load = (
    name: (typeof runtimeFiles)[number],
    offline: boolean,
    signal: AbortSignal | undefined,
    onBytes?: (bytes: number) => void
  ) => {
    const url = urls[name];
    const file = url.pathname.slice(url.pathname.lastIndexOf('/') + 1);
    return readVerifiedAsset(new URL('.', url), {files: {[file]: pins.files[name]!}}, file, {
      cache,
      offline,
      signal,
      ...(onBytes ? {onBytes} : {})
    });
  };
  const onBytes = options.onProgress
    ? (bytes: number) => {
        loadedBytes += bytes;
        options.onProgress!({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes, phase: 'bundled-assets'});
      }
    : undefined;
  const [wasm, graphBytes, inventoryBytes, tokenBytes, tokenizerConfig] = await Promise.all(
    runtimeFiles.map((name) => load(name, offline, signal, onBytes))
  );
  async function verifyCachedFiles(caller: AbortSignal) {
    for (const name of runtimeFiles) await load(name, true, caller);
  }

  async function ensureCachedFiles(caller: AbortSignal) {
    if (!cache) throw new WebDecisionError('ASSET_LOAD_FAILED', 'Gemma 4 v1 requires a cache for offline preparation.');
    if (!offline) for (const name of runtimeFiles) await load(name, false, caller);
    await verifyCachedFiles(caller);
  }
  const text = <T,>(bytes: ArrayBuffer) => JSON.parse(new TextDecoder().decode(bytes)) as T;
  const graph = text<Gemma4V1Graph>(graphBytes!);
  const {maxLength} = policy;
  if (maxLength > graph.maxLength)
    throw new WebDecisionError(
      'INVALID_INPUT',
      `maxLength must be from 1 to the pinned ${graph.maxLength}-position model capacity.`
    );
  const inventory = text<Gemma4V1Inventory>(inventoryBytes!);
  const bindings = [...graph.final, ...graph.ple, ...graph.layers.flatMap(({project, tail}) => [...project, ...tail])];
  const source = createGemma4V1Source(inventory, bindings, {cache, offline});
  if (offline && !(await source.isOfflineReady(signal)))
    throw new WebDecisionError('ASSET_LOAD_FAILED', 'Gemma 4 v1 offline weights are incomplete.');
  const tokenizer = new Tokenizer(text(tokenBytes!), text(tokenizerConfig!));
  signal?.throwIfAborted();
  const gpu = await requestGpu(maxLength);
  let runtime: Instance | undefined;
  try {
    signal?.throwIfAborted();
    runtime = await instantiate(wasm!, createPolyfillWASI());
    runtime.initWebGPU(gpu);
    const stages = createGemma4V1TvmStages(runtime, runtime.webgpu(), source, gpu);
    const decoder = createGemma4V1Decoder(graph, stages, source, (tokens) => checkGemma4V1Buffer(gpu.limits, tokens));
    const session = createGemma4V1Session(tokenizer, decoder, policy);
    signal?.throwIfAborted();
    const closed = new AbortController();
    const combined = (caller?: AbortSignal) => (caller ? AbortSignal.any([closed.signal, caller]) : closed.signal);
    return {
      ...session,
      async isOfflineReady(caller?: AbortSignal) {
        const signal = combined(caller);
        if (!(await source.isOfflineReady(signal))) return false;
        try {
          await verifyCachedFiles(signal);
          return true;
        } catch {
          signal.throwIfAborted();
          return false;
        }
      },
      async prepareOffline(caller?: AbortSignal) {
        const signal = combined(caller);
        await ensureCachedFiles(signal);
        await source.prepareOffline(signal);
      },
      dispose() {
        closed.abort(new WebDecisionError('SESSION_CLOSED', 'The Strands Gemma 4 v1 model has been disposed.'));
        return session.dispose();
      }
    };
  } catch (error) {
    try {
      runtime?.dispose();
    } finally {
      gpu.destroy();
    }
    throw error;
  }
}
