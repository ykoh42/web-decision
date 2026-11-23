import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {createGemma4V2ImageInserter} from '../../packages/web-decision/src/strands/gemma4-v2-image-runner.mts';
import {loadGemma4V2VisionWeights} from '../../packages/web-decision/src/strands/gemma4-v2-vision-bindings.mts';
import {loadGemma4V2Model} from '../../packages/web-decision/src/strands/gemma4-v2-loader.mts';
import {createGemma4V2Runner} from '../../packages/web-decision/src/strands/gemma4-v2-runner.mts';
import {transposedSplitGraphs} from '../../packages/web-decision/src/strands/gemma4-v2-split-adapter.mts';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};

type Request = Parameters<Awaited<ReturnType<typeof loadGemma4V2Model>>['systemOne']>[0];
type Snapshot = {count: number; bytes: number};
declare global {
  interface Window {
    runGemma4ImageLifecycle: (request: Request, split?: boolean) => Promise<unknown>;
  }
}

function trackStorage(device: GPUDevice) {
  const live = new Map<GPUBuffer, number>();
  const original = device.createBuffer.bind(device);
  Object.defineProperty(device, 'createBuffer', {
    configurable: true,
    value(descriptor: GPUBufferDescriptor) {
      const buffer = original(descriptor);
      // WebGPU's STORAGE usage bit; exclude persistent uniform and readback pools.
      if (descriptor.usage & 0x80) {
        live.set(buffer, Number(descriptor.size));
        const destroy = buffer.destroy.bind(buffer);
        Object.defineProperty(buffer, 'destroy', {
          configurable: true,
          value() {
            live.delete(buffer);
            destroy();
          }
        });
      }
      return buffer;
    }
  });
  return (): Snapshot => ({count: live.size, bytes: [...live.values()].reduce((sum, value) => sum + value, 0)});
}

async function data(name: string) {
  const response = await fetch('/strands-v2-lifecycle-assets/' + name);
  if (!response.ok) throw new Error(`Missing lifecycle graph ${name}`);
  return response.arrayBuffer();
}

window.runGemma4ImageLifecycle = async (request, split = false) => {
  if (!request.images || request.images.length < 2) throw new Error('At least two real images are required.');
  let armed: AbortController | undefined;
  let imageSyncs = 0;
  let snapshot: (() => Snapshot) | undefined;
  let gpu: GPUDevice | undefined;
  let visionLoads = 0;
  let sourceReleases = 0;
  let splitLayers = 0;
  let cancellingDecoder = false;
  const openSource = (options: Parameters<typeof openGemma4V2SourceFromInventory>[3]) =>
    openGemma4V2SourceFromInventory(
      inventory as never,
      'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
      new URL('/strands-v2-lifecycle-source/', location.href),
      options
    );
  const session = await loadGemma4V2Model(
    {cache: null, offline: false},
    {
      openSource,
      async createRunner({source, readout}) {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical WebGPU adapter required');
        gpu = await adapter.requestDevice();
        gpu.pushErrorScope('validation');
        snapshot = trackStorage(gpu);
        const runtime = await instantiate(await data('linked.wasm'), createPolyfillWASI());
        let scoped = false;
        try {
          runtime.initWebGPU(gpu);
          runtime.beginScope();
          scoped = true;
          const device = runtime.webgpu();
          const originalSync = device.sync.bind(device);
          Object.defineProperty(device, 'sync', {
            configurable: true,
            value: async () => {
              await originalSync();
              if (armed && (!split || cancellingDecoder) && ++imageSyncs === 2) {
                const controller = armed;
                armed = undefined;
                controller.abort();
              }
            }
          });
          const actualVm = runtime.createVirtualMachine(device);
          const vm = {
            getFunction(name: string) {
              const fn = actualVm.getFunction(name);
              if (typeof fn !== 'function') return fn;
              return (...args: unknown[]) => {
                if (/^split(?:0|4|15|19)attn_packed_main$/u.test(name)) {
                  splitLayers++;
                  if (armed) cancellingDecoder = true;
                }
                return fn(...args);
              };
            }
          };
          const trackedSource = {
            ...source,
            releaseCachedMemory() {
              sourceReleases++;
              return source.releaseCachedMemory();
            }
          };
          const insertImages = await createGemma4V2ImageInserter(
            runtime,
            device,
            vm,
            trackedSource,
            undefined,
            async (...args) => {
              visionLoads++;
              return loadGemma4V2VisionWeights(...args);
            }
          );
          return await createGemma4V2Runner({
            runtime: runtime as never,
            device,
            vm,
            source: trackedSource,
            readout,
            insertImages,
            localFrequency: new Float32Array(await data('local.bin')),
            globalFrequency: new Float32Array(await data('global.bin')),
            ...(split ? {splitExecution: {graphs: transposedSplitGraphs(), chunkTokens: 64}} : {}),
            dispose() {
              try {
                runtime.endScope();
              } finally {
                runtime.dispose();
              }
            }
          });
        } catch (error) {
          if (scoped) runtime.endScope();
          runtime.dispose();
          throw error;
        }
      }
    }
  );
  try {
    if (!snapshot || !gpu) throw new Error('GPU tracking was not initialized.');
    const initial = snapshot();
    const first = await session.systemOne(request);
    const firstLayers = splitLayers;
    const afterFirst = snapshot();
    const firstLoads = visionLoads;
    await session.releaseCachedMemory();
    const afterRelease = snapshot();
    const releasesAfterFirst = sourceReleases;
    const controller = new AbortController();
    armed = controller;
    imageSyncs = 0;
    cancellingDecoder = false;
    let cancelled = false;
    try {
      await session.systemOne(request, {signal: controller.signal});
    } catch (error) {
      if (!(error instanceof DOMException) || error.name !== 'AbortError') throw error;
      cancelled = true;
    }
    const afterCancel = snapshot();
    const cancelLayers = splitLayers;
    const cancelLoads = visionLoads;
    await session.releaseCachedMemory();
    const afterCancelRelease = snapshot();
    const retry = await session.systemOne(request);
    const retryLayers = splitLayers;
    const afterRetry = snapshot();
    const retryLoads = visionLoads;
    await session.releaseCachedMemory();
    const afterFinalRelease = snapshot();
    const totalReleases = sourceReleases;
    const error = await gpu.popErrorScope();
    if (error) throw new Error(error.message);
    await session.dispose();
    const afterDispose = snapshot();
    return {
      first,
      retry,
      cancelled,
      imageSyncs,
      firstLayers,
      cancelLayers,
      retryLayers,
      initial,
      afterFirst,
      afterRelease,
      afterCancel,
      afterCancelRelease,
      afterRetry,
      afterFinalRelease,
      afterDispose,
      firstLoads,
      cancelLoads,
      retryLoads,
      releasesAfterFirst,
      totalReleases
    };
  } finally {
    await session.dispose();
  }
};
