import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {createGemma4V2Runner} from '../../packages/web-decision/src/strands/gemma4-v2-runner.mts';
import {transposedSplitGraphs} from '../../packages/web-decision/src/strands/gemma4-v2-split-adapter.mts';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};

declare global {
  interface Window {
    runOptimizedFull: () => Promise<unknown>;
  }
}
async function data(name: string) {
  const response = await fetch(`/strands-v2-optimized-full-assets/${name}`);
  if (!response.ok) throw new Error(`Missing ${name}`);
  return response.arrayBuffer();
}
window.runOptimizedFull = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical GPU required');
  const gpu = await adapter.requestDevice();
  const runtime = await instantiate(await data('linked.wasm'), createPolyfillWASI());
  const source = await openGemma4V2SourceFromInventory(
    inventory as never,
    'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
    new URL('/strands-v2-optimized-full-source/', location.href)
  );
  try {
    runtime.initWebGPU(gpu);
    const device = runtime.webgpu();
    runtime.beginScope();
    try {
      const underlyingVm = runtime.createVirtualMachine(device);
      let layers = 0;
      const vm = {
        getFunction(name: string) {
          const fn = underlyingVm.getFunction(name);
          if (typeof fn !== 'function') return fn;
          return (...args: unknown[]) => {
            if (/^split(?:0|4|15|19)attn_packed_main$/u.test(name)) layers++;
            return fn(...args);
          };
        }
      };
      const base = {
        runtime: runtime as never,
        device,
        vm,
        source,
        insertImages: async (word: never, input: {images: unknown[]}) => {
          if (input.images.length) throw new Error('text only');
          return word;
        },
        readout: new Uint8Array(await data('readout.bin')),
        localFrequency: new Float32Array(await data('local.bin')),
        globalFrequency: new Float32Array(await data('global.bin')),
        dispose() {}
      };
      async function run(optimized: boolean) {
        const runner = await createGemma4V2Runner({
          ...base,
          ...(optimized ? {splitExecution: {graphs: transposedSplitGraphs(), chunkTokens: 4}} : {})
        } as never);
        try {
          return Array.from(
            await runner.run({
              stateIds: [2, 3, 4, 5, 6, 7, 8, 9],
              questionIds: [10, 11, 12, 13, 14],
              images: [],
              imageSpans: []
            })
          );
        } finally {
          await runner.dispose();
        }
      }
      gpu.pushErrorScope('validation');
      const baseline = await run(false);
      const candidate = await run(true);
      const validation = await gpu.popErrorScope();
      if (validation) throw new Error(validation.message);
      let maxError = 0,
        sum = 0;
      for (let i = 0; i < baseline.length; i++) {
        const diff = Math.abs(baseline[i]! - candidate[i]!);
        if (!Number.isFinite(diff)) throw new Error('non-finite');
        maxError = Math.max(maxError, diff);
        sum += diff;
      }
      return {tokens: 13, chunk: 4, layers, logits: baseline.length, maxError, meanError: sum / baseline.length};
    } finally {
      runtime.endScope();
    }
  } finally {
    source.dispose();
    runtime.dispose();
    gpu.destroy();
  }
};
