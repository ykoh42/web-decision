import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {createGemma4V2ImagePreparer} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';
import {createGemma4V2ImageInserter} from '../../packages/web-decision/src/strands/gemma4-v2-image-runner.mts';
import {createGemma4V2Runner} from '../../packages/web-decision/src/strands/gemma4-v2-runner.mts';
import {transposedSplitGraphs} from '../../packages/web-decision/src/strands/gemma4-v2-split-adapter.mts';
import {finishGemma4V2Answer} from '../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import type {Tensor} from '../../packages/web-decision/src/runtime/tvm-types.js';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};

type Input = {state: number[]; questions: number[][]; images: {start: number; end: number}[]};
declare global {
  interface Window {
    runGemma4V2TwoImageE2E: (input: Input, split?: boolean) => Promise<unknown>;
  }
}
async function data(name: string) {
  const response = await fetch('/strands-v2-image-e2e-assets/' + name);
  if (!response.ok) throw new Error(`Missing image runner fixture ${name}`);
  return response.arrayBuffer();
}
function sampleImage(reverse: boolean) {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas is unavailable');
  context.fillStyle = reverse ? 'rgb(15,40,210)' : 'rgb(220,20,30)';
  context.fillRect(0, 0, 32, 64);
  context.fillStyle = reverse ? 'rgb(220,20,30)' : 'rgb(15,40,210)';
  context.fillRect(32, 0, 32, 64);
  return canvas.toDataURL('image/png').split(',')[1]!;
}
window.runGemma4V2TwoImageE2E = async (input, split = false) => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical WebGPU adapter required');
  const gpu = await adapter.requestDevice();
  gpu.pushErrorScope('validation');
  const runtime = await instantiate(await data('linked.wasm'), createPolyfillWASI());
  const source = await openGemma4V2SourceFromInventory(
    inventory as never,
    'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
    new URL('/strands-v2-image-e2e-source/', location.href)
  );
  let scoped = false;
  try {
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scoped = true;
    const device = runtime.webgpu();
    const vm = runtime.createVirtualMachine(device);
    const debugHosts: Array<{name: string; tensor: Tensor & {dispose(): void}}> = [];
    const scopedRuntime = runtime as typeof runtime & {detachFromCurrentScope<T>(value: T): T};
    const capture = (name: string, value: Tensor) => {
      const host = runtime.empty(value.shape, 'float32', runtime.cpu()).copyFrom(value) as Tensor & {dispose(): void};
      scopedRuntime.detachFromCurrentScope(host);
      debugHosts.push({name, tensor: host});
    };
    let layerIndex = 0;
    let pleIndex = 0;
    const debugVm = {
      getFunction(name: string) {
        const func = vm.getFunction(name);
        if (typeof func !== 'function') return func;
        return (...args: unknown[]) => {
          const isLayer = split ? /^split(?:0|4|15|19)attn_packed_main$/u.test(name) : name.startsWith('layer');
          const index = isLayer ? layerIndex++ : -1;
          const relative = index - 35;
          if (!split && [0, 4, 34].includes(relative)) capture(`layer${relative}-hidden`, args[0] as Tensor);
          const result = func(...args);
          if (!split && name === 'ple_packed_main') {
            const relativePle = pleIndex++ - 35;
            if ([0, 4, 34].includes(relativePle)) capture(`ple${relativePle}`, result as Tensor);
          }
          if (!split && [0, 4, 34].includes(relative)) {
            const value = relative < 15 ? (result as {get(index: number): Tensor}).get(0) : (result as Tensor);
            capture(`layer${relative}-output`, value);
          }
          return result;
        };
      }
    };
    const preparer = createGemma4V2ImagePreparer();
    const images = [await preparer.prepareImage(sampleImage(false)), await preparer.prepareImage(sampleImage(true))];
    const insertImages = await createGemma4V2ImageInserter(runtime, device, vm, source);
    const runner = await createGemma4V2Runner({
      runtime: runtime as never,
      device,
      vm: debugVm,
      source,
      insertImages,
      readout: new Uint8Array(await data('readout.bin')),
      localFrequency: new Float32Array(await data('local.bin')),
      globalFrequency: new Float32Array(await data('global.bin')),
      ...(split ? {splitExecution: {graphs: transposedSplitGraphs(), chunkTokens: 64}} : {}),
      dispose() {}
    });
    try {
      const results = [];
      for (const questionIds of input.questions) {
        const logits = await runner.run({
          stateIds: input.state,
          questionIds,
          images: images.map((item) => item.value),
          imageSpans: input.images
        });
        results.push({
          logits: Array.from(logits),
          answer: finishGemma4V2Answer(
            {kind: 'choice', options: ['red', 'blue'], descriptions: ['red', 'blue']},
            logits,
            {temperature: 1.5456, softcap: 30}
          )
        });
      }
      const error = await gpu.popErrorScope();
      if (error) throw new Error(error.message);
      const differences: Record<string, {max: number; mean: number; rms: number}> = {};
      for (const {name, tensor} of debugHosts) {
        const reference = new Float32Array(await data(`debug-${name}.bin`));
        const actual = tensor.toArray() as Float32Array;
        if (reference.length !== actual.length) throw new Error(`Bad debug shape: ${name}`);
        let max = 0;
        let sum = 0;
        let squared = 0;
        for (let i = 0; i < reference.length; i++) {
          const difference = Math.abs(reference[i]! - actual[i]!);
          max = Math.max(max, difference);
          sum += difference;
          squared += difference * difference;
        }
        differences[name] = {max, mean: sum / reference.length, rms: Math.sqrt(squared / reference.length)};
        tensor.dispose();
      }
      return {tokens: images.map((image) => image.tokens), results, differences, layers: layerIndex};
    } finally {
      await runner.dispose();
    }
  } finally {
    if (scoped) runtime.endScope();
    runtime.dispose();
    gpu.destroy();
    source.dispose();
  }
};
