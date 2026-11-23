import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {createGemma4V2ImagePreparer} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';
import {insertGemma4V2VisionEmbedding} from '../../packages/web-decision/src/strands/gemma4-v2-vision.mts';
import type {Tensor} from '../../packages/web-decision/src/runtime/tvm-types.js';

type ImageResult = {width: number; height: number; tokens: number; pixelsSha256: string; positionsSha256: string};
type Insertion = {name: string; n: number; spans: number[][]};
declare global {
  interface Window {
    gemma4ImageMatrix: (inputs: string[]) => Promise<ImageResult[]>;
    gemma4InvalidImages: (inputs: string[]) => Promise<boolean[]>;
    gemma4ImageInsertions: (cases: Insertion[]) => Promise<string[]>;
  }
}

async function hash(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', copy)), (value) =>
    value.toString(16).padStart(2, '0')
  ).join('');
}

window.gemma4ImageMatrix = async (inputs) => {
  const preparer = createGemma4V2ImagePreparer();
  const output: ImageResult[] = [];
  for (const input of inputs) {
    const {tokens, value} = await preparer.prepareImage(input);
    output.push({
      width: value.width,
      height: value.height,
      tokens,
      pixelsSha256: await hash(new Uint8Array(value.pixels.buffer)),
      positionsSha256: await hash(new Uint8Array(value.positions.buffer))
    });
  }
  return output;
};

window.gemma4InvalidImages = async (inputs) => {
  const preparer = createGemma4V2ImagePreparer();
  const output: boolean[] = [];
  for (const input of inputs) {
    try {
      await preparer.prepareImage(input);
      output.push(false);
    } catch {
      output.push(true);
    }
  }
  return output;
};

window.gemma4ImageInsertions = async (cases) => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical WebGPU adapter required');
  const gpu = await adapter.requestDevice();
  gpu.pushErrorScope('validation');
  const response = await fetch('/strands-v2-image-matrix/linked.wasm');
  if (!response.ok) throw new Error('Missing unified TVM graph');
  const runtime = await instantiate(await response.arrayBuffer(), createPolyfillWASI());
  try {
    runtime.initWebGPU(gpu);
    const device = runtime.webgpu();
    runtime.beginScope();
    try {
      const vm = runtime.createVirtualMachine(device);
      const run = (...args: Tensor[]) => vm.getFunction('image_insert_main')(...args) as Tensor;
      const output: string[] = [];
      for (const item of cases) {
        const initial = Float32Array.from({length: item.n * 1536}, (_, index) => ((index % 113) - 56) / 37);
        let hidden = runtime.empty([1, item.n, 1536], 'float32', device).copyFrom(initial);
        for (const [index, span] of item.spans.entries()) {
          const [start, count] = span as [number, number];
          const values = Float32Array.from({length: count * 1536}, (_, i) => ((i % 89) - 44) / 17 + index * 0.5);
          const image = runtime.empty([count, 1536], 'float32', device).copyFrom(values);
          hidden = insertGemma4V2VisionEmbedding(runtime, device, run, hidden, image, start);
        }
        const host = runtime.empty([1, item.n, 1536], 'float32', runtime.cpu()).copyFrom(hidden);
        await device.sync();
        const result = host.toArray() as Float32Array;
        output.push(await hash(new Uint8Array(result.buffer, result.byteOffset, result.byteLength)));
      }
      const error = await gpu.popErrorScope();
      if (error) throw new Error(error.message);
      return output;
    } finally {
      runtime.endScope();
    }
  } finally {
    runtime.dispose();
    gpu.destroy();
  }
};
