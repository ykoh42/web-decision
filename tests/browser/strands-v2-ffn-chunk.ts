import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import type {DLDevice, Instance, Tensor, VirtualMachine} from '../../packages/web-decision/src/runtime/tvm-types.js';

type Fixture = {name: string; dtype: string; shape: number[]; bytes: number};
type Result = {chunk: number; tokens: number; chunks: number; maxReferenceError: number; meanReferenceError: number};
declare global {
  interface Window {
    runGemma4FfnChunks: () => Promise<{results: Result[]; maxBufferSize: number; maxStorageBufferBindingSize: number}>;
  }
}

async function file(name: string): Promise<ArrayBuffer> {
  const response = await fetch(`/strands-v2-ffn-chunk/${name}`);
  if (!response.ok) throw new Error(`Missing FFN test fixture ${name}.`);
  return response.arrayBuffer();
}

async function runChunk(
  runtime: Instance,
  device: DLDevice,
  vm: VirtualMachine,
  weights: Tensor[],
  hidden: Float32Array,
  perLayer: Float32Array,
  scale: Float32Array,
  start: number,
  length: number
): Promise<Float32Array> {
  runtime.beginScope();
  try {
    const h = runtime
      .empty([1, length, 1536], 'float32', device)
      .copyFrom(hidden.subarray(start * 1536, (start + length) * 1536));
    const p = runtime
      .empty([1, length, 256], 'float32', device)
      .copyFrom(perLayer.subarray(start * 256, (start + length) * 256));
    const s = runtime.empty([1], 'float32', device).copyFrom(scale);
    const result = vm.getFunction('packed_main')(h, p, s, ...weights) as Tensor;
    if (result.shape.join(',') !== `1,${length},1536`) throw new Error('Unexpected FFN chunk shape.');
    const host = runtime.empty([1, length, 1536], 'float32', runtime.cpu()).copyFrom(result);
    await device.sync();
    return (host.toArray() as Float32Array).slice();
  } finally {
    runtime.endScope();
  }
}

async function runCase(
  runtime: Instance,
  device: DLDevice,
  vm: VirtualMachine,
  weights: Tensor[],
  hidden: Float32Array,
  perLayer: Float32Array,
  scale: Float32Array,
  expected: Float32Array,
  chunk: number
): Promise<Result> {
  const tokens = 7;
  let maxReferenceError = 0;
  let sum = 0;
  let count = 0;
  let chunks = 0;
  for (let start = 0; start < tokens; start += chunk) {
    const length = Math.min(chunk, tokens - start);
    const values = await runChunk(runtime, device, vm, weights, hidden, perLayer, scale, start, length);
    for (let i = 0; i < values.length; i++) {
      const actual = values[i]!;
      if (!Number.isFinite(actual)) throw new Error('FFN chunk returned a non-finite value.');
      const difference = Math.abs(actual - expected[start * 1536 + i]!);
      maxReferenceError = Math.max(maxReferenceError, difference);
      sum += difference;
      count++;
    }
    chunks++;
  }
  return {chunk, tokens, chunks, maxReferenceError, meanReferenceError: sum / count};
}

window.runGemma4FfnChunks = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('A physical WebGPU adapter is required.');
  const {maxBufferSize, maxStorageBufferBindingSize} = adapter.limits;
  const requiredBytes = 12288 * 1536 * Float32Array.BYTES_PER_ELEMENT;
  if (Math.min(maxBufferSize, maxStorageBufferBindingSize) < requiredBytes)
    throw new Error(`Adapter cannot fit one unpacked FFN weight: ${requiredBytes} bytes.`);
  const gpu = await adapter.requestDevice({
    requiredLimits: {maxBufferSize: requiredBytes, maxStorageBufferBindingSize: requiredBytes}
  });
  let runtime: Awaited<ReturnType<typeof instantiate>> | undefined;
  let scoped = false;
  try {
    const records = JSON.parse(new TextDecoder().decode(await file('inputs.json'))) as Fixture[];
    if (records.length !== 11) throw new Error('Unexpected FFN fixture input count.');
    const hidden = new Float32Array(await file('input-0.bin'));
    const perLayer = new Float32Array(await file('input-1.bin'));
    const scale = new Float32Array(await file('input-2.bin'));
    const expected = new Float32Array(await file('expected.bin'));
    runtime = await instantiate(await file('tail.wasm'), createPolyfillWASI());
    runtime.initWebGPU(gpu);
    const device = runtime.webgpu();
    runtime.beginScope();
    scoped = true;
    const vm = runtime.createVirtualMachine(device);
    const weights: Tensor[] = [];
    for (const record of records.slice(3)) {
      const bytes = await file(record.name);
      if (record.dtype !== 'uint32' || bytes.byteLength !== record.bytes)
        throw new Error('Invalid FFN weight fixture.');
      weights.push(runtime.empty(record.shape, 'uint32', device).copyFrom(new Uint32Array(bytes)));
    }
    gpu.pushErrorScope('validation');
    const results: Result[] = [];
    try {
      for (const chunk of [1, 2, 3, 6, 7])
        results.push(await runCase(runtime, device, vm, weights, hidden, perLayer, scale, expected, chunk));
    } catch (error) {
      await gpu.popErrorScope().catch(() => {});
      throw error;
    }
    const validation = await gpu.popErrorScope();
    if (validation) throw new Error(`WebGPU validation failed: ${validation.message}`);
    return {results, maxBufferSize, maxStorageBufferBindingSize};
  } finally {
    if (scoped) runtime?.endScope();
    runtime?.dispose();
    gpu.destroy();
  }
};
