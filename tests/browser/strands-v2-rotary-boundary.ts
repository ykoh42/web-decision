import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import type {DLDevice, Instance, Tensor, VirtualMachine} from '../../packages/web-decision/src/runtime/tvm-types.js';
import {rotaryAllocationBytes} from './strands-v2-rotary-limits.ts';

type Case = {length: number; status: 'passed' | 'unsupported'; maxError?: number; requiredBytes: number};
type Result = {adapterLimits: {maxBufferSize: number; maxStorageBufferBindingSize: number}; cases: Case[]};

declare global {
  interface Window {
    runGemma4RotaryBoundary: () => Promise<Result>;
  }
}

const lengths = [131071, 131072, 131073] as const;
const columns = [0, 1, 127, 255];

function sampleError(rows: Int32Array, frequency: Float32Array, values: Float32Array): number {
  let maxError = 0;
  for (let row = 0; row < rows.length; row++) {
    for (let column = 0; column < columns.length; column++) {
      const expected = Math.cos(Math.fround(rows[row]! * frequency[columns[column]! % 128]!));
      const actual = values[row * columns.length + column]!;
      if (!Number.isFinite(actual)) throw new Error('Non-finite rotary sample.');
      maxError = Math.max(maxError, Math.abs(actual - expected));
    }
  }
  return maxError;
}

async function runCase(
  runtime: Instance,
  device: DLDevice,
  vm: VirtualMachine,
  gpu: GPUDevice,
  length: number
): Promise<Case> {
  gpu.pushErrorScope('validation');
  runtime.beginScope();
  let maxError = 0;
  try {
    const ids = Int32Array.from({length}, (_, i) => i);
    const frequency = Float32Array.from({length: 128}, (_, i) => (i + 1) / 65536);
    const rows = Int32Array.from([0, Math.floor(length / 2), length - 2, length - 1]);
    const tensor = (shape: number[], dtype: string, data: Int32Array | Float32Array) =>
      runtime.empty(shape, dtype, device).copyFrom(data);
    const output = vm.getFunction('local_cos')(
      tensor([length], 'int32', ids),
      tensor([128], 'float32', frequency)
    ) as Tensor;
    if (output.dtype !== 'float32' || output.shape.join(',') !== `1,${length},256`)
      throw new Error(`Unexpected rotary output shape at ${length}.`);
    const samples = vm.getFunction('sample_main')(
      output,
      tensor([4], 'int32', rows),
      tensor([4], 'int32', Int32Array.from(columns))
    ) as Tensor;
    const host = runtime.empty([4, 4], 'float32', runtime.cpu()).copyFrom(samples);
    await device.sync();
    maxError = sampleError(rows, frequency, host.toArray() as Float32Array);
    if (maxError > 0.02) throw new Error(`Rotary samples differ at ${length}: ${maxError}.`);
  } catch (error) {
    await gpu.popErrorScope().catch(() => {});
    throw error;
  } finally {
    runtime.endScope();
  }
  const validation = await gpu.popErrorScope();
  if (validation) throw new Error(`WebGPU validation failed: ${validation.message}`);
  return {length, status: 'passed', maxError, requiredBytes: rotaryAllocationBytes(length)};
}

window.runGemma4RotaryBoundary = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('A physical WebGPU adapter is required.');
  const {maxBufferSize, maxStorageBufferBindingSize} = adapter.limits;
  const limit = Math.min(maxBufferSize, maxStorageBufferBindingSize);
  const supported = lengths.filter((length) => rotaryAllocationBytes(length) <= limit);
  const cases: Case[] = lengths
    .filter((length) => rotaryAllocationBytes(length) > limit)
    .map((length) => ({length, status: 'unsupported', requiredBytes: rotaryAllocationBytes(length)}));
  const adapterLimits = {maxBufferSize, maxStorageBufferBindingSize};
  if (!supported.length) return {adapterLimits, cases};
  const largest = rotaryAllocationBytes(supported.at(-1)!);
  const gpu = await adapter.requestDevice({
    requiredLimits: {maxBufferSize: largest, maxStorageBufferBindingSize: largest}
  });
  let runtime: Awaited<ReturnType<typeof instantiate>> | undefined;
  let scoped = false;
  try {
    const response = await fetch('/strands-v2-rotary-boundary/graph.wasm');
    if (!response.ok) throw new Error('The rotary boundary test graph is unavailable.');
    runtime = await instantiate(await response.arrayBuffer(), createPolyfillWASI());
    runtime.initWebGPU(gpu);
    const device = runtime.webgpu();
    runtime.beginScope();
    scoped = true;
    const vm = runtime.createVirtualMachine(device);
    for (const length of supported) cases.push(await runCase(runtime, device, vm, gpu, length));
    cases.sort((left, right) => left.length - right.length);
    return {adapterLimits, cases};
  } finally {
    if (scoped) runtime?.endScope();
    runtime?.dispose();
    gpu.destroy();
  }
};
