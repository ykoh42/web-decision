import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import type {DLDevice, Instance, Tensor, VirtualMachine} from '../../packages/web-decision/src/runtime/tvm-types.js';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import {loadGemma4V2VisionWeights} from '../../packages/web-decision/src/strands/gemma4-v2-vision-bindings.mts';
import {
  gemma4V2VisionGeometry,
  type Gemma4V2VisionWeights
} from '../../packages/web-decision/src/strands/gemma4-v2-vision.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};
import {compareVisionOutputs, visionPerformancePlan} from './strands-v2-vision-performance-plan.mts';

type InputTensors = {pixels: Tensor; positions: Tensor; cos: Tensor; sin: Tensor; mask: Tensor; indices: Tensor};
type TrialResult = {
  interval: 2 | 8;
  warmup: boolean;
  elapsedMs: number;
  maxAbsoluteError: number;
  meanAbsoluteError: number;
};
type Failure = {stage: string; error: string; loss: {reason: string; message: string} | null; uncaptured: string[]};
declare global {
  interface Window {
    runVisionPerformance: (
      base: string,
      rounds: number
    ) => Promise<{
      userAgent: string;
      adapter: {fallback: boolean | null; limits: {maxBufferSize: number; maxStorageBufferBindingSize: number}};
      trials: TrialResult[];
      failure: Failure | null;
    }>;
  }
}

async function fixture(base: string, name: string): Promise<ArrayBuffer> {
  const response = await fetch(new URL(name, base));
  if (!response.ok) throw new Error(`Cannot load ${name}: ${response.status}`);
  return response.arrayBuffer();
}

function makeInputs(runtime: Instance, device: DLDevice, pixels: ArrayBuffer, positions: ArrayBuffer): InputTensors {
  const image = {pixels: new Float32Array(pixels), positions: new Int32Array(positions), tokens: 256};
  const geometry = gemma4V2VisionGeometry(image);
  const tensor = (shape: number[], dtype: string, values: Float32Array | Int32Array) =>
    runtime.empty(shape, dtype, device).copyFrom(values);
  return {
    pixels: tensor([1, 2520, 768], 'float32', image.pixels),
    positions: tensor([1, 2520, 2], 'int32', image.positions),
    cos: tensor([1, 2520, 64], 'float32', geometry.cos),
    sin: tensor([1, 2520, 64], 'float32', geometry.sin),
    mask: tensor([1, 1, 1, 2520], 'float32', geometry.mask),
    indices: tensor([256, 9], 'int32', geometry.indices)
  };
}

function endTrialScope(runtime: Instance, failed: boolean): void {
  try {
    runtime.endScope();
  } catch (error) {
    if (!failed) throw error;
  }
}

async function runTrial(
  runtime: Instance,
  device: DLDevice,
  vm: VirtualMachine,
  weights: Gemma4V2VisionWeights,
  input: InputTensors,
  interval: 2 | 8
) {
  runtime.beginScope();
  let failed = false;
  try {
    const invoke = (name: string, ...args: Tensor[]) => vm.getFunction(name)(...args) as Tensor;
    const start = performance.now();
    let hidden = invoke('vision_patch_main', input.pixels, input.positions, ...weights.patch);
    for (let index = 0; index < weights.layers.length; index++) {
      hidden = invoke(
        'vision_encoder_main',
        hidden,
        input.cos,
        input.sin,
        input.positions,
        input.mask,
        ...weights.layers[index]!
      );
      if ((index + 1) % interval === 0) await device.sync();
    }
    const pooled = invoke('vision_pool_main', hidden, input.indices);
    const result = invoke('vision_projection_main', pooled, ...weights.projection);
    await device.sync();
    const elapsedMs = performance.now() - start;
    if (result.shape.join(',') !== '256,1536' || result.dtype !== 'float32')
      throw new Error('Unexpected vision output.');
    const host = runtime.empty([256, 1536], 'float32', runtime.cpu()).copyFrom(result);
    await device.sync();
    return {elapsedMs, output: (host.toArray() as Float32Array).slice()};
  } catch (error) {
    failed = true;
    throw error;
  } finally {
    endTrialScope(runtime, failed);
  }
}

window.runVisionPerformance = async (base, rounds) => {
  const plan = visionPerformancePlan(rounds);
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable.');
  const fallback =
    adapter.info?.isFallbackAdapter ??
    (adapter as GPUAdapter & {isFallbackAdapter?: boolean}).isFallbackAdapter ??
    null;
  if (fallback !== false) throw new Error('A verified non-fallback adapter is required.');
  const gpu = await adapter.requestDevice();
  const uncaptured: string[] = [];
  let loss: {reason: string; message: string} | null = null;
  let closing = false;
  void gpu.lost.then((info) => {
    if (!closing || info.reason !== 'destroyed') loss = {reason: info.reason, message: info.message};
  });
  gpu.addEventListener('uncapturederror', (event: Event) =>
    uncaptured.push(String((event as Event & {error?: Error}).error))
  );
  const report = {
    userAgent: navigator.userAgent,
    adapter: {
      fallback,
      limits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
      }
    },
    trials: [] as TrialResult[],
    failure: null as Failure | null
  };
  let stage = 'load';
  let runtime: Awaited<ReturnType<typeof instantiate>> | undefined;
  let source: Awaited<ReturnType<typeof openGemma4V2SourceFromInventory>> | undefined;
  let scoped = false;
  try {
    const [wasm, pixels, positions] = await Promise.all(
      ['graph.wasm', 'input-pixels.bin', 'input-positions.bin'].map((name) => fixture(base, name))
    );
    runtime = await instantiate(wasm!, createPolyfillWASI());
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scoped = true;
    const device = runtime.webgpu();
    const vm = runtime.createVirtualMachine(device);
    source = await openGemma4V2SourceFromInventory(
      inventory as never,
      'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
      new URL(base)
    );
    stage = 'load-vision-weights';
    const weights = await loadGemma4V2VisionWeights(runtime, device, source);
    const inputs = makeInputs(runtime, device, pixels!, positions!);
    await device.sync();
    let reference: Float32Array | undefined;
    for (const trial of plan) {
      stage = `${trial.warmup ? 'warmup' : 'measure'}-interval-${trial.interval}`;
      const result = await runTrial(runtime, device, vm, weights, inputs, trial.interval);
      if (loss) throw new Error('WebGPU device lost during trial.');
      reference ??= result.output;
      const difference = compareVisionOutputs(reference, result.output);
      report.trials.push({
        ...trial,
        elapsedMs: result.elapsedMs,
        maxAbsoluteError: difference.maxAbsoluteError,
        meanAbsoluteError: difference.meanAbsoluteError
      });
    }
  } catch (error) {
    await Promise.race([gpu.lost, new Promise((resolve) => setTimeout(resolve, 250))]);
    report.failure = {stage, error: String(error), loss, uncaptured};
  } finally {
    closing = true;
    for (const action of [
      () => {
        if (scoped) runtime?.endScope();
      },
      () => runtime?.dispose(),
      () => gpu.destroy(),
      () => source?.dispose()
    ]) {
      try {
        action();
      } catch (error) {
        report.failure ??= {stage: 'cleanup', error: String(error), loss, uncaptured};
      }
    }
  }
  return report;
};
