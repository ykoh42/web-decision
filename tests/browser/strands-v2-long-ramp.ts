import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {createGemma4V2Runner} from '../../packages/web-decision/src/strands/gemma4-v2-runner.mts';
import {
  queryTiledSplitGraphs,
  transposedSplitGraphs
} from '../../packages/web-decision/src/strands/gemma4-v2-split-adapter.mts';
import {finishGemma4V2Answer} from '../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};
import type {Tensor, VirtualMachine} from '../../packages/web-decision/src/runtime/tvm-types.js';
import {captureLastHiddenRows} from './strands-v2-hidden-capture.mts';

type Input = {
  state: number[];
  question: number[];
  chunkTokens: number;
  mode: 'both' | 'baseline' | 'split';
  hostReadout: boolean;
  captureHidden?: boolean;
  queryTiles?: boolean;
};
declare global {
  interface Window {
    runGemma4LongRamp: (input: Input) => Promise<unknown>;
  }
}

async function asset(name: string) {
  const response = await fetch(`/strands-v2-long-ramp-assets/${name}`);
  if (!response.ok) throw new Error(`Missing long input fixture: ${name}`);
  return response.arrayBuffer();
}

function trackBuffers(device: GPUDevice) {
  const live = new Map<GPUBuffer, {bytes: number; storage: boolean}>();
  const original = device.createBuffer.bind(device);
  let bytes = 0;
  let storageBytes = 0;
  let peak = 0;
  let storagePeak = 0;
  function snapshot() {
    peak = Math.max(peak, bytes);
    storagePeak = Math.max(storagePeak, storageBytes);
    return {count: live.size, bytes, storageBytes, peakBytes: peak, peakStorageBytes: storagePeak};
  }
  Object.defineProperty(device, 'createBuffer', {
    configurable: true,
    value(descriptor: GPUBufferDescriptor) {
      const buffer = original(descriptor);
      const item = {bytes: Number(descriptor.size), storage: Boolean(descriptor.usage & 0x80)};
      live.set(buffer, item);
      bytes += item.bytes;
      if (item.storage) storageBytes += item.bytes;
      const destroy = buffer.destroy.bind(buffer);
      Object.defineProperty(buffer, 'destroy', {
        configurable: true,
        value() {
          const prior = live.get(buffer);
          if (prior) {
            bytes -= prior.bytes;
            if (prior.storage) storageBytes -= prior.bytes;
            live.delete(buffer);
          }
          destroy();
        }
      });
      snapshot();
      return buffer;
    }
  });
  return {
    snapshot,
    resetPeak() {
      peak = 0;
      storagePeak = 0;
      snapshot();
    }
  };
}

function trackLayers(
  underlyingVm: VirtualMachine,
  queryTiles: boolean,
  capture?: ReturnType<typeof captureLastHiddenRows>
) {
  let layers = 0;
  return {
    vm: {
      getFunction(name: string) {
        const fn = underlyingVm.getFunction(name);
        if (typeof fn !== 'function') return fn;
        return (...args: unknown[]) => {
          if (
            queryTiles
              ? name === 'ple_packed_main'
              : /^(?:split(?:0|4|15|19)attn|layer(?:0|4|15|19))_packed_main$/u.test(name)
          ) {
            capture?.queue(layers === 0 ? 'embedding' : `layer${layers - 1}`, args[0] as Tensor);
            layers++;
          }
          if (!capture || name !== 'readout_packed_main') return fn(...args);
          capture.queue(`layer${layers - 1}`, args[0] as Tensor);
          const result = fn(...args);
          capture.queue('finalNorm', result as Tensor);
          return result;
        };
      }
    },
    reset() {
      layers = 0;
    },
    count() {
      return layers;
    }
  };
}

function cleanupRuntime(
  scoped: boolean,
  runtime: Awaited<ReturnType<typeof instantiate>>,
  gpu: GPUDevice,
  source: Awaited<ReturnType<typeof openGemma4V2SourceFromInventory>>,
  failed: boolean
) {
  const actions = [
    () => {
      if (scoped) runtime.endScope();
    },
    () => runtime.dispose(),
    () => gpu.destroy(),
    () => source.dispose()
  ];
  for (const action of actions) {
    try {
      action();
    } catch (error) {
      if (!failed) {
        failed = true;
        console.error('LONG_RAMP_CLEANUP', error);
      }
    }
  }
  return failed;
}

function answerPhase<T extends {logits: number[] | null}>(phase: T | undefined) {
  if (!phase) return null;
  const question = {kind: 'choice' as const, options: ['yes', 'no'], descriptions: ['yes', 'no']};
  const scoring = {temperature: 1.5456, softcap: 30};
  return {...phase, answer: phase.logits ? finishGemma4V2Answer(question, phase.logits, scoring) : null};
}

window.runGemma4LongRamp = async ({
  state,
  question,
  chunkTokens,
  mode,
  hostReadout,
  captureHidden,
  queryTiles = false
}) => {
  if (captureHidden && (queryTiles || mode !== 'split' || !hostReadout || state.length + question.length > 1024))
    throw new Error('Hidden capture requires one split host-readout run of at most 1024 tokens.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter !== false) throw new Error('Physical WebGPU adapter required');
  const limits = adapter.limits;
  const gpu = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup
    }
  });
  const tracked = trackBuffers(gpu);
  const runtime = await instantiate(await asset('linked.wasm'), createPolyfillWASI());
  const source = await openGemma4V2SourceFromInventory(
    inventory as never,
    'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
    new URL('/strands-v2-long-ramp-source/', location.href)
  );
  let scoped = false;
  let failed = false;
  let output: Record<string, unknown> | undefined;
  try {
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scoped = true;
    const device = runtime.webgpu();
    const capture = captureHidden ? captureLastHiddenRows(runtime as never, device) : undefined;
    const layers = trackLayers(runtime.createVirtualMachine(device), queryTiles, capture);
    const base = {
      runtime: runtime as never,
      device,
      vm: layers.vm,
      source,
      insertImages: async (word: never) => word,
      readout: new Uint8Array(await asset('readout.bin')),
      localFrequency: new Float32Array(await asset('local.bin')),
      globalFrequency: new Float32Array(await asset('global.bin')),
      ...(hostReadout ? {hostReadout: true} : {}),
      dispose() {}
    };
    async function run(split: boolean) {
      layers.reset();
      tracked.resetPeak();
      const started = performance.now();
      const heap = () =>
        (performance as Performance & {memory?: {usedJSHeapSize: number}}).memory?.usedJSHeapSize ?? null;
      const heapBefore = heap();
      gpu.pushErrorScope('internal');
      gpu.pushErrorScope('out-of-memory');
      gpu.pushErrorScope('validation');
      let result: ArrayLike<number> | undefined;
      let failure: unknown;
      try {
        const runner = await createGemma4V2Runner({
          ...base,
          ...(split || queryTiles
            ? {
                splitExecution: {
                  graphs: split && queryTiles ? queryTiledSplitGraphs() : transposedSplitGraphs(),
                  chunkTokens
                }
              }
            : {})
        } as never);
        try {
          result = await runner.run({stateIds: state, questionIds: question, images: [], imageSpans: []});
        } finally {
          await runner.dispose();
        }
      } catch (error) {
        failure = error;
      }
      const hidden = await capture?.finish();
      const errors: string[] = [];
      for (const kind of ['validation', 'out-of-memory', 'internal']) {
        try {
          const error = await gpu.popErrorScope();
          if (error) errors.push(`${kind}: ${error.message}`);
        } catch (error) {
          errors.push(`${kind}: ${String(error)}`);
        }
      }
      const detail = {
        mode: split ? 'split' : 'baseline',
        durationMs: performance.now() - started,
        jsHeapBeforeBytes: heapBefore,
        jsHeapAfterBytes: heap(),
        layers: layers.count(),
        buffers: tracked.snapshot(),
        error: failure ? String(failure) : null,
        gpuErrors: errors,
        ...(hidden ? {hidden} : {}),
        logits: result ? Array.from(result) : null
      };
      const {hidden: _hidden, ...summary} = detail;
      console.log(`LONG_RAMP_PHASE ${JSON.stringify(summary)}`);
      return detail;
    }
    const baseline = mode === 'split' ? undefined : await run(false);
    const baselinePassed = !baseline || (baseline.logits && !baseline.error && baseline.gpuErrors.length === 0);
    const split = mode !== 'baseline' && baselinePassed ? await run(true) : undefined;
    output = {
      experiment:
        mode === 'both'
          ? 'baseline then split on the same runtime; cached allocations persist between phases'
          : `single ${mode} phase`,
      tokens: state.length + question.length,
      hostReadout,
      queryTiles,
      stateTokens: state.length,
      questionTokens: question.length,
      chunkTokens,
      adapter: {
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension
      },
      baseline: answerPhase(baseline),
      split: answerPhase(split)
    };
  } catch (error) {
    failed = true;
    throw error;
  } finally {
    failed = cleanupRuntime(scoped, runtime, gpu, source, failed);
  }
  if (failed) throw new Error('Long input GPU cleanup failed.');
  return {...output, buffersAfterDispose: tracked.snapshot()};
};
