import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {createGemma4V2ImagePreparer} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';
import {createGemma4V2ImageInserter} from '../../packages/web-decision/src/strands/gemma4-v2-image-runner.mts';
import {createGemma4V2Runner} from '../../packages/web-decision/src/strands/gemma4-v2-runner.mts';
import {transposedSplitGraphs} from '../../packages/web-decision/src/strands/gemma4-v2-split-adapter.mts';
import {finishGemma4V2Answer} from '../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import {withGemma4V2Cleanup} from '../../packages/web-decision/src/strands/gemma4-v2-cleanup.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};

type Input = {state: number[]; questions: number[][]; images: {start: number; end: number}[]};
declare global {
  interface Window {
    runGemma4V2ImageE2E: (input: Input, split?: boolean, hostReadout?: boolean) => Promise<unknown>;
    checkGemma4V2HostReadout: () => Promise<number>;
  }
}

async function data(name: string) {
  const response = await fetch('/strands-v2-image-e2e-assets/' + name);
  if (!response.ok) throw new Error(`Missing image runner fixture ${name}`);
  return response.arrayBuffer();
}

function sampleImage() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas is unavailable');
  context.fillStyle = 'rgb(220,20,30)';
  context.fillRect(0, 0, 32, 64);
  context.fillStyle = 'rgb(15,40,210)';
  context.fillRect(32, 0, 32, 64);
  return canvas.toDataURL('image/png').split(',')[1]!;
}

function cleanup(actions: Array<() => void>, failed: boolean) {
  let failure: unknown;
  let cleanupFailed = false;
  for (const action of actions) {
    try {
      action();
    } catch (error) {
      if (!cleanupFailed) {
        failure = error;
        cleanupFailed = true;
      }
    }
  }
  if (!failed && cleanupFailed) throw failure;
}

window.runGemma4V2ImageE2E = async (input, split = false, hostReadout = false) => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('A WebGPU adapter is required');
  const fallback =
    adapter.info?.isFallbackAdapter ??
    (adapter as GPUAdapter & {isFallbackAdapter?: boolean}).isFallbackAdapter ??
    null;
  if (fallback !== false) throw new Error('A verified non-fallback WebGPU adapter is required');
  const gpu = await adapter.requestDevice();
  gpu.pushErrorScope('validation');
  const runtime = await instantiate(await data('linked.wasm'), createPolyfillWASI());
  const source = await openGemma4V2SourceFromInventory(
    inventory as never,
    'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
    new URL('/strands-v2-image-e2e-source/', location.href)
  );
  let scoped = false;
  let failed = false;
  try {
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scoped = true;
    const device = runtime.webgpu();
    const actualVm = runtime.createVirtualMachine(device);
    let layers = 0;
    const vm = {
      getFunction(name: string) {
        const fn = actualVm.getFunction(name);
        if (typeof fn !== 'function') return fn;
        return (...args: unknown[]) => {
          if (/^split(?:0|4|15|19)attn_packed_main$/u.test(name)) layers++;
          return fn(...args);
        };
      }
    };
    const image = await createGemma4V2ImagePreparer().prepareImage(sampleImage());
    const insertImages = await createGemma4V2ImageInserter(runtime, device, vm, source);
    const runner = await createGemma4V2Runner({
      runtime: runtime as never,
      device,
      vm,
      source,
      insertImages,
      readout: new Uint8Array(await data('readout.bin')),
      hostReadout,
      localFrequency: new Float32Array(await data('local.bin')),
      globalFrequency: new Float32Array(await data('global.bin')),
      ...(split ? {splitExecution: {graphs: transposedSplitGraphs(), chunkTokens: 64}} : {}),
      dispose() {}
    });
    return await withGemma4V2Cleanup(
      async () => {
        const logits = await runner.run({
          stateIds: input.state,
          questionIds: input.questions[0]!,
          images: [image.value],
          imageSpans: input.images
        });
        const error = await gpu.popErrorScope();
        if (error) throw new Error(error.message);
        return {
          tokens: image.tokens,
          layers,
          logits: Array.from(logits),
          answer: finishGemma4V2Answer(
            {kind: 'choice', options: ['red', 'blue'], descriptions: ['red', 'blue']},
            logits,
            {temperature: 1.5456, softcap: 30}
          )
        };
      },
      async () => {
        await runner.dispose();
      }
    );
  } catch (error) {
    failed = true;
    throw error;
  } finally {
    cleanup(
      [
        () => {
          if (scoped) runtime.endScope();
        },
        () => runtime.dispose(),
        () => gpu.destroy(),
        () => source.dispose()
      ],
      failed
    );
  }
};

window.checkGemma4V2HostReadout = async () => {
  const [wasm, readout, hidden, expected] = await Promise.all(
    ['linked.wasm', 'readout.bin', 'fp64-hidden.f32le', 'fp64-logits.f32le'].map(data)
  );
  if (hidden!.byteLength !== 6 * 1536 * 4 || expected!.byteLength !== 6 * 255 * 4)
    throw new Error('Invalid frozen FP64 readout fixture.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('A physical WebGPU adapter is required.');
  const gpu = await adapter.requestDevice();
  const runtime = await instantiate(wasm!, createPolyfillWASI());
  runtime.initWebGPU(gpu);
  runtime.beginScope();
  try {
    const cpu = runtime.cpu();
    const weights = runtime.empty([195840], 'uint32', cpu).copyFromRawBytes(new Uint8Array(readout!));
    const input = runtime.empty([1, 1536], 'float32', cpu);
    const output = runtime.empty([1, 255], 'float32', cpu);
    const accumulate = runtime.systemLib().getFunction('readout_accumulate');
    let count = 0;
    for (let row = 0; row < 6; row++) {
      input.copyFromRawBytes(new Uint8Array(hidden!, row * 6144, 6144));
      accumulate(input, weights, output);
      const bytes = output.toRawBytes();
      const original = new Uint8Array(expected!, row * 1020, 1020);
      if (bytes.some((byte, i) => byte !== original[i])) throw new Error(`FP64 readout differs on row ${row}.`);
      count += 255;
    }
    return count;
  } finally {
    runtime.endScope();
    runtime.dispose();
    gpu.destroy();
  }
};
