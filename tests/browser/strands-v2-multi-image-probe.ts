import {createPolyfillWASI, instantiate} from '../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {createGemma4V2ImagePreparer} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';
import {createGemma4V2ImageInserter} from '../../packages/web-decision/src/strands/gemma4-v2-image-runner.mts';
import {createGemma4V2Runner} from '../../packages/web-decision/src/strands/gemma4-v2-runner.mts';
import {finishGemma4V2Answer} from '../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};

type Input = {state: number[]; questions: number[][]; images: {start: number; end: number}[]; encoded: string[]};
declare global {
  interface Window {
    runGemma4V2MultipleImages: (input: Input) => Promise<unknown>;
  }
}
async function data(name: string) {
  const response = await fetch('/strands-v2-multi-image-assets/' + name);
  if (!response.ok) throw new Error(`Missing multi-image fixture ${name}`);
  return response.arrayBuffer();
}

window.runGemma4V2MultipleImages = async (input) => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical WebGPU adapter required');
  const gpu = await adapter.requestDevice();
  gpu.pushErrorScope('validation');
  const runtime = await instantiate(await data('linked.wasm'), createPolyfillWASI());
  const source = await openGemma4V2SourceFromInventory(
    inventory as never,
    'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
    new URL('/strands-v2-multi-image-source/', location.href)
  );
  let scoped = false;
  try {
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    scoped = true;
    const device = runtime.webgpu();
    const vm = runtime.createVirtualMachine(device);
    const preparer = createGemma4V2ImagePreparer();
    const images = [];
    for (const encoded of input.encoded) images.push(await preparer.prepareImage(encoded));
    const insertImages = await createGemma4V2ImageInserter(runtime, device, vm, source);
    const runner = await createGemma4V2Runner({
      runtime: runtime as never,
      device,
      vm,
      source,
      insertImages,
      readout: new Uint8Array(await data('readout.bin')),
      localFrequency: new Float32Array(await data('local.bin')),
      globalFrequency: new Float32Array(await data('global.bin')),
      dispose() {}
    });
    try {
      const logits = await runner.run({
        stateIds: input.state,
        questionIds: input.questions[0]!,
        imageSpans: input.images,
        images: images.map((image) => image.value)
      });
      const error = await gpu.popErrorScope();
      if (error) throw new Error(error.message);
      return {
        tokens: images.map((image) => image.tokens),
        logits: Array.from(logits),
        answer: finishGemma4V2Answer(
          {kind: 'choice', options: ['green', 'yellow'], descriptions: ['green', 'yellow']},
          logits,
          {temperature: 1.5456, softcap: 30}
        )
      };
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
