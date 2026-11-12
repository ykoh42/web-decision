import {createKananaTvmInferenceSession} from '../../packages/web-decision/src/kanana/tvm-session.js';
import {setupKananaTvmRuntime} from '../../packages/web-decision/src/kanana/tvm-runtime.js';
import {openKananaWeightSource} from '../../packages/web-decision/src/kanana/weight-source.mjs';
import {detectGPUDevice} from '../../packages/web-decision/src/kanana/tvm-webgpu.js';

type TvmJs = {
  instantiate(
    bytes: ArrayBuffer,
    imports: Record<string, unknown>,
    logger: (message: string) => void
  ): Promise<unknown>;
  createPolyfillWASI(): Record<string, unknown>;
};

type ParityCase = Readonly<{
  name: string;
  sequenceLength: number;
  inputIds: number[];
  attentionMask: number[];
  imageEmbeddings: number[];
  imagePositions: number[];
  positionIds: number[];
  referenceValues: number[];
}>;

type FullSmokeInference = Awaited<ReturnType<typeof createKananaTvmInferenceSession>>;

function assertSuccessfulResponses(responses: readonly Response[]): void {
  for (const response of responses)
    if (!response.ok) throw new Error(`Kanana full smoke asset request failed: HTTP ${response.status}.`);
}

function selectParityCases(cases: readonly ParityCase[]): ParityCase[] {
  const selected = [
    cases.find((item) => item.name === 'no-images' && item.sequenceLength === 1),
    cases.find((item) => item.name === 'no-images' && item.sequenceLength === 8),
    cases.find((item) => item.name === 'two-image-tokens' && item.sequenceLength === 8)
  ];
  if (selected.some((item) => !item)) throw new Error('Full Kanana parity cases are incomplete.');
  return selected as ParityCase[];
}

async function runParityCase(inference: FullSmokeInference, item: ParityCase) {
  const multimodal = item.imagePositions.length
    ? {
        imageEmbeddings: {
          shape: [item.imagePositions.length, 2560] as const,
          values: item.imageEmbeddings
        },
        imagePositions: item.imagePositions,
        positionIds: item.positionIds
      }
    : undefined;
  const started = performance.now();
  const output = await inference.infer(item.inputIds, item.attentionMask, multimodal);
  if (output.values.length !== item.referenceValues.length)
    throw new Error(`Kanana output length differs for ${item.name} at ${item.sequenceLength} tokens.`);
  let maxAbsDelta = 0;
  let squaredDelta = 0;
  for (let index = 0; index < item.referenceValues.length; index++) {
    const delta = output.values[index]! - item.referenceValues[index]!;
    maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta));
    squaredDelta += delta * delta;
  }
  return {
    name: item.name,
    sequenceLength: item.sequenceLength,
    outputShape: output.shape,
    maxAbsDelta,
    rmse: Math.sqrt(squaredDelta / item.referenceValues.length),
    inferenceMilliseconds: performance.now() - started
  };
}

async function runParityCases(inference: FullSmokeInference, cases: readonly ParityCase[]) {
  const results = [];
  for (const item of cases) results.push(await runParityCase(inference, item));
  return results;
}

export async function runKananaOnlineAttentionFullSmoke(assetsBase: string, checkpointBase: string) {
  const tvmjs = (window as Window & {tvmjs?: TvmJs}).tvmjs;
  if (!tvmjs) throw new Error('Apache TVM JS runtime was not loaded.');
  const gpu = await detectGPUDevice();
  if (!gpu) throw new Error('This browser does not expose WebGPU.');
  const webgpuErrors: string[] = [];
  gpu.device.addEventListener('uncapturederror', (event) => {
    const error = (event as Event & {error?: {message?: string}}).error;
    webgpuErrors.push(error?.message ?? 'Unspecified WebGPU error.');
  });

  const [wasmResponse, weightsResponse, rangeResponse, casesResponse] = await Promise.all([
    fetch(new URL('kanana-text-model.wasm', assetsBase)),
    fetch(new URL('external-weight-manifest.json', assetsBase)),
    fetch(new URL('range-manifest.json', assetsBase)),
    fetch(new URL('sequence-parity-cases.json', assetsBase))
  ]);
  assertSuccessfulResponses([wasmResponse, weightsResponse, rangeResponse, casesResponse]);

  const wasm = await wasmResponse.arrayBuffer();
  const weightManifest = (await weightsResponse.json()) as {weights: readonly unknown[]};
  const rangeManifest = await rangeResponse.json();
  const parityCases = (await casesResponse.json()) as {cases: readonly ParityCase[]};
  if (!Array.isArray(weightManifest.weights) || weightManifest.weights.length !== 354)
    throw new Error('The full Kanana text graph must contain exactly 354 graph weights.');

  let loadedWeightBytes = 0;
  let source: Awaited<ReturnType<typeof openKananaWeightSource>> | undefined;
  let setup: Awaited<ReturnType<typeof setupKananaTvmRuntime>> | undefined;
  let inference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>> | undefined;
  try {
    source = await openKananaWeightSource({
      base: checkpointBase,
      manifest: rangeManifest,
      requiredWeights: weightManifest.weights as never,
      cache: null,
      onBytes: (bytes) => {
        loadedWeightBytes += bytes;
      }
    });
    setup = await setupKananaTvmRuntime({
      instantiate: tvmjs.instantiate as never,
      createPolyfillWASI: tvmjs.createPolyfillWASI,
      device: gpu.device,
      bytes: wasm
    });
    inference = await createKananaTvmInferenceSession({
      setup: setup as Parameters<typeof createKananaTvmInferenceSession>[0]['setup'],
      weightSource: {
        manifest: {weights: weightManifest.weights as never},
        async readWeight(record) {
          if (!source) throw new Error('Kanana verified checkpoint source was disposed.');
          return (await source.readTensor(record.name)).bytes;
        }
      }
    });
    setup = undefined;

    const sequenceCases = await runParityCases(inference, selectParityCases(parityCases.cases));
    return {
      adapter: gpu.adapter.info,
      features: Array.from(gpu.device.features),
      loadedWeightBytes,
      sequenceCases,
      webgpuErrors
    };
  } finally {
    if (inference) await inference.dispose();
    else if (setup) setup.dispose();
    gpu.device.destroy();
  }
}
