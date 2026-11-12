import {createKananaTvmInferenceSession} from '../../packages/web-decision/src/kanana/tvm-session.js';
import {setupKananaTvmRuntime} from '../../packages/web-decision/src/kanana/tvm-runtime.js';
import {detectGPUDevice} from '../../packages/web-decision/src/kanana/tvm-webgpu.js';

type TvmJs = {
  instantiate(
    bytes: ArrayBuffer,
    imports: Record<string, unknown>,
    logger: (message: string) => void
  ): Promise<unknown>;
  createPolyfillWASI(): Record<string, unknown>;
};

type SequenceCase = Readonly<{
  sequenceLength: number;
  inputIds: number[];
  attentionMask: number[];
  referenceValues: number[];
}>;

type SmokeInference = Awaited<ReturnType<typeof createKananaTvmInferenceSession>>;

function assertSuccessfulResponses(responses: readonly Response[]): void {
  for (const response of responses)
    if (!response.ok) throw new Error(`Kanana smoke asset request failed: HTTP ${response.status}.`);
}

async function verifySmokeWeights(
  weights: Uint8Array,
  records: readonly {
    name: string;
    length: number;
    assetOffset: number;
    assetLength: number;
    assetSha256: string;
    bytes?: Uint8Array;
  }[]
): Promise<void> {
  for (const record of records) {
    const range = weights.subarray(record.assetOffset, record.assetOffset + record.assetLength);
    if (range.byteLength !== record.length || record.assetLength !== record.length)
      throw new Error(`Kanana smoke weight byte length differs for ${record.name}.`);
    const digest = await crypto.subtle.digest('SHA-256', range.slice().buffer as ArrayBuffer);
    const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
    if (hash !== record.assetSha256) throw new Error(`Kanana smoke weight hash differs for ${record.name}.`);
    record.bytes = range;
  }
}

async function runSequenceCase(
  inference: SmokeInference,
  item: SequenceCase | undefined,
  sequenceLength: number,
  referenceInputIds: readonly number[]
) {
  if (!item) throw new Error(`Kanana CPU reference has no length-${sequenceLength} case.`);
  if (sequenceLength === 8 && item.inputIds.join(',') !== referenceInputIds.join(','))
    throw new Error('Kanana eight-token reference does not match the parity case.');
  const started = performance.now();
  const output = await inference.infer(item.inputIds, item.attentionMask);
  let maxAbsDelta = 0;
  let squaredDelta = 0;
  for (let index = 0; index < item.referenceValues.length; index++) {
    const delta = output.values[index]! - item.referenceValues[index]!;
    maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta));
    squaredDelta += delta * delta;
  }
  return {
    sequenceLength,
    outputShape: output.shape,
    maxAbsDelta,
    rmse: Math.sqrt(squaredDelta / item.referenceValues.length),
    inferenceMilliseconds: performance.now() - started
  };
}

async function runBoundaryCase(inference: SmokeInference, sequenceLength: number) {
  const inputIds = Array.from({length: sequenceLength}, (_, index) => (index % 127) + 1);
  const attentionMask = Array.from({length: sequenceLength}, () => 1);
  const started = performance.now();
  const output = await inference.infer(inputIds, attentionMask);
  const sampleValues = [...output.values.slice(0, 4), ...output.values.slice(-4)];
  if (!sampleValues.every(Number.isFinite))
    throw new Error(`Kanana boundary inference returned non-finite values at ${sequenceLength} tokens.`);
  return {
    sequenceLength,
    outputShape: output.shape,
    sampleValues,
    inferenceMilliseconds: performance.now() - started
  };
}

export async function runKananaOnlineAttentionSmoke(assetsBase: string, referenceUrl: string) {
  const tvmjs = (window as Window & {tvmjs?: TvmJs}).tvmjs;
  if (!tvmjs) throw new Error('Apache TVM JS runtime was not loaded.');
  const gpu = await detectGPUDevice();
  if (!gpu) throw new Error('This browser does not expose WebGPU.');
  const webgpuErrors: string[] = [];
  gpu.device.addEventListener('uncapturederror', (event) => {
    const error = (event as Event & {error?: {message?: string}}).error;
    webgpuErrors.push(error?.message ?? 'Unspecified WebGPU error.');
  });

  const [wasmResponse, manifestResponse, weightsResponse, referenceResponse, casesResponse] = await Promise.all([
    fetch(new URL('kanana-text-model.wasm', assetsBase)),
    fetch(new URL('external-weight-manifest.json', assetsBase)),
    fetch(new URL('external-weights.bin', assetsBase)),
    fetch(referenceUrl),
    fetch(new URL('sequence-parity-cases.json', assetsBase))
  ]);
  assertSuccessfulResponses([wasmResponse, manifestResponse, weightsResponse, referenceResponse, casesResponse]);

  const manifest = (await manifestResponse.json()) as {
    weights: readonly (Readonly<{
      name: string;
      parameterName: string;
      shape: readonly number[];
      dtype: string;
      length: number;
      assetOffset: number;
      assetLength: number;
      assetSha256: string;
    }> & {bytes?: Uint8Array})[];
  };
  const weights = new Uint8Array(await weightsResponse.arrayBuffer());
  const wasm = await wasmResponse.arrayBuffer();
  const reference = (await referenceResponse.json()) as {
    inputIds: number[];
  };
  const parityCases = (await casesResponse.json()) as {cases: readonly SequenceCase[]};
  if (!Array.isArray(manifest.weights) || manifest.weights.length !== 13)
    throw new Error('The smoke graph must contain exactly thirteen one-layer weights.');
  await verifySmokeWeights(weights, manifest.weights);

  let setup: Awaited<ReturnType<typeof setupKananaTvmRuntime>> | undefined;
  let inference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>> | undefined;
  try {
    setup = await setupKananaTvmRuntime({
      instantiate: tvmjs.instantiate as never,
      createPolyfillWASI: tvmjs.createPolyfillWASI,
      device: gpu.device,
      bytes: wasm
    });
    inference = await createKananaTvmInferenceSession({
      setup: setup as Parameters<typeof createKananaTvmInferenceSession>[0]['setup'],
      weightSource: {weights: manifest.weights}
    });
    setup = undefined;
    const sequenceCases = await Promise.all(
      [1, 8].map((sequenceLength) =>
        runSequenceCase(
          inference!,
          parityCases.cases.find((candidate) => candidate.sequenceLength === sequenceLength),
          sequenceLength,
          reference.inputIds
        )
      )
    );
    const boundaryCases = [];
    for (const sequenceLength of [20_479, 20_480]) boundaryCases.push(await runBoundaryCase(inference, sequenceLength));
    return {
      adapter: gpu.adapter.info,
      features: Array.from(gpu.device.features),
      sequenceCases,
      boundaryCases,
      webgpuErrors
    };
  } finally {
    if (inference) await inference.dispose();
    if (setup) setup.dispose();
    gpu.device.destroy();
  }
}
