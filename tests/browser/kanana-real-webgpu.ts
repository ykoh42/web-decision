import {createKananaDecisionSession} from '../../packages/web-decision/src/kanana/decision-session.js';
import {KANANA_REVISION} from '../../packages/web-decision/src/kanana/contract.mjs';
import {createKananaTokenizer} from '../../packages/web-decision/src/kanana/tokenizer-asset.mjs';
import {openKananaSource} from '../../packages/web-decision/src/kanana/source.mjs';
import {
  openKananaWeightSource as openPinnedKananaWeightSource,
  openSafetensorsWeightSource as openSafetensorsSource
} from '../../packages/web-decision/src/kanana/weight-source.mjs';
import {
  createKananaTvmInferenceSession,
  type KananaTvmInferenceSessionOptions
} from '../../packages/web-decision/src/kanana/tvm-session.js';
import {setupKananaTvmRuntime} from '../../packages/web-decision/src/kanana/tvm-runtime.js';

type TvmJs = {
  instantiate(
    bytes: ArrayBuffer,
    imports: Record<string, unknown>,
    logger: (message: string) => void
  ): Promise<unknown>;
  createPolyfillWASI(): Record<string, unknown>;
  detectGPUDevice(): Promise<{adapter: GPUAdapter; device: GPUDevice} | undefined>;
};

type ValidationWeightRecord = {
  name: string;
  parameterName?: string;
  shape: readonly number[];
  dtype: string;
  length: number;
};

type ValidationWeightSource = {
  records: readonly ValidationWeightRecord[];
  readTensor(name: string): Promise<{bytes: Uint8Array}>;
};

type ValidationWeightOptions = {
  base: string;
  manifest: Record<string, unknown>;
  requiredWeights: readonly Record<string, unknown>[];
  cache: false;
};

type ValidationWeightOpener = (options: ValidationWeightOptions) => Promise<ValidationWeightSource>;

const openKananaWeights = openPinnedKananaWeightSource as unknown as ValidationWeightOpener;
const openSafetensorsWeights = openSafetensorsSource as unknown as ValidationWeightOpener;

declare global {
  interface Window {
    tvmjs?: TvmJs;
    __kananaProgress?(event: Record<string, unknown>): Promise<void>;
  }
}

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}: HTTP ${response.status}.`);
  return (await response.json()) as T;
}

type KananaDecisionReference = {
  request: {
    state: unknown;
    questions: unknown;
    state_format: string;
    temperature: number;
  };
  result: {answers: Record<string, {choice?: string; probabilities?: Record<string, number>}>};
};

type TvmInference = Awaited<ReturnType<typeof createKananaTvmInferenceSession>>;

async function inferRealHiddenState(
  inference: TvmInference,
  reference: {inputIds: number[]; attentionMask: number[]},
  stages: {name: string; milliseconds: number}[],
  device: GPUDevice,
  adapter: GPUAdapter,
  webgpuErrors: string[],
  cleanupErrors: string[]
) {
  try {
    const hiddenState = await timed('real hidden state', stages, () =>
      inference.infer(reference.inputIds, reference.attentionMask)
    );
    return {kind: 'success' as const, hiddenState};
  } catch (error) {
    const lost = await device.lost;
    return {
      kind: 'execution-failure' as const,
      result: {
        kind: 'execution-failure' as const,
        model: 'kakaocorp/kanana-2-v-decide-3b-2611',
        revision: KANANA_REVISION,
        error: describeError(error),
        deviceLost: {reason: lost.reason, message: lost.message},
        gpu: {
          vendor: adapter.info?.vendor ?? '',
          architecture: adapter.info?.architecture ?? '',
          maxBufferSize: device.limits.maxBufferSize,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          features: Array.from(device.features)
        },
        webgpuErrors,
        cleanupErrors,
        stages
      }
    };
  }
}

async function runDecisionReference(
  decision: ReturnType<typeof createKananaDecisionSession>,
  reference: KananaDecisionReference,
  stages: {name: string; milliseconds: number}[]
) {
  const request = {
    state: reference.request.state,
    questions: reference.request.questions,
    stateFormat: reference.request.state_format,
    temperature: reference.request.temperature
  };
  const result = await timed('real text decision', stages, () => decision.systemOne(request));
  const answer = (result.answers as Record<string, {choice?: string; probabilities?: Record<string, number>}>)[
    'delivery_status'
  ];
  const expected = reference.result.answers.delivery_status;
  if (!expected?.probabilities) throw new Error('Kanana decision reference is missing probabilities.');
  if (!answer || typeof answer.choice !== 'string' || !answer.probabilities)
    throw new Error('Kanana returned an invalid choice answer.');
  return {
    answer,
    expected,
    probabilityDeltas: Object.fromEntries(
      Object.keys(expected.probabilities ?? {}).map((key) => [
        key,
        Math.abs((answer.probabilities?.[key] ?? Number.NaN) - (expected.probabilities?.[key] ?? Number.NaN))
      ])
    ),
    usage: result.usage
  };
}

async function timed<T>(stage: string, stages: {name: string; milliseconds: number}[], run: () => Promise<T>) {
  const started = performance.now();
  await window.__kananaProgress?.({stage, state: 'started'});
  const result = await run();
  const milliseconds = Math.round(performance.now() - started);
  stages.push({name: stage, milliseconds});
  await window.__kananaProgress?.({stage, state: 'completed', milliseconds});
  return result;
}

function compareValues(actual: Float32Array, expected: number[]) {
  if (actual.length !== expected.length) throw new Error('Kanana hidden state length differs from its CPU reference.');
  let maxAbsDelta = 0;
  let sumSquaredDelta = 0;
  for (let index = 0; index < actual.length; index++) {
    const delta = actual[index]! - expected[index]!;
    maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta));
    sumSquaredDelta += delta * delta;
  }
  return {maxAbsDelta, rmse: Math.sqrt(sumSquaredDelta / actual.length)};
}

function describeError(error: unknown): unknown {
  if (error instanceof AggregateError)
    return {
      name: error.name,
      message: error.message,
      errors: Array.from(error.errors, describeError),
      cause: error.cause === undefined ? undefined : describeError(error.cause)
    };
  if (error instanceof Error)
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
      cause: error.cause === undefined ? undefined : describeError(error.cause)
    };
  return String(error);
}

export async function runKananaRealWebGpu(
  assetsBase: string,
  pytorchReferenceUrl: string,
  decisionReferenceUrl: string
) {
  const tvmjs = window.tvmjs;
  if (!tvmjs) throw new Error('Apache TVM JS runtime bundle is not loaded.');

  const stages: {name: string; milliseconds: number}[] = [];
  const base = new URL(assetsBase);
  if (!base.pathname.endsWith('/')) base.pathname += '/';

  const assets = await timed('verified tokenizer assets', stages, async () => {
    const source = await openKananaSource({base: base.href, cacheEnabled: false});
    const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    return {
      config: JSON.parse(new TextDecoder().decode(configBytes)) as Record<string, unknown>,
      tokenizerConfig: JSON.parse(new TextDecoder().decode(tokenizerConfigBytes)) as Record<string, unknown>,
      tokenizerJson: JSON.parse(new TextDecoder().decode(tokenizerBytes)) as Record<string, unknown>
    };
  });
  const tokenizer = createKananaTokenizer(assets.tokenizerJson, assets.tokenizerConfig, assets.config);

  const [rangeManifest, graphManifest, pytorchReference] = await Promise.all([
    timed('range manifests', stages, () =>
      readJson<Record<string, unknown>>(new URL('range-manifest.json', base).href)
    ),
    readJson<{weights: readonly Record<string, unknown>[]}>(new URL('external-weight-manifest.json', base).href),
    readJson<{
      inputIds: number[];
      attentionMask: number[];
      shape: number[];
      values: number[];
      mode: string;
    }>(pytorchReferenceUrl)
  ]);
  const fullTextGraph = graphManifest.weights.length === 354;
  const decisionReference = fullTextGraph ? await readJson<KananaDecisionReference>(decisionReferenceUrl) : undefined;
  const weightSource =
    graphManifest.weights.length === 354
      ? await openKananaWeights({
          base: base.href,
          manifest: rangeManifest,
          requiredWeights: graphManifest.weights,
          cache: false
        })
      : await openSafetensorsWeights({
          base: base.href,
          manifest: rangeManifest,
          requiredWeights: graphManifest.weights,
          cache: false
        });

  const gpuSetup = await tvmjs.detectGPUDevice();
  if (!gpuSetup) throw new Error('This browser does not expose WebGPU.');
  const {adapter, device} = gpuSetup;
  const webgpuErrors: string[] = [];
  const cleanupErrors: string[] = [];
  device.addEventListener('uncapturederror', (event) => {
    const error = (event as Event & {error?: {message?: string}}).error;
    webgpuErrors.push(error?.message ?? 'Unspecified WebGPU error.');
  });

  const wasmResponse = await fetch(new URL('kanana-text-model.wasm', base));
  if (!wasmResponse.ok) throw new Error(`Could not read Kanana TVM WASM: HTTP ${wasmResponse.status}.`);
  const wasmBytes = await wasmResponse.arrayBuffer();
  const setup = (await timed('TVM WebGPU runtime', stages, () =>
    setupKananaTvmRuntime({
      instantiate: tvmjs.instantiate as never,
      createPolyfillWASI: tvmjs.createPolyfillWASI,
      device,
      bytes: wasmBytes
    })
  )) as unknown as KananaTvmInferenceSessionOptions['setup'];

  let inference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>> | undefined;
  let decision: ReturnType<typeof createKananaDecisionSession> | undefined;
  let loadedWeights = 0;
  let loadedBytes = 0;
  try {
    inference = await timed('verified graph weights', stages, () =>
      createKananaTvmInferenceSession({
        setup,
        weightSource: {
          manifest: {weights: weightSource.records},
          async readWeight(record) {
            const weight = await weightSource.readTensor(record.name);
            loadedWeights++;
            loadedBytes += record.length;
            if (loadedWeights % 16 === 0 || loadedWeights === weightSource.records.length)
              await window.__kananaProgress?.({
                stage: 'weight loading',
                state: 'in_progress',
                loadedWeights,
                totalWeights: weightSource.records.length,
                loadedBytes
              });
            return weight.bytes;
          }
        }
      })
    );
    const hiddenAttempt = await inferRealHiddenState(
      inference,
      pytorchReference,
      stages,
      device,
      adapter,
      webgpuErrors,
      cleanupErrors
    );
    if (hiddenAttempt.kind === 'execution-failure') return hiddenAttempt.result;
    const {hiddenState} = hiddenAttempt;
    const hiddenStateComparison = compareValues(hiddenState.values, pytorchReference.values);
    const common = {
      model: 'kakaocorp/kanana-2-v-decide-3b-2611',
      revision: KANANA_REVISION,
      hiddenState: {shape: hiddenState.shape, ...hiddenStateComparison},
      gpu: {
        vendor: adapter.info?.vendor ?? '',
        architecture: adapter.info?.architecture ?? '',
        maxBufferSize: device.limits.maxBufferSize,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        features: Array.from(device.features)
      },
      webgpuErrors,
      cleanupErrors,
      stages
    };

    if (!fullTextGraph) {
      return {
        kind: 'hidden-state-probe' as const,
        ...common,
        mode: pytorchReference.mode
      };
    }

    decision = createKananaDecisionSession({
      tokenizer,
      infer: ({inputIds, attentionMask}) => inference!.infer(inputIds, attentionMask),
      dispose: () => inference!.dispose()
    });

    if (!decisionReference) throw new Error('The full Kanana decision reference is missing.');
    const decisionResult = await runDecisionReference(decision, decisionReference, stages);
    return {
      kind: 'decision' as const,
      ...common,
      ...decisionResult
    };
  } finally {
    try {
      if (decision) await decision.dispose();
      else if (inference) await inference.dispose();
      else setup.dispose();
    } catch (error) {
      cleanupErrors.push(error instanceof Error ? error.message : String(error));
    }
    device.destroy();
  }
}
