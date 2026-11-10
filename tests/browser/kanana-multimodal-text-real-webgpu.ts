import {
  createKananaTvmInferenceSession,
  type KananaTvmInferenceSessionOptions
} from '../../packages/web-decision/src/kanana/tvm-session.js';
import {KANANA_IMAGE_TOKEN_ID} from '../../packages/web-decision/src/kanana/contract.mjs';
import {openKananaWeightSource} from '../../packages/web-decision/src/kanana/weight-source.mjs';
import {answerKananaQuestion} from '../../packages/web-decision/src/kanana/protocol.ts';
import {
  dotKananaVectors,
  poolKananaHiddenStates,
  scaleKananaLogits
} from '../../packages/web-decision/src/kanana/scoring.ts';
import {setupKananaTvmRuntime} from '../../packages/web-decision/src/kanana/tvm-runtime.js';

type SafetensorsWeightSource = {
  readTensor(name: string): Promise<{bytes: Uint8Array}>;
};

type SafetensorsWeightOpener = (options: {
  base: string;
  manifest: Record<string, unknown>;
  requiredWeights: readonly Record<string, unknown>[];
  cache: false;
}) => Promise<SafetensorsWeightSource>;

const openWeights = openKananaWeightSource as unknown as SafetensorsWeightOpener;

type TvmJs = {
  instantiate(
    bytes: ArrayBuffer,
    imports: Record<string, unknown>,
    logger: (message: string) => void
  ): Promise<unknown>;
  createPolyfillWASI(): Record<string, unknown>;
  detectGPUDevice(): Promise<{adapter: GPUAdapter; device: GPUDevice} | undefined>;
};

type ExternalWeight = {
  name: string;
  parameterName: string;
  shape: number[];
  dtype: string;
  length: number;
  assetOffset?: number;
  assetLength?: number;
  assetSha256?: string;
};

type SequenceParityCase = {
  name: string;
  sequenceLength: number;
  shape: number[];
  inputIds: number[];
  attentionMask: number[];
  imagePositions: number[];
  positionIds: number[];
  referenceValues: number[];
};

type DecisionProbe = {
  decisionTemperature: number;
  embeddingDimension: number;
  cases: {
    questionId: string;
    question: Record<string, unknown>;
    candidateKeys: string[];
    answer: Record<string, unknown>;
    inputs: {role: string; inputIds: number[]; referenceVector: number[]}[];
  }[];
};

type DecisionProbeCase = DecisionProbe['cases'][number];

type MultimodalTextReport = {
  modelId: string;
  modelRevision: string;
  mode: string;
  checkpointLoaded: boolean;
  multimodalInputs: boolean;
  weightSource: string;
  externalWeightAssetBytes?: number;
  externalWeightAssetSha256?: string;
  effectiveTextConfig: {vocabSize: number; embeddingDimension: number};
};

type TextReference = {
  inputIds: number[];
  attentionMask: number[];
  shape: number[];
  values: number[];
  imagePositions: number[];
  positionIds: number[][][];
  imageEmbeddings: number[][];
};

type TextGraphMode = {
  reducedGraph: boolean;
  checkpointOneLayerGraph: boolean;
  checkpointFullGraph: boolean;
  maxAbsTolerance: number;
  rmseTolerance: number;
  classification: string;
};

type MultimodalTextAssets = {
  base: URL;
  report: MultimodalTextReport;
  manifest: {weights: ExternalWeight[]};
  sequenceParity: {cases: SequenceParityCase[]};
  reference: TextReference;
  wasmBytes: Uint8Array;
  bundledWeights: Uint8Array | undefined;
  decisionProbe: DecisionProbe | undefined;
  mode: TextGraphMode;
};

declare global {
  interface Window {
    tvmjs?: TvmJs;
  }
}

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}: HTTP ${response.status}.`);
  return (await response.json()) as T;
}

async function readBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}: HTTP ${response.status}.`);
  return new Uint8Array(await response.arrayBuffer());
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', copy.buffer);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
}

function compare(actual: Float32Array, expected: readonly number[]) {
  if (actual.length !== expected.length) throw new Error('Text graph output differs from the PyTorch reference shape.');
  let maxAbsDelta = 0;
  let sumSquaredDelta = 0;
  for (let index = 0; index < actual.length; index++) {
    const delta = actual[index]! - expected[index]!;
    maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta));
    sumSquaredDelta += delta * delta;
  }
  return {maxAbsDelta, rmse: Math.sqrt(sumSquaredDelta / actual.length)};
}

function linearPositionIds(sequenceLength: number): Int32Array {
  const positions = Int32Array.from({length: sequenceLength}, (_, index) => index);
  const result = new Int32Array(4 * sequenceLength);
  for (let axis = 0; axis < 4; axis++) result.set(positions, axis * sequenceLength);
  return result;
}

function validateDecisionProbe(probe: DecisionProbe): void {
  if (!Number.isSafeInteger(probe.embeddingDimension) || probe.embeddingDimension < 1)
    throw new Error('The pinned decision fixture has an invalid embedding dimension.');
}

function validateDecisionProbeCase(probeCase: DecisionProbeCase): void {
  if (
    probeCase.inputs.length !== probeCase.candidateKeys.length + 1 ||
    probeCase.inputs[0]?.role !== 'query' ||
    probeCase.inputs.slice(1).some(({role}) => role !== 'candidate')
  )
    throw new Error(`The ${probeCase.questionId} fixture has inconsistent query and candidate inputs.`);
}

async function embedProbeInput(
  inference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>>,
  input: DecisionProbeCase['inputs'][number],
  embeddingDimension: number
): Promise<Float32Array> {
  const inputIds = Int32Array.from(input.inputIds);
  const attentionMask = new Int32Array(inputIds.length).fill(1);
  const hidden = await inference.infer(inputIds, attentionMask, {
    imageEmbeddings: {shape: [0, embeddingDimension], values: new Float32Array(0)},
    imagePositions: new Int32Array(0),
    positionIds: linearPositionIds(inputIds.length)
  });
  return poolKananaHiddenStates(
    hidden.values,
    attentionMask,
    hidden.shape as readonly [number, number, number],
    embeddingDimension
  ).values;
}

async function embedProbeInputs(
  inference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>>,
  inputs: DecisionProbeCase['inputs'],
  embeddingDimension: number
): Promise<Float32Array[]> {
  const vectors: Float32Array[] = [];
  for (const input of inputs) vectors.push(await embedProbeInput(inference, input, embeddingDimension));
  return vectors;
}

function calculateProbeScores(
  probeCase: DecisionProbeCase,
  actualVectors: readonly Float32Array[],
  decisionTemperature: number
) {
  const referenceQuery = Float32Array.from(probeCase.inputs[0]!.referenceVector);
  const actualQuery = actualVectors[0]!;
  const referenceScores = scaleKananaLogits(
    probeCase.inputs.slice(1).map(({referenceVector}) => dotKananaVectors(referenceVector, referenceQuery)),
    decisionTemperature
  );
  const actualScores = scaleKananaLogits(
    actualVectors.slice(1).map((vector) => dotKananaVectors(vector, actualQuery)),
    decisionTemperature
  );
  return {referenceScores, actualScores};
}

function answerKeyFor(type: unknown): string {
  return type === 'choice' ? 'choice' : type === 'score' ? 'score' : 'noul';
}

function validateReferenceAnswer(
  probeCase: DecisionProbeCase,
  expectedAnswer: Record<string, unknown>,
  answerKey: string
): void {
  const type = probeCase.question.type;
  if (type === 'choice' && expectedAnswer.choice !== probeCase.answer.choice)
    throw new Error(`The ${probeCase.questionId} fixture scores do not reproduce the pinned choice.`);
  if (
    (type === 'score' || type === 'noul') &&
    Math.abs((expectedAnswer[answerKey] as number) - (probeCase.answer[answerKey] as number)) > 1e-5
  )
    throw new Error(`The ${probeCase.questionId} fixture scores do not reproduce the pinned answer.`);
}

function maxProbabilityDelta(
  keys: readonly string[],
  expected: Record<string, number> | undefined,
  actual: Record<string, number> | undefined
): number | null {
  if (!expected || !actual) return null;
  return Math.max(...keys.map((key) => Math.abs(actual[key]! - expected[key]!)));
}

function compareProbeAnswers(
  probeCase: DecisionProbeCase,
  expectedAnswer: Record<string, unknown>,
  actualAnswer: Record<string, unknown>,
  answerKey: string
) {
  const type = probeCase.question.type;
  const expectedProbabilities = expectedAnswer.probabilities as Record<string, number> | undefined;
  const actualProbabilities = actualAnswer.probabilities as Record<string, number> | undefined;
  const maxAbsProbabilityDelta = maxProbabilityDelta(
    probeCase.candidateKeys,
    expectedProbabilities,
    actualProbabilities
  );
  const answerDelta =
    type === 'choice' ? 0 : Math.abs((actualAnswer[answerKey] as number) - (probeCase.answer[answerKey] as number));
  const confidenceDelta =
    typeof expectedAnswer.confidence === 'number' && typeof actualAnswer.confidence === 'number'
      ? Math.abs(actualAnswer.confidence - expectedAnswer.confidence)
      : null;
  const withinOutputTolerance =
    (type === 'choice'
      ? actualAnswer.choice === probeCase.answer.choice
      : answerDelta < (type === 'score' ? 0.003 : 0.002)) &&
    (maxAbsProbabilityDelta === null || maxAbsProbabilityDelta < 0.002) &&
    (confidenceDelta === null || confidenceDelta < 0.002);
  return {maxAbsProbabilityDelta, answerDelta, confidenceDelta, withinOutputTolerance};
}

async function runDecisionProbeCase(
  inference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>>,
  probe: DecisionProbe,
  probeCase: DecisionProbeCase
) {
  validateDecisionProbeCase(probeCase);
  const actualVectors = await embedProbeInputs(inference, probeCase.inputs, probe.embeddingDimension);
  const {referenceScores, actualScores} = calculateProbeScores(probeCase, actualVectors, probe.decisionTemperature);
  const expectedAnswer = answerKananaQuestion(probeCase.question, probeCase.candidateKeys, referenceScores);
  const actualAnswer = answerKananaQuestion(probeCase.question, probeCase.candidateKeys, actualScores);
  const answerKey = answerKeyFor(probeCase.question.type);
  validateReferenceAnswer(probeCase, expectedAnswer, answerKey);
  const scoreDelta = compare(Float32Array.from(actualScores), referenceScores);
  const answerDeltas = compareProbeAnswers(probeCase, expectedAnswer, actualAnswer, answerKey);
  return {
    questionId: probeCase.questionId,
    type: probeCase.question.type,
    expectedAnswer: probeCase.answer,
    actualAnswer,
    answerKey,
    expectedValue: probeCase.answer[answerKey],
    actualValue: actualAnswer[answerKey],
    ...answerDeltas,
    candidateKeys: probeCase.candidateKeys,
    expectedScores: referenceScores,
    actualScores,
    maxAbsScoreDelta: scoreDelta.maxAbsDelta,
    scoreRmse: scoreDelta.rmse
  };
}

async function runDecisionProbe(
  inference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>>,
  probe: DecisionProbe
) {
  validateDecisionProbe(probe);
  const cases: Awaited<ReturnType<typeof runDecisionProbeCase>>[] = [];
  for (const probeCase of probe.cases) cases.push(await runDecisionProbeCase(inference, probe, probeCase));
  return {
    cases,
    withinOutputTolerance: cases.every(({withinOutputTolerance}) => withinOutputTolerance),
    maxAbsScoreDelta: Math.max(...cases.map(({maxAbsScoreDelta}) => maxAbsScoreDelta)),
    maxAbsProbabilityDelta: Math.max(...cases.map(({maxAbsProbabilityDelta}) => maxAbsProbabilityDelta ?? 0)),
    maxAbsAnswerDelta: Math.max(...cases.map(({answerDelta}) => answerDelta)),
    maxAbsConfidenceDelta: Math.max(...cases.map(({confidenceDelta}) => confidenceDelta ?? 0))
  };
}

function normalizeAssetsBase(assetsBase: string): URL {
  const base = new URL(assetsBase);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return base;
}

function classifyTextGraph(report: MultimodalTextReport): TextGraphMode {
  const reducedGraph = report.mode === 'reduced' && !report.checkpointLoaded;
  const checkpointOneLayerGraph = report.mode === 'checkpoint_one_layer' && report.checkpointLoaded;
  const checkpointFullGraph = report.mode === 'checkpoint_full' && report.checkpointLoaded;
  if ((!reducedGraph && !checkpointOneLayerGraph && !checkpointFullGraph) || !report.multimodalInputs)
    throw new Error('The WebGPU graph must be a reduced, one-layer, or full-checkpoint multimodal graph.');
  if (report.weightSource === 'pinned-safetensors' && !checkpointFullGraph)
    throw new Error('Only the pinned full-checkpoint graph can read weights from Safetensors ranges.');
  return {
    reducedGraph,
    checkpointOneLayerGraph,
    checkpointFullGraph,
    maxAbsTolerance: checkpointFullGraph ? 0.12 : 0.03,
    rmseTolerance: checkpointFullGraph ? 0.02 : 0.005,
    classification: checkpointFullGraph
      ? 'pinned-checkpoint full 32-layer multimodal hidden-state parity'
      : checkpointOneLayerGraph
        ? 'pinned-checkpoint one-text-layer multimodal parity; not full 32-layer inference'
        : 'reduced random-init multimodal runtime smoke; not checkpoint inference'
  };
}

async function loadMultimodalTextAssets(assetsBase: string): Promise<MultimodalTextAssets> {
  const base = normalizeAssetsBase(assetsBase);
  const report = await readJson<MultimodalTextReport>(new URL('prototype-report.json', base).href);
  const [manifest, sequenceParity, reference, wasmBytes] = await Promise.all([
    readJson<{weights: ExternalWeight[]}>(new URL('external-weight-manifest.json', base).href),
    readJson<{cases: SequenceParityCase[]}>(new URL('sequence-parity-cases.json', base).href),
    readJson<TextReference>(new URL('pytorch-reference.json', base).href),
    readBytes(new URL('kanana-text-model.wasm', base).href)
  ]);
  const bundledWeights =
    report.weightSource === 'pinned-safetensors'
      ? undefined
      : await readBytes(new URL('external-weights.bin', base).href);
  const mode = classifyTextGraph(report);
  const decisionProbe = mode.checkpointFullGraph
    ? await readJson<DecisionProbe>(new URL('decision-probe.json', base).href)
    : undefined;
  return {base, report, manifest, sequenceParity, reference, wasmBytes, bundledWeights, decisionProbe, mode};
}

async function verifyBundledWeightAsset(
  report: MultimodalTextReport,
  bundledWeights: Uint8Array | undefined
): Promise<void> {
  if (bundledWeights) {
    if ((await sha256(bundledWeights)) !== report.externalWeightAssetSha256)
      throw new Error('The graph weight payload failed its SHA-256 check.');
    if (bundledWeights.byteLength !== report.externalWeightAssetBytes)
      throw new Error('The graph weight payload length differs from its build report.');
  }
}

type TvmInference = Awaited<ReturnType<typeof createKananaTvmInferenceSession>>;
type RuntimeSetup = Awaited<ReturnType<typeof setupKananaTvmRuntime>>;

async function readSafetensorsSource(assets: MultimodalTextAssets): Promise<SafetensorsWeightSource | undefined> {
  if (assets.report.weightSource !== 'pinned-safetensors') return undefined;
  return openWeights({
    base: assets.base.href,
    manifest: await readJson<Record<string, unknown>>(new URL('range-manifest.json', assets.base).href),
    requiredWeights: assets.manifest.weights.map(({name, parameterName, shape, dtype, length}) => ({
      name,
      parameterName,
      shape,
      dtype,
      length
    })),
    cache: false
  });
}

async function readBundledWeight(
  record: {name: string},
  weights: ReadonlyMap<string, ExternalWeight>,
  bundledWeights: Uint8Array | undefined
): Promise<Uint8Array> {
  const weight = weights.get(record.name);
  if (!weight || !bundledWeights || weight.assetOffset === undefined || !weight.assetSha256)
    throw new Error(`The graph has no verified payload for ${record.name}.`);
  const assetLength = weight.assetLength ?? weight.length;
  const bytes = bundledWeights.subarray(weight.assetOffset, weight.assetOffset + assetLength);
  if (bytes.byteLength !== weight.length || (await sha256(bytes)) !== weight.assetSha256)
    throw new Error(`The graph weight ${record.name} failed its length or SHA-256 check.`);
  return bytes;
}

async function createTextInferenceSession(assets: MultimodalTextAssets, setup: RuntimeSetup): Promise<TvmInference> {
  const weights = new Map(assets.manifest.weights.map((weight) => [weight.name, weight]));
  const safetensorsSource = await readSafetensorsSource(assets);
  return createKananaTvmInferenceSession({
    setup: setup as unknown as KananaTvmInferenceSessionOptions['setup'],
    embeddingSize: assets.report.effectiveTextConfig.embeddingDimension,
    imageTokenId: assets.mode.reducedGraph ? 63 : KANANA_IMAGE_TOKEN_ID,
    weightSource: {
      manifest: {weights: assets.manifest.weights},
      async readWeight(record) {
        if (safetensorsSource) return (await safetensorsSource.readTensor(record.name)).bytes;
        return readBundledWeight(record, weights, assets.bundledWeights);
      }
    }
  });
}

function assertWebGpuClean(errors: readonly string[]): void {
  if (errors.length) throw new Error(`WebGPU reported errors: ${errors.join('; ')}`);
}

function assertOutputShape(actual: readonly number[], expected: readonly number[]): void {
  if (actual.some((dimension, index) => dimension !== expected[index]))
    throw new Error(`TVM returned [${actual}] instead of [${expected}].`);
}

function assertOutputTolerance(
  comparison: ReturnType<typeof compare>,
  mode: TextGraphMode,
  label: string,
  details: Record<string, unknown> = {}
): void {
  if (comparison.maxAbsDelta >= mode.maxAbsTolerance || comparison.rmse >= mode.rmseTolerance)
    throw new Error(
      `${label}: ${JSON.stringify({
        ...comparison,
        maxAbsTolerance: mode.maxAbsTolerance,
        rmseTolerance: mode.rmseTolerance,
        ...details
      })}.`
    );
}

async function runPrimaryParity(
  inference: TvmInference,
  assets: MultimodalTextAssets,
  device: GPUDevice,
  webgpuErrors: string[]
) {
  const {reference} = assets;
  const hidden = await inference.infer(reference.inputIds, reference.attentionMask, {
    imageEmbeddings: {
      shape: [reference.imageEmbeddings.length, assets.report.effectiveTextConfig.embeddingDimension],
      values: Float32Array.from(reference.imageEmbeddings.flat())
    },
    imagePositions: reference.imagePositions,
    positionIds: Int32Array.from(reference.positionIds.flat(2))
  });
  await device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assertWebGpuClean(webgpuErrors);
  const comparison = compare(hidden.values, reference.values);
  assertOutputShape(hidden.shape, reference.shape);
  assertOutputTolerance(comparison, assets.mode, 'Multimodal parity exceeded tolerance', {
    actual: hidden.values.slice(0, 12),
    expected: reference.values.slice(0, 12)
  });
  return {hidden, comparison};
}

async function runDynamicParityCases(
  inference: TvmInference,
  assets: MultimodalTextAssets,
  device: GPUDevice,
  webgpuErrors: string[]
): Promise<{sequenceLength: number; maxAbsDelta: number; rmse: number}[]> {
  const dynamicCases: {sequenceLength: number; maxAbsDelta: number; rmse: number}[] = [];
  const embeddingDimension = assets.report.effectiveTextConfig.embeddingDimension;
  const cases = assets.sequenceParity.cases.filter(({imagePositions}) => imagePositions.length === 0);
  for (const testCase of cases) {
    const dynamicOutput = await inference.infer(testCase.inputIds, testCase.attentionMask, {
      imageEmbeddings: {shape: [0, embeddingDimension], values: new Float32Array(0)},
      imagePositions: testCase.imagePositions,
      positionIds: Int32Array.from(testCase.positionIds)
    });
    await device.queue.onSubmittedWorkDone();
    assertWebGpuClean(webgpuErrors);
    const dynamicComparison = compare(dynamicOutput.values, testCase.referenceValues);
    assertOutputShape(dynamicOutput.shape, testCase.shape);
    assertOutputTolerance(
      dynamicComparison,
      assets.mode,
      `Dynamic no-image parity exceeded tolerance at length ${testCase.sequenceLength}`
    );
    dynamicCases.push({sequenceLength: testCase.sequenceLength, ...dynamicComparison});
  }
  return dynamicCases;
}

function installWebGpuErrorCapture(device: GPUDevice): string[] {
  const webgpuErrors: string[] = [];
  device.addEventListener('uncapturederror', (event) => {
    const error = (event as Event & {error?: {message?: string}}).error;
    webgpuErrors.push(error?.message ?? 'Unspecified WebGPU error.');
  });
  return webgpuErrors;
}

function createTextParityReport(
  assets: MultimodalTextAssets,
  adapter: GPUAdapter,
  hidden: Awaited<ReturnType<TvmInference['infer']>>,
  comparison: ReturnType<typeof compare>,
  dynamicCases: {sequenceLength: number; maxAbsDelta: number; rmse: number}[],
  decisionScoreParity: Awaited<ReturnType<typeof runDecisionProbe>> | undefined,
  webgpuErrors: string[]
) {
  return {
    model: assets.report.modelId,
    revision: assets.report.modelRevision,
    classification: assets.mode.classification,
    gpu: {vendor: adapter.info?.vendor ?? '', architecture: adapter.info?.architecture ?? ''},
    weights: assets.manifest.weights.length,
    loadedWeightBytes:
      assets.bundledWeights?.byteLength ?? assets.manifest.weights.reduce((total, weight) => total + weight.length, 0),
    shape: hidden.shape,
    ...comparison,
    dynamicCases,
    ...(decisionScoreParity ? {decisionScoreParity} : {}),
    webgpuErrors
  };
}

export async function runKananaMultimodalTextRealWebGpu(assetsBase: string) {
  const tvmjs = window.tvmjs;
  if (!tvmjs) throw new Error('Apache TVM JS runtime bundle is not loaded.');
  const assets = await loadMultimodalTextAssets(assetsBase);
  await verifyBundledWeightAsset(assets.report, assets.bundledWeights);
  const adapterSetup = await tvmjs.detectGPUDevice();
  if (!adapterSetup) throw new Error('This browser does not expose WebGPU.');
  const {adapter, device} = adapterSetup;
  const webgpuErrors = installWebGpuErrorCapture(device);
  const setup = await setupKananaTvmRuntime({
    instantiate: tvmjs.instantiate as never,
    createPolyfillWASI: tvmjs.createPolyfillWASI,
    device,
    bytes: assets.wasmBytes
  });
  let inference: TvmInference | undefined;
  try {
    inference = await createTextInferenceSession(assets, setup);
    const {hidden, comparison} = await runPrimaryParity(inference, assets, device, webgpuErrors);
    const dynamicCases = await runDynamicParityCases(inference, assets, device, webgpuErrors);
    const decisionScoreParity = assets.decisionProbe
      ? await runDecisionProbe(inference, assets.decisionProbe)
      : undefined;
    return createTextParityReport(assets, adapter, hidden, comparison, dynamicCases, decisionScoreParity, webgpuErrors);
  } finally {
    if (inference) await inference.dispose();
    else setup.dispose();
  }
}
