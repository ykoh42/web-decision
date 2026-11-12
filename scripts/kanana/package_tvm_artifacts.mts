#!/usr/bin/env node
import {createHash} from 'node:crypto';
import {chmod, copyFile, mkdir, readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {KANANA_MODEL_ID, KANANA_REPOSITORY, KANANA_REVISION} from '../../packages/web-decision/src/kanana/contract.mjs';
import {KANANA_SAFETENSORS_SPEC} from '../../packages/web-decision/src/kanana/safetensors-spec.mjs';

const DEFAULT_OUTPUT = 'packages/web-decision/assets/kanana';
const FILES = Object.freeze([
  ['kanana-text-model.wasm', 'kanana-text-model.wasm', 'textGraph'],
  ['text-weight-manifest.json', 'external-weight-manifest.json', 'textGraph'],
  ['kanana-vision-model.wasm', 'kanana-vision-model.wasm', 'visionGraph'],
  ['vision-weight-manifest.json', 'external-weight-manifest.json', 'visionGraph'],
  ['safetensors-range-manifest.json', 'range-manifest.json', 'textGraph']
] as const);

type PackageOptions = {
  textDirectory?: string;
  visionDirectory?: string;
  outputDirectory: string;
  check: boolean;
};
type JsonObject = Record<string, unknown>;
type BuildReport = JsonObject & {
  modelId?: unknown;
  repository?: unknown;
  modelRevision?: unknown;
  revision?: unknown;
  checkpointSha256?: unknown;
  expectedCheckpointSha256?: unknown;
  mode?: unknown;
  checkpointLoaded?: unknown;
  fullWidthDimensionsUnchanged?: unknown;
  onlyLayerCountReduced?: unknown;
  weightSource?: unknown;
  onnxImportOrExportUsed?: unknown;
  versions?: unknown;
  parameterCount?: unknown;
  sourceParameterDtype?: unknown;
  runtimeParameterDtype?: unknown;
  computeDtype?: unknown;
  float32ComputeModules?: unknown;
  effectiveTextConfig?: unknown;
  onlineAttentionGraphRewrite?: unknown;
  webgpuAttentionLowering?: unknown;
  relaxCpuRun?: unknown;
  relaxWebgpuBuild?: unknown;
  relaxWebgpuWasmExport?: unknown;
  sequenceCases?: unknown;
  maxPatchCount?: unknown;
};
type BuildManifest = JsonObject & {
  modelId?: unknown;
  modelRevision?: unknown;
  weights?: unknown;
};
type RangeManifest = JsonObject & {
  model?: unknown;
  repository?: unknown;
  revision?: unknown;
  path?: unknown;
  totalBytes?: unknown;
  headerSha256?: unknown;
  tensorCount?: unknown;
};
type ArtifactFile = {name: string; source: string; bytes: Uint8Array; sha256: string};

function parseArguments(values: string[]): Required<PackageOptions> {
  const options: PackageOptions = {outputDirectory: DEFAULT_OUTPUT, check: false};
  for (let index = 0; index < values.length; index++) {
    const name = values[index];
    if (name === '--check') {
      options.check = true;
      continue;
    }
    const value = values[index + 1];
    if (!value || value.startsWith('--')) throw new TypeError(`${name} needs a path.`);
    if (name === '--text-dir') options.textDirectory = value;
    else if (name === '--vision-dir') options.visionDirectory = value;
    else if (name === '--output-dir') options.outputDirectory = value;
    else throw new TypeError(`Unknown option: ${name}`);
    index++;
  }
  if (!options.textDirectory || !options.visionDirectory)
    throw new TypeError('Pass --text-dir and --vision-dir for the verified TVM build outputs.');
  return {
    ...options,
    textDirectory: options.textDirectory,
    visionDirectory: options.visionDirectory
  } as Required<PackageOptions>;
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

async function json(path: string): Promise<JsonObject> {
  const value: unknown = JSON.parse(await readFile(path, 'utf8'));
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    throw new TypeError(`${path} must contain a JSON object.`);
  return value as JsonObject;
}

function requireCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function isRecord(value: unknown): value is JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function verifyCpuSequenceCases(report: BuildReport): void {
  requireCondition(report.relaxCpuRun === 'passed', 'Kanana text graph CPU Relax run must pass.');
  requireCondition(report.relaxWebgpuBuild === 'passed', 'Kanana text graph WebGPU Relax build must pass.');
  requireCondition(report.relaxWebgpuWasmExport === 'passed', 'Kanana text graph WebGPU WASM export must pass.');

  const sequenceCases = report.sequenceCases;
  requireCondition(
    Array.isArray(sequenceCases) && sequenceCases.length === 3,
    'Kanana text report must contain the three required CPU sequence parity cases.'
  );
  const requiredCases = [
    {name: 'no-images', sequenceLength: 1, imageTokens: 0},
    {name: 'no-images', sequenceLength: 8, imageTokens: 0},
    {name: 'two-image-tokens', sequenceLength: 8, imageTokens: 2}
  ];
  for (const expected of requiredCases) {
    const matchingCases = (sequenceCases as unknown[]).filter(
      (value): value is JsonObject =>
        isRecord(value) &&
        value.name === expected.name &&
        value.sequenceLength === expected.sequenceLength &&
        value.imageTokens === expected.imageTokens
    );
    requireCondition(
      matchingCases.length === 1,
      `Kanana CPU report must contain exactly one ${expected.name} sequence-length-${expected.sequenceLength} case.`
    );
    const parityCase = matchingCases[0];
    if (!parityCase) throw new Error('Kanana CPU parity case disappeared after validation.');
    const maxAbsDelta = parityCase.maxAbsDelta;
    const rmse = parityCase.rmse;
    requireCondition(
      typeof maxAbsDelta === 'number' && Number.isFinite(maxAbsDelta) && maxAbsDelta < 0.03,
      `Kanana ${expected.name} sequence-length-${expected.sequenceLength} max absolute error must be finite and below 0.03.`
    );
    requireCondition(
      typeof rmse === 'number' && Number.isFinite(rmse) && rmse < 0.005,
      `Kanana ${expected.name} sequence-length-${expected.sequenceLength} RMSE must be finite and below 0.005.`
    );
  }
}

function verifyBuildReports(
  textReport: BuildReport,
  visionReport: BuildReport,
  textManifest: BuildManifest,
  visionManifest: BuildManifest,
  rangeManifest: RangeManifest
): void {
  for (const report of [textReport, visionReport]) {
    requireCondition(
      (report.repository ?? report.modelId) === KANANA_REPOSITORY,
      'Kanana artifact model ID differs from the package pin.'
    );
    requireCondition(
      (report.modelRevision ?? report.revision) === KANANA_REVISION,
      'Kanana artifact revision differs from the latest package pin.'
    );
    requireCondition(
      (report.checkpointSha256 ?? report.expectedCheckpointSha256) === KANANA_SAFETENSORS_SPEC.fileSha256,
      'Kanana artifact checkpoint hash differs from the package pin.'
    );
  }
  requireCondition(
    textReport.mode === 'checkpoint_full' && textReport.checkpointLoaded === true,
    'Kanana text graph must use the full pinned checkpoint.'
  );
  requireCondition(
    textReport.fullWidthDimensionsUnchanged === true && textReport.onlyLayerCountReduced === false,
    'Kanana text graph must preserve all pinned model dimensions and layers.'
  );
  requireCondition(
    textReport.weightSource === 'pinned-safetensors' && textReport.onnxImportOrExportUsed === false,
    'Kanana text graph must load pinned Safetensors weights through native TVM.'
  );
  requireCondition(
    textReport.dynamicSequence === true && textReport.exportedSequenceLengthUpperBound === null,
    'Kanana text graph must preserve runtime sequence length without exporting a max-length allocation bound.'
  );
  requireCondition(
    textReport.sourceParameterDtype === 'bfloat16' && textReport.runtimeParameterDtype === 'float16',
    'Kanana text graph must store the pinned BF16 parameters as float16.'
  );
  requireCondition(textReport.computeDtype === 'float32', 'Kanana text graph compute dtype must be float32.');
  const effectiveTextConfig = textReport.effectiveTextConfig;
  requireCondition(
    isRecord(effectiveTextConfig) && effectiveTextConfig.attentionImplementation === 'web_decision_online',
    'Kanana text graph must use the verified model-specific online attention path.'
  );
  const attentionRewrite = textReport.onlineAttentionGraphRewrite;
  requireCondition(
    isRecord(attentionRewrite) &&
      attentionRewrite.attentionOps === 32 &&
      attentionRewrite.defaultAttentionDecompositionRemoved === true,
    'Kanana text graph must replace all 32 attention operators before TVM Relax import.'
  );
  const attentionLowering = textReport.webgpuAttentionLowering;
  requireCondition(
    isRecord(attentionLowering) &&
      attentionLowering.function === 'kanana_online_attention' &&
      attentionLowering.algorithm === 'tiled_online_softmax' &&
      attentionLowering.sequenceDimension === 'dynamic' &&
      attentionLowering.queryTile === 16 &&
      attentionLowering.keyTile === 16 &&
      attentionLowering.threadsPerBlock === 128 &&
      attentionLowering.causalKeyLimit === 'query_tile_end' &&
      attentionLowering.intermediateScoreMatrix === false &&
      attentionLowering.loweredAttentionOps === 32,
    'Kanana WebGPU graph must lower all 32 attention operators without a sequence-square score matrix.'
  );
  verifyCpuSequenceCases(textReport);
  const float32ComputeModules = textReport.float32ComputeModules;
  requireCondition(
    isRecord(float32ComputeModules) &&
      float32ComputeModules.linearLayers === 224 &&
      float32ComputeModules.rmsNormLayers === 129,
    'Kanana text graph must report float32 compute for all 224 Linear and 129 RMSNorm modules.'
  );
  requireCondition(textManifest.modelId === KANANA_REPOSITORY, 'Kanana text graph manifest has the wrong model.');
  requireCondition(
    textManifest.modelRevision === KANANA_REVISION,
    'Kanana text graph manifest has the wrong revision.'
  );
  requireCondition(
    Array.isArray(textManifest.weights) && textManifest.weights.length === 354,
    'Kanana text graph must have all 354 graph weights.'
  );
  requireCondition(
    (textManifest.weights as unknown[]).every(
      (weight) => isRecord(weight) && weight.dtype === 'float16' && weight.sourceDtype === 'bfloat16'
    ),
    'Kanana text graph weights must be float16 values converted from pinned BF16 source weights.'
  );
  requireCondition(
    Array.isArray(visionManifest.weights) && visionManifest.weights.length === 333,
    'Kanana vision graph must have all 333 graph weights.'
  );
  requireCondition(
    rangeManifest.model === KANANA_MODEL_ID &&
      rangeManifest.repository === KANANA_REPOSITORY &&
      rangeManifest.revision === KANANA_REVISION &&
      rangeManifest.path === KANANA_SAFETENSORS_SPEC.path &&
      rangeManifest.totalBytes === KANANA_SAFETENSORS_SPEC.totalBytes &&
      rangeManifest.headerSha256 === KANANA_SAFETENSORS_SPEC.headerSha256 &&
      rangeManifest.tensorCount === KANANA_SAFETENSORS_SPEC.tensorCount,
    'Kanana Safetensors range manifest differs from the pinned source.'
  );
}

async function collectFiles(
  textDirectory: string,
  visionDirectory: string
): Promise<{
  files: ArtifactFile[];
  textReport: BuildReport;
  visionReport: BuildReport;
}> {
  const sources = new Map([
    ['textGraph', resolve(textDirectory)],
    ['visionGraph', resolve(visionDirectory)]
  ]);
  const result: ArtifactFile[] = [];
  for (const [name, sourceName, graph] of FILES) {
    const sourceDirectory = sources.get(graph);
    if (!sourceDirectory) throw new Error(`Unknown artifact graph: ${graph}`);
    const source = resolve(sourceDirectory, sourceName);
    const bytes = await readFile(source);
    result.push({name, source, bytes, sha256: sha256(bytes)});
  }
  const [textRange, visionRange, textReport, visionReport, textManifest, visionManifest] = await Promise.all([
    json(resolve(textDirectory, 'range-manifest.json')),
    json(resolve(visionDirectory, 'range-manifest.json')),
    json(resolve(textDirectory, 'prototype-report.json')),
    json(resolve(visionDirectory, 'build-report.json')),
    json(resolve(textDirectory, 'external-weight-manifest.json')),
    json(resolve(visionDirectory, 'external-weight-manifest.json'))
  ]);
  requireCondition(
    sha256(Buffer.from(JSON.stringify(textRange))) === sha256(Buffer.from(JSON.stringify(visionRange))),
    'Kanana text and vision builds must use the same Safetensors range manifest.'
  );
  verifyBuildReports(
    textReport as BuildReport,
    visionReport as BuildReport,
    textManifest as BuildManifest,
    visionManifest as BuildManifest,
    textRange as RangeManifest
  );
  return {files: result, textReport: textReport as BuildReport, visionReport: visionReport as BuildReport};
}

function packageManifest(files: ArtifactFile[], textReport: BuildReport, visionReport: BuildReport): JsonObject {
  const versions = textReport.versions;
  const tvmVersion =
    versions !== null && typeof versions === 'object' && !Array.isArray(versions)
      ? (versions as JsonObject).tvm
      : undefined;
  return {
    format: 'web-decision-kanana-tvm-artifacts-v1',
    modelId: KANANA_MODEL_ID,
    repository: KANANA_REPOSITORY,
    revision: KANANA_REVISION,
    checkpointSha256: KANANA_SAFETENSORS_SPEC.fileSha256,
    modifications: {
      modifiedBy: 'web-decision',
      files: FILES.map(([name]) => name),
      description: 'web-decision generated TVM WebGPU modules and weight manifests from the pinned Kanana checkpoint.'
    },
    runtime: {
      tvm: tvmVersion,
      textGraph: {
        mode: textReport.mode,
        weights: textReport.parameterCount,
        graphWeights: 354,
        parameterDtype: textReport.runtimeParameterDtype,
        computeDtype: 'float32',
        attention: {
          implementation: 'web_decision_online',
          lowering: 'kanana_online_attention',
          algorithm: 'tiled_online_softmax',
          sequenceDimension: 'dynamic',
          queryTile: 16,
          keyTile: 16,
          threadsPerBlock: 128,
          causalKeyLimit: 'query_tile_end',
          intermediateScoreMatrix: false
        }
      },
      visionGraph: {graphWeights: 333, maxPatchCount: visionReport.maxPatchCount}
    },
    files: Object.fromEntries(files.map(({name, bytes, sha256}) => [name, {bytes: bytes.byteLength, sha256}]))
  };
}

async function verifyOrWrite(
  outputDirectory: string,
  files: ArtifactFile[],
  manifest: JsonObject,
  check: boolean
): Promise<void> {
  const destination = resolve(outputDirectory);
  if (!check) await mkdir(destination, {recursive: true});
  for (const {name, source, bytes, sha256: expectedSha256} of files) {
    const path = resolve(destination, name);
    if (check) {
      const existing = await readFile(path);
      requireCondition(
        existing.byteLength === bytes.byteLength && sha256(existing) === expectedSha256,
        `Packaged Kanana artifact is stale: ${name}.`
      );
    } else if (sha256(await readFile(source)) === expectedSha256) {
      await copyFile(source, path);
      await chmod(path, 0o644);
    }
  }

  const manifestPath = resolve(destination, 'manifest.json');
  const expected = `${JSON.stringify(manifest, null, 2)}\n`;
  if (check) {
    requireCondition((await readFile(manifestPath, 'utf8')) === expected, 'Packaged Kanana manifest is stale.');
  } else {
    await writeFile(manifestPath, expected);
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const {files, textReport, visionReport} = await collectFiles(options.textDirectory, options.visionDirectory);
  const manifest = packageManifest(files, textReport, visionReport);
  await verifyOrWrite(options.outputDirectory, files, manifest, options.check);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
