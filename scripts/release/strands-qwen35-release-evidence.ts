import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream, readFileSync} from 'node:fs';
import {lstat, mkdir, readFile, realpath, stat, writeFile} from 'node:fs/promises';
import {dirname, resolve, sep} from 'node:path';
import {isDeepStrictEqual} from 'node:util';
import type {ArchiveInfo} from './verify-release-evidence.ts';
import {canonicalSha256, verifyArchiveInstall} from './verify-release-evidence.ts';

const QWEN35_ASSET_MANIFEST_BYTES = readFileSync(
  new URL('../../packages/web-decision/assets/qwen35/manifest.json', import.meta.url)
);
const QWEN35_ASSET_MANIFEST_VALUE = JSON.parse(QWEN35_ASSET_MANIFEST_BYTES.toString('utf8')) as unknown;
const QWEN35_ASSET_MANIFEST_RAW_SHA256 = createHash('sha256').update(QWEN35_ASSET_MANIFEST_BYTES).digest('hex');
const QWEN35_PREFIX_CACHE_REPORT_PATH =
  'scripts/strands/qwen35/reports/qwen35-prefix-cache-equivalence-cpu-r1-2026-11-16.json';
const QWEN35_PREFIX_CACHE_REPORT_BYTES = readFileSync(
  new URL(
    '../../scripts/strands/qwen35/reports/qwen35-prefix-cache-equivalence-cpu-r1-2026-11-16.json',
    import.meta.url
  )
);
const QWEN35_PREFIX_CACHE_REPORT_SHA256 = createHash('sha256').update(QWEN35_PREFIX_CACHE_REPORT_BYTES).digest('hex');
const QWEN35_PREFIX_CACHE_REPORT = JSON.parse(QWEN35_PREFIX_CACHE_REPORT_BYTES.toString('utf8')) as RecordValue;
export const QWEN35_MIXED_VISION_REFERENCE = Object.freeze({
  path: 'references/qwen35-image-counts-2-3-source-oracle-r1-2026-11-17/source-reference.json',
  sha256: 'b46bd0cfdc26a30335d6d791e5361c82ae52b222e41a48ecbc2040566680aeeb'
});
export const QWEN35_OFFICIAL_VISION_REFERENCE = Object.freeze({
  path: 'references/qwen35-official-vision-source-reference-r2-2026-11-18/vision-reference.json',
  sha256: '8eeb699becc1020a3ebd314c32e018335f1d4a95a84188e2ec936a4e81cf9569'
});
export const QWEN35_B3_S4096_SOURCE_REFERENCE = Object.freeze({
  path: 'scripts/strands/qwen35/reports/qwen35-full-pointer-b3-s4096-source-reference-r1-2026-11-16.json',
  sha256: 'd18bf9ef341447d34c519da817d140dc818f9a270df4ca06142fe9991d687900'
});
const QWEN35_B3_S4096_PROTOCOL_REFERENCE = Object.freeze({
  path: 'scripts/strands/qwen35/reports/qwen35-source-reference-cpu-r3-mixed-2026-11-15.json',
  sha256: 'e66e7adedadd68932726e0a8dce5e5f603ead2bb96356fa9586081dcc41a97e2'
});
const QWEN35_B3_S4096_LABELS = Object.freeze(['Node.js', 'TVM WebGPU', 'WebGPU']);
const QWEN35_B3_S4096_PROTOCOL_SLOT_LABELS = Object.freeze(['WebGPU', 'Node.js', 'TVM WebGPU']);
const QWEN35_B3_S4096_EXPECTED_CHOICES = Object.freeze(['TVM WebGPU', 'TVM WebGPU', 'TVM WebGPU']);
const QWEN35_B3_S4096_PROBABILITIES = Object.freeze([
  Object.freeze([0.10017982870340347, 0.8295549154281616, 0.0702652782201767]),
  Object.freeze([0.10017982870340347, 0.8295549154281616, 0.0702652782201767]),
  Object.freeze([0.10017982870340347, 0.8295549154281616, 0.0702652782201767])
]);

export const QWEN35_RELEASE_MODEL = Object.freeze({
  // Strands/SystemOneEngine reports the upstream repository name as its model.
  modelId: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
  // The web-decision catalog uses its own stable, curated identifier.
  catalogModelId: 'strands-decider-2b-qwen3.5-v1-2610',
  assetManifestSha256: canonicalSha256(QWEN35_ASSET_MANIFEST_VALUE),
  repository: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
  revision: '216d1ccb46a9e9b036eb9187b248db98480f77d4',
  baseRepository: 'Qwen/Qwen3.5-2B-Base',
  baseRevision: 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
  strandsRepository: 'StrandsAgents/strands-decider',
  strandsRevision: '13c05149265dc404c09d2469636f71b1320e1713',
  maxLength: 4096,
  maxChoiceOptions: 255,
  maxScoreLevels: 10,
  maxImages: 4
});

const HASH_PATTERN = /^[a-f0-9]{64}$/u;
const SOURCE_RUNNER = 'StrandsDeciderModel.load/SystemOneEngine.evaluate';
const VISION_SOURCE_RUNNER = 'StrandsDeciderModel.load/VisionEngine.evaluate';
const MIXED_VISION_SOURCE_RUNNER = 'VisionDeciderModel.load/VisionEngine.evaluate';
const MIXED_VISION_SOURCE_FILE_SHA256 = '705238d4f294b9724bee9026d3365594cd075ce1bf3b4fd0aa97dc49a80cbdd5';
const MIXED_VISION_RESPONSE_MODEL = 'strands-decider-0.2.1.dev7+g13c0514';
const MIXED_VISION_IMAGE_PINS = Object.freeze({
  'vision-input/yellow-square-portrait-320x416.png': {
    bytes: 975,
    sha256: '8e7f591f12138a7aabfa826e0aaea218294032b3fecae3d6fa19342c0f23c9f6'
  },
  'vision-input/green-circle-landscape-416x320.png': {
    bytes: 1209,
    sha256: 'df6e8d522206f7762704b1c16d462332a3f64e7092ac6d323a9dabb409ff6859'
  },
  'vision-input/white-triangle-small-128x128.png': {
    bytes: 358,
    sha256: '43308dd51f240a9933f2f15d23771b0fb69f83dde61dfc235d8f2abd20223ee9'
  }
});
const OUTPUT_ROUNDING_TOLERANCE = 0.0001;
const TEXT_SOURCE_DERIVED_TOLERANCE = 0.00915062427520752;

type RecordValue = Record<string, unknown>;
type AssetMetadata = Readonly<{bytes: number; sha256: string}>;
type SourceExpected = Readonly<{
  model: string;
  usage: Readonly<{input_tokens: number; output_tokens: number}>;
  answers: Record<string, RecordValue>;
}>;
type CandidateUsageExpectation = Readonly<{
  usage: Readonly<{input_tokens: number; output_tokens: number}>;
  mode: 'observed-default-cache-source' | 'derived-source-prefix-groups';
  evidence: RecordValue;
}>;
type SourceCase = Readonly<{
  name: string;
  request: RecordValue;
  choiceLabels?: readonly string[];
  requestMetadata?: RecordValue;
  imageFiles?: readonly Readonly<{path: string; sha256: string; bytes: number; mimeType: string}>[];
  imageGeometry?: readonly Readonly<{width: number; height: number; longSide: number; patchRows: number}>[];
  expected: SourceExpected;
  candidateUsageExpectation?: CandidateUsageExpectation;
  precisionReference?: SourceExpected;
  precisionEnvelope?: RecordValue;
}>;

export interface Qwen35CandidateInputs {
  readonly artifactRoot: string;
  readonly evidenceInput: RecordValue;
  readonly evidenceInputSha256: string;
  readonly manifestPath: string;
  readonly manifestSha256: string;
  readonly manifestRawSha256: string;
  readonly assetManifestSha256: string;
  readonly manifest: RecordValue;
  readonly manifestFiles: Readonly<Record<string, AssetMetadata>>;
  readonly sourceCases: readonly SourceCase[];
  readonly visionCases: readonly SourceCase[];
  readonly boundaryCases: Readonly<Record<string, SourceCase | Readonly<Record<string, unknown>>>>;
  readonly b3S4096Case: Qwen35B3S4096SourceCase;
  readonly fixtureSha256: Readonly<Record<string, string>>;
  readonly artifactSha256: Readonly<Record<string, string>>;
}

export interface Qwen35B3S4096SourceCase {
  readonly name: 'batch-3-context-4096';
  readonly request: RecordValue;
  readonly questionNames: readonly string[];
  readonly labels: readonly string[];
  readonly expectedChoices: readonly string[];
  readonly batchSize: 3;
  readonly sequenceLength: 4096;
  readonly graphInputTokens: 12288;
  readonly expectedProbabilities: readonly (readonly number[])[];
  readonly probabilityTolerance: number;
  readonly expectedUsage: Readonly<{input_tokens: 4250; output_tokens: 3}>;
  readonly sourceReference: Readonly<{path: string; sha256: string}>;
}

export interface Qwen35TvmMainCallSites {
  readonly r19: true;
  readonly vision: true;
  readonly tvmRuntimeDispatch: true;
}

export function assertQwen35TvmMainCallSites(
  textRunnerSourceValue: unknown,
  visionRunnerSourceValue: unknown,
  tvmRuntimeSourceValue: unknown
): Qwen35TvmMainCallSites {
  const textRunnerSource = text(textRunnerSourceValue, 'Qwen exact-package text runner source');
  const visionRunnerSource = text(visionRunnerSourceValue, 'Qwen exact-package vision runner source');
  const tvmRuntimeSource = text(tvmRuntimeSourceValue, 'Qwen exact-package TVM runtime source');
  assert.match(
    textRunnerSource,
    /createVirtualMachine\(\s*device\s*\)\s*\.getFunction\(\s*["']main["']\s*\)/u,
    'Qwen exact package has no R19 TVM VM main function call site.'
  );
  assert.match(
    textRunnerSource,
    /await\s+main\(\s*\.\.\.\s*args\s*\)/u,
    'Qwen exact package does not invoke the R19 TVM VM main function.'
  );
  assert.match(
    visionRunnerSource,
    /createVirtualMachine\(\s*runtime\.webgpu\(\)\s*\)\s*\.getFunction\(\s*["']main["']\s*\)/u,
    'Qwen exact package has no vision TVM VM main function call site.'
  );
  assert.match(
    visionRunnerSource,
    /runImage\(\s*runtime\s*,\s*main\s*,/u,
    'Qwen exact package does not pass vision inputs to the TVM VM main function.'
  );
  assert.match(
    tvmRuntimeSource,
    /compute\.dispatchWorkgroups\(\s*workDim\[0\]\s*,\s*workDim\[1\]\s*,\s*workDim\[2\]\s*\)/u,
    'Pinned TVM.js runtime has no WebGPU compute dispatch call site.'
  );
  return {r19: true, vision: true, tvmRuntimeDispatch: true};
}

function assertMixedVisionProcessor(value: unknown): void {
  const processor = record(value, 'Qwen mixed-vision processor metadata');
  assert.equal(processor.name, 'Qwen2VLImageProcessorPil', 'Qwen mixed-vision processor mismatch.');
  assert.equal(processor.transformersVersion, '5.22.2', 'Qwen mixed-vision Transformers version mismatch.');
  assert.equal(processor.pillowVersion, '13.0.0', 'Qwen mixed-vision Pillow version mismatch.');
  assert.equal(
    processor.configSha256,
    '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
    'Qwen mixed-vision processor config pin mismatch.'
  );
  assert.equal(processor.imageLongSide, 448, 'Qwen mixed-vision long-side limit mismatch.');
  assert.equal(processor.maxImages, QWEN35_RELEASE_MODEL.maxImages, 'Qwen mixed-vision image-count limit mismatch.');
  assert.equal(processor.maxImagePixels, 16_777_216, 'Qwen mixed-vision maximum image-pixel limit mismatch.');
  assert.equal(processor.minPixels, 65_536, 'Qwen mixed-vision minimum pixel limit mismatch.');
  assert.equal(processor.maxPixels, 16_777_216, 'Qwen mixed-vision maximum pixel limit mismatch.');
  assert.equal(processor.patchSize, 16, 'Qwen mixed-vision patch-size mismatch.');
  assert.equal(processor.resizeFactor, 32, 'Qwen mixed-vision resize-factor mismatch.');
}

function assertMixedVisionImagePins(cases: readonly SourceCase[]): void {
  const foundImages = new Set<string>();
  for (const item of cases) {
    assert.equal(item.imageGeometry?.length, item.imageFiles?.length, `${item.name} image geometry is incomplete.`);
    assert.ok(
      (item.imageGeometry ?? []).every(({longSide}) => longSide <= 448),
      `${item.name} exceeds the pinned mixed-vision image envelope.`
    );
    for (const image of item.imageFiles ?? []) {
      const expected = MIXED_VISION_IMAGE_PINS[image.path as keyof typeof MIXED_VISION_IMAGE_PINS];
      assert.ok(expected, `Qwen mixed-vision image path is not pinned: ${image.path}`);
      assert.equal(image.bytes, expected.bytes, `Qwen mixed-vision image size pin mismatch: ${image.path}`);
      assert.equal(image.sha256, expected.sha256, `Qwen mixed-vision image digest pin mismatch: ${image.path}`);
      foundImages.add(image.path);
    }
  }
  assert.deepEqual(
    [...foundImages].sort(),
    Object.keys(MIXED_VISION_IMAGE_PINS).sort(),
    'Qwen mixed-vision cases must cover the pinned source images.'
  );
}

export function assertQwen35MixedVisionSourceReference(value: unknown): readonly SourceCase[] {
  const reference = record(value, 'Qwen mixed-vision source reference');
  assert.equal(reference.format, 'qwen35-image-counts-2-3-source-reference-v1', 'Unknown Qwen mixed-vision format.');
  assertQwen35PassedSourceReference(reference, 'Qwen mixed-vision source reference');
  const source = record(reference.source, 'Qwen mixed-vision source pins');
  assertSourcePins(source, 'Qwen mixed-vision reference', MIXED_VISION_SOURCE_RUNNER);
  assert.equal(source.strandsVersion, '0.2.1.dev7+g13c0514', 'Qwen mixed-vision Strands version mismatch.');
  assert.equal(
    source.sourceVisionFileSha256,
    MIXED_VISION_SOURCE_FILE_SHA256,
    'Qwen mixed-vision source file digest mismatch.'
  );
  assert.equal(source.responseModelName, MIXED_VISION_RESPONSE_MODEL);
  assert.equal(
    source.cpuPrecision,
    'BF16 checkpoint source result plus standard upstream CPU FP32 result',
    'Qwen mixed-vision source precision contract mismatch.'
  );
  assertMixedVisionProcessor(reference.processor);
  const toolchain = record(reference.toolchain, 'Qwen mixed-vision toolchain pins');
  assert.equal(toolchain.python, '3.11.11');
  assert.equal(toolchain.torch, '2.8.0');
  assert.equal(toolchain.transformers, '5.22.2');
  assert.equal(toolchain.pillow, '13.0.0');
  const scope = record(reference.scope, 'Qwen mixed-vision source scope');
  assert.equal(scope.modelInferencePerformed, true, 'Qwen mixed-vision source must record inference.');
  assert.equal(scope.gpuUsed, false, 'Qwen mixed-vision source must be CPU-only.');
  assert.equal(scope.tvmUsed, false, 'Qwen mixed-vision source must be independent of TVM.');
  assert.equal(scope.sourceCpuUpcastApplied, true, 'Qwen mixed-vision source must preserve its FP32 reference run.');
  const cases = Array.isArray(reference.cases)
    ? reference.cases.map((item, index) =>
        sourceCase(item, `Qwen mixed-vision source case ${index}`, MIXED_VISION_RESPONSE_MODEL)
      )
    : [];
  assert.deepEqual(
    cases.map(({name, imageFiles}) => [name, imageFiles?.length]),
    [
      ['two-image-mixed-aspect', 2],
      ['three-image-mixed-aspect', 3]
    ],
    'Qwen mixed-vision reference must contain its pinned two and three image cases in order.'
  );
  for (const item of cases) {
    assert.ok(item.precisionReference, `${item.name} is missing the FP32 precision reference.`);
    assert.ok(item.precisionEnvelope, `${item.name} is missing its complete BF16/FP32 precision envelope.`);
  }
  assertMixedVisionImagePins(cases);
  return cases;
}

export async function loadQwen35MixedVisionSourceReference(
  artifactRoot: string,
  value: unknown
): Promise<Readonly<{path: string; sha256: string; cases: readonly SourceCase[]}>> {
  const entry = record(value, 'Qwen mixed-vision reference entry');
  assert.equal(entry.path, QWEN35_MIXED_VISION_REFERENCE.path, 'Qwen mixed-vision reference path is not pinned.');
  assert.equal(entry.sha256, QWEN35_MIXED_VISION_REFERENCE.sha256, 'Qwen mixed-vision reference digest is not pinned.');
  const root = await realpath(resolve(artifactRoot));
  const pinned = await readPinnedJson(root, entry.path, entry.sha256, 'Qwen mixed-vision source reference');
  const cases = assertQwen35MixedVisionSourceReference(pinned.value);
  for (const item of cases)
    for (const image of item.imageFiles ?? []) {
      const bytes = await readRegularFile(root, image.path);
      assert.equal(bytes.byteLength, image.bytes, `Qwen mixed-vision image size differs: ${image.path}`);
      assert.equal(sha256(bytes), image.sha256, `Qwen mixed-vision image digest differs: ${image.path}`);
    }
  return {path: pinned.path, sha256: pinned.sha256, cases};
}

export interface Qwen35CandidatePackage {
  readonly archive: ArchiveInfo;
  readonly installedRoot: string;
}

function record(value: unknown, description: string): RecordValue {
  assert.ok(value !== null && typeof value === 'object' && !Array.isArray(value), `${description} must be an object.`);
  return value as RecordValue;
}

function text(value: unknown, description: string): string {
  assert.ok(typeof value === 'string' && value.length > 0, `${description} is required.`);
  return value as string;
}

function digest(value: unknown, description: string): string {
  assert.ok(typeof value === 'string' && HASH_PATTERN.test(value), `${description} must be a SHA-256 digest.`);
  return value as string;
}

function safeRelativePath(value: unknown, description: string): string {
  const path = text(value, description);
  assert.ok(!path.includes('\\') && !path.startsWith('/') && !path.includes('\0'), `${description} is unsafe.`);
  const parts = path.split('/');
  assert.ok(
    parts.every((part) => part && part !== '.' && part !== '..'),
    `${description} is unsafe.`
  );
  return parts.join('/');
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

async function readRegularFile(root: string, relativePath: string): Promise<Buffer> {
  const path = resolve(root, ...safeRelativePath(relativePath, 'Qwen input file').split('/'));
  const info = await lstat(path);
  assert.ok(info.isFile() && !info.isSymbolicLink(), `Qwen input must be a regular file: ${relativePath}`);
  const canonicalRoot = await realpath(root);
  const canonicalPath = await realpath(path);
  assert.ok(
    canonicalPath.startsWith(`${canonicalRoot}${sep}`),
    `Qwen input escapes the build artifact root: ${relativePath}`
  );
  return readFile(canonicalPath);
}

async function hashRegularFile(root: string, relativePath: string): Promise<AssetMetadata> {
  const path = resolve(root, ...safeRelativePath(relativePath, 'Qwen artifact file').split('/'));
  const info = await lstat(path);
  assert.ok(info.isFile() && !info.isSymbolicLink(), `Qwen artifact must be a regular file: ${relativePath}`);
  const canonicalRoot = await realpath(root);
  const canonicalPath = await realpath(path);
  assert.ok(
    canonicalPath.startsWith(`${canonicalRoot}${sep}`),
    `Qwen artifact escapes the build root: ${relativePath}`
  );
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(canonicalPath)) {
    const data = chunk as Buffer;
    bytes += data.byteLength;
    hash.update(data);
  }
  return {bytes, sha256: hash.digest('hex')};
}

function assertSourcePins(value: unknown, description: string, expectedRunner = SOURCE_RUNNER): void {
  const source = record(value, `${description} source`);
  assert.equal(source.baseRepository, QWEN35_RELEASE_MODEL.baseRepository, `${description} base repository mismatch.`);
  assert.equal(source.baseRevision, QWEN35_RELEASE_MODEL.baseRevision, `${description} base revision mismatch.`);
  assert.equal(
    source.checkpointRepository,
    QWEN35_RELEASE_MODEL.repository,
    `${description} checkpoint repository mismatch.`
  );
  assert.equal(
    source.checkpointRevision,
    QWEN35_RELEASE_MODEL.revision,
    `${description} checkpoint revision mismatch.`
  );
  assert.equal(
    source.strandsRepository,
    QWEN35_RELEASE_MODEL.strandsRepository,
    `${description} Strands repository mismatch.`
  );
  assert.equal(
    source.strandsRevision,
    QWEN35_RELEASE_MODEL.strandsRevision,
    `${description} Strands revision mismatch.`
  );
  assert.equal(source.runner, expectedRunner, `${description} must use the pinned upstream Strands inference path.`);
  assert.equal(source.weightsLoaded, true, `${description} must load the real pinned checkpoint weights.`);
  assert.equal(source.tvmUsed, false, `${description} must be an independent upstream source reference.`);
}

export function assertQwen35PassedSourceReference(value: unknown, description: string): RecordValue {
  const reference = record(value, description);
  assert.equal(
    reference.status,
    'passed-cpu-source-reference',
    `${description} is not a passing upstream source reference.`
  );
  return reference;
}

function assertAnswer(value: unknown, description: string): RecordValue {
  const answer = record(value, description);
  assert.ok(
    ['choice', 'score', 'noul'].includes(String(answer.type)),
    `${description} has an unsupported answer type.`
  );
  if (answer.type === 'choice') {
    const choice = text(answer.choice, `${description} choice`);
    const probabilities = assertProbabilityMap(answer.probabilities, `${description} probabilities`);
    assert.ok(Object.hasOwn(probabilities, choice), `${description} choice is absent from its probabilities.`);
    assert.ok(Number.isFinite(answer.confidence), `${description} confidence is invalid.`);
  } else if (answer.type === 'score') {
    assert.ok(Number.isFinite(answer.score), `${description} score is invalid.`);
    const probabilities = assertProbabilityMap(answer.probabilities, `${description} probabilities`);
    assert.deepEqual(
      Object.keys(record(answer.legend, `${description} legend`)),
      Object.keys(probabilities),
      `${description} legend order differs from its probabilities.`
    );
    assert.ok(Number.isFinite(answer.confidence), `${description} confidence is invalid.`);
  } else {
    assert.ok(
      Number.isFinite(answer.noul) && Number(answer.noul) >= 0 && Number(answer.noul) <= 1,
      `${description} noul value is invalid.`
    );
  }
  return answer;
}

function assertProbabilityMap(value: unknown, description: string): RecordValue {
  const probabilities = record(value, description);
  const values = Object.values(probabilities);
  assert.ok(values.length >= 2, `${description} must contain at least two options.`);
  assert.ok(
    values.every((item) => Number.isFinite(item) && Number(item) >= 0 && Number(item) <= 1),
    `${description} contains an invalid probability.`
  );
  assert.ok(
    Math.abs(values.reduce<number>((sum, item) => sum + Number(item), 0) - 1) <= 0.001,
    `${description} is not normalized.`
  );
  return probabilities;
}

function assertRequest(value: unknown, description: string): RecordValue {
  const request = record(value, description);
  assert.ok(Object.hasOwn(request, 'state'), `${description} is missing state.`);
  const questions = questionEntries(request.questions, `${description} questions`);
  assert.ok(questions.length > 0, `${description} has no questions.`);
  for (const [name, rawQuestion] of questions) {
    const question = record(rawQuestion, `${description} question ${name}`);
    assert.ok(
      ['choice', 'score', 'noul'].includes(String(question.type)),
      `${description} question ${name} has an invalid type.`
    );
    assert.ok(Object.hasOwn(question, 'instructions'), `${description} question ${name} instructions are required.`);
    if (question.type === 'choice') {
      const criteria = record(question.criteria, `${description} question ${name} criteria`);
      const optionCount = Object.keys(criteria).length;
      assert.ok(optionCount >= 2, `${description} choice question must have at least two options.`);
    } else if (question.type === 'score') {
      assert.ok(Array.isArray(question.criteria), `${description} score criteria must be an array.`);
      assert.ok(question.criteria.length >= 2, `${description} score question must have at least two levels.`);
      assert.ok(
        question.criteria.every((level) => typeof level === 'string'),
        `${description} score criteria levels must be strings.`
      );
    } else if (question.criteria !== undefined && question.criteria !== null) {
      const criteria = record(question.criteria, `${description} question ${name} criteria`);
      assert.ok(
        Object.keys(criteria).every((key) => key === 'true' || key === 'false'),
        `${description} noul criteria keys must be true or false.`
      );
    }
  }
  return request;
}

function questionEntries(value: unknown, description: string): Array<[string, unknown]> {
  if (Array.isArray(value)) {
    const entries: Array<[string, unknown]> = value.map((item, index) => {
      assert.ok(
        Array.isArray(item) && item.length === 2,
        `${description} entry ${index} must be a [name, question] pair.`
      );
      return [text(item[0], `${description} entry ${index} name`), item[1]];
    });
    assert.equal(
      new Set(entries.map(([name]) => name)).size,
      entries.length,
      `${description} contains duplicate question names.`
    );
    return entries;
  }
  const entries = Object.entries(record(value, description));
  const hasIndexKey = entries.some(([name]) => /^(?:0|[1-9]\d*)$/u.test(name) && Number(name) < 4_294_967_295);
  assert.ok(
    !hasIndexKey,
    `${description} with integer-like names must use ordered [name, question] entries so the browser can pass a Map.`
  );
  return entries;
}

function choiceCount(value: unknown, description: string): number {
  return Object.keys(record(value, description)).length;
}

function expectedResult(
  value: unknown,
  request: RecordValue,
  description: string,
  sourceModel: string = QWEN35_RELEASE_MODEL.modelId
): SourceExpected {
  const expected = record(value, `${description} expected output`);
  assert.equal(expected.model, sourceModel, `${description} reference model ID mismatch.`);
  const usage = record(expected.usage, `${description} usage`);
  assert.ok(
    Number.isSafeInteger(usage.input_tokens) && Number(usage.input_tokens) >= 0,
    `${description} input token count is invalid.`
  );
  assert.ok(
    Number.isSafeInteger(usage.output_tokens) && Number(usage.output_tokens) >= 0,
    `${description} output token count is invalid.`
  );
  const questions = new Map(questionEntries(request.questions, 'request questions'));
  assert.equal(
    usage.output_tokens,
    questions.size,
    `${description} output token count must equal the number of source questions.`
  );
  const expectedAnswers = record(expected.answers, `${description} answers`);
  assert.deepEqual(
    Object.keys(expectedAnswers).sort(),
    [...questions.keys()].sort(),
    `${description} answer keys differ from request question names.`
  );
  for (const [name, answer] of Object.entries(expectedAnswers)) {
    const expectedAnswer = assertAnswer(answer, `${description} answer ${name}`);
    const requestQuestion = record(questions.get(name), `${description} question ${name}`);
    assert.equal(
      expectedAnswer.type,
      requestQuestion.type,
      `${description} answer ${name} does not match its request type.`
    );
    if (requestQuestion.type === 'choice' && expectedAnswer.type === 'choice') {
      const criteria = record(requestQuestion.criteria, `${description} question ${name} criteria`);
      assert.deepEqual(
        Object.keys(record(expectedAnswer.probabilities, `${description} answer ${name} probabilities`)),
        Object.keys(criteria),
        `${description} choice probabilities differ from request options.`
      );
      assert.ok(
        Number(expectedAnswer.confidence) >= 0 && Number(expectedAnswer.confidence) <= 1,
        `${description} choice confidence must be between zero and one.`
      );
    } else if (requestQuestion.type === 'score' && expectedAnswer.type === 'score') {
      const criteria = requestQuestion.criteria as unknown[];
      const expectedLevels = criteria.map((_, index) => String(index));
      assert.deepEqual(
        Object.keys(record(expectedAnswer.probabilities, `${description} answer ${name} probabilities`)),
        expectedLevels,
        `${description} score probabilities differ from request levels.`
      );
      assert.deepEqual(
        Object.keys(record(expectedAnswer.legend, `${description} answer ${name} legend`)),
        expectedLevels,
        `${description} score legend differs from request levels.`
      );
      assert.deepEqual(
        expectedAnswer.legend,
        Object.fromEntries(criteria.map((level, index) => [String(index), String(level).trim()])),
        `${description} score legend differs from request criteria.`
      );
      assert.ok(
        Number(expectedAnswer.score) >= 0 && Number(expectedAnswer.score) <= criteria.length - 1,
        `${description} score is outside the requested level range.`
      );
      assert.ok(
        Number(expectedAnswer.confidence) >= 0 && Number(expectedAnswer.confidence) <= 1,
        `${description} score confidence must be between zero and one.`
      );
    }
  }
  return {
    model: QWEN35_RELEASE_MODEL.modelId,
    usage: usage as SourceExpected['usage'],
    answers: expectedAnswers as Record<string, RecordValue>
  };
}

function usageExpectation(
  value: unknown,
  expected: SourceExpected,
  sourceCaseValue: RecordValue,
  description: string
): CandidateUsageExpectation {
  const item = record(value, `${description} candidate usage expectation`);
  assert.ok(
    item.mode === 'observed-default-cache-source' || item.mode === 'derived-source-prefix-groups',
    `${description} usage mode is unsupported.`
  );
  const usage = record(item.usage, `${description} candidate usage`);
  assert.ok(
    Number.isSafeInteger(usage.input_tokens) && Number(usage.input_tokens) >= 0,
    `${description} candidate input token count is invalid.`
  );
  assert.ok(
    Number.isSafeInteger(usage.output_tokens) && Number(usage.output_tokens) >= 0,
    `${description} candidate output token count is invalid.`
  );
  const evidence = record(item.evidence, `${description} candidate usage evidence`);
  assert.equal(evidence.reportPath, QWEN35_PREFIX_CACHE_REPORT_PATH, `${description} cache evidence path changed.`);
  assert.equal(evidence.reportSha256, QWEN35_PREFIX_CACHE_REPORT_SHA256, `${description} cache evidence hash changed.`);
  const cachedDefault = record(QWEN35_PREFIX_CACHE_REPORT.cachedDefault, 'Pinned Qwen cache source result');
  const uncached = record(QWEN35_PREFIX_CACHE_REPORT.uncached, 'Pinned Qwen uncached source result');
  assert.equal(cachedDefault.usePrefixCache, true, 'Pinned cache source must use the default prefix-cache path.');
  assert.equal(
    record(QWEN35_PREFIX_CACHE_REPORT.request, 'Cache source request').requestIsIdenticalAcrossPaths,
    true,
    'Pinned cache source must use the same request in both paths.'
  );
  assert.equal(
    record(QWEN35_PREFIX_CACHE_REPORT.candidateUsageExpectation, 'Pinned cache candidate usage expectation')
      .accountingContract,
    "The package counts the prefix once per max_batch/max_batch_tokens source group; this is semantic usage, not the TVM graph's repeated full-row work.",
    'Pinned cache source accounting contract changed.'
  );
  const mode = item.mode as CandidateUsageExpectation['mode'];
  if (mode === 'observed-default-cache-source') {
    assert.deepEqual(
      expected.usage,
      uncached.usage,
      `${description} uncached source usage differs from the cache report.`
    );
    assert.equal(evidence.cachedUsageField, 'cachedDefault.usage', `${description} cache usage field changed.`);
    assert.equal(evidence.uncachedUsageField, 'uncached.usage', `${description} uncached usage field changed.`);
    assert.equal(
      record(QWEN35_PREFIX_CACHE_REPORT.m4FixtureComparison, 'Pinned cache fixture comparison')
        .referenceFixtureInputIdsMatchBothSourceAndCachedComparison,
      true,
      'Pinned cache source must match the saved release input IDs.'
    );
    assert.deepEqual(
      usage,
      cachedDefault.usage,
      `${description} candidate usage differs from the pinned cached source.`
    );
  } else {
    assert.equal(
      evidence.accounting,
      'pinned-source-default-prefix-cache',
      `${description} cache accounting basis changed.`
    );
    assert.equal(evidence.maxBatch, 32, `${description} source max batch changed.`);
    assert.equal(cachedDefault.usePrefixCache, true, `${description} cache mode is not pinned as enabled.`);
    const groups = evidence.groupSizes;
    assert.ok(Array.isArray(groups) && groups.length > 0, `${description} cache groups are missing.`);
    assert.ok(
      groups.every((size) => Number.isSafeInteger(size) && Number(size) > 0),
      `${description} cache group sizes are invalid.`
    );
    const prefixTokens = evidence.prefixTokensPerGroup;
    assert.ok(
      Array.isArray(prefixTokens) && prefixTokens.length === groups.length,
      `${description} prefix token lengths do not match source groups.`
    );
    assert.ok(
      prefixTokens.every((size) => Number.isSafeInteger(size) && Number(size) > 0),
      `${description} prefix token lengths are invalid.`
    );
    assert.equal(
      groups.reduce((sum, size) => sum + Number(size), 0),
      expected.usage.output_tokens,
      `${description} cache groups do not cover source questions.`
    );
    assert.equal(evidence.uncachedInputWorkTokens, expected.usage.input_tokens);
    const reusedPrefixTokens = groups.reduce(
      (sum, size, index) => sum + Number(prefixTokens[index]) * (Number(size) - 1),
      0
    );
    assert.equal(
      usage.input_tokens,
      Number(evidence.uncachedInputWorkTokens) - reusedPrefixTokens,
      `${description} cached token count does not follow the pinned source grouping formula.`
    );
    assert.equal(usage.output_tokens, expected.usage.output_tokens);
    const sourceInput = record(sourceCaseValue.input, `${description} captured source input`);
    const sourceExecution = record(sourceCaseValue.sourceExecution, `${description} source execution`);
    assert.deepEqual(
      sourceExecution.forwardBatchSizes,
      groups,
      `${description} cache groups differ from upstream source chunk sizes.`
    );
    const inputIds = sourceInput.inputIds;
    const attentionMask = sourceInput.attentionMask;
    assert.ok(
      Array.isArray(inputIds) && Array.isArray(attentionMask),
      `${description} captured token rows are missing.`
    );
    assert.equal(inputIds.length, expected.usage.output_tokens);
    assert.equal(attentionMask.length, expected.usage.output_tokens);
    assert.equal(
      inputIds.reduce((sum: number, row: unknown, index: number) => {
        assert.ok(Array.isArray(row), `${description} captured row ${index} is invalid.`);
        const mask = attentionMask[index];
        assert.ok(Array.isArray(mask), `${description} captured mask ${index} is invalid.`);
        return sum + mask.reduce((rowSum: number, value: unknown) => rowSum + Number(value), 0);
      }, 0),
      evidence.uncachedInputWorkTokens,
      `${description} captured masks differ from the uncached work count.`
    );
    let rowStart = 0;
    const firstPrefixes: number[][] = [];
    for (const [index, rawSize] of groups.entries()) {
      const size = Number(rawSize);
      const prefixSize: number = Number(prefixTokens[index]);
      const rows = inputIds.slice(rowStart, rowStart + size) as number[][];
      const masks = attentionMask.slice(rowStart, rowStart + size) as number[][];
      assert.equal(rows.length, size, `${description} captured source group ${index} is incomplete.`);
      const first = rows[0]!.slice(0, prefixSize);
      for (const [rowIndex, row] of rows.entries())
        assert.deepEqual(
          row.slice(0, prefixSize),
          first,
          `${description} source group ${index} row ${rowIndex} differs in its shared state prefix.`
        );
      assert.ok(
        masks.every((mask) => mask.slice(0, prefixSize).every((token) => token === 1)),
        `${description} source group ${index} shared prefix is masked.`
      );
      firstPrefixes.push(first);
      rowStart += size;
    }
    assert.ok(
      firstPrefixes.every((prefix) => JSON.stringify(prefix) === JSON.stringify(firstPrefixes[0])),
      `${description} groups do not share the same fitted state prefix.`
    );
    assert.equal(rowStart, expected.usage.output_tokens);
  }
  return {
    mode,
    usage: usage as CandidateUsageExpectation['usage'],
    evidence
  };
}

function assertPrecisionReferenceAligned(
  primary: SourceExpected,
  alternate: SourceExpected,
  description: string
): void {
  assert.equal(alternate.model, primary.model, `${description} precision reference model differs.`);
  assert.deepEqual(alternate.usage, primary.usage, `${description} precision reference usage differs.`);
  assert.deepEqual(
    Object.keys(alternate.answers),
    Object.keys(primary.answers),
    `${description} precision reference answer order differs.`
  );
  for (const [name, primaryValue] of Object.entries(primary.answers)) {
    const alternateValue = assertAnswer(alternate.answers[name], `${description} precision reference answer ${name}`);
    assert.equal(
      alternateValue.type,
      primaryValue.type,
      `${description} precision reference type differs for ${name}.`
    );
    if (primaryValue.type === 'choice') {
      assert.equal(
        alternateValue.choice,
        primaryValue.choice,
        `${description} precision reference choice differs for ${name}.`
      );
      assert.deepEqual(
        Object.keys(record(alternateValue.probabilities, `${description} alternate probabilities`)),
        Object.keys(record(primaryValue.probabilities, `${description} primary probabilities`)),
        `${description} precision reference probability order differs for ${name}.`
      );
    } else if (primaryValue.type === 'score') {
      assert.deepEqual(
        alternateValue.legend,
        primaryValue.legend,
        `${description} precision reference legend differs for ${name}.`
      );
      assert.deepEqual(
        Object.keys(record(alternateValue.probabilities, `${description} alternate probabilities`)),
        Object.keys(record(primaryValue.probabilities, `${description} primary probabilities`)),
        `${description} precision reference probability order differs for ${name}.`
      );
    }
  }
}

function buildPrecisionEnvelope(primary: SourceExpected, alternate: SourceExpected): RecordValue {
  const metric = (bf16Value: unknown, fp32Value: unknown) => {
    assert.ok(
      Number.isFinite(bf16Value) && Number.isFinite(fp32Value),
      'Source precision envelope values must be finite.'
    );
    const absoluteDelta = Math.abs(Number(bf16Value) - Number(fp32Value));
    return {
      bf16: Number(bf16Value),
      fp32: Number(fp32Value),
      absoluteDelta,
      allowedDelta: absoluteDelta + OUTPUT_ROUNDING_TOLERANCE
    };
  };
  const answers: RecordValue = {};
  for (const [name, answer] of Object.entries(primary.answers)) {
    const fp32Answer = alternate.answers[name]!;
    const answerEnvelope: RecordValue = {};
    if (answer.type === 'choice' && fp32Answer.type === 'choice') {
      answerEnvelope.confidence = metric(answer.confidence, fp32Answer.confidence);
      const bf16Probabilities = record(answer.probabilities, `BF16 ${name} probabilities`);
      const fp32Probabilities = record(fp32Answer.probabilities, `FP32 ${name} probabilities`);
      answerEnvelope.probabilities = Object.fromEntries(
        Object.entries(bf16Probabilities).map(([key, value]) => [key, metric(value, fp32Probabilities[key])])
      );
    } else if (answer.type === 'score' && fp32Answer.type === 'score') {
      answerEnvelope.score = metric(answer.score, fp32Answer.score);
      answerEnvelope.confidence = metric(answer.confidence, fp32Answer.confidence);
      const bf16Probabilities = record(answer.probabilities, `BF16 ${name} probabilities`);
      const fp32Probabilities = record(fp32Answer.probabilities, `FP32 ${name} probabilities`);
      answerEnvelope.probabilities = Object.fromEntries(
        Object.entries(bf16Probabilities).map(([key, value]) => [key, metric(value, fp32Probabilities[key])])
      );
    } else if (answer.type === 'noul' && fp32Answer.type === 'noul') {
      answerEnvelope.noul = metric(answer.noul, fp32Answer.noul);
    }
    answers[name] = answerEnvelope;
  }
  return {
    formula: 'abs(bf16-fp32)+roundingTolerance',
    roundingTolerance: OUTPUT_ROUNDING_TOLERANCE,
    answers
  };
}

function sourceChoiceLabels(value: unknown, request: RecordValue, description: string): string[] | undefined {
  if (value === undefined) return undefined;
  assert.ok(Array.isArray(value), `${description} choiceLabels must be an array.`);
  const choiceLabels = value.map((label, index) => text(label, `${description} choice label ${index}`));
  const choiceQuestion = questionEntries(request.questions, `${description} questions`).find(
    ([, value]) => record(value, `${description} question`).type === 'choice'
  );
  assert.ok(choiceQuestion, `${description} declares choiceLabels without a choice question.`);
  const criteria = record(
    record(choiceQuestion[1], `${description} choice question`).criteria,
    `${description} choice criteria`
  );
  assert.deepEqual(
    choiceLabels,
    Object.keys(criteria),
    `${description} choiceLabels differ from request criteria order.`
  );
  return choiceLabels;
}

function sourcePrecisionEvidence(
  item: RecordValue,
  expected: SourceExpected,
  request: RecordValue,
  description: string,
  sourceModel: string
): Pick<SourceCase, 'precisionReference' | 'precisionEnvelope'> {
  const precisionReference =
    item.precisionReference === undefined
      ? undefined
      : expectedResult(item.precisionReference, request, `${description} precision reference`, sourceModel);
  const precisionEnvelope =
    item.precisionEnvelope === undefined
      ? undefined
      : record(item.precisionEnvelope, `${description} precision envelope`);
  if (precisionReference) {
    assertPrecisionReferenceAligned(expected, precisionReference, description);
    assert.deepEqual(
      precisionEnvelope,
      buildPrecisionEnvelope(expected, precisionReference),
      `${description} source precision envelope differs from its BF16/FP32 references.`
    );
  } else {
    assert.equal(precisionEnvelope, undefined, `${description} precision envelope requires a precision reference.`);
  }
  return {
    ...(precisionReference ? {precisionReference} : {}),
    ...(precisionEnvelope ? {precisionEnvelope} : {})
  };
}

function sourceImageFiles(imageFiles: unknown, description: string): SourceCase['imageFiles'] {
  if (imageFiles === undefined) return undefined;
  assert.ok(Array.isArray(imageFiles), `${description} imageFiles must be an array.`);
  assert.ok(imageFiles.length <= QWEN35_RELEASE_MODEL.maxImages, `${description} exceeds the source image limit.`);
  for (const [index, rawFile] of imageFiles.entries()) {
    const file = record(rawFile, `${description} image ${index}`);
    safeRelativePath(file.path, `${description} image ${index} path`);
    digest(file.sha256, `${description} image ${index} SHA-256`);
    assert.ok(
      Number.isSafeInteger(file.bytes) && Number(file.bytes) > 0,
      `${description} image ${index} byte length is invalid.`
    );
    text(file.mimeType, `${description} image ${index} MIME type`);
  }
  return imageFiles as NonNullable<SourceCase['imageFiles']>;
}

function sourceImageGeometry(
  imageGeometry: unknown,
  imageCount: number,
  description: string
): SourceCase['imageGeometry'] {
  if (imageGeometry === undefined) return undefined;
  assert.ok(Array.isArray(imageGeometry), `${description} imageGeometry must be an array.`);
  assert.equal(imageGeometry.length, imageCount, `${description} image geometry count differs from image count.`);
  for (const [index, rawGeometry] of imageGeometry.entries()) {
    const geometry = record(rawGeometry, `${description} image geometry ${index}`);
    for (const field of ['width', 'height', 'longSide', 'patchRows'])
      assert.ok(
        Number.isSafeInteger(geometry[field]) && Number(geometry[field]) > 0,
        `${description} image geometry ${index} has invalid ${field}.`
      );
    assert.equal(
      geometry.longSide,
      Math.max(Number(geometry.width), Number(geometry.height)),
      `${description} image geometry ${index} longSide is inconsistent.`
    );
  }
  return imageGeometry as NonNullable<SourceCase['imageGeometry']>;
}

function sourceCase(
  value: unknown,
  description: string,
  sourceModel: string = QWEN35_RELEASE_MODEL.modelId
): SourceCase {
  const item = record(value, description);
  const request = assertRequest(item.request, `${description} request`);
  const choiceLabels = sourceChoiceLabels(item.choiceLabels, request, description);
  const expected = expectedResult(item.expected, request, description, sourceModel);
  const candidateUsageExpectation =
    item.candidateUsageExpectation === undefined
      ? undefined
      : usageExpectation(item.candidateUsageExpectation, expected, item, description);
  const requestMetadata =
    item.requestMetadata === undefined ? undefined : record(item.requestMetadata, `${description} request metadata`);
  const precision = sourcePrecisionEvidence(item, expected, request, description, sourceModel);
  const imageFiles = sourceImageFiles(item.imageFiles, description);
  const imageGeometry = sourceImageGeometry(item.imageGeometry, imageFiles?.length ?? 0, description);
  return {
    name: text(item.name, `${description} name`),
    request,
    ...(choiceLabels ? {choiceLabels} : {}),
    expected,
    ...(candidateUsageExpectation ? {candidateUsageExpectation} : {}),
    ...(requestMetadata ? {requestMetadata} : {}),
    ...precision,
    ...(imageFiles ? {imageFiles} : {}),
    ...(imageGeometry ? {imageGeometry} : {})
  };
}

export function assertQwen35SourceCase(value: unknown, description: string): void {
  sourceCase(value, description);
}

function parseJson(bytes: Buffer, description: string): RecordValue {
  let value: unknown;
  try {
    value = JSON.parse(bytes.toString('utf8'));
  } catch (cause) {
    throw new Error(`${description} is not valid JSON.`, {cause});
  }
  return record(value, description);
}

async function readPinnedJson(root: string, pathValue: unknown, shaValue: unknown, description: string) {
  const path = safeRelativePath(pathValue, `${description} path`);
  const bytes = await readRegularFile(root, path);
  const actual = sha256(bytes);
  assert.equal(actual, digest(shaValue, `${description} SHA-256`), `${description} changed from its build pin.`);
  return {path, bytes, sha256: actual, value: parseJson(bytes, description)};
}

export function assertQwen35B3S4096SourceReference(value: unknown): Qwen35B3S4096SourceCase {
  const reference = record(value, 'Qwen batch-three max-context source reference');
  assert.equal(
    reference.format,
    'qwen35-full-pointer-b3-s4096-cpu-source-reference-v1',
    'Unknown Qwen B3×S4096 source reference format.'
  );
  assert.equal(reference.status, 'passed-cpu-source-reference', 'Qwen B3×S4096 source reference did not pass.');
  const scope = record(reference.scope, 'Qwen B3×S4096 source scope');
  assert.equal(scope.realPinnedCheckpointLoaded, true, 'Qwen B3×S4096 reference must load the pinned checkpoint.');
  assert.equal(scope.sourceDecisionForwardPerformed, true, 'Qwen B3×S4096 reference must run source inference.');
  assert.equal(scope.gpuUsed, false, 'Qwen B3×S4096 is an independent CPU source reference.');
  assert.equal(scope.tvmUsed, false, 'Qwen B3×S4096 source reference must not claim TVM execution.');

  const request = record(reference.request, 'Qwen B3×S4096 source request');
  assert.equal(request.apiModel, 'strands-decider-latest', 'Qwen B3×S4096 source request API model changed.');
  assert.equal(request.sourceDerivedInput, true, 'Qwen B3×S4096 input must be source-derived.');
  const state = record(request.state, 'Qwen B3×S4096 state');
  assert.equal(typeof state.context, 'string', 'Qwen B3×S4096 state must contain full natural-language context.');
  const renderedState = text(request.renderedState, 'Qwen B3×S4096 rendered source state');
  const renderedStateMatch = /^<state>\n([\s\S]*)\n<\/state>\n$/u.exec(renderedState);
  assert.ok(renderedStateMatch, 'Qwen B3×S4096 rendered source state has invalid delimiters.');
  const publicState = record(JSON.parse(renderedStateMatch[1]!), 'Qwen B3×S4096 public state');
  assert.deepEqual(publicState, state, 'Qwen B3×S4096 public state data differs from source metadata.');
  assert.equal(
    renderedState,
    `<state>\n${JSON.stringify(publicState, null, 2)}\n</state>\n`,
    'Qwen B3×S4096 public state differs from the pinned source prompt.'
  );

  const questionNames = ['runtime', 'runtime_copy_2', 'runtime_copy_3'];
  assert.deepEqual(request.questionOrder, questionNames, 'Qwen B3×S4096 source question order changed.');
  assert.ok(Array.isArray(request.questions), 'Qwen B3×S4096 source questions must be an array.');
  assert.equal(request.questions.length, 3, 'Qwen B3×S4096 source reference must contain three questions.');
  const questions = request.questions.map((rawQuestion, index) => {
    const item = record(rawQuestion, `Qwen B3×S4096 question ${index}`);
    assert.equal(item.name, questionNames[index], `Qwen B3×S4096 question ${index} name changed.`);
    const question = record(item.question, `Qwen B3×S4096 typed question ${index}`);
    assert.equal(question.type, 'choice', `Qwen B3×S4096 question ${index} must remain a choice.`);
    const criteria = record(question.criteria, `Qwen B3×S4096 criteria ${index}`);
    assert.deepEqual(
      Object.keys(criteria),
      QWEN35_B3_S4096_LABELS,
      `Qwen B3×S4096 question ${index} option label order changed.`
    );
    return [questionNames[index]!, question] as const;
  });
  assert.ok(
    questions.slice(1).every(([, question]) => isDeepStrictEqual(question, questions[0]?.[1])),
    'Qwen B3×S4096 source rows must preserve the pinned identical typed request.'
  );

  const derivation = record(reference.sourceRequestDerivation, 'Qwen B3×S4096 source derivation');
  assert.equal(derivation.requestedSequenceTokens, 4096, 'Qwen B3×S4096 source derivation must request 4096 tokens.');
  assert.equal(derivation.rowsAreIdenticallyRendered, true, 'Qwen B3×S4096 rows must match source rendering.');
  assert.equal(derivation.coverage, 'identical-row batch-shape coverage; not a row-independence test');
  assert.equal(
    derivation.sourceReportPath,
    'scripts/strands/qwen35/reports/qwen35-source-reference-cpu-r3-mixed-2026-11-15.json',
    'Qwen B3×S4096 source derivation report path changed.'
  );
  assert.equal(
    derivation.sourceReportSha256,
    'e66e7adedadd68932726e0a8dce5e5f603ead2bb96356fa9586081dcc41a97e2',
    'Qwen B3×S4096 source derivation report hash changed.'
  );

  const forward = record(reference.sourceDerivedForward, 'Qwen B3×S4096 source forward');
  assert.equal(forward.actualBatch, 3, 'Qwen B3×S4096 source forward must keep all three rows.');
  assert.equal(forward.actualSequence, QWEN35_RELEASE_MODEL.maxLength, 'Qwen B3×S4096 must keep 4096 tokens.');
  assert.equal(forward.stateTokens, 4019, 'Qwen B3×S4096 source state token count changed.');
  assert.deepEqual(forward.questionTokens, [77, 77, 77], 'Qwen B3×S4096 source question token counts changed.');
  assert.deepEqual(forward.sourceEngineGrouping, {
    chunkCount: 1,
    chunkRowSlice: [0, 3],
    maxBatch: 32,
    maxBatchTokens: null,
    strictWindow: false
  });
  const inputHashes = forward.inputIdsSha256ByRow;
  assert.ok(Array.isArray(inputHashes) && inputHashes.length === 3, 'Qwen B3×S4096 must pin three input rows.');
  for (const [index, hash] of inputHashes.entries()) digest(hash, `Qwen B3×S4096 input row ${index} SHA-256`);
  assert.equal(new Set(inputHashes).size, 1, 'Qwen B3×S4096 source input rows must be identical.');

  const outputs = record(reference.candidateOutputs, 'Qwen B3×S4096 source outputs');
  assert.equal(outputs.probabilitiesFinite, true, 'Qwen B3×S4096 source probabilities must be finite.');
  assert.equal(outputs.probabilitiesShapeMatchesSource, true, 'Qwen B3×S4096 source probability shape must match.');
  assert.deepEqual(outputs.probabilitiesShape, [3, 3], 'Qwen B3×S4096 source probabilities must be 3 × 3.');
  assert.ok(Array.isArray(outputs.probabilities), 'Qwen B3×S4096 source probabilities are required.');
  const expectedProbabilities = outputs.probabilities.map((rawRow, rowIndex) => {
    assert.ok(Array.isArray(rawRow), `Qwen B3×S4096 source probability row ${rowIndex} must be an array.`);
    assert.equal(rawRow.length, 3, `Qwen B3×S4096 source probability row ${rowIndex} must have three options.`);
    const row = rawRow.map((value, optionIndex) => {
      assert.ok(
        typeof value === 'number' && Number.isFinite(value),
        `Qwen B3×S4096 source probability ${rowIndex}/${optionIndex} must be finite.`
      );
      return value;
    });
    assert.ok(Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) <= 1e-6, 'Source probabilities must sum to 1.');
    return row;
  });
  assert.equal(expectedProbabilities.length, 3, 'Qwen B3×S4096 source output must contain three rows.');
  assert.ok(
    expectedProbabilities.every((row) => isDeepStrictEqual(row, expectedProbabilities[0])),
    'Qwen B3×S4096 identical source questions must retain their pinned probability rows.'
  );
  assert.deepEqual(
    expectedProbabilities,
    QWEN35_B3_S4096_PROBABILITIES,
    'Qwen B3×S4096 source probabilities differ from the independently pinned output matrix.'
  );
  const expectedChoices = expectedProbabilities.map((row, rowIndex) => {
    const choice = QWEN35_B3_S4096_LABELS[row.indexOf(Math.max(...row))];
    assert.equal(
      choice,
      QWEN35_B3_S4096_EXPECTED_CHOICES[rowIndex],
      `Qwen B3×S4096 source choice mapping changed for row ${rowIndex}.`
    );
    return choice!;
  });
  const toleranceCalibration = record(reference.sourceToleranceCalibration, 'Qwen B3×S4096 tolerance calibration');
  assert.equal(toleranceCalibration.diagnosticToleranceMultiplier, 3);
  assert.equal(toleranceCalibration.probabilitiesMaxAbs, 0.00305020809173584);
  const probabilityTolerance = Number(toleranceCalibration.probabilitiesMaxAbs) * 3;
  assert.equal(
    probabilityTolerance,
    TEXT_SOURCE_DERIVED_TOLERANCE,
    'Qwen B3×S4096 probability tolerance differs from the pinned source calibration.'
  );

  return {
    name: 'batch-3-context-4096',
    request: {
      state: publicState,
      questions: questions.map(([name, question]) => [name, question])
    },
    questionNames,
    labels: QWEN35_B3_S4096_LABELS,
    expectedChoices,
    batchSize: 3,
    sequenceLength: 4096,
    graphInputTokens: 12_288,
    expectedProbabilities,
    probabilityTolerance,
    expectedUsage: {input_tokens: 4_250, output_tokens: 3},
    sourceReference: QWEN35_B3_S4096_SOURCE_REFERENCE
  };
}

export async function loadQwen35B3S4096SourceReference(): Promise<
  Qwen35B3S4096SourceCase & {readonly rawReference: RecordValue}
> {
  const bytes = await readFile(
    new URL(
      '../strands/qwen35/reports/qwen35-full-pointer-b3-s4096-source-reference-r1-2026-11-16.json',
      import.meta.url
    )
  );
  assert.equal(
    sha256(bytes),
    QWEN35_B3_S4096_SOURCE_REFERENCE.sha256,
    'Qwen B3×S4096 CPU source reference changed from its pin.'
  );
  const rawReference = parseJson(bytes, 'Qwen B3×S4096 CPU source reference');
  const sourceCase = assertQwen35B3S4096SourceReference(rawReference);
  const protocolBytes = await readFile(
    new URL('../strands/qwen35/reports/qwen35-source-reference-cpu-r3-mixed-2026-11-15.json', import.meta.url)
  );
  assert.equal(
    sha256(protocolBytes),
    QWEN35_B3_S4096_PROTOCOL_REFERENCE.sha256,
    'Qwen B3×S4096 typed-question source reference changed from its pin.'
  );
  const protocolValue = JSON.parse(
    protocolBytes.toString('utf8').replace(/(^|[\s:,\u005B])(-Infinity|Infinity|NaN)(?=$|[\s,\]}])/gmu, '$1null')
  ) as unknown;
  const protocol = record(protocolValue, 'Qwen B3×S4096 typed-question source reference');
  const protocolRequest = record(protocol.request, 'Qwen B3×S4096 typed-question request');
  assert.ok(Array.isArray(protocolRequest.questions), 'Qwen B3×S4096 typed-question reference must contain rows.');
  const runtimeQuestion = protocolRequest.questions
    .map((rawQuestion, index) => record(rawQuestion, `Qwen B3×S4096 protocol question ${index}`))
    .find(({name}) => name === 'runtime');
  assert.ok(runtimeQuestion, 'Qwen B3×S4096 protocol reference must contain the original runtime question.');
  assert.deepEqual(runtimeQuestion.slotLabelsByPosition, QWEN35_B3_S4096_PROTOCOL_SLOT_LABELS);
  const publicQuestions = sourceCase.request.questions;
  assert.ok(Array.isArray(publicQuestions), 'Qwen B3×S4096 public question pairs must be an array.');
  const firstPublicQuestion = publicQuestions[0];
  assert.ok(Array.isArray(firstPublicQuestion), 'Qwen B3×S4096 public question pair must be a tuple.');
  assert.deepEqual(runtimeQuestion.question, firstPublicQuestion[1]);
  return {...sourceCase, rawReference};
}

async function assertAssetFiles(root: string, manifest: RecordValue): Promise<Record<string, AssetMetadata>> {
  assert.equal(manifest.model, QWEN35_RELEASE_MODEL.modelId, 'Generated artifact manifest model ID mismatch.');
  assert.equal(manifest.backend, 'tvm-webgpu', 'Qwen candidate artifacts must target TVM WebGPU.');
  const runtime = record(manifest.runtime, 'Generated artifact runtime metadata');
  text(runtime.tvm, 'Generated artifact TVM version');
  assert.ok(
    typeof runtime.tvmSourceRevision === 'string' && /^[a-f0-9]{40}$/u.test(runtime.tvmSourceRevision),
    'Generated artifact TVM source revision must be a commit.'
  );
  assert.deepEqual(runtime.tvmPatches, [], 'Pinned upstream TVM must remain unpatched.');
  assert.equal(runtime.onnxImportOrExportUsed, false, 'Qwen candidate artifacts must use direct TVM import.');
  const rawFiles = record(manifest.files, 'Generated artifact file map');
  assert.ok(Object.keys(rawFiles).length > 0, 'Generated artifact manifest is empty.');
  const files: Record<string, AssetMetadata> = {};
  let hasWasm = false;
  let hasWeights = false;
  for (const [rawPath, rawMetadata] of Object.entries(rawFiles)) {
    const path = safeRelativePath(rawPath, 'Generated artifact file');
    assert.ok(!/\.onnx(?:$|\.)/iu.test(path), `Qwen candidate artifact uses ONNX: ${path}`);
    const metadata = record(rawMetadata, `Generated artifact metadata ${path}`);
    assert.ok(
      Number.isSafeInteger(metadata.bytes) && Number(metadata.bytes) >= 0,
      `Invalid generated artifact size: ${path}`
    );
    const expectedSha = digest(metadata.sha256, `Generated artifact ${path} SHA-256`);
    const actual = await hashRegularFile(root, path);
    assert.equal(actual.bytes, metadata.bytes, `Generated artifact size differs for ${path}.`);
    assert.equal(actual.sha256, expectedSha, `Generated artifact digest differs for ${path}.`);
    hasWasm ||= path.endsWith('.wasm');
    hasWeights ||= /(?:weight|param|tensor|safetensor|\.bin(?:$|\.))/iu.test(path);
    files[path] = {bytes: Number(metadata.bytes), sha256: expectedSha};
  }
  assert.ok(hasWasm, 'Generated Qwen artifacts must include a TVM WebAssembly module.');
  assert.ok(hasWeights, 'Generated Qwen artifacts must include real model-weight data.');
  return files;
}

function assertMixedTypedCases(cases: readonly SourceCase[], description: string): void {
  const types = new Set<string>();
  for (const item of cases)
    for (const [name, question] of questionEntries(item.request.questions, `${description} questions`)) {
      types.add(String(record(question, `${description} question ${name}`).type));
    }
  assert.deepEqual(
    [...types].sort(),
    ['choice', 'noul', 'score'],
    `${description} must cover choice, score, and noul.`
  );
}

export function assertQwen35VisionImageCountCoverage(
  cases: readonly Readonly<{imageFiles?: readonly unknown[]}>[]
): void {
  const imageCounts = new Set(cases.map(({imageFiles}) => imageFiles?.length));
  for (const count of [1, 2, 3, 4])
    assert.ok(imageCounts.has(count), `Qwen M4 gate requires a pinned ${count}-image vision case.`);
}

export async function loadQwen35CandidateInputs(artifactRoot: string): Promise<Qwen35CandidateInputs> {
  const root = await realpath(resolve(artifactRoot));
  assert.ok((await stat(root)).isDirectory(), 'Qwen build artifact root must be a directory.');
  const evidenceBytes = await readRegularFile(root, 'release-harness.json');
  const evidenceInput = parseJson(evidenceBytes, 'Qwen release-harness.json');
  assert.equal(evidenceInput.schemaVersion, 1, 'Unsupported Qwen release harness input schema.');
  assert.equal(evidenceInput.modelId, QWEN35_RELEASE_MODEL.modelId, 'Qwen release harness model ID mismatch.');
  assertSourcePins(evidenceInput.source, 'Qwen release harness');
  const manifestEntry = record(evidenceInput.manifest, 'Qwen release manifest entry');
  const pinnedManifest = await readPinnedJson(root, manifestEntry.path, manifestEntry.sha256, 'Qwen artifact manifest');
  const manifestFiles = await assertAssetFiles(root, pinnedManifest.value);
  const manifestSha256 = canonicalSha256(pinnedManifest.value);
  const manifestSource = record(pinnedManifest.value.sourceEvidence, 'Qwen artifact manifest source evidence');
  assert.equal(
    manifestSource.assetLockPath,
    'assets/qwen35/manifest.json',
    'Qwen artifact manifest must identify the pinned package asset manifest.'
  );
  assert.equal(
    manifestSource.assetLockSha256,
    QWEN35_ASSET_MANIFEST_RAW_SHA256,
    'Qwen artifact manifest package asset source differs from the repository pin.'
  );
  const references = record(evidenceInput.references, 'Qwen release references');
  const textReference = await readPinnedJson(
    root,
    record(references.text, 'Qwen text reference').path,
    record(references.text, 'Qwen text reference').sha256,
    'Qwen text source reference'
  );
  const visionReferenceEntry = record(references.vision, 'Qwen vision reference');
  assert.equal(
    visionReferenceEntry.path,
    QWEN35_OFFICIAL_VISION_REFERENCE.path,
    'Qwen official vision reference path is not pinned.'
  );
  assert.equal(
    visionReferenceEntry.sha256,
    QWEN35_OFFICIAL_VISION_REFERENCE.sha256,
    'Qwen official vision reference digest is not pinned.'
  );
  const visionReference = await readPinnedJson(
    root,
    visionReferenceEntry.path,
    visionReferenceEntry.sha256,
    'Qwen vision source reference'
  );
  const mixedVisionReference =
    references.mixedVision === undefined
      ? undefined
      : await loadQwen35MixedVisionSourceReference(root, references.mixedVision);
  const boundaryReference = await readPinnedJson(
    root,
    record(references.boundaries, 'Qwen boundary reference').path,
    record(references.boundaries, 'Qwen boundary reference').sha256,
    'Qwen boundary source reference'
  );

  for (const [name, reference] of [
    ['text', textReference.value],
    ['vision', visionReference.value],
    ['boundaries', boundaryReference.value]
  ] as const)
    assertSourcePins(
      reference.source,
      `Qwen ${name} reference`,
      name === 'vision' ? VISION_SOURCE_RUNNER : SOURCE_RUNNER
    );
  assert.equal(
    textReference.value.format,
    'qwen35-system-one-source-reference-v1',
    'Unknown Qwen text reference format.'
  );
  assertQwen35PassedSourceReference(textReference.value, 'Qwen text source reference');
  const sourceCases = Array.isArray(textReference.value.cases)
    ? textReference.value.cases.map((item, index) => sourceCase(item, `Qwen text source case ${index}`))
    : [];
  assert.ok(sourceCases.length > 0, 'Qwen text source reference has no cases.');
  assertMixedTypedCases(sourceCases, 'Qwen text source reference');
  assert.equal(
    visionReference.value.format,
    'qwen35-official-vision-source-reference-v2',
    'Unknown Qwen vision reference format.'
  );
  assertQwen35PassedSourceReference(visionReference.value, 'Qwen official vision source reference');
  const processor = record(visionReference.value.processor, 'Qwen official vision processor metadata');
  assert.equal(
    processor.name,
    'Qwen2VLImageProcessorPil',
    'Qwen vision fixture must use the official pinned image processor.'
  );
  assert.equal(
    processor.transformersVersion,
    '5.22.2',
    'Qwen vision fixture must use the pinned Transformers processor.'
  );
  assert.equal(processor.pillowVersion, '13.0.0', 'Qwen vision fixture must use the pinned Pillow processor.');
  digest(processor.configSha256, 'Qwen official vision processor configuration SHA-256');
  assert.equal(
    processor.configSha256,
    '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
    'Qwen vision processor config pin mismatch.'
  );
  assert.equal(processor.imageLongSide, 448, 'Qwen vision envelope must preserve the pinned 448px geometry.');
  assert.equal(processor.maxImages, QWEN35_RELEASE_MODEL.maxImages, 'Qwen vision image-count limit mismatch.');
  assert.equal(processor.minPixels, 65_536, 'Qwen vision minimum pixel limit mismatch.');
  assert.equal(processor.maxPixels, 16_777_216, 'Qwen vision maximum pixel limit mismatch.');
  assert.equal(processor.patchSize, 16, 'Qwen vision patch-size mismatch.');
  assert.equal(processor.resizeFactor, 32, 'Qwen vision resize-factor mismatch.');
  const officialVisionCases = Array.isArray(visionReference.value.cases)
    ? visionReference.value.cases.map((item, index) => sourceCase(item, `Qwen vision source case ${index}`))
    : [];
  assert.ok(officialVisionCases.length > 0, 'Qwen vision source reference has no cases.');
  assert.ok(
    officialVisionCases.some(
      (item) => item.imageFiles?.length === 1 && item.imageGeometry?.some(({longSide}) => longSide === 448)
    ),
    'Qwen official vision reference must include a pinned 448px one-image inference.'
  );
  assert.ok(
    officialVisionCases.some((item) => item.imageFiles?.length === QWEN35_RELEASE_MODEL.maxImages),
    'Qwen official vision reference must include the four-image boundary.'
  );
  const visionCases = [...officialVisionCases, ...(mixedVisionReference?.cases ?? [])];
  for (const item of visionCases) {
    assert.equal(
      item.imageGeometry?.length,
      item.imageFiles?.length,
      `${item.name} must pin processor geometry for every image.`
    );
    assert.ok(
      (item.imageGeometry ?? []).every(({longSide}) => longSide <= 448),
      `${item.name} exceeds the pinned 448px image envelope.`
    );
    for (const image of item.imageFiles ?? []) {
      const imageBytes = await readRegularFile(root, image.path);
      assert.equal(imageBytes.byteLength, image.bytes, `Qwen vision image size differs: ${image.path}`);
      assert.equal(sha256(imageBytes), image.sha256, `Qwen vision image digest differs: ${image.path}`);
    }
  }
  assert.equal(
    boundaryReference.value.format,
    'qwen35-system-one-boundary-source-reference-v1',
    'Unknown Qwen boundary reference format.'
  );
  assertQwen35PassedSourceReference(boundaryReference.value, 'Qwen boundary source reference');
  const boundaryCases = record(boundaryReference.value.cases, 'Qwen boundary reference cases');
  const contextAtLimit = sourceCase(boundaryCases.contextAtLimit, 'Qwen max-context source case');
  const contextBeyondLimit = sourceCase(boundaryCases.contextBeyondLimit, 'Qwen over-context source case');
  const contextAtLimitRequestMetadata = record(contextAtLimit.requestMetadata, 'Qwen max-context request metadata');
  const contextBeyondLimitRequestMetadata = record(
    contextBeyondLimit.requestMetadata,
    'Qwen over-context request metadata'
  );
  assert.equal(
    contextAtLimit.expected.usage.input_tokens,
    QWEN35_RELEASE_MODEL.maxLength,
    'Qwen max-context source case must use exactly 4096 tokens.'
  );
  assert.equal(
    contextBeyondLimit.expected.usage.input_tokens,
    QWEN35_RELEASE_MODEL.maxLength,
    'Qwen over-context case must record the source-fitted 4096-token input.'
  );
  assert.equal(
    contextAtLimitRequestMetadata.requestedInputTokens,
    QWEN35_RELEASE_MODEL.maxLength,
    'Qwen max-context request token count must be 4096.'
  );
  assert.equal(
    contextBeyondLimitRequestMetadata.requestedInputTokens,
    QWEN35_RELEASE_MODEL.maxLength + 1,
    'Qwen over-context request must be 4097 tokens.'
  );
  const choiceAtLimit = sourceCase(boundaryCases.choiceOptionsAtLimit, 'Qwen max-choice-options source case');
  const choiceBeyondLimit = record(boundaryCases.choiceOptionsBeyondLimit, 'Qwen over-choice-options source case');
  const choiceBeyondRequest = assertRequest(choiceBeyondLimit.request, 'Qwen over-choice-options request');
  assert.equal(
    choiceBeyondLimit.expectedErrorCode,
    'INVALID_INPUT',
    'Qwen 256-option source case must reject as invalid input.'
  );
  assert.equal(
    choiceBeyondLimit.requestedOptionCount,
    QWEN35_RELEASE_MODEL.maxChoiceOptions + 1,
    'Qwen over-choice case must have 256 options.'
  );
  const overChoiceQuestion = questionEntries(choiceBeyondRequest.questions, 'Qwen over-choice questions')
    .map(([name, question]) => record(question, `Qwen over-choice question ${name}`))
    .find((question) => question.type === 'choice');
  assert.ok(overChoiceQuestion, 'Qwen over-choice source case has no choice question.');
  assert.equal(
    choiceCount(overChoiceQuestion.criteria, 'Qwen over-choice criteria'),
    QWEN35_RELEASE_MODEL.maxChoiceOptions + 1,
    'Qwen over-choice request must contain exactly 256 options.'
  );
  const choiceQuestion = questionEntries(choiceAtLimit.request.questions, 'Qwen max-choice questions')
    .map(([name, question]) => record(question, `Qwen max-choice question ${name}`))
    .find((question) => question.type === 'choice');
  assert.ok(choiceQuestion, 'Qwen max-choice source case has no choice question.');
  assert.equal(
    choiceCount(choiceQuestion.criteria, 'Qwen max-choice criteria'),
    QWEN35_RELEASE_MODEL.maxChoiceOptions,
    'Qwen max-choice source case must have exactly 255 options.'
  );
  assert.equal(
    choiceBeyondLimit.requestedOptionCount,
    QWEN35_RELEASE_MODEL.maxChoiceOptions + 1,
    'Qwen over-choice case must have 256 options.'
  );
  const scoreAtLimit = sourceCase(boundaryCases.scoreLevelsAtLimit, 'Qwen max-score-levels source case');
  const scoreQuestion = questionEntries(scoreAtLimit.request.questions, 'Qwen max-score questions')
    .map(([name, question]) => record(question, `Qwen max-score question ${name}`))
    .find((question) => question.type === 'score');
  assert.ok(scoreQuestion, 'Qwen max-score source case has no score question.');
  assert.ok(Array.isArray(scoreQuestion.criteria), 'Qwen max-score criteria must be an array.');
  assert.equal(
    scoreQuestion.criteria.length,
    QWEN35_RELEASE_MODEL.maxScoreLevels,
    'Qwen max-score source case must have exactly ten levels.'
  );
  const scoreBeyondLimit = record(boundaryCases.scoreLevelsBeyondLimit, 'Qwen over-score-levels source case');
  const scoreBeyondRequest = assertRequest(scoreBeyondLimit.request, 'Qwen over-score-levels request');
  assert.equal(
    scoreBeyondLimit.expectedErrorCode,
    'INVALID_INPUT',
    'Qwen 11-level score case must reject as invalid input.'
  );
  const overScoreQuestion = questionEntries(scoreBeyondRequest.questions, 'Qwen over-score questions')
    .map(([name, question]) => record(question, `Qwen over-score question ${name}`))
    .find((question) => question.type === 'score');
  assert.ok(overScoreQuestion, 'Qwen over-score source case has no score question.');
  assert.ok(Array.isArray(overScoreQuestion.criteria), 'Qwen over-score criteria must be an array.');
  assert.equal(
    overScoreQuestion.criteria.length,
    QWEN35_RELEASE_MODEL.maxScoreLevels + 1,
    'Qwen over-score request must contain exactly eleven levels.'
  );
  const batchAtLimit = sourceCase(boundaryCases.batchAtLimit, 'Qwen max-batch source case');
  const batchBeyondLimit = sourceCase(boundaryCases.batchBeyondLimit, 'Qwen over-batch source case');
  assert.equal(
    Object.keys(record(batchAtLimit.request.questions, 'Qwen max-batch questions')).length,
    32,
    'Qwen batch-at-limit case must contain 32 questions.'
  );
  assert.equal(
    Object.keys(record(batchBeyondLimit.request.questions, 'Qwen over-batch questions')).length,
    33,
    'Qwen batch-over-limit case must contain 33 questions.'
  );
  const batchSourceRun = record(
    record(boundaryReference.value.sourceRuns, 'Qwen boundary source runs').batch,
    'Qwen batch source run'
  );
  for (const [description, item] of [
    ['Qwen max-batch source case', batchAtLimit],
    ['Qwen over-batch source case', batchBeyondLimit]
  ] as const) {
    assert.ok(item.candidateUsageExpectation, `${description} has no default-cache usage expectation.`);
    assert.equal(
      item.candidateUsageExpectation.mode,
      'derived-source-prefix-groups',
      `${description} must retain uncached source usage and use the explicit cache derivation.`
    );
    assert.equal(
      item.candidateUsageExpectation.evidence.sourceRunPath,
      batchSourceRun.reportPath,
      `${description} source-run path differs from the boundary report pin.`
    );
    assert.equal(
      item.candidateUsageExpectation.evidence.sourceRunSha256,
      batchSourceRun.reportSha256,
      `${description} source-run hash differs from the boundary report pin.`
    );
  }
  const imageCountBeyondLimit = record(boundaryCases.imageCountBeyondLimit, 'Qwen over-image-count source case');
  const imageRequest = assertRequest(imageCountBeyondLimit.request, 'Qwen over-image-count request');
  assert.equal(
    imageCountBeyondLimit.expectedErrorCode,
    'INVALID_INPUT',
    'Qwen five-image source case must reject as invalid input.'
  );
  assert.ok(
    Array.isArray(imageCountBeyondLimit.imageFiles),
    'Qwen over-image-count source case must identify image files.'
  );
  assert.equal(
    imageCountBeyondLimit.imageFiles.length,
    QWEN35_RELEASE_MODEL.maxImages + 1,
    'Qwen over-image-count source case must contain five images.'
  );
  const rejectedImages = imageCountBeyondLimit.imageFiles.map((rawFile, index) => {
    const file = record(rawFile, `Qwen over-image-count image ${index}`);
    const image = {
      path: safeRelativePath(file.path, `Qwen over-image-count image ${index} path`),
      sha256: digest(file.sha256, `Qwen over-image-count image ${index} SHA-256`),
      bytes: Number(file.bytes),
      mimeType: text(file.mimeType, `Qwen over-image-count image ${index} MIME type`)
    };
    assert.ok(
      Number.isSafeInteger(image.bytes) && image.bytes > 0,
      `Qwen over-image-count image size is invalid: ${image.path}`
    );
    return image;
  });
  for (const image of rejectedImages) {
    const imageBytes = await readRegularFile(root, image.path);
    assert.equal(imageBytes.byteLength, image.bytes, `Qwen over-image-count image size differs: ${image.path}`);
    assert.equal(sha256(imageBytes), image.sha256, `Qwen over-image-count image digest differs: ${image.path}`);
  }

  const fileRefs = {
    manifest: pinnedManifest,
    text: textReference,
    vision: visionReference,
    ...(mixedVisionReference ? {mixedVision: mixedVisionReference} : {}),
    boundaries: boundaryReference
  };
  const fixtureSha256 = Object.fromEntries(
    Object.entries(fileRefs)
      .filter(([name]) => name !== 'manifest')
      .map(([name, value]) => [name, value.sha256])
  );
  const artifactSha256 = Object.fromEntries(
    Object.entries(manifestFiles).map(([path, metadata]) => [path, metadata.sha256])
  );
  const b3S4096Case = await loadQwen35B3S4096SourceReference();
  return {
    artifactRoot: root,
    evidenceInput,
    evidenceInputSha256: sha256(evidenceBytes),
    manifestPath: pinnedManifest.path,
    manifestSha256,
    manifestRawSha256: pinnedManifest.sha256,
    assetManifestSha256: QWEN35_RELEASE_MODEL.assetManifestSha256,
    manifest: pinnedManifest.value,
    manifestFiles,
    sourceCases,
    visionCases,
    boundaryCases: {
      contextAtLimit,
      contextBeyondLimit,
      choiceOptionsAtLimit: choiceAtLimit,
      choiceOptionsBeyondLimit: choiceBeyondLimit,
      scoreLevelsAtLimit: scoreAtLimit,
      scoreLevelsBeyondLimit: scoreBeyondLimit,
      batchAtLimit,
      batchBeyondLimit,
      imageCountBeyondLimit: {
        name: text(imageCountBeyondLimit.name, 'Qwen over-image-count case name'),
        request: imageRequest,
        imageFiles: rejectedImages,
        expectedErrorCode: 'INVALID_INPUT'
      }
    },
    b3S4096Case,
    fixtureSha256,
    artifactSha256
  };
}

export async function verifyQwen35ExactPackage(
  archivePath: string,
  installedPackagePath: string
): Promise<Qwen35CandidatePackage> {
  assert.ok(archivePath.endsWith('.tgz'), 'Qwen release archive must be an npm .tgz package.');
  const archive = await verifyArchiveInstall(archivePath, installedPackagePath);
  assert.equal(archive.packageName, 'web-decision', 'Qwen release archive package name mismatch.');
  assert.ok(
    /^\d+\.\d+\.\d+$/u.test(archive.packageVersion) || archive.packageVersion === '0.0.0-qwen35-candidate',
    'Qwen exact-package run must use a stable version or the explicit local candidate version.'
  );
  const installedRoot = await realpath(resolve(installedPackagePath));
  assert.equal(
    archive.installedRoot,
    installedRoot,
    'Installed package root differs from the archive verifier result.'
  );
  return {archive, installedRoot};
}

interface DecimalValue {
  readonly coefficient: bigint;
  readonly scale: number;
}

function decimalValue(value: number): DecimalValue {
  const match = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/iu.exec(String(value));
  assert.ok(match, `Expected a finite decimal number, got ${String(value)}.`);
  const fraction = match[3] ?? '';
  const exponent = Number(match[4] ?? '0');
  let coefficient = BigInt(`${match[2]!}${fraction}`);
  if (match[1] === '-') coefficient = -coefficient;
  let scale = fraction.length - exponent;
  if (scale < 0) {
    coefficient *= 10n ** BigInt(-scale);
    scale = 0;
  }
  while (scale > 0 && coefficient % 10n === 0n) {
    coefficient /= 10n;
    scale -= 1;
  }
  return {coefficient, scale};
}

function scaledDecimalValue(value: DecimalValue, scale: number): bigint {
  return value.coefficient * 10n ** BigInt(scale - value.scale);
}

interface PrecisionEnvelopeComparison {
  readonly absoluteDelta: number;
  readonly allowedDelta: number;
  readonly withinTolerance: boolean;
}

function comparePrecisionEnvelope(
  actualValue: unknown,
  expectedValue: unknown,
  precisionReferenceValue: unknown,
  description: string
): PrecisionEnvelopeComparison {
  const actual = Number(actualValue);
  const expected = Number(expectedValue);
  const precisionReference = precisionReferenceValue === undefined ? undefined : Number(precisionReferenceValue);
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${description} precision values must be finite.`);
  if (precisionReference !== undefined)
    assert.ok(Number.isFinite(precisionReference), `${description} precision values must be finite.`);

  const actualDecimal = decimalValue(actual);
  const expectedDecimal = decimalValue(expected);
  const precisionDecimal = precisionReference === undefined ? undefined : decimalValue(precisionReference);
  const roundingTolerance = decimalValue(OUTPUT_ROUNDING_TOLERANCE);
  const textTolerance = precisionDecimal
    ? undefined
    : decimalValue(TEXT_SOURCE_DERIVED_TOLERANCE + OUTPUT_ROUNDING_TOLERANCE);
  const scale = Math.max(
    actualDecimal.scale,
    expectedDecimal.scale,
    precisionDecimal?.scale ?? 0,
    roundingTolerance.scale,
    textTolerance?.scale ?? 0
  );
  const actualUnits = scaledDecimalValue(actualDecimal, scale);
  const expectedUnits = scaledDecimalValue(expectedDecimal, scale);
  const allowedUnits = precisionDecimal
    ? absBigInt(expectedUnits - scaledDecimalValue(precisionDecimal, scale)) +
      scaledDecimalValue(roundingTolerance, scale)
    : scaledDecimalValue(textTolerance!, scale);
  const absoluteDeltaUnits = absBigInt(actualUnits - expectedUnits);
  return {
    absoluteDelta: Number(`${absoluteDeltaUnits}e-${scale}`),
    allowedDelta: Number(`${allowedUnits}e-${scale}`),
    withinTolerance: absoluteDeltaUnits <= allowedUnits
  };
}

function absBigInt(value: bigint): bigint {
  return value < 0n ? -value : value;
}

function compareAnswerConfidence(
  actual: RecordValue,
  expected: RecordValue,
  precision: RecordValue | undefined,
  description: string
): void {
  assert.ok(
    comparePrecisionEnvelope(actual.confidence, expected.confidence, precision?.confidence, `${description} confidence`)
      .withinTolerance,
    `${description} confidence differs from the source precision envelope.`
  );
}

function compareChoiceAnswer(
  actual: RecordValue,
  expected: RecordValue,
  precision: RecordValue | undefined,
  description: string
): void {
  assert.equal(actual.choice, expected.choice, `${description} choice differs from source.`);
  if (precision)
    assert.equal(precision.choice, expected.choice, `${description} source precision changed the choice label.`);
  compareAnswerConfidence(actual, expected, precision, description);
  compareProbabilityMaps(actual.probabilities, expected.probabilities, description, precision?.probabilities);
}

function compareScoreAnswer(
  actual: RecordValue,
  expected: RecordValue,
  precision: RecordValue | undefined,
  description: string
): void {
  assert.ok(
    comparePrecisionEnvelope(actual.score, expected.score, precision?.score, `${description} score`).withinTolerance,
    `${description} score differs from the source precision envelope.`
  );
  compareAnswerConfidence(actual, expected, precision, description);
  assert.deepEqual(actual.legend, expected.legend, `${description} score legend differs from source.`);
  if (precision)
    assert.deepEqual(precision.legend, expected.legend, `${description} precision reference changed the score legend.`);
  compareProbabilityMaps(actual.probabilities, expected.probabilities, description, precision?.probabilities);
}

export function compareQwen35Answer(
  actualValue: unknown,
  expectedValue: unknown,
  description: string,
  precisionReferenceValue?: unknown
): void {
  const actual = assertAnswer(actualValue, `${description} observed answer`);
  const expected = assertAnswer(expectedValue, `${description} source answer`);
  const precisionReference =
    precisionReferenceValue === undefined
      ? undefined
      : assertAnswer(precisionReferenceValue, `${description} alternate precision source answer`);
  assert.equal(actual.type, expected.type, `${description} typed answer differs from source.`);
  if (precisionReference) {
    assert.equal(
      precisionReference.type,
      expected.type,
      `${description} precision reference type differs from source.`
    );
  }
  if (actual.type === 'choice' && expected.type === 'choice') {
    compareChoiceAnswer(
      actual,
      expected,
      precisionReference?.type === 'choice' ? precisionReference : undefined,
      description
    );
  } else if (actual.type === 'score' && expected.type === 'score') {
    compareScoreAnswer(
      actual,
      expected,
      precisionReference?.type === 'score' ? precisionReference : undefined,
      description
    );
  } else if (actual.type === 'noul' && expected.type === 'noul') {
    const precisionNoul = precisionReference?.type === 'noul' ? precisionReference.noul : undefined;
    assert.ok(
      comparePrecisionEnvelope(actual.noul, expected.noul, precisionNoul, `${description} noul`).withinTolerance,
      `${description} noul value differs from the source precision envelope.`
    );
  }
}

function compareProbabilityMaps(
  actualValue: unknown,
  expectedValue: unknown,
  description: string,
  precisionReferenceValue?: unknown
): void {
  const actual = assertProbabilityMap(actualValue, `${description} observed probabilities`);
  const expected = assertProbabilityMap(expectedValue, `${description} source probabilities`);
  const precisionReference =
    precisionReferenceValue === undefined
      ? undefined
      : assertProbabilityMap(precisionReferenceValue, `${description} alternate precision probabilities`);
  assert.deepEqual(Object.keys(actual), Object.keys(expected), `${description} probability order differs from source.`);
  if (precisionReference)
    assert.deepEqual(
      Object.keys(precisionReference),
      Object.keys(expected),
      `${description} source precision probability order differs.`
    );
  for (const key of Object.keys(expected)) {
    assert.ok(
      comparePrecisionEnvelope(
        actual[key],
        expected[key],
        precisionReference?.[key],
        `${description} probability ${key}`
      ).withinTolerance,
      `${description} probability differs from the source precision envelope for ${key}.`
    );
  }
}

export interface Qwen35ParityCheck {
  readonly candidate: unknown;
  readonly source: unknown;
  readonly matches: boolean | null;
}

export interface Qwen35NumericParityCheck extends Qwen35ParityCheck {
  readonly precisionReference: unknown;
  readonly absoluteDelta: number | null;
  readonly allowedDelta: number | null;
  readonly withinTolerance: boolean | null;
}

export interface Qwen35AnswerParityDiagnostic {
  readonly name: string;
  readonly status: 'passed' | 'failed' | 'incomplete';
  readonly candidate: unknown;
  readonly source: unknown;
  readonly precisionReference: unknown;
  readonly validity: Readonly<{
    candidate: boolean;
    source: boolean;
    precisionReference: boolean | null;
  }>;
  readonly checks: Readonly<Record<string, Qwen35ParityCheck | null>>;
  readonly numeric: Readonly<Record<string, Qwen35NumericParityCheck>>;
  readonly probabilities: Readonly<Record<string, Qwen35NumericParityCheck>>;
}

export interface Qwen35ResultParityDiagnostic {
  readonly name: string;
  readonly status: 'passed' | 'failed' | 'incomplete';
  readonly candidateOutput: unknown;
  readonly sourceOutput: unknown;
  readonly precisionReference: unknown;
  readonly candidateUsageExpectation: unknown;
  readonly checks: Readonly<Record<string, Qwen35ParityCheck | null>>;
  readonly answers: readonly Qwen35AnswerParityDiagnostic[];
}

function diagnosticRecord(value: unknown): RecordValue | undefined {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as RecordValue) : undefined;
}

function diagnosticCheck(candidate: unknown, source: unknown): Qwen35ParityCheck {
  return {
    candidate: candidate === undefined ? null : candidate,
    source: source === undefined ? null : source,
    matches: candidate === undefined || source === undefined ? null : isDeepStrictEqual(candidate, source)
  };
}

function diagnosticNumber(candidate: unknown, source: unknown, precisionReference: unknown): Qwen35NumericParityCheck {
  const candidateIsNumber = typeof candidate === 'number' && Number.isFinite(candidate);
  const sourceIsNumber = typeof source === 'number' && Number.isFinite(source);
  const precisionIsNumber = typeof precisionReference === 'number' && Number.isFinite(precisionReference);
  const envelope =
    candidateIsNumber && sourceIsNumber && (precisionReference === undefined || precisionIsNumber)
      ? comparePrecisionEnvelope(candidate, source, precisionReference, 'Qwen numeric parity diagnostic')
      : undefined;
  const absoluteDelta = envelope?.absoluteDelta ?? null;
  const allowedDelta = envelope?.allowedDelta ?? null;
  const withinTolerance = envelope?.withinTolerance ?? null;
  return {
    candidate: candidate === undefined ? null : candidate,
    source: source === undefined ? null : source,
    matches: withinTolerance,
    precisionReference: precisionReference === undefined ? null : precisionReference,
    absoluteDelta,
    allowedDelta,
    withinTolerance
  };
}

function validDiagnosticProbabilities(value: unknown): boolean {
  const probabilities = diagnosticRecord(value);
  if (!probabilities) return false;
  const values = Object.values(probabilities);
  return (
    values.length >= 2 &&
    values.every((item) => typeof item === 'number' && Number.isFinite(item) && item >= 0 && item <= 1) &&
    Math.abs(values.reduce<number>((sum, item) => sum + Number(item), 0) - 1) <= 0.001
  );
}

function validDiagnosticAnswer(value: unknown): boolean {
  const answer = diagnosticRecord(value);
  if (!answer) return false;
  if (answer.type === 'choice') {
    const probabilities = diagnosticRecord(answer.probabilities);
    return (
      typeof answer.choice === 'string' &&
      validDiagnosticProbabilities(answer.probabilities) &&
      Boolean(probabilities && Object.hasOwn(probabilities, answer.choice)) &&
      typeof answer.confidence === 'number' &&
      Number.isFinite(answer.confidence)
    );
  }
  if (answer.type === 'score') {
    const probabilities = diagnosticRecord(answer.probabilities);
    const legend = diagnosticRecord(answer.legend);
    return (
      typeof answer.score === 'number' &&
      Number.isFinite(answer.score) &&
      typeof answer.confidence === 'number' &&
      Number.isFinite(answer.confidence) &&
      validDiagnosticProbabilities(answer.probabilities) &&
      Boolean(probabilities && legend && isDeepStrictEqual(Object.keys(legend), Object.keys(probabilities)))
    );
  }
  return (
    answer.type === 'noul' &&
    typeof answer.noul === 'number' &&
    Number.isFinite(answer.noul) &&
    answer.noul >= 0 &&
    answer.noul <= 1
  );
}

function parityStatus(checks: readonly (boolean | null)[]): 'passed' | 'failed' | 'incomplete' {
  if (checks.includes(false)) return 'failed';
  if (checks.length > 0 && checks.every((check) => check === true)) return 'passed';
  return 'incomplete';
}

function answerProbabilityDiagnostics(
  candidate: RecordValue | undefined,
  source: RecordValue,
  precisionReference: RecordValue | undefined,
  checks: Record<string, Qwen35ParityCheck | null>
): Record<string, Qwen35NumericParityCheck> {
  const candidateProbabilities = diagnosticRecord(candidate?.probabilities);
  const sourceProbabilities = diagnosticRecord(source.probabilities);
  const precisionProbabilities = diagnosticRecord(precisionReference?.probabilities);
  checks.probabilityOrder = diagnosticCheck(
    candidateProbabilities ? Object.keys(candidateProbabilities) : undefined,
    sourceProbabilities ? Object.keys(sourceProbabilities) : undefined
  );
  checks.precisionProbabilityOrder = precisionReference
    ? diagnosticCheck(
        precisionProbabilities ? Object.keys(precisionProbabilities) : undefined,
        sourceProbabilities ? Object.keys(sourceProbabilities) : undefined
      )
    : null;
  const probabilities: Record<string, Qwen35NumericParityCheck> = {};
  for (const [key, value] of Object.entries(sourceProbabilities ?? {}))
    probabilities[key] = diagnosticNumber(candidateProbabilities?.[key], value, precisionProbabilities?.[key]);
  return probabilities;
}

function answerParityDiagnostic(
  name: string,
  candidateValue: unknown,
  sourceValue: unknown,
  precisionValue: unknown
): Qwen35AnswerParityDiagnostic {
  const candidate = diagnosticRecord(candidateValue);
  const source = diagnosticRecord(sourceValue);
  const precisionReference = diagnosticRecord(precisionValue);
  const checks: Record<string, Qwen35ParityCheck | null> = {
    type: diagnosticCheck(candidate?.type, source?.type),
    precisionType: precisionReference ? diagnosticCheck(precisionReference.type, source?.type) : null
  };
  const numeric: Record<string, Qwen35NumericParityCheck> = {};
  let probabilities: Record<string, Qwen35NumericParityCheck> = {};

  switch (source?.type) {
    case 'choice':
      checks.choice = diagnosticCheck(candidate?.choice, source.choice);
      checks.precisionChoice = precisionReference ? diagnosticCheck(precisionReference.choice, source.choice) : null;
      numeric.confidence = diagnosticNumber(candidate?.confidence, source.confidence, precisionReference?.confidence);
      probabilities = answerProbabilityDiagnostics(candidate, source, precisionReference, checks);
      break;
    case 'score':
      numeric.score = diagnosticNumber(candidate?.score, source.score, precisionReference?.score);
      numeric.confidence = diagnosticNumber(candidate?.confidence, source.confidence, precisionReference?.confidence);
      checks.legend = diagnosticCheck(candidate?.legend, source.legend);
      checks.precisionLegend = precisionReference ? diagnosticCheck(precisionReference.legend, source.legend) : null;
      probabilities = answerProbabilityDiagnostics(candidate, source, precisionReference, checks);
      break;
    case 'noul':
      numeric.noul = diagnosticNumber(candidate?.noul, source.noul, precisionReference?.noul);
      break;
  }

  const validity = {
    candidate: validDiagnosticAnswer(candidateValue),
    source: validDiagnosticAnswer(sourceValue),
    precisionReference: precisionValue === undefined ? null : validDiagnosticAnswer(precisionValue)
  };
  const statusChecks: Array<boolean | null> = [
    validity.candidate,
    validity.source,
    ...(validity.precisionReference === null ? [] : [validity.precisionReference]),
    ...Object.entries(checks)
      .filter(([key]) => precisionReference !== undefined || !key.startsWith('precision'))
      .map(([, check]) => check?.matches ?? null),
    ...Object.values(numeric).map((check) => check.withinTolerance),
    ...Object.values(probabilities).map((check) => check.withinTolerance)
  ];

  return {
    name,
    status: parityStatus(statusChecks),
    candidate: candidateValue ?? null,
    source: sourceValue ?? null,
    precisionReference: precisionValue ?? null,
    validity,
    checks,
    numeric,
    probabilities
  };
}

export function buildQwen35ResultParityDiagnostic(
  candidateValue: unknown,
  sourceValue: unknown,
  name: string,
  precisionReferenceValue?: unknown,
  candidateUsageExpectationValue?: unknown
): Qwen35ResultParityDiagnostic {
  const candidate = diagnosticRecord(candidateValue);
  const source = diagnosticRecord(sourceValue);
  const precisionReference = diagnosticRecord(precisionReferenceValue);
  const usageExpectation = diagnosticRecord(candidateUsageExpectationValue);
  const candidateUsage = usageExpectation?.usage === undefined ? source?.usage : usageExpectation.usage;
  const candidateAnswers = diagnosticRecord(candidate?.answers);
  const sourceAnswers = diagnosticRecord(source?.answers);
  const precisionAnswers = diagnosticRecord(precisionReference?.answers);
  const checks: Record<string, Qwen35ParityCheck | null> = {
    candidateModel: diagnosticCheck(candidate?.model, QWEN35_RELEASE_MODEL.modelId),
    sourceModel: diagnosticCheck(source?.model, QWEN35_RELEASE_MODEL.modelId),
    precisionModel: precisionReference ? diagnosticCheck(precisionReference.model, source?.model) : null,
    usage: diagnosticCheck(candidate?.usage, candidateUsage),
    precisionUsage: precisionReference ? diagnosticCheck(precisionReference.usage, source?.usage) : null,
    answerOrder: diagnosticCheck(
      candidateAnswers ? Object.keys(candidateAnswers) : undefined,
      sourceAnswers ? Object.keys(sourceAnswers) : undefined
    ),
    precisionAnswerOrder: precisionReference
      ? diagnosticCheck(
          precisionAnswers ? Object.keys(precisionAnswers) : undefined,
          sourceAnswers ? Object.keys(sourceAnswers) : undefined
        )
      : null
  };
  const answers = Object.entries(sourceAnswers ?? {}).map(([answerName, sourceAnswer]) =>
    answerParityDiagnostic(answerName, candidateAnswers?.[answerName], sourceAnswer, precisionAnswers?.[answerName])
  );
  const statusChecks: Array<boolean | null> = [
    ...Object.entries(checks)
      .filter(([key]) => precisionReference !== undefined || !key.startsWith('precision'))
      .map(([, check]) => check?.matches ?? null),
    ...answers.map(({status}) => (status === 'passed' ? true : status === 'failed' ? false : null))
  ];

  return {
    name,
    status: parityStatus(statusChecks),
    candidateOutput: candidateValue === undefined ? null : candidateValue,
    sourceOutput: sourceValue === undefined ? null : sourceValue,
    precisionReference: precisionReferenceValue === undefined ? null : precisionReferenceValue,
    candidateUsageExpectation: candidateUsageExpectationValue === undefined ? null : candidateUsageExpectationValue,
    checks,
    answers
  };
}

export async function writeQwen35OnlineDiagnostic(path: string, value: RecordValue): Promise<void> {
  const outputPath = resolve(path);
  await mkdir(dirname(outputPath), {recursive: true});
  await writeFile(outputPath, `${JSON.stringify(value, null, 2)}\n`);
}

export function compareQwen35Result(
  actualValue: unknown,
  expectedValue: unknown,
  description: string,
  precisionReferenceValue?: unknown,
  candidateUsageExpectationValue?: unknown
): void {
  const actual = record(actualValue, `${description} observed result`);
  const expected = record(expectedValue, `${description} source result`);
  const precisionReference =
    precisionReferenceValue === undefined
      ? undefined
      : record(precisionReferenceValue, `${description} alternate precision source result`);
  const usageExpectation =
    candidateUsageExpectationValue === undefined
      ? undefined
      : record(candidateUsageExpectationValue, `${description} candidate usage expectation`);
  assert.equal(actual.model, QWEN35_RELEASE_MODEL.modelId, `${description} model ID mismatch.`);
  assert.equal(expected.model, QWEN35_RELEASE_MODEL.modelId, `${description} source model ID mismatch.`);
  const expectedUsage = usageExpectation
    ? record(usageExpectation.usage, `${description} candidate usage`)
    : expected.usage;
  assert.deepEqual(
    actual.usage,
    expectedUsage,
    usageExpectation
      ? `${description} usage differs from the pinned default-cache source expectation.`
      : `${description} usage differs from upstream source.`
  );
  if (precisionReference) {
    assert.equal(precisionReference.model, expected.model, `${description} precision source model differs.`);
    assert.deepEqual(precisionReference.usage, expected.usage, `${description} precision source usage differs.`);
  }
  const actualAnswers = record(actual.answers, `${description} observed answers`);
  const expectedAnswers = record(expected.answers, `${description} expected answers`);
  assert.deepEqual(
    Object.keys(actualAnswers),
    Object.keys(expectedAnswers),
    `${description} answer order differs from source.`
  );
  const precisionAnswers = precisionReference
    ? record(precisionReference.answers, `${description} precision source answers`)
    : undefined;
  if (precisionAnswers)
    assert.deepEqual(
      Object.keys(precisionAnswers),
      Object.keys(expectedAnswers),
      `${description} precision source answer order differs.`
    );
  for (const [name, answer] of Object.entries(expectedAnswers))
    compareQwen35Answer(actualAnswers[name], answer, `${description}/${name}`, precisionAnswers?.[name]);
}

export function compareQwen35B3S4096Result(
  actualValue: unknown,
  source: Qwen35B3S4096SourceCase,
  description = 'Qwen B3×S4096'
): void {
  const actual = record(actualValue, `${description} observed result`);
  assert.equal(actual.model, QWEN35_RELEASE_MODEL.modelId, `${description} model ID mismatch.`);
  assert.deepEqual(actual.usage, source.expectedUsage, `${description} usage differs from the pinned source boundary.`);
  const answers = record(actual.answers, `${description} answers`);
  assert.deepEqual(Object.keys(answers), source.questionNames, `${description} answer rows differ from source.`);
  for (const [rowIndex, name] of source.questionNames.entries()) {
    const answer = record(answers[name], `${description}/${name} answer`);
    assert.equal(answer.type, 'choice', `${description}/${name} typed result differs from source.`);
    const probabilities = record(answer.probabilities, `${description}/${name} probabilities`);
    assert.deepEqual(
      Object.keys(probabilities),
      source.labels,
      `${description}/${name} option order differs from source.`
    );
    const expectedRow = source.expectedProbabilities[rowIndex]!;
    const expectedChoice = source.expectedChoices[rowIndex];
    assert.equal(answer.choice, expectedChoice, `${description}/${name} choice differs from source.`);
    for (const [optionIndex, label] of source.labels.entries()) {
      const observed = probabilities[label];
      const expected = expectedRow[optionIndex]!;
      assert.ok(
        typeof observed === 'number' &&
          Number.isFinite(observed) &&
          Math.abs(observed - expected) <= source.probabilityTolerance,
        `${description}/${name} probability ${label} differs from source beyond ${source.probabilityTolerance}.`
      );
    }
  }
}

export function assertQwen35OfflineParity(online: unknown, offline: unknown): void {
  assert.deepEqual(offline, online, 'Offline Qwen inference must reproduce the online source-parity result exactly.');
}

export function assertQwen35OfflineResults(online: readonly unknown[], offlineValue: unknown): void {
  assert.ok(Array.isArray(offlineValue), 'Offline Qwen results must be an array.');
  assert.equal(offlineValue.length, online.length, 'Offline Qwen result count differs from online inference.');
  for (const [index, expectedValue] of online.entries()) {
    const expected = record(expectedValue, `Online Qwen result ${index}`);
    const observed = record(offlineValue[index], `Offline Qwen result ${index}`);
    assert.equal(observed.name, expected.name, `Offline Qwen case order differs at index ${index}.`);
    assertQwen35OfflineParity(expected.result, observed.result);
  }
}

export function assertQwen35EvidenceRecord(value: unknown): RecordValue {
  const evidence = record(value, 'Qwen browser evidence');
  assert.equal(evidence.schemaVersion, 1, 'Unsupported Qwen browser evidence schema.');
  assert.equal(
    evidence.status,
    'candidate-browser-evidence-passed',
    'Qwen candidate evidence is not a release verification.'
  );
  assert.equal(evidence.modelId, QWEN35_RELEASE_MODEL.catalogModelId, 'Qwen browser evidence model ID mismatch.');
  const packageEvidence = record(evidence.package, 'Qwen exact package evidence');
  assert.equal(packageEvidence.name, 'web-decision', 'Qwen release package name mismatch.');
  assert.ok(
    typeof packageEvidence.version === 'string' &&
      (/^\d+\.\d+\.\d+$/u.test(packageEvidence.version) || packageEvidence.version === '0.0.0-qwen35-candidate'),
    'Qwen exact-package evidence must record a stable version or the explicit local candidate version.'
  );
  digest(packageEvidence.archiveSha256, 'Qwen archive SHA-256');
  digest(packageEvidence.packageTreeSha256, 'Qwen package tree SHA-256');
  text(packageEvidence.bundlePath, 'Qwen public bundle path');
  digest(packageEvidence.bundleSha256, 'Qwen bundle SHA-256');
  text(packageEvidence.candidateBundlePath, 'Qwen candidate bundle path');
  digest(packageEvidence.candidateBundleSha256, 'Qwen candidate bundle SHA-256');
  assert.equal(
    packageEvidence.bundlePath,
    packageEvidence.candidateBundlePath,
    'Qwen candidate evidence must execute the exact public package bundle.'
  );
  assert.equal(
    packageEvidence.bundleSha256,
    packageEvidence.candidateBundleSha256,
    'Qwen candidate evidence bundle digest differs from the public package bundle.'
  );
  assert.equal(
    packageEvidence.exactInstallMatch,
    true,
    'Qwen browser evidence must use the exact npm package install.'
  );
  const source = record(evidence.source, 'Qwen source pin evidence');
  assertSourcePins(source, 'Qwen browser evidence');
  assert.deepEqual(
    source.b3S4096SourceReference,
    QWEN35_B3_S4096_SOURCE_REFERENCE,
    'Qwen B3×S4096 evidence must identify the pinned independent source report.'
  );
  const candidateExecution = record(evidence.candidateExecution, 'Qwen candidate execution evidence');
  assert.equal(candidateExecution.backend, 'tvm-webgpu', 'Qwen candidate must execute on direct TVM WebGPU.');
  assert.equal(candidateExecution.runtime, 'TVM.js', 'Qwen candidate must use the pinned TVM.js runtime.');
  assert.deepEqual(
    candidateExecution.graphsExecuted,
    {r19: true, vision: true},
    'Qwen browser evidence must include executed TVM R19 and vision graphs.'
  );
  assert.deepEqual(
    candidateExecution.graphMainCallSites,
    {r19: true, vision: true, tvmRuntimeDispatch: true},
    'Qwen browser evidence must verify both compiled TVM main call sites and the pinned runtime dispatch.'
  );
  const graphDispatchCounts = record(candidateExecution.graphDispatchCounts, 'Qwen WebGPU graph dispatch counts');
  const tvmRuntimeDispatchCounts = record(
    candidateExecution.tvmRuntimeDispatchCounts,
    'Qwen TVM.js runtime dispatch counts'
  );
  const graphDispatchCallers = record(candidateExecution.graphDispatchCallers, 'Qwen TVM dispatch callers');
  for (const group of ['text', 'vision', 'batch-3-context-4096']) {
    assert.ok(
      Number.isSafeInteger(graphDispatchCounts[group]) && Number(graphDispatchCounts[group]) > 0,
      `Qwen browser evidence must observe WebGPU dispatches during ${group}.`
    );
    assert.ok(
      Number.isSafeInteger(tvmRuntimeDispatchCounts[group]) && Number(tvmRuntimeDispatchCounts[group]) > 0,
      `Qwen browser evidence must observe pinned TVM.js dispatches during ${group}.`
    );
    assert.match(
      text(graphDispatchCallers[group], `Qwen ${group} TVM.js dispatch caller`),
      /tvm\.generated/u,
      `Qwen ${group} dispatch must originate in the packaged TVM.js runtime.`
    );
  }
  assert.equal(candidateExecution.tvmUsed, true, 'Qwen candidate TVM WebGPU execution must be recorded as used.');
  const artifacts = record(evidence.artifacts, 'Qwen artifact evidence');
  digest(artifacts.manifestSha256, 'Qwen artifact manifest SHA-256');
  assert.equal(
    digest(artifacts.assetManifestSha256, 'Qwen package asset manifest SHA-256'),
    QWEN35_RELEASE_MODEL.assetManifestSha256,
    'Qwen package descriptor manifest differs from its pinned asset manifest.'
  );
  record(artifacts.filesSha256, 'Qwen artifact file digests');
  const hardware = record(evidence.hardware, 'Qwen hardware evidence');
  assert.equal(hardware.platform, 'darwin', 'Qwen M4 evidence must run on macOS.');
  assert.equal(hardware.arch, 'arm64', 'Qwen M4 evidence must run on Apple silicon.');
  assert.equal(hardware.chipType, 'Apple M4 Pro', 'Qwen M4 evidence must identify Apple M4 Pro host hardware.');
  const gpu = record(evidence.gpu, 'Qwen WebGPU provenance');
  assert.equal(gpu.api, 'webgpu', 'Qwen browser evidence must use WebGPU.');
  assert.equal(gpu.isFallbackAdapter, false, 'Qwen browser evidence must use a hardware adapter.');
  assert.match(
    String(record(gpu.adapter, 'Qwen WebGPU adapter info').vendor),
    /apple/iu,
    'Qwen WebGPU adapter must report Apple as its vendor.'
  );
  assert.match(String(gpu.renderer), /Apple M4 Pro/iu, 'Qwen GPU renderer must identify Apple M4 Pro.');
  assert.match(
    String(gpu.chromeGpuStatus),
    /Hardware accelerated/iu,
    'Qwen GPU report must show hardware acceleration.'
  );
  const browser = record(evidence.browser, 'Qwen browser identity');
  assert.equal(browser.name, 'chromium', 'Qwen browser evidence must use Chromium.');
  assert.ok(
    browser.channel === 'chromium' || browser.channel === 'chrome',
    'Qwen browser evidence must identify its Chromium or Chrome channel.'
  );
  assert.match(
    text(browser.version, 'Qwen browser version'),
    /^\d+(?:\.\d+){2,3}$/u,
    'Qwen Chrome version must be recorded in dotted numeric form.'
  );
  assert.match(String(browser.userAgent), /Chrome\/\d/iu, 'Qwen browser evidence must record a Chrome user agent.');
  const checks = record(evidence.checks, 'Qwen release checks');
  for (const name of [
    'curatedPublicApi',
    'sourceParity',
    'typedChoiceScoreNoul',
    'officialVision',
    'maximumContextBoundary',
    'maximumChoiceOptionsBoundary',
    'maximumScoreLevelsBoundary',
    'maximumImageCountBoundary',
    'batchBoundaries',
    'fullContextBatchBoundary',
    'onlineInference',
    'offlineCacheReuse'
  ])
    assert.equal(checks[name], true, `Qwen release check did not pass: ${name}.`);
  const fullContextBatch = record(evidence.fullContextBatch, 'Qwen full-context batch evidence');
  assert.deepEqual(
    fullContextBatch.sourceReference,
    QWEN35_B3_S4096_SOURCE_REFERENCE,
    'Qwen full-context batch must use the pinned independent source report.'
  );
  assert.equal(fullContextBatch.batchSize, 3, 'Qwen full-context batch evidence must preserve all three rows.');
  assert.equal(
    fullContextBatch.sequenceLength,
    QWEN35_RELEASE_MODEL.maxLength,
    'Qwen full-context batch evidence must preserve the 4096-token limit.'
  );
  assert.deepEqual(
    fullContextBatch.expectedUsage,
    {input_tokens: 4_250, output_tokens: 3},
    'Qwen full-context batch evidence usage differs from the source boundary.'
  );
  assert.deepEqual(
    fullContextBatch.expectedProbabilities,
    QWEN35_B3_S4096_PROBABILITIES,
    'Qwen full-context batch source probabilities differ from the independent source pin.'
  );
  assert.equal(fullContextBatch.parityPassed, true, 'Qwen B3×S4096 public output parity did not pass.');
  compareQwen35B3S4096Result(fullContextBatch.result, {
    name: 'batch-3-context-4096',
    request: {},
    questionNames: ['runtime', 'runtime_copy_2', 'runtime_copy_3'],
    labels: QWEN35_B3_S4096_LABELS,
    expectedChoices: QWEN35_B3_S4096_EXPECTED_CHOICES,
    batchSize: 3,
    sequenceLength: QWEN35_RELEASE_MODEL.maxLength,
    graphInputTokens: 12_288,
    expectedProbabilities: QWEN35_B3_S4096_PROBABILITIES,
    probabilityTolerance: TEXT_SOURCE_DERIVED_TOLERANCE,
    expectedUsage: {input_tokens: 4_250, output_tokens: 3},
    sourceReference: QWEN35_B3_S4096_SOURCE_REFERENCE
  });
  const results = record(evidence.results, 'Qwen browser results');
  assert.ok(
    Array.isArray(results.b3S4096) && results.b3S4096.length === 1,
    'Qwen evidence must contain one B3×S4096 result.'
  );
  const onlineB3S4096 = record(results.b3S4096[0], 'Qwen online B3×S4096 result');
  assert.equal(onlineB3S4096.name, 'batch-3-context-4096', 'Qwen online B3×S4096 case name changed.');
  assert.deepEqual(onlineB3S4096.result, fullContextBatch.result, 'Qwen B3×S4096 evidence outputs disagree.');
  assert.ok(Array.isArray(results.offline), 'Qwen evidence must include offline cache-reuse results.');
  const offlineB3S4096 = results.offline
    .map((rawResult, index) => record(rawResult, `Qwen offline result ${index}`))
    .filter(({name}) => name === 'batch-3-context-4096');
  assert.equal(offlineB3S4096.length, 1, 'Qwen offline evidence must include the B3×S4096 case once.');
  assertQwen35OfflineResults([onlineB3S4096], offlineB3S4096);
  assert.ok(Array.isArray(evidence.onlineNetworkRequests), 'Qwen online network request evidence is required.');
  const onlineRequests = evidence.onlineNetworkRequests.map((rawRequest, index) => {
    const request = record(rawRequest, `Qwen online network request ${index}`);
    return {
      url: text(request.url, `Qwen online network request ${index} URL`),
      method: text(request.method, `Qwen online network request ${index} method`).toUpperCase(),
      rangeHeader: request.rangeHeader === null ? null : text(request.rangeHeader, `Qwen online request ${index} Range`)
    };
  });
  const rangedHuggingFaceRequests = onlineRequests.filter((request) => {
    let url: URL;
    try {
      url = new URL(request.url);
    } catch {
      return false;
    }
    const hostname = url.hostname.toLowerCase();
    const upstreamHost =
      hostname === 'hf.co' ||
      hostname.endsWith('.hf.co') ||
      hostname === 'huggingface.co' ||
      hostname.endsWith('.huggingface.co');
    const range = /^bytes=(\d+)-(\d*)$/u.exec(request.rangeHeader ?? '');
    if (!range) return false;
    const start = Number(range[1]);
    const end = range[2] === '' ? undefined : Number(range[2]);
    return (
      url.protocol === 'https:' &&
      upstreamHost &&
      request.method === 'GET' &&
      Number.isSafeInteger(start) &&
      (end === undefined || (Number.isSafeInteger(end) && end >= start))
    );
  });
  assert.ok(
    rangedHuggingFaceRequests.length > 0,
    'Qwen online load must record a ranged GET to Hugging Face or hf.co after clearing the model cache.'
  );
  assert.deepEqual(
    evidence.rangedHuggingFaceRequests,
    rangedHuggingFaceRequests,
    'Qwen ranged Hugging Face request summary differs from the captured request log.'
  );
  assert.deepEqual(evidence.offlineNetworkAttempts, [], 'Offline Qwen inference made network requests.');
  return evidence;
}

export function resolveQwen35ArtifactPath(root: string, relativePath: string): string {
  const path = safeRelativePath(relativePath, 'Qwen artifact path');
  const result = resolve(root, ...path.split('/'));
  assert.ok(result.startsWith(`${resolve(root)}${sep}`), 'Qwen artifact path escapes the generated build root.');
  return result;
}
