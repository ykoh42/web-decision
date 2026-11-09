import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {createReadStream} from 'node:fs';
import {lstat, readFile, readdir, realpath} from 'node:fs/promises';
import {basename, dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {verifyTvmBuildEvidence} from './tvm-build-evidence.ts';
import {
  EXPANDED_INFERENCE_FIXTURE_PATH,
  EXPANDED_INFERENCE_FIXTURE_SHA256,
  loadExpandedInferenceFixture,
  validateExpandedInferenceEvidence,
  validateExpandedInferenceFixture,
  validateExpandedInferencePolicy,
  type ExpandedInferenceFixture,
  type ExpandedInferencePolicy
} from '../../tests/browser/kev06-expanded-inference.ts';

type AnyRecord = Record<string, any>;
type RequiredFiles = string[];

interface BrowserRequirement {
  name: string;
  project: string;
}

interface EvidencePolicy {
  schemaVersion: 2;
  evidencePrefix: string;
  testFile: string;
  expandedInference: ExpandedInferencePolicy;
  browsers: BrowserRequirement[];
  requireNonFallbackAdapter: true;
  requirePhysicalDevice: true;
  requiredApi: 'webgpu';
}

interface ModelRequirement {
  modelId: string;
  backend: {
    name: 'tvm-webgpu';
    requiredFeatures: string[];
    requiredLimits: Record<string, number>;
  };
  manifestSha256: string;
  manifestPath: string;
  requiredFiles: RequiredFiles;
  manifestFiles: Record<string, AssetMetadata>;
  bundledRuntimeFiles: BundledRuntimeFile[];
}

interface AssetMetadata {
  bytes: number;
  sha256: string;
}

interface BundledRuntimeFile extends AssetMetadata {
  manifestName: string;
  sourcePath: string;
}

interface AssetManifest extends AnyRecord {
  model?: string;
  modelId?: string;
  backend?: string;
  files?: Record<string, AssetMetadata>;
}

interface ArchiveManifestEvidence {
  path: string;
  manifest: AssetManifest;
  manifestSha256: string;
  filesSha256: string;
  files: Record<string, AssetMetadata>;
}

interface ArchiveInfo {
  archivePath: string;
  archiveSha256: string;
  installedRoot: string;
  packageName: string;
  packageVersion: string;
  packageTreeSha256: string;
  bundlePath: string;
  bundleSha256: string;
  installedPackagePathSha256: string;
  entryPathSha256: string;
  archiveFiles: string[];
}

interface PlaywrightTestResult extends AnyRecord {
  status?: string;
  stdout?: Array<string | {text?: string}>;
  errors?: unknown[];
}

interface PlaywrightTest extends AnyRecord {
  expectedStatus?: string;
  status?: string;
  projectName?: string;
  results?: PlaywrightTestResult[];
}

interface PlaywrightSpec extends AnyRecord {
  file?: string;
  title?: string;
  tests?: PlaywrightTest[];
}

interface PlaywrightSuite extends AnyRecord {
  file?: string;
  specs?: PlaywrightSpec[];
  suites?: PlaywrightSuite[];
}

interface PlaywrightReport extends AnyRecord {
  suites?: PlaywrightSuite[];
  errors?: unknown[];
  stats?: AnyRecord;
}

interface PlaywrightTestItem {
  file?: string;
  title?: string;
  test: PlaywrightTest;
}

interface VerificationContext {
  archive: ArchiveInfo;
  catalogSha256: string;
  manifests: Map<string, ArchiveManifestEvidence>;
  models: ModelRequirement[];
  policy: EvidencePolicy;
  workspaceRoot: string;
}

interface ReleaseEvidenceOptions {
  archivePath: string;
  installedPackagePath: string;
  playwrightReportPath: string;
  buildEvidencePath?: string;
  workspaceRoot?: string;
  sourcePackageRoot?: string;
}

interface ArtifactVerificationOptions extends ReleaseEvidenceOptions {
  policy: EvidencePolicy;
}

interface PublicCatalogDescriptor {
  modelId: string;
  backend: ModelRequirement['backend'];
  manifestSha256: string;
  requiredFiles: string[];
}

interface CatalogArtifactLock {
  manifest: {
    sourcePath: string;
    canonicalSha256: string;
    files: Record<string, AssetMetadata>;
  };
  bundledRuntimeFiles: BundledRuntimeFile[];
}

interface ModelSpecFile {
  schemaVersion: number;
  models: Array<{modelId: string; description: AnyRecord}>;
}

interface ArtifactLockFile {
  schemaVersion: number;
  byModelId: Record<string, CatalogArtifactLock>;
}

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const WORKSPACE_ROOT = resolve(process.env.WEB_DECISION_WORKSPACE_ROOT ?? resolve(SCRIPT_DIRECTORY, '../..'));
const RELEASE_SCRIPT_DIRECTORY = join(WORKSPACE_ROOT, 'scripts/release');
const POLICY_PATH = join(RELEASE_SCRIPT_DIRECTORY, 'release-evidence-policy.json');
export const RELEASE_EVIDENCE_POLICY_SHA256 = 'e9cc0f4ed17d22485057c3b213f6c8ba7694c94c95956d3afcb55440fe04d4bc';
export const BASELINE_REFERENCE = Object.freeze({
  packageSha256: '5aef3331e41a67e20fac9fbc00e47074d02e1feeef4e0c7e6f4b13fdc87bca6f',
  fixtureSha256: '91297c6ebf012170b95c61bb5a053b6ed86e4e47a129fe5eaaa681d4ec8c74cd',
  scope: 'baseline-preservation-only'
});
export const BASELINE_EXPECTED_OUTPUT = Object.freeze({
  selected: 'billing',
  optionOrder: Object.freeze(['billing', 'shipping']),
  logits: Object.freeze([4.62109375, -0.78466796875]),
  probabilities: Object.freeze([0.9955294431288633, 0.004470556871136836]),
  maxAbsoluteLogitDifference: 0.05,
  maxAbsoluteProbabilityDifference: 0.0005
});
export const UPSTREAM_REFERENCE = Object.freeze({
  fixtureSha256: '656e8403984e57f24dd38f08d461188bde7802288c369a88dac0b29319bada10',
  modelId: 'kev-0.6b-q4f16',
  graphSha256: '26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8',
  weightsSha256: '777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b',
  scope: 'single-input-upstream-onnx-cpu-reference'
});
export const UPSTREAM_EXPECTED_OUTPUT = Object.freeze({
  logits: Object.freeze([4.56640625, -0.8623046875]),
  probabilities: Object.freeze([0.9956304274292445, 0.0043695725707555465]),
  maxAbsoluteLogitDifference: 0.15,
  maxAbsoluteProbabilityDifference: 0.001
});
const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024;
const TARBALL = process.platform === 'win32' ? 'tar.exe' : 'tar';

export function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

export function canonicalPathSha256(path: string): string {
  const normalized = resolve(path).replaceAll('\\', '/');
  return sha256(process.platform === 'win32' ? normalized.toLowerCase() : normalized);
}

export function canonical<T>(value: T): T {
  if (Array.isArray(value)) return value.map(canonical) as T;
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(record)
        .sort()
        .map((key) => [key, canonical(record[key])])
    ) as T;
  }
  return value;
}

export function canonicalSha256(value: unknown): string {
  return sha256(Buffer.from(`${JSON.stringify(canonical(value))}\n`));
}

function fail(condition: unknown, message: string): asserts condition {
  assert.ok(condition, message);
}

function normalizedRelativePath(path: unknown, description: string): string {
  fail(typeof path === 'string' && path.length > 0, `${description} must be a non-empty path.`);
  fail(!path.includes('\\') && !isAbsolute(path), `${description} must be a safe relative path: ${path}`);
  const segments = path.split('/');
  fail(
    segments.every((segment) => segment && segment !== '.' && segment !== '..'),
    `${description} contains an unsafe segment: ${path}`
  );
  return segments.join('/');
}

function isPathWithin(root: string, target: string): boolean {
  const path = relative(resolve(root), resolve(target));
  return path === '' || (path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}

async function assertCanonicalDirectory(path: string, description: string): Promise<string> {
  const absolute = resolve(path);
  const info = await lstat(absolute);
  fail(info.isDirectory() && !info.isSymbolicLink(), `${description} must be a regular directory: ${absolute}`);
  fail((await realpath(absolute)) === absolute, `${description} must use its canonical path: ${absolute}`);
  return absolute;
}

async function hashFile(path: string): Promise<string> {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest('hex');
}

export function validatePolicy(input: unknown): EvidencePolicy {
  fail(
    input !== null && typeof input === 'object' && !Array.isArray(input),
    'Release evidence policy must be an object.'
  );
  const policy = input as AnyRecord;
  fail(policy.schemaVersion === 2, 'Unsupported release evidence policy schema.');
  fail(typeof policy.evidencePrefix === 'string' && policy.evidencePrefix.length > 0, 'Evidence prefix is required.');
  const testFile = normalizedRelativePath(policy.testFile, 'Policy test file');
  fail(testFile.startsWith('tests/browser/'), 'The release evidence spec must be under tests/browser/.');
  const expandedInference = validateExpandedInferencePolicy(policy.expandedInference);
  fail(policy.requireNonFallbackAdapter === true, 'The policy must require a non-fallback GPU adapter.');
  fail(policy.requirePhysicalDevice === true, 'The policy must require physical hardware metadata.');
  fail(policy.requiredApi === 'webgpu', 'The policy must require WebGPU.');
  fail(Array.isArray(policy.browsers) && policy.browsers.length > 0, 'The browser matrix must not be empty.');
  const browserList = policy.browsers as BrowserRequirement[];

  const names = new Set();
  const projects = new Set();
  for (const browser of browserList) {
    fail(browser && typeof browser.name === 'string' && browser.name.length > 0, 'Every browser needs a name.');
    fail(
      typeof browser.project === 'string' && browser.project.length > 0,
      `Browser ${browser.name} needs a Playwright project.`
    );
    fail(!names.has(browser.name), `Duplicate browser requirement: ${browser.name}`);
    fail(!projects.has(browser.project), `Duplicate Playwright project: ${browser.project}`);
    names.add(browser.name);
    projects.add(browser.project);
  }

  return {
    schemaVersion: 2,
    evidencePrefix: policy.evidencePrefix,
    testFile,
    expandedInference,
    browsers: browserList.map((browser) => ({...browser})),
    requireNonFallbackAdapter: true,
    requirePhysicalDevice: true,
    requiredApi: 'webgpu'
  };
}

export function validatePolicyBytes(bytes: Buffer, expectedSha256: string): EvidencePolicy {
  const actualSha256 = sha256(bytes);
  assert.equal(actualSha256, expectedSha256, 'Release evidence policy differs from its committed pin.');
  return validatePolicy(JSON.parse(bytes.toString('utf8')));
}

async function loadTrustedPolicy(workspaceRoot: string): Promise<EvidencePolicy> {
  const policyPath = resolve(workspaceRoot, relative(WORKSPACE_ROOT, POLICY_PATH));
  const bytes = await readFile(policyPath);
  const relativePolicyPath = relative(workspaceRoot, policyPath).split(sep).join('/');
  execFileSync('git', ['ls-files', '--error-unmatch', relativePolicyPath], {cwd: workspaceRoot, stdio: 'ignore'});
  const committed = execFileSync('git', ['show', `HEAD:${relativePolicyPath}`], {cwd: workspaceRoot});
  assert.deepEqual(bytes, committed, 'Release evidence policy must match the committed version.');
  return validatePolicyBytes(bytes, RELEASE_EVIDENCE_POLICY_SHA256);
}

function assertCommittedCatalogSources(workspaceRoot: string): void {
  const paths = ['packages/web-decision/src', 'packages/web-decision/assets'];
  execFileSync('git', ['diff', '--quiet', 'HEAD', '--', ...paths], {
    cwd: workspaceRoot,
    stdio: 'ignore'
  });
  const untracked = execFileSync('git', ['ls-files', '--others', '--exclude-standard', '--', ...paths], {
    cwd: workspaceRoot,
    encoding: 'utf8'
  });
  assert.equal(untracked, '', 'Model catalog and asset inputs must be committed.');
}

async function readCommittedFixture(workspaceRoot: string, path: string, digest: string): Promise<AnyRecord> {
  const fixturePath = resolve(workspaceRoot, path);
  const info = await lstat(fixturePath);
  fail(info.isFile() && !info.isSymbolicLink(), `Reference fixture must be a regular file: ${path}`);
  const bytes = await readFile(fixturePath);
  assert.equal(sha256(bytes), digest, `Reference fixture digest mismatch: ${path}`);
  execFileSync('git', ['ls-files', '--error-unmatch', path], {cwd: workspaceRoot, stdio: 'ignore'});
  const committed = execFileSync('git', ['show', `HEAD:${path}`], {cwd: workspaceRoot});
  assert.deepEqual(bytes, committed, `Reference fixture differs from its committed version: ${path}`);
  return JSON.parse(bytes.toString('utf8')) as AnyRecord;
}

async function assertCommittedReferenceFixtures(workspaceRoot: string): Promise<void> {
  const baseline = await readCommittedFixture(
    workspaceRoot,
    'tests/fixtures/kev06-main-baseline.json',
    BASELINE_REFERENCE.fixtureSha256
  );
  assert.equal(baseline.scope, BASELINE_REFERENCE.scope, 'Baseline fixture scope mismatch.');
  assert.equal(baseline.packageSha256, BASELINE_REFERENCE.packageSha256, 'Baseline package fixture pin mismatch.');
  assert.equal(baseline.selected, BASELINE_EXPECTED_OUTPUT.selected, 'Baseline selected option mismatch.');
  assert.deepEqual(
    baseline.request?.questions?.[0]?.options,
    BASELINE_EXPECTED_OUTPUT.optionOrder,
    'Baseline fixture option order mismatch.'
  );
  fail(Array.isArray(baseline.probabilities), 'Baseline fixture has no probability outputs.');
  const baselineProbabilities = baseline.probabilities as Array<{logit: number; probability: number}>;
  assert.deepEqual(
    baselineProbabilities.map(({logit}) => logit),
    BASELINE_EXPECTED_OUTPUT.logits,
    'Baseline fixture logits mismatch.'
  );
  assert.deepEqual(
    baselineProbabilities.map(({probability}) => probability),
    BASELINE_EXPECTED_OUTPUT.probabilities,
    'Baseline fixture probabilities mismatch.'
  );

  const upstream = await readCommittedFixture(
    workspaceRoot,
    'tests/fixtures/kev06-upstream-onnx-reference.json',
    UPSTREAM_REFERENCE.fixtureSha256
  );
  assert.equal(upstream.scope, UPSTREAM_REFERENCE.scope, 'Upstream fixture scope mismatch.');
  assert.equal(upstream.modelId, UPSTREAM_REFERENCE.modelId, 'Upstream fixture model ID mismatch.');
  assert.equal(upstream.graphSha256, UPSTREAM_REFERENCE.graphSha256, 'Upstream graph fixture pin mismatch.');
  assert.equal(upstream.weightsSha256, UPSTREAM_REFERENCE.weightsSha256, 'Upstream weights fixture pin mismatch.');
  assert.deepEqual(upstream.logits, UPSTREAM_EXPECTED_OUTPUT.logits, 'Upstream fixture logits mismatch.');
  assert.deepEqual(
    upstream.probabilities,
    UPSTREAM_EXPECTED_OUTPUT.probabilities,
    'Upstream fixture probabilities mismatch.'
  );
  assert.equal(
    upstream.maxAbsoluteLogitDifference,
    UPSTREAM_EXPECTED_OUTPUT.maxAbsoluteLogitDifference,
    'Upstream logit tolerance mismatch.'
  );
  assert.equal(
    upstream.maxAbsoluteProbabilityDifference,
    UPSTREAM_EXPECTED_OUTPUT.maxAbsoluteProbabilityDifference,
    'Upstream probability tolerance mismatch.'
  );

  const expanded = await readCommittedFixture(
    workspaceRoot,
    EXPANDED_INFERENCE_FIXTURE_PATH,
    EXPANDED_INFERENCE_FIXTURE_SHA256
  );
  validateExpandedInferenceFixture(expanded as unknown as ExpandedInferenceFixture);
  assert.equal(
    expanded.tokenizerSha256,
    upstream.tokenizerSha256,
    'Expanded tokenizer differs from the pinned upstream fixture.'
  );
  assert.equal(
    expanded.sourceGraphSha256,
    upstream.graphSha256,
    'Expanded ONNX graph differs from the pinned fixture.'
  );
  assert.equal(
    expanded.sourceWeightsSha256,
    upstream.weightsSha256,
    'Expanded ONNX weights differ from the pinned fixture.'
  );
  assert.deepEqual(
    expanded.decisionCases[0]?.tokenIds,
    upstream.tokenIds,
    'Expanded baseline token IDs differ from the existing one-input reference.'
  );
  assert.deepEqual(
    expanded.decisionCases[0]?.optionIndices,
    upstream.optionIndices,
    'Expanded baseline option indices differ from the existing one-input reference.'
  );
}

function assertRegularArchiveMembers(archivePath: string): void {
  const listing = execFileSync(TARBALL, ['-tvzf', archivePath], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
  const entries = listing.split(/\r?\n/u).filter(Boolean);
  fail(entries.length > 0, 'npm archive is empty.');
  for (const entry of entries) {
    fail(entry[0] === '-' || entry[0] === 'd', `npm archive contains a non-regular member: ${entry}`);
  }
}

function listArchiveFiles(archivePath: string): string[] {
  assertRegularArchiveMembers(archivePath);
  const output = execFileSync(TARBALL, ['-tzf', archivePath], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
  const files = [];
  const seen = new Set();

  for (const rawEntry of output.split(/\r?\n/u).filter(Boolean)) {
    const isDirectory = rawEntry.endsWith('/');
    const entry = isDirectory ? rawEntry.slice(0, -1) : rawEntry;
    if (entry === 'package') continue;
    fail(entry.startsWith('package/'), `Unexpected npm archive path: ${rawEntry}`);
    const name = normalizedRelativePath(entry.slice('package/'.length), 'Archive member');
    if (isDirectory) continue;
    fail(!seen.has(name), `Duplicate npm archive member: ${name}`);
    seen.add(name);
    files.push(name);
  }

  fail(files.includes('package.json'), 'npm archive is missing package/package.json.');
  return files.sort();
}

function readArchiveMember(archivePath: string, name: string): Buffer {
  const bytes = execFileSync(TARBALL, ['-xOzf', archivePath, `package/${name}`], {
    maxBuffer: MAX_ARCHIVE_MEMBER_BYTES
  });
  fail(bytes.length <= MAX_ARCHIVE_MEMBER_BYTES, `Archive member exceeds the verifier limit: ${name}`);
  return bytes;
}

async function listInstalledFiles(directory: string, prefix = ''): Promise<string[]> {
  const files = [];
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const child = join(directory, entry.name);
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...(await listInstalledFiles(child, name)));
    else {
      const info = await lstat(child);
      fail(info.isFile() && !info.isSymbolicLink(), `Installed package contains a non-regular file: ${name}`);
      files.push(normalizedRelativePath(name, 'Installed package member'));
    }
  }
  return files.sort();
}

export async function verifyArchiveInstall(archivePath: string, installedPackagePath: string): Promise<ArchiveInfo> {
  const archive = resolve(archivePath);
  fail(archive.toLowerCase().endsWith('.tgz'), 'The candidate npm archive must use the .tgz extension.');
  const archiveInfo = await lstat(archive);
  fail(archiveInfo.isFile() && !archiveInfo.isSymbolicLink(), 'The release archive must be a regular file.');
  const installedRoot = await assertCanonicalDirectory(installedPackagePath, 'Installed package');
  const archiveFiles = listArchiveFiles(archive);
  const installedFiles = await listInstalledFiles(installedRoot);
  assert.deepEqual(installedFiles, archiveFiles, 'Installed package file list differs from the release archive.');

  const members = [];
  for (const name of archiveFiles) {
    const archiveBytes = readArchiveMember(archive, name);
    const installedBytes = await readFile(join(installedRoot, ...name.split('/')));
    assert.deepEqual(installedBytes, archiveBytes, `Installed package file differs from the release archive: ${name}`);
    members.push({path: name, bytes: archiveBytes.length, sha256: sha256(archiveBytes)});
  }

  const packageJson = JSON.parse(readArchiveMember(archive, 'package.json').toString('utf8'));
  fail(packageJson.name === 'web-decision', 'Archive is not the web-decision package.');
  fail(typeof packageJson.version === 'string' && packageJson.version.length > 0, 'Package version is missing.');
  const bundlePath = resolve(installedRoot, packageEntryPath(packageJson));
  fail(isPathWithin(installedRoot, bundlePath), 'Package entrypoint escapes the installed archive.');
  const bundleBytes = await readFile(bundlePath);
  const inventorySha256 = canonicalSha256(members);

  return {
    archivePath: archive,
    archiveSha256: await hashFile(archive),
    installedRoot,
    packageName: packageJson.name,
    packageVersion: packageJson.version,
    packageTreeSha256: inventorySha256,
    bundlePath,
    bundleSha256: sha256(bundleBytes),
    installedPackagePathSha256: canonicalPathSha256(installedRoot),
    entryPathSha256: canonicalPathSha256(bundlePath),
    archiveFiles
  };
}

function normalizeRequiredFiles(value: unknown, modelId: string): RequiredFiles {
  fail(Array.isArray(value) && value.length > 0, `Model ${modelId} has no required asset files.`);
  const paths = (value as unknown[]).map((path) => normalizedRelativePath(path, `Required asset for ${modelId}`));
  fail(new Set(paths).size === paths.length, `Model ${modelId} has duplicate required asset files.`);
  return paths.sort();
}

function normalizeCatalogDescriptor(modelId: string, input: unknown): PublicCatalogDescriptor {
  const descriptor = input as AnyRecord;
  fail(descriptor && typeof descriptor === 'object', `Model ${modelId} has no descriptor.`);
  assert.equal(descriptor.modelId, modelId, `Catalog descriptor ID mismatch for ${modelId}.`);
  const backend = descriptor.backend as AnyRecord;
  fail(backend && backend.name === 'tvm-webgpu', `Catalog model ${modelId} is not a TVM WebGPU route.`);
  const assets = descriptor.assets as AnyRecord;
  fail(assets && SHA256_PATTERN.test(assets.manifestSha256), `Model ${modelId} has no pinned manifest SHA-256.`);
  fail(Array.isArray(backend.requiredFeatures), `Model ${modelId} must declare required GPU features.`);
  const requiredFeatures = [...new Set(backend.requiredFeatures as string[])].sort();
  fail(
    requiredFeatures.every((feature) => typeof feature === 'string' && feature.length > 0),
    `Model ${modelId} has an invalid required GPU feature.`
  );
  const requiredLimits = backend.requiredLimits ?? {};
  fail(
    requiredLimits && typeof requiredLimits === 'object' && !Array.isArray(requiredLimits),
    `Model ${modelId} has invalid required GPU limits.`
  );
  for (const [name, minimum] of Object.entries(requiredLimits)) {
    fail(
      name.length > 0 && typeof minimum === 'number' && Number.isSafeInteger(minimum) && minimum > 0,
      `Model ${modelId} has an invalid required GPU limit: ${name}.`
    );
  }
  return {
    modelId,
    backend: {
      name: 'tvm-webgpu',
      requiredFeatures,
      requiredLimits: requiredLimits as Record<string, number>
    },
    manifestSha256: assets.manifestSha256 as string,
    requiredFiles: normalizeRequiredFiles(assets.requiredFiles, modelId)
  };
}

export async function loadPublicCatalog(packageRoot: string, entryPath: string): Promise<PublicCatalogDescriptor[]> {
  const absoluteRoot = await assertCanonicalDirectory(packageRoot, 'Model catalog package');
  const entry = resolve(absoluteRoot, entryPath);
  fail(isPathWithin(absoluteRoot, entry), 'Catalog entrypoint escapes its package root.');
  const entryInfo = await lstat(entry);
  fail(entryInfo.isFile() && !entryInfo.isSymbolicLink(), 'Catalog entrypoint must be a regular file.');
  fail((await realpath(entry)) === entry, 'Catalog entrypoint must use its canonical path.');
  const moduleUrl = pathToFileURL(entry);
  moduleUrl.searchParams.set('release-catalog-check', `${Date.now()}-${Math.random()}`);
  const api = (await import(moduleUrl.href)) as {
    listModels: () => unknown[];
    describeModel: (modelId: string) => unknown;
  };
  fail(
    typeof api.listModels === 'function' && typeof api.describeModel === 'function',
    'Release verification requires the public listModels()/describeModel(id) API.'
  );
  const listed = api.listModels();
  fail(Array.isArray(listed) && listed.length > 0, 'The public model catalog must not be empty.');
  const ids = listed.map((item: unknown) => (item as AnyRecord | null)?.modelId);
  fail(
    ids.every((id) => typeof id === 'string' && id.length > 0),
    'Every public catalog item needs a modelId.'
  );
  fail(new Set(ids).size === ids.length, 'The public model catalog contains duplicate IDs.');

  const descriptors: PublicCatalogDescriptor[] = [];
  for (const id of ids as string[]) {
    const listedItem = listed.find((item: unknown) => (item as AnyRecord | null)?.modelId === id);
    const describedItem = api.describeModel(id);
    assert.deepEqual(
      canonical(listedItem),
      canonical(describedItem),
      `listModels()/describeModel() disagree for ${id}.`
    );
    descriptors.push(normalizeCatalogDescriptor(id, describedItem));
  }
  return descriptors.sort((left, right) => left.modelId.localeCompare(right.modelId));
}

export async function loadInstalledCatalog(packageRoot: string): Promise<PublicCatalogDescriptor[]> {
  const metadata = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8'));
  return loadPublicCatalog(packageRoot, packageEntryPath(metadata));
}

async function loadCommittedCatalog(packageRoot: string): Promise<ModelRequirement[]> {
  const catalogRoot = join(packageRoot, 'src/catalog');
  const modelSpec = JSON.parse(await readFile(join(catalogRoot, 'model-spec.json'), 'utf8')) as ModelSpecFile;
  const artifactLock = JSON.parse(await readFile(join(catalogRoot, 'artifacts.lock.json'), 'utf8')) as ArtifactLockFile;
  fail(modelSpec.schemaVersion === 1 && artifactLock.schemaVersion === 1, 'Unsupported generated catalog schema.');
  fail(Array.isArray(modelSpec.models) && modelSpec.models.length > 0, 'Committed model spec must not be empty.');
  fail(artifactLock.byModelId && typeof artifactLock.byModelId === 'object', 'Committed artifact lock is missing.');

  const ids = modelSpec.models.map(({modelId}) => modelId);
  fail(
    ids.every((modelId) => typeof modelId === 'string' && modelId.length > 0),
    'Every model spec item needs an ID.'
  );
  fail(new Set(ids).size === ids.length, 'Committed model spec contains duplicate IDs.');
  assert.deepEqual(
    Object.keys(artifactLock.byModelId).sort(),
    [...ids].sort(),
    'Artifact lock IDs must exactly match the committed model spec.'
  );

  const models = modelSpec.models.map(({modelId, description}) => {
    const artifacts = artifactLock.byModelId[modelId];
    fail(artifacts !== undefined, `Artifact lock is missing model ${modelId}.`);
    const assets = description.assets as AnyRecord;
    const backend = description.backend as AnyRecord;
    fail(assets && backend, `Committed model ${modelId} is missing assets or backend metadata.`);
    const manifest = artifacts.manifest;
    const manifestSha256 = manifest.canonicalSha256;
    fail(SHA256_PATTERN.test(manifestSha256), `Artifact lock has no canonical manifest digest for ${modelId}.`);
    assert.equal(manifest.sourcePath, normalizedRelativePath(manifest.sourcePath, `Manifest path for ${modelId}`));
    assert.equal(
      assets.manifestSha256 ?? manifestSha256,
      manifestSha256,
      `Catalog manifest digest differs from the artifact lock for ${modelId}.`
    );
    fail(
      !Object.hasOwn(assets, 'requiredFiles'),
      `Model ${modelId} must derive requiredFiles from its pinned manifest, not maintain a separate source list.`
    );
    const manifestFiles = normalizeManifestFiles({files: manifest.files}, modelId);
    const requiredFiles = normalizeRequiredFiles(['manifest.json', ...Object.keys(manifestFiles)], modelId);
    for (const path of requiredFiles) {
      if (path !== 'manifest.json')
        fail(Object.hasOwn(manifestFiles, path), `Required file ${path} is absent from the manifest for ${modelId}.`);
    }
    const bundledRuntimeFiles = artifacts.bundledRuntimeFiles;
    fail(
      Array.isArray(bundledRuntimeFiles) && bundledRuntimeFiles.length > 0,
      `Artifact lock has no bundled runtime files for ${modelId}.`
    );
    const bundledNames = new Set<string>();
    for (const file of bundledRuntimeFiles) {
      const name = normalizedRelativePath(file.manifestName, `Bundled file name for ${modelId}`);
      normalizedRelativePath(file.sourcePath, `Bundled source path for ${modelId}`);
      fail(Object.hasOwn(manifestFiles, name), `Bundled file ${name} is absent from the manifest for ${modelId}.`);
      assert.deepEqual(
        file,
        {manifestName: name, sourcePath: file.sourcePath, ...manifestFiles[name]},
        `Bundled file lock differs from the manifest for ${modelId}/${name}.`
      );
      fail(!bundledNames.has(name), `Duplicate bundled file for ${modelId}/${name}.`);
      bundledNames.add(name);
    }
    const publicModel = normalizeCatalogDescriptor(modelId, {
      modelId,
      ...description,
      assets: {...assets, manifestSha256, requiredFiles}
    });
    return {
      ...publicModel,
      manifestPath: manifest.sourcePath,
      manifestFiles,
      bundledRuntimeFiles
    };
  });
  return models.sort((left, right) => left.modelId.localeCompare(right.modelId));
}

function manifestModelId(manifest: AssetManifest): string | undefined {
  return manifest?.modelId ?? manifest?.model;
}

function normalizeManifestFiles(manifest: AssetManifest, modelId: string): Record<string, AssetMetadata> {
  fail(manifest && manifest.files && typeof manifest.files === 'object', `Manifest for ${modelId} has no file map.`);
  const entries = Object.entries(manifest.files as Record<string, AssetMetadata>).sort(([left], [right]) =>
    left.localeCompare(right)
  );
  fail(entries.length > 0, `Manifest for ${modelId} has no model artifacts.`);
  for (const [path, metadata] of entries) {
    normalizedRelativePath(path, `Manifest file for ${modelId}`);
    fail(
      metadata && Number.isSafeInteger(metadata.bytes) && metadata.bytes >= 0,
      `Invalid byte length for ${modelId}/${path}.`
    );
    fail(
      typeof metadata.sha256 === 'string' && SHA256_PATTERN.test(metadata.sha256),
      `Invalid file SHA-256 for ${modelId}/${path}.`
    );
  }
  return Object.fromEntries(entries);
}

function validateRequiredFiles(model: ModelRequirement, files: Record<string, AssetMetadata>): void {
  for (const path of model.requiredFiles) {
    if (path !== 'manifest.json')
      fail(Object.hasOwn(files, path), `Pinned manifest for ${model.modelId} omits required file ${path}.`);
  }
}

export function validateArtifactManifest(
  model: ModelRequirement,
  manifest: AssetManifest
): {
  manifestSha256: string;
  filesSha256: string;
  files: Record<string, AssetMetadata>;
} {
  assert.equal(manifestModelId(manifest), model.modelId, `Artifact manifest ID mismatch for ${model.modelId}.`);
  const digest = canonicalSha256(manifest);
  assert.equal(
    digest,
    model.manifestSha256,
    `Artifact manifest SHA differs from the catalog pin for ${model.modelId}.`
  );
  const files = normalizeManifestFiles(manifest, model.modelId);
  assert.deepEqual(
    canonical(files),
    canonical(model.manifestFiles),
    `Manifest file map differs from the generated lock for ${model.modelId}.`
  );
  validateRequiredFiles(model, files);
  return {manifestSha256: digest, filesSha256: canonicalSha256(files), files};
}

export async function readArchiveModelManifests(
  archiveInfo: ArchiveInfo,
  models: ModelRequirement[]
): Promise<Map<string, ArchiveManifestEvidence>> {
  const expected = new Map(models.map((model) => [model.modelId, model]));
  const candidates = new Map<string, Array<{path: string; manifest: AssetManifest}>>();
  for (const path of archiveInfo.archiveFiles.filter((name) => basename(name) === 'manifest.json')) {
    const bytes = readArchiveMember(archiveInfo.archivePath, path);
    let manifest: AssetManifest;
    try {
      manifest = JSON.parse(bytes.toString('utf8'));
    } catch (error) {
      throw new Error(`Invalid static manifest in npm archive: ${path}`, {cause: error});
    }
    const modelId = manifestModelId(manifest);
    if (!modelId) continue;
    if (manifest.backend === 'tvm-webgpu')
      fail(expected.has(modelId), `Archive contains an uncurated TVM manifest: ${modelId}`);
    if (!expected.has(modelId)) continue;
    const matches = candidates.get(modelId) ?? [];
    matches.push({path, manifest});
    candidates.set(modelId, matches);
  }

  const result = new Map<string, ArchiveManifestEvidence>();
  for (const model of models) {
    const matches = candidates.get(model.modelId) ?? [];
    const wanted = normalizedRelativePath(model.manifestPath, `Manifest path for ${model.modelId}`);
    fail(
      archiveInfo.archiveFiles.includes(wanted),
      `Archive is missing the pinned manifest path for ${model.modelId}.`
    );
    assert.equal(
      matches.filter((item) => item.path === wanted).length,
      1,
      `Archive is missing the pinned manifest path for ${model.modelId}.`
    );
    assert.equal(matches.length, 1, `Archive must contain exactly one static manifest for ${model.modelId}.`);
    const item = matches[0];
    fail(item !== undefined, `Archive is missing the static manifest for ${model.modelId}.`);
    const validated = validateArtifactManifest(model, item.manifest);
    result.set(model.modelId, {...item, ...validated});
  }
  return result;
}

export async function collectReleaseEvidenceInputs({
  archivePath,
  installedPackagePath,
  sourcePackageRoot = join(WORKSPACE_ROOT, 'packages/web-decision')
}: {
  archivePath: string;
  installedPackagePath: string;
  sourcePackageRoot?: string;
}): Promise<{
  archive: ArchiveInfo;
  models: ModelRequirement[];
  catalogSha256: string;
  manifests: Map<string, ArchiveManifestEvidence>;
}> {
  const archive = await verifyArchiveInstall(archivePath, installedPackagePath);
  const models = await loadCommittedCatalog(sourcePackageRoot);
  const expectedPublicCatalog = models.map(({modelId, backend, manifestSha256, requiredFiles}) => ({
    modelId,
    backend,
    manifestSha256,
    requiredFiles
  }));
  const sourceCatalog = await loadPublicCatalog(sourcePackageRoot, 'dist/index.js');
  const installedCatalog = await loadInstalledCatalog(archive.installedRoot);
  assert.deepEqual(
    sourceCatalog,
    expectedPublicCatalog,
    'Built public catalog differs from the committed model/artifact inputs.'
  );
  assert.deepEqual(
    installedCatalog,
    expectedPublicCatalog,
    'Installed public catalog differs from the committed model/artifact inputs.'
  );
  const manifests = await readArchiveModelManifests(archive, models);
  return {archive, models, catalogSha256: canonicalSha256(models), manifests};
}

function collectPlaywrightTests(
  suites: PlaywrightSuite[] | undefined,
  result: PlaywrightTestItem[] = []
): PlaywrightTestItem[] {
  for (const suite of suites ?? []) {
    for (const spec of suite.specs ?? []) {
      for (const test of spec.tests ?? []) {
        const item: PlaywrightTestItem = {test};
        const file = spec.file ?? suite.file;
        if (file !== undefined) item.file = file;
        if (spec.title !== undefined) item.title = spec.title;
        result.push(item);
      }
    }
    collectPlaywrightTests(suite.suites, result);
  }
  return result;
}

function playwrightStdout(result: PlaywrightTestResult): string {
  return (result.stdout ?? []).map((item) => (typeof item === 'string' ? item : (item?.text ?? ''))).join('');
}

function recordFromTest(test: PlaywrightTest, evidencePrefix: string): AnyRecord {
  const results = test.results ?? [];
  assert.equal(results.length, 1, 'Each required GPU test must have exactly one execution and no retry.');
  assert.equal(test.expectedStatus, 'passed', 'Every release GPU test must expect a pass.');
  assert.equal(test.status, 'expected', 'Every release GPU test must pass without skip or retry.');
  const result = results[0];
  fail(result !== undefined, 'Release GPU test has no execution result.');
  assert.equal(result.status, 'passed', 'Release GPU test did not pass.');
  assert.deepEqual(result.errors ?? [], [], 'Release GPU test contains errors.');
  const lines = playwrightStdout(result)
    .split(/\r?\n/u)
    .filter((line) => line.startsWith(evidencePrefix));
  assert.equal(lines.length, 1, 'Each passing release GPU test must emit exactly one structured evidence record.');
  const line = lines[0];
  fail(line !== undefined, 'Release GPU test has no structured evidence record.');
  return JSON.parse(line.slice(evidencePrefix.length));
}

function requireText(value: unknown, description: string): void {
  fail(typeof value === 'string' && value.trim().length > 0, `${description} is required.`);
}

function validateObservedFiles(value: unknown, model: ModelRequirement): void {
  fail(Array.isArray(value) && value.length > 0, `Browser evidence has no verified files for ${model.modelId}.`);
  const observed: Record<string, AssetMetadata> = {};
  for (const item of value as unknown[]) {
    const file = item as AnyRecord;
    const path = normalizedRelativePath(file?.path, `Verified file for ${model.modelId}`);
    fail(!Object.hasOwn(observed, path), `Browser evidence repeats ${model.modelId}/${path}.`);
    fail(
      Number.isSafeInteger(file.bytes) && file.bytes >= 0,
      `Invalid verified byte length for ${model.modelId}/${path}.`
    );
    fail(
      typeof file.sha256 === 'string' && SHA256_PATTERN.test(file.sha256),
      `Invalid verified SHA-256 for ${model.modelId}/${path}.`
    );
    observed[path] = {bytes: file.bytes, sha256: file.sha256};
  }
  assert.deepEqual(
    canonical(observed),
    canonical(model.manifestFiles),
    `Browser cache verification differs from every pinned manifest file for ${model.modelId}.`
  );
}

function validateHardware(
  record: AnyRecord,
  model: ModelRequirement,
  browser: BrowserRequirement,
  policy: EvidencePolicy
): void {
  assert.equal(record.browser?.name, browser.name, `Browser evidence mismatch for ${model.modelId}.`);
  requireText(record.browser?.version, `Browser version for ${model.modelId}`);
  requireText(record.browser?.userAgent, `Browser user agent for ${model.modelId}`);
  const device = record.device;
  validateDeviceEvidence(device, model.modelId);
  if (policy.requirePhysicalDevice)
    assert.equal(device.isVirtual, false, `GPU test for ${model.modelId} must use physical hardware.`);
  const gpu = record.gpu;
  assert.equal(gpu?.api, policy.requiredApi, `GPU API mismatch for ${model.modelId}.`);
  assert.equal(gpu.isFallbackAdapter, false, `GPU test for ${model.modelId} used a fallback adapter.`);
  validateAdapterInfo(gpu.adapterInfo, model.modelId);
  validateGpuWmiEvidence(gpu, model.modelId);
  fail(Array.isArray(gpu.features), `GPU features are missing for ${model.modelId}.`);
  const requiredFeatures =
    model.modelId === UPSTREAM_REFERENCE.modelId
      ? [...new Set([...model.backend.requiredFeatures, 'shader-f16'])]
      : model.backend.requiredFeatures;
  for (const feature of requiredFeatures) {
    fail(gpu.features.includes(feature), `GPU for ${model.modelId} does not report required feature ${feature}.`);
  }
  validateLimits(gpu.limits, model.backend.requiredLimits, model.modelId);
}

function validateDeviceEvidence(device: AnyRecord, modelId: string): void {
  const host = device?.evidence?.host;
  const os = device?.evidence?.os;
  for (const field of ['manufacturer', 'model'])
    requireText(host?.raw?.[field === 'manufacturer' ? 'Manufacturer' : 'Model'], `WMI host ${field} for ${modelId}`);
  assert.equal(host?.source, 'windows-wmi/Win32_ComputerSystem', `Host evidence source is required for ${modelId}.`);
  assert.equal(
    device.manufacturer,
    host.raw.Manufacturer,
    `Host manufacturer differs from WMI evidence for ${modelId}.`
  );
  assert.equal(device.model, host.raw.Model, `Host model differs from WMI evidence for ${modelId}.`);
  fail(
    !/(?:virtual machine|vmware|virtualbox|qemu|\bkvm\b|\bxen\b|hyper-v|parallels)/iu.test(
      `${device.manufacturer} ${device.model}`
    ),
    `GPU test for ${modelId} appears to run in a virtual machine.`
  );
  assert.equal(os?.source, 'node:os', `OS evidence source is required for ${modelId}.`);
  assert.equal(device.os, os?.raw?.version, `OS name differs from runtime evidence for ${modelId}.`);
  assert.equal(device.osVersion, os?.raw?.release, `OS version differs from runtime evidence for ${modelId}.`);
  assert.equal(
    device.architecture,
    os?.raw?.architecture,
    `Architecture differs from runtime evidence for ${modelId}.`
  );
  requireText(device.os, `Device OS for ${modelId}`);
  requireText(device.osVersion, `Device OS version for ${modelId}`);
  requireText(device.architecture, `Device architecture for ${modelId}`);
}

function validateAdapterInfo(adapterInfo: AnyRecord, modelId: string): void {
  assert.equal(
    adapterInfo?.source,
    'navigator.gpu.requestAdapter().info',
    `Adapter info source is required for ${modelId}.`
  );
  fail(
    adapterInfo.value && typeof adapterInfo.value === 'object' && !Array.isArray(adapterInfo.value),
    `Adapter info must be recorded as an object for ${modelId}.`
  );
}

function validateGpuWmiEvidence(gpu: AnyRecord, modelId: string): void {
  const evidence = gpu.hardwareEvidence;
  assert.equal(
    evidence?.source,
    'windows-wmi/Win32_VideoController',
    `GPU hardware source is required for ${modelId}.`
  );
  const raw = evidence.raw;
  const reported = gpu.hardware;
  const fields: Array<[string, string]> = [
    ['manufacturer', 'AdapterCompatibility'],
    ['model', 'Name'],
    ['driverVersion', 'DriverVersion'],
    ['pnpDeviceId', 'PNPDeviceID']
  ];
  for (const [reportedField, rawField] of fields) {
    requireText(raw?.[rawField], `WMI GPU ${rawField} for ${modelId}`);
    assert.equal(
      reported?.[reportedField],
      raw[rawField],
      `GPU ${reportedField} differs from WMI evidence for ${modelId}.`
    );
  }
}

function validateLimits(actual: AnyRecord, required: Record<string, number>, modelId: string): void {
  fail(actual && typeof actual === 'object', `GPU limits are missing for ${modelId}.`);
  for (const [name, minimum] of Object.entries(required)) {
    fail(Number.isFinite(actual[name]) && actual[name] >= minimum, `GPU does not meet ${name} for ${modelId}.`);
  }
}

function validateTestEvidence(
  record: AnyRecord,
  model: ModelRequirement,
  browser: BrowserRequirement,
  context: VerificationContext
): void {
  assert.equal(record.schemaVersion, 2, 'Unsupported browser evidence schema.');
  assert.equal(record.testKind, 'real-model-webgpu', 'Evidence must come from a real-model WebGPU test.');
  for (const field of ['requestedId', 'loadedId', 'describedId']) {
    assert.equal(record.model?.[field], model.modelId, `Test model ${field} mismatch for ${model.modelId}.`);
  }
  assert.equal(
    record.model?.assetManifestSha256,
    model.manifestSha256,
    `Runtime model descriptor manifest pin mismatch for ${model.modelId}.`
  );
  const artifact = record.artifact;
  assert.equal(artifact?.modelId, model.modelId, `Artifact identity mismatch for ${model.modelId}.`);
  const archiveManifest = context.manifests.get(model.modelId);
  assert.ok(archiveManifest, `Static archive manifest is missing for ${model.modelId}.`);
  assert.equal(
    artifact?.manifestPath,
    archiveManifest.path,
    `Browser evidence refers to a different static manifest path for ${model.modelId}.`
  );
  const actualArtifact = validateArtifactManifest(model, artifact.manifest);
  assert.equal(
    artifact.manifestSha256,
    actualArtifact.manifestSha256,
    `Reported manifest SHA mismatch for ${model.modelId}.`
  );
  assert.equal(
    artifact.filesSha256,
    actualArtifact.filesSha256,
    `Reported artifact file digest mismatch for ${model.modelId}.`
  );
  validateObservedFiles(artifact.verifiedFiles, model);
  assert.deepEqual(
    canonical(artifact.manifest),
    canonical(archiveManifest.manifest),
    `Browser artifact manifest does not match the archive-pinned manifest for ${model.modelId}.`
  );
  assert.equal(record.catalogSha256, context.catalogSha256, 'Browser evidence refers to a different model catalog.');
  assert.equal(record.archive?.sha256, context.archive.archiveSha256, 'Browser test used a different package archive.');
  assert.equal(
    record.archive?.packageTreeSha256,
    context.archive.packageTreeSha256,
    'Browser test used a different installed package tree.'
  );
  assert.equal(
    record.archive?.bundleSha256,
    context.archive.bundleSha256,
    'Browser test used a different package bundle.'
  );
  assert.equal(
    record.archive?.installedPackagePathSha256,
    context.archive.installedPackagePathSha256,
    'Browser evidence points to a different installed package directory.'
  );
  assert.equal(
    record.archive?.entryPathSha256,
    context.archive.entryPathSha256,
    'Browser evidence points to a different installed package entrypoint.'
  );
  assert.equal(
    record.archive?.packageName,
    context.archive.packageName,
    'Browser package name does not match the archive.'
  );
  assert.equal(
    record.archive?.packageVersion,
    context.archive.packageVersion,
    'Browser package version does not match the archive.'
  );
  validateInferenceEvidence(record.inference, model);
  const expandedFixture = loadExpandedInferenceFixture(context.workspaceRoot, context.policy.expandedInference);
  validateExpandedInferenceEvidence(
    record.expandedInference,
    expandedFixture.fixture,
    expandedFixture.fixtureSha256,
    model.manifestSha256
  );
  validateHardware(record, model, browser, context.policy);
}

function compareValuesWithinTolerance(
  actual: unknown,
  expected: readonly number[],
  tolerance: number,
  description: string
): void {
  fail(Array.isArray(actual) && actual.length === expected.length, `${description} has the wrong number of values.`);
  for (const [index, expectedValue] of expected.entries()) {
    const actualValue = actual[index];
    fail(Number.isFinite(actualValue), `${description} value ${index} is not finite.`);
    fail(
      Math.abs(actualValue - expectedValue) <= tolerance,
      `${description} value ${index} differs from the pinned reference by more than ${tolerance}.`
    );
  }
}

function validateObservedInferenceOutput(output: AnyRecord, modelId: string, mode: string): void {
  fail(
    !Object.hasOwn(output, 'optionIndices'),
    `${mode} evidence must not report option indices that the browser API does not expose for ${modelId}.`
  );
  assert.deepEqual(
    output.optionOrder,
    BASELINE_EXPECTED_OUTPUT.optionOrder,
    `${mode} output option order differs from the reference input for ${modelId}.`
  );
  assert.equal(
    output.selected,
    BASELINE_EXPECTED_OUTPUT.selected,
    `${mode} output selection differs from the baseline fixture for ${modelId}.`
  );
  compareValuesWithinTolerance(
    output.logits,
    BASELINE_EXPECTED_OUTPUT.logits,
    BASELINE_EXPECTED_OUTPUT.maxAbsoluteLogitDifference,
    `${mode} baseline logits for ${modelId}`
  );
  compareValuesWithinTolerance(
    output.probabilities,
    BASELINE_EXPECTED_OUTPUT.probabilities,
    BASELINE_EXPECTED_OUTPUT.maxAbsoluteProbabilityDifference,
    `${mode} baseline probabilities for ${modelId}`
  );
  compareValuesWithinTolerance(
    output.logits,
    UPSTREAM_EXPECTED_OUTPUT.logits,
    UPSTREAM_EXPECTED_OUTPUT.maxAbsoluteLogitDifference,
    `${mode} upstream logits for ${modelId}`
  );
  compareValuesWithinTolerance(
    output.probabilities,
    UPSTREAM_EXPECTED_OUTPUT.probabilities,
    UPSTREAM_EXPECTED_OUTPUT.maxAbsoluteProbabilityDifference,
    `${mode} upstream probabilities for ${modelId}`
  );
}

function validateInferenceEvidence(inference: AnyRecord, model: ModelRequirement): void {
  for (const mode of ['online', 'offline']) {
    assert.equal(inference?.[mode]?.completed, true, `${mode} inference did not complete for ${model.modelId}.`);
    assert.equal(
      inference[mode].modelId,
      model.modelId,
      `${mode} inference used a different model for ${model.modelId}.`
    );
    assert.equal(
      inference[mode].backend,
      'tvm-webgpu',
      `${mode} inference did not use TVM WebGPU for ${model.modelId}.`
    );
    assert.equal(
      inference[mode].assetManifestSha256,
      model.manifestSha256,
      `${mode} inference used a different artifact manifest for ${model.modelId}.`
    );
  }
  assert.equal(
    model.modelId,
    UPSTREAM_REFERENCE.modelId,
    'No upstream reference fixture is pinned for this curated model.'
  );
  validateObservedInferenceOutput(inference.online, model.modelId, 'Online');
  validateObservedInferenceOutput(inference.offline, model.modelId, 'Offline');
  assert.equal(
    inference.online.referenceMatched,
    true,
    `Online output did not match the pinned baseline for ${model.modelId}.`
  );
  assert.equal(
    inference.offline.matchesOnline,
    true,
    `Offline output did not match online output for ${model.modelId}.`
  );
  for (const field of ['selected', 'optionOrder', 'logits', 'probabilities']) {
    assert.deepEqual(
      inference.offline[field],
      inference.online[field],
      `Offline ${field} differ from online output for ${model.modelId}.`
    );
  }
  assert.equal(
    inference.reference?.packageSha256,
    BASELINE_REFERENCE.packageSha256,
    'Baseline package archive pin mismatch.'
  );
  assert.equal(inference.reference?.fixtureSha256, BASELINE_REFERENCE.fixtureSha256, 'Baseline fixture pin mismatch.');
  assert.equal(
    inference.reference?.scope,
    BASELINE_REFERENCE.scope,
    'Reference evidence scope must remain baseline preservation only.'
  );
  assert.equal(
    inference.upstreamReference?.matched,
    true,
    `Online output did not match the independent upstream CPU reference for ${model.modelId}.`
  );
  assert.equal(
    inference.upstreamReference?.modelId,
    UPSTREAM_REFERENCE.modelId,
    'Upstream reference model ID mismatch.'
  );
  assert.equal(
    inference.upstreamReference?.fixtureSha256,
    UPSTREAM_REFERENCE.fixtureSha256,
    'Upstream reference fixture pin mismatch.'
  );
  assert.equal(
    inference.upstreamReference?.graphSha256,
    UPSTREAM_REFERENCE.graphSha256,
    'Upstream ONNX graph pin mismatch.'
  );
  assert.equal(
    inference.upstreamReference?.weightsSha256,
    UPSTREAM_REFERENCE.weightsSha256,
    'Upstream weights pin mismatch.'
  );
  assert.equal(
    inference.upstreamReference?.scope,
    UPSTREAM_REFERENCE.scope,
    'Upstream reference scope must remain limited to one input.'
  );
  assert.deepEqual(
    inference.offline.networkAttempts,
    [],
    `Offline inference attempted a network request for ${model.modelId}.`
  );
}

export function validatePlaywrightReport(
  report: PlaywrightReport,
  context: VerificationContext
): {modelCount: number; browserCount: number; caseCount: number} {
  const tests = collectPlaywrightTests(report.suites);
  const reportRoot = report.config?.rootDir;
  fail(typeof reportRoot === 'string' && isAbsolute(reportRoot), 'Playwright report has no absolute test root.');
  const models = context.models;
  const browsers = context.policy.browsers;
  const expectedCount = models.length * browsers.length;
  fail(report.stats && typeof report.stats === 'object', 'Playwright report has no summary statistics.');
  assert.equal(
    report.stats.expected,
    expectedCount,
    'Playwright pass count differs from the required model/browser matrix.'
  );
  assert.equal(report.stats.skipped, 0, 'Skipped tests block release evidence.');
  assert.equal(report.stats.unexpected, 0, 'Failed or unexpected tests block release evidence.');
  assert.equal(report.stats.flaky, 0, 'Flaky tests block release evidence.');
  assert.deepEqual(report.errors ?? [], [], 'Playwright report contains global errors.');
  assert.equal(tests.length, expectedCount, 'Playwright report must contain exactly one test per required case.');

  const modelById = new Map<string, ModelRequirement>(models.map((model) => [model.modelId, model]));
  const browserByProject = new Map<string, BrowserRequirement>(browsers.map((browser) => [browser.project, browser]));
  const cases = new Set();
  for (const item of tests) {
    const file = item.file;
    fail(typeof file === 'string' && file.length > 0, 'Playwright test file is missing.');
    assert.equal(
      resolve(reportRoot, file),
      resolve(context.workspaceRoot, context.policy.testFile),
      `Only the dedicated release GPU test is accepted: ${file}`
    );
    fail(typeof item.test.projectName === 'string', 'Playwright test is missing its browser project.');
    const browser = browserByProject.get(item.test.projectName);
    assert.ok(browser, `Unexpected Playwright project: ${item.test.projectName}`);
    const record = recordFromTest(item.test, context.policy.evidencePrefix);
    const model = modelById.get(record.model?.requestedId);
    assert.ok(model, `Evidence refers to an uncurated model: ${record.model?.requestedId}`);
    assert.ok(String(item.title).includes(model.modelId), `Test title does not identify model ${model.modelId}.`);
    const key = `${model.modelId}\0${browser.name}`;
    assert.ok(!cases.has(key), `Duplicate browser evidence for ${model.modelId}/${browser.name}.`);
    validateTestEvidence(record, model, browser, context);
    cases.add(key);
  }

  for (const model of models) {
    for (const browser of browsers) {
      assert.ok(
        cases.has(`${model.modelId}\0${browser.name}`),
        `Missing GPU evidence for ${model.modelId}/${browser.name}.`
      );
    }
  }
  return {modelCount: models.length, browserCount: browsers.length, caseCount: cases.size};
}

export async function verifyReleaseArtifacts({
  archivePath,
  installedPackagePath,
  playwrightReportPath,
  workspaceRoot = WORKSPACE_ROOT,
  sourcePackageRoot = join(workspaceRoot, 'packages/web-decision'),
  policy
}: ArtifactVerificationOptions): Promise<AnyRecord> {
  const checkedPolicy = validatePolicy(policy);
  const inputs = await collectReleaseEvidenceInputs({archivePath, installedPackagePath, sourcePackageRoot});
  const report = JSON.parse(await readFile(playwrightReportPath, 'utf8'));
  const browserResults = validatePlaywrightReport(report, {
    ...inputs,
    models: inputs.models,
    policy: checkedPolicy,
    workspaceRoot
  });
  const finalArchive = await verifyArchiveInstall(archivePath, installedPackagePath);
  assert.equal(
    finalArchive.archiveSha256,
    inputs.archive.archiveSha256,
    'Archive changed during evidence verification.'
  );
  assert.equal(
    finalArchive.packageTreeSha256,
    inputs.archive.packageTreeSha256,
    'Installed package changed during evidence verification.'
  );
  assert.equal(
    finalArchive.bundleSha256,
    inputs.archive.bundleSha256,
    'Installed bundle changed during evidence verification.'
  );
  return {
    archiveSha256: inputs.archive.archiveSha256,
    packageTreeSha256: inputs.archive.packageTreeSha256,
    bundleSha256: inputs.archive.bundleSha256,
    packageName: inputs.archive.packageName,
    packageVersion: inputs.archive.packageVersion,
    catalogSha256: inputs.catalogSha256,
    models: inputs.models.map((model) => model.modelId),
    browserResults
  };
}

export async function verifyReleaseEvidence(options: ReleaseEvidenceOptions): Promise<AnyRecord> {
  const root = resolve(options.workspaceRoot ?? WORKSPACE_ROOT);
  const policy = await loadTrustedPolicy(root);
  await assertCommittedReferenceFixtures(root);
  assertCommittedCatalogSources(root);
  fail(options.buildEvidencePath, 'TVM build evidence is required.');
  const sourceRoot = options.sourcePackageRoot ?? join(root, 'packages/web-decision');
  const models = await loadCommittedCatalog(sourceRoot);
  const buildModels = await Promise.all(
    models.map(async (model) => {
      const manifest: unknown = JSON.parse(await readFile(join(sourceRoot, model.manifestPath), 'utf8'));
      fail(
        typeof manifest === 'object' &&
          manifest !== null &&
          !Array.isArray(manifest) &&
          'tvmVersion' in manifest &&
          typeof manifest.tvmVersion === 'string',
        `No TVM version is recorded for ${model.modelId}.`
      );
      assert.equal(canonicalSha256(manifest), model.manifestSha256, `Source manifest changed for ${model.modelId}.`);
      const sourceArtifacts = Object.fromEntries(
        model.bundledRuntimeFiles.map(({manifestName, sourcePath}) => [manifestName, join(sourceRoot, sourcePath)])
      );
      return {...model, tvmVersion: manifest.tvmVersion, sourceArtifacts};
    })
  );
  const forbiddenRoots = [sourceRoot, options.installedPackagePath];
  const builds = await verifyTvmBuildEvidence(resolve(options.buildEvidencePath), buildModels, forbiddenRoots);
  const result = await verifyReleaseArtifacts({...options, workspaceRoot: root, policy});
  assert.deepEqual(
    await verifyTvmBuildEvidence(resolve(options.buildEvidencePath), buildModels, forbiddenRoots),
    builds,
    'TVM build evidence changed during verification.'
  );
  return {...result, builds};
}

function packageEntryPath(metadata: AnyRecord): string {
  const exported = metadata.exports?.['.'];
  const entry =
    typeof exported === 'string'
      ? exported
      : (exported?.import ?? exported?.default ?? metadata.module ?? 'dist/index.js');
  return normalizedRelativePath(entry.replace(/^\.\//u, ''), 'Package import entrypoint');
}

function parseArguments(
  args: string[]
): Pick<ReleaseEvidenceOptions, 'archivePath' | 'installedPackagePath' | 'playwrightReportPath' | 'buildEvidencePath'> {
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    fail(typeof key === 'string', 'Missing option name.');
    fail(
      ['--archive', '--installed-package', '--playwright-report', '--build-evidence'].includes(key),
      `Unknown option: ${key}`
    );
    fail(typeof value === 'string' && !value.startsWith('--'), `Missing value for ${key}.`);
    fail(!values.has(key), `Duplicate option: ${key}`);
    values.set(key, value);
  }
  for (const key of ['--archive', '--installed-package', '--playwright-report', '--build-evidence']) {
    fail(values.has(key), `Required option is missing: ${key}`);
  }
  const requiredValue = (key: string): string => {
    const value = values.get(key);
    fail(value, `Required option is missing: ${key}`);
    return value;
  };
  return {
    archivePath: requiredValue('--archive'),
    installedPackagePath: requiredValue('--installed-package'),
    playwrightReportPath: requiredValue('--playwright-report'),
    buildEvidencePath: requiredValue('--build-evidence')
  };
}

async function main() {
  const result = await verifyReleaseEvidence(parseArguments(process.argv.slice(2)));
  console.log(JSON.stringify(result, null, 2));
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  main().catch((error) => {
    console.error(`Release evidence verification failed: ${error.message}`);
    process.exitCode = 1;
  });
}
