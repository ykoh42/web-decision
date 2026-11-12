import {createHash} from 'node:crypto';
import {createReadStream, type Stats} from 'node:fs';
import {readFile, readdir, realpath, stat} from 'node:fs/promises';
import {dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';

interface AssetMetadata {
  readonly bytes: number;
  readonly sha256: string;
}

export interface TvmBuildModel {
  readonly modelId: string;
  readonly manifestFiles: Readonly<Record<string, AssetMetadata>>;
  readonly tvmVersion: string;
  readonly tvmSourceRevision?: string;
  readonly sourceArtifacts?: Readonly<Record<string, string>>;
  readonly reportRequirements?: Readonly<Record<string, Readonly<Record<string, string | number | boolean>>>>;
}

interface BuildRecord {
  readonly modelId: string;
  readonly artifact: string;
  readonly buildRoot: string;
  readonly wasm: string;
  readonly report: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireCondition(value: unknown, message: string): asserts value {
  if (!value) throw new Error(message);
}

function parseBuildRecords(value: unknown): BuildRecord[] {
  requireCondition(
    isRecord(value) && value.schemaVersion === 1 && Array.isArray(value.builds),
    'Invalid TVM build evidence.'
  );
  return value.builds.map((entry: unknown) => {
    requireCondition(isRecord(entry), 'Invalid TVM build evidence record.');
    const {modelId, artifact, buildRoot, wasm, report} = entry;
    requireCondition(
      [modelId, artifact, buildRoot, wasm, report].every((item) => typeof item === 'string' && item.trim().length > 0),
      'TVM build evidence records require modelId, artifact, buildRoot, wasm and report.'
    );
    return {
      modelId: modelId as string,
      artifact: artifact as string,
      buildRoot: buildRoot as string,
      wasm: wasm as string,
      report: report as string
    };
  });
}

async function hashFile(path: string): Promise<string> {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest('hex');
}

async function containsFileIdentity(root: string, file: {dev: number; ino: number}): Promise<boolean> {
  if (file.ino === 0) return false;
  for (const entry of await readdir(root, {withFileTypes: true})) {
    const path = join(root, entry.name);
    if (entry.isDirectory() && (await containsFileIdentity(path, file))) return true;
    if (!entry.isFile()) continue;
    const candidate = await stat(path);
    if (candidate.ino !== 0 && candidate.ino === file.ino && candidate.dev === file.dev) return true;
  }
  return false;
}

/** Compare supplied TVM build artifacts with the packaged model pins. */
function inside(root: string, path: string): boolean {
  const value = relative(root, path);
  return value === '' || (value !== '..' && !value.startsWith(`..${sep}`) && !isAbsolute(value));
}

interface ExpectedArtifact extends AssetMetadata {
  readonly tvmVersion: string;
  readonly tvmSourceRevision: string | undefined;
  readonly sourcePath: string | undefined;
  readonly reportRequirements: Readonly<Record<string, string | number | boolean>>;
}

interface BuildPaths {
  readonly buildRoot: string;
  readonly wasmPath: string;
  readonly reportPath: string;
}

function expectedArtifacts(models: readonly TvmBuildModel[]): Map<string, ExpectedArtifact> {
  const expected = new Map<string, ExpectedArtifact>();
  for (const model of models) {
    const artifacts = Object.entries(model.manifestFiles).filter(([name]) => name.endsWith('.wasm'));
    requireCondition(artifacts.length > 0, `No TVM WASM artifact is recorded for ${model.modelId}.`);
    requireCondition(model.tvmVersion.length > 0, `No TVM version is recorded for ${model.modelId}.`);
    for (const [name, metadata] of artifacts) {
      expected.set(`${model.modelId}\0${name}`, {
        ...metadata,
        tvmVersion: model.tvmVersion,
        tvmSourceRevision: model.tvmSourceRevision,
        sourcePath: model.sourceArtifacts?.[name],
        reportRequirements: model.reportRequirements?.[name] ?? {}
      });
    }
  }
  return expected;
}

async function resolveBuildPaths(
  evidencePath: string,
  entry: BuildRecord,
  blockedRoots: readonly string[]
): Promise<BuildPaths> {
  const buildRoot = await realpath(resolve(dirname(evidencePath), entry.buildRoot));
  const wasmPath = await realpath(resolve(buildRoot, entry.wasm));
  const reportPath = await realpath(resolve(buildRoot, entry.report));
  requireCondition(
    inside(buildRoot, wasmPath) && inside(buildRoot, reportPath),
    `TVM build files must be inside their build root for ${entry.modelId}/${entry.artifact}.`
  );
  requireCondition(
    !blockedRoots.some((root) => inside(root, wasmPath) || inside(root, reportPath)),
    `TVM build evidence must be outside the source and installed packages for ${entry.modelId}/${entry.artifact}.`
  );
  return {buildRoot, wasmPath, reportPath};
}

async function readBuildReport(
  reportPath: string,
  entry: BuildRecord
): Promise<{report: Record<string, unknown>; reportSha256: string}> {
  const bytes = await readFile(reportPath);
  const report: unknown = JSON.parse(bytes.toString('utf8'));
  requireCondition(
    isRecord(report) &&
      report.evidenceSchema === 'web-decision-tvm-build-v1' &&
      report.modelId === entry.modelId &&
      report.artifact === entry.artifact &&
      report.status === 'success' &&
      report.target === 'webgpu / wasm32' &&
      typeof report.wasm === 'string' &&
      !isAbsolute(report.wasm),
    `Invalid TVM build report for ${entry.modelId}/${entry.artifact}.`
  );
  return {
    report: report as Record<string, unknown>,
    reportSha256: createHash('sha256').update(bytes).digest('hex')
  };
}

async function assertReportMatchesArtifact(
  report: Record<string, unknown>,
  buildRoot: string,
  wasmPath: string,
  entry: BuildRecord
): Promise<void> {
  requireCondition(
    (await realpath(resolve(buildRoot, report.wasm as string))) === wasmPath,
    `TVM build report points to a different WASM for ${entry.modelId}/${entry.artifact}.`
  );
}

async function assertNotPackagedArtifact(
  sourcePath: string | undefined,
  blockedRoots: readonly string[],
  file: Stats,
  entry: BuildRecord
): Promise<void> {
  if (sourcePath) {
    const source = await stat(sourcePath);
    requireCondition(
      file.ino === 0 || source.ino === 0 || file.ino !== source.ino || file.dev !== source.dev,
      `TVM build WASM is the packaged source file for ${entry.modelId}/${entry.artifact}.`
    );
  }
  for (const root of blockedRoots) {
    requireCondition(
      !(await containsFileIdentity(root, file)),
      `TVM build WASM is the packaged source file or installed file for ${entry.modelId}/${entry.artifact}.`
    );
  }
}

async function verifyArtifactBytes(
  wasmPath: string,
  wanted: ExpectedArtifact,
  entry: BuildRecord,
  report: Record<string, unknown>,
  file: Stats
): Promise<string> {
  requireCondition(
    file.isFile() && file.size === wanted.bytes,
    `Rebuilt TVM WASM size differs for ${entry.modelId}/${entry.artifact}.`
  );
  const actual = await hashFile(wasmPath);
  requireCondition(
    actual === wanted.sha256 && report.wasmSha256 === actual && report.wasmBytes === file.size,
    `Rebuilt TVM WASM digest differs for ${entry.modelId}/${entry.artifact}.`
  );
  return actual;
}

function validateReportRequirements(
  report: Record<string, unknown>,
  wanted: ExpectedArtifact,
  entry: BuildRecord
): void {
  requireCondition(
    report.embeddedBuildRoot === false && report.tvmVersion === wanted.tvmVersion,
    `Incomplete TVM build report for ${entry.modelId}/${entry.artifact}.`
  );
  for (const [field, value] of Object.entries(wanted.reportRequirements)) {
    requireCondition(
      report[field] === value,
      `TVM build report ${field} differs for ${entry.modelId}/${entry.artifact}.`
    );
  }
  if (wanted.tvmSourceRevision) {
    const environment = report.buildEnvironment;
    const nativeTvm = isRecord(environment) ? environment.nativeTvm : undefined;
    const source = isRecord(nativeTvm) ? nativeTvm.source : undefined;
    requireCondition(
      isRecord(source) && source.revision === wanted.tvmSourceRevision,
      `TVM build report source revision differs for ${entry.modelId}/${entry.artifact}.`
    );
    requireCondition(
      Array.isArray(source.patches) && source.patches.length === 0,
      `TVM build report uses local source patches for ${entry.modelId}/${entry.artifact}.`
    );
    const pythonPackage = isRecord(nativeTvm) ? nativeTvm.pythonPackage : undefined;
    requireCondition(
      isRecord(pythonPackage) && pythonPackage.version === wanted.tvmVersion,
      `TVM build report Python package version differs for ${entry.modelId}/${entry.artifact}.`
    );
  }
}

async function verifyBuildRecord(
  evidencePath: string,
  entry: BuildRecord,
  wanted: ExpectedArtifact,
  blockedRoots: readonly string[],
  evidenceSha256: string
) {
  const paths = await resolveBuildPaths(evidencePath, entry, blockedRoots);
  const {report, reportSha256} = await readBuildReport(paths.reportPath, entry);
  await assertReportMatchesArtifact(report, paths.buildRoot, paths.wasmPath, entry);
  const file = await stat(paths.wasmPath);
  await assertNotPackagedArtifact(wanted.sourcePath, blockedRoots, file, entry);
  const actual = await verifyArtifactBytes(paths.wasmPath, wanted, entry, report, file);
  validateReportRequirements(report, wanted, entry);
  return {modelId: entry.modelId, artifact: entry.artifact, sha256: actual, evidenceSha256, reportSha256};
}

export async function verifyTvmBuildEvidence(
  evidencePath: string,
  models: readonly TvmBuildModel[],
  forbiddenRoots: readonly string[] = []
) {
  const evidenceBytes = await readFile(evidencePath);
  const evidenceSha256 = createHash('sha256').update(evidenceBytes).digest('hex');
  const evidence = parseBuildRecords(JSON.parse(evidenceBytes.toString('utf8')) as unknown);
  const expected = expectedArtifacts(models);
  const blockedRoots = await Promise.all(forbiddenRoots.map((root) => realpath(root)));
  requireCondition(
    expected.size > 0 && evidence.length === expected.size,
    'Every curated TVM WASM needs one independent build record.'
  );
  const seen = new Set<string>();
  const results = [];
  for (const entry of evidence) {
    const key = `${entry.modelId}\0${entry.artifact}`;
    const wanted = expected.get(key);
    requireCondition(
      wanted && !seen.has(key),
      `Unexpected or duplicate TVM build evidence for ${entry.modelId}/${entry.artifact}.`
    );
    seen.add(key);
    results.push(await verifyBuildRecord(evidencePath, entry, wanted, blockedRoots, evidenceSha256));
  }
  return results.sort((a, b) => a.modelId.localeCompare(b.modelId) || a.artifact.localeCompare(b.artifact));
}
