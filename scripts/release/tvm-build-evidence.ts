import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {readFile, realpath, stat} from 'node:fs/promises';
import {dirname, isAbsolute, relative, resolve, sep} from 'node:path';

interface AssetMetadata {
  readonly bytes: number;
  readonly sha256: string;
}

export interface TvmBuildModel {
  readonly modelId: string;
  readonly manifestFiles: Readonly<Record<string, AssetMetadata>>;
  readonly tvmVersion: string;
  readonly sourceArtifacts?: Readonly<Record<string, string>>;
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

/** Compare supplied TVM build artifacts with the packaged model pins. */
function inside(root: string, path: string): boolean {
  const value = relative(root, path);
  return value === '' || (value !== '..' && !value.startsWith(`..${sep}`) && !isAbsolute(value));
}

export async function verifyTvmBuildEvidence(
  evidencePath: string,
  models: readonly TvmBuildModel[],
  forbiddenRoots: readonly string[] = []
) {
  const evidenceBytes = await readFile(evidencePath);
  const evidenceSha256 = createHash('sha256').update(evidenceBytes).digest('hex');
  const evidence = parseBuildRecords(JSON.parse(evidenceBytes.toString('utf8')) as unknown);
  const expected = new Map<string, AssetMetadata & {tvmVersion: string; sourcePath: string | undefined}>();
  const blockedRoots = await Promise.all(forbiddenRoots.map((root) => realpath(root)));
  for (const model of models) {
    const artifacts = Object.entries(model.manifestFiles).filter(([name]) => name.endsWith('.wasm'));
    requireCondition(artifacts.length > 0, `No TVM WASM artifact is recorded for ${model.modelId}.`);
    requireCondition(model.tvmVersion.length > 0, `No TVM version is recorded for ${model.modelId}.`);
    for (const [name, metadata] of artifacts)
      expected.set(`${model.modelId}\0${name}`, {
        ...metadata,
        tvmVersion: model.tvmVersion,
        sourcePath: model.sourceArtifacts?.[name]
      });
  }
  requireCondition(
    expected.size > 0 && evidence.length === expected.size,
    'Every curated TVM WASM needs one independent build record.'
  );
  const seen = new Set<string>();
  const results: Array<{
    modelId: string;
    artifact: string;
    sha256: string;
    evidenceSha256: string;
    reportSha256: string;
  }> = [];
  for (const entry of evidence) {
    const key = `${entry.modelId}\0${entry.artifact}`;
    const wanted = expected.get(key);
    requireCondition(
      wanted && !seen.has(key),
      `Unexpected or duplicate TVM build evidence for ${entry.modelId}/${entry.artifact}.`
    );
    seen.add(key);
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
    const reportBytes = await readFile(reportPath);
    const reportSha256 = createHash('sha256').update(reportBytes).digest('hex');
    const report: unknown = JSON.parse(reportBytes.toString('utf8'));
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
    requireCondition(
      (await realpath(resolve(buildRoot, report.wasm))) === wasmPath,
      `TVM build report points to a different WASM for ${entry.modelId}/${entry.artifact}.`
    );
    const file = await stat(wasmPath);
    if (wanted.sourcePath) {
      const source = await stat(wanted.sourcePath);
      requireCondition(
        file.ino === 0 || source.ino === 0 || file.ino !== source.ino || file.dev !== source.dev,
        `TVM build WASM is the packaged source file for ${entry.modelId}/${entry.artifact}.`
      );
    }
    requireCondition(
      file.isFile() && file.size === wanted.bytes,
      `Rebuilt TVM WASM size differs for ${entry.modelId}/${entry.artifact}.`
    );
    const actual = await hashFile(wasmPath);
    requireCondition(
      actual === wanted.sha256 && report.wasmSha256 === actual && report.wasmBytes === file.size,
      `Rebuilt TVM WASM digest differs for ${entry.modelId}/${entry.artifact}.`
    );
    requireCondition(
      report.embeddedBuildRoot === false && report.tvmVersion === wanted.tvmVersion,
      `Incomplete TVM build report for ${entry.modelId}/${entry.artifact}.`
    );
    results.push({modelId: entry.modelId, artifact: entry.artifact, sha256: actual, evidenceSha256, reportSha256});
  }
  return results.sort((a, b) => a.modelId.localeCompare(b.modelId) || a.artifact.localeCompare(b.artifact));
}
