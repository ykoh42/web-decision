import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {copyFile, lstat, mkdir, readFile, realpath, readdir, stat, writeFile} from 'node:fs/promises';
import {dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'vite';
import {canonical, canonicalSha256} from '../../release/verify-release-evidence.ts';
import {verifyQwen35ExactPackage} from '../../release/strands-qwen35-release-evidence.ts';

const workspaceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const packageRoot = join(workspaceRoot, 'packages/web-decision');
const modelId = 'strands-decider-2b-qwen3.5-v1-2610';
const candidateVersion = '0.0.0-qwen35-candidate';

type RecordValue = Record<string, unknown>;
type ModelSpecFile = Readonly<{models: readonly Readonly<{modelId: string; description: RecordValue}>[]}>;
type ArtifactLock = Readonly<{
  byModelId: Readonly<
    Record<
      string,
      Readonly<{
        manifest: Readonly<{
          sourcePath: string;
          rawSha256: string;
          canonicalSha256: string;
          canonicalBytes: number;
          files: Readonly<Record<string, Readonly<{bytes: number; sha256: string}>>>;
        }>;
      }>
    >
  >;
}>;

export function isSameOrInside(path: string, root: string): boolean {
  const pathFromRoot = relative(root, path);
  return (
    pathFromRoot === '' || (pathFromRoot !== '..' && !pathFromRoot.startsWith(`..${sep}`) && !isAbsolute(pathFromRoot))
  );
}

function assertSafeRelativePath(sourcePath: string, description: string): void {
  assert.ok(
    sourcePath.length > 0 &&
      !isAbsolute(sourcePath) &&
      !sourcePath.includes('\\') &&
      !sourcePath.includes('\0') &&
      !/^[a-z]:/iu.test(sourcePath),
    `${description} must be a safe relative path.`
  );
  assert.ok(
    sourcePath.split('/').every((segment) => segment.length > 0 && segment !== '.' && segment !== '..'),
    `${description} must be a safe relative path.`
  );
}

export async function resolvePackageFilePath(
  sourcePackageRoot: string,
  sourcePath: string,
  description: string
): Promise<string> {
  assertSafeRelativePath(sourcePath, description);
  const segments = sourcePath.split('/');

  const packageRoot = await realpath(sourcePackageRoot);
  const candidatePath = resolve(packageRoot, ...segments);
  assert.ok(isSameOrInside(candidatePath, packageRoot), `${description} escapes the package root.`);
  const canonicalPath = await realpath(candidatePath);
  assert.ok(isSameOrInside(canonicalPath, packageRoot), `${description} resolves outside the package root.`);
  assert.ok((await stat(canonicalPath)).isFile(), `${description} must resolve to a regular file.`);
  return canonicalPath;
}

export async function validateCandidateOutputDirectory(
  outputDirectory: string,
  sourceWorkspaceRoot = workspaceRoot
): Promise<string> {
  const absolute = resolve(outputDirectory);
  let info;
  try {
    info = await lstat(absolute);
  } catch {
    throw new Error('Qwen candidate output must be an existing empty directory.');
  }
  if (info.isSymbolicLink() || !info.isDirectory()) throw new Error('Qwen candidate output must be a real directory.');
  if ((await readdir(absolute)).length > 0) throw new Error('Qwen candidate output must be empty.');

  const [outputRoot, sourceRoot] = await Promise.all([realpath(absolute), realpath(sourceWorkspaceRoot)]);
  if (isSameOrInside(outputRoot, sourceRoot) || isSameOrInside(sourceRoot, outputRoot))
    throw new Error('Qwen candidate output must be outside the repository.');
  return outputRoot;
}

export function buildCandidateCatalogModule(entry: Readonly<{modelId: string; description: RecordValue}>): string {
  return [
    `const entry = ${JSON.stringify(entry)};`,
    'function deepFreeze(value) {',
    '  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;',
    '  for (const child of Object.values(value)) deepFreeze(child);',
    '  return Object.freeze(value);',
    '}',
    'const catalog = deepFreeze([entry]);',
    'export const APPROVED_MODEL_IDS = Object.freeze(catalog.map(({modelId}) => modelId));',
    'export const MODEL_CATALOG = catalog;',
    'export const MODEL_CATALOG_BY_ID = Object.freeze(Object.fromEntries(catalog.map((item) => [item.modelId, item])));',
    ''
  ].join('\n');
}

export function buildCandidateAdapterModule(): string {
  return [
    'export const APPROVED_ADAPTER_LOADERS = Object.freeze({',
    '  "strands-qwen35": () => import("@web-decision/qwen35-candidate-adapter")',
    '});',
    ''
  ].join('\n');
}

export function candidatePackageMetadata(dependencies: RecordValue): RecordValue {
  return {
    name: 'web-decision',
    version: candidateVersion,
    private: true,
    description: 'UNPUBLISHABLE internal Qwen3.5 candidate for exact-package verification.',
    type: 'module',
    types: './dist/index.d.ts',
    exports: {'.': {types: './dist/index.d.ts', import: './dist/index.js'}},
    files: ['dist'],
    dependencies
  };
}

async function stageCandidateDeclarations(dist: string): Promise<void> {
  const source = join(packageRoot, 'src');
  const curated = join(dist, 'curated');
  const kanana = join(dist, 'kanana');
  // Preserve source-relative imports and use the package's nominal float declaration throughout.
  await Promise.all([mkdir(curated, {recursive: true}), mkdir(kanana, {recursive: true})]);
  await Promise.all([
    copyFile(join(source, 'strands-qwen35-candidate-index.d.ts'), join(dist, 'index.d.ts')),
    copyFile(
      join(source, 'curated/strands-qwen35-candidate-types.d.ts'),
      join(curated, 'strands-qwen35-candidate-types.d.ts')
    ),
    copyFile(join(source, 'curated/public-types.d.ts'), join(curated, 'public-types.d.ts')),
    writeFile(
      join(curated, 'public-model-id.generated.d.ts'),
      `export type PublicModelId = ${JSON.stringify(modelId)};\n`
    ),
    writeFile(join(kanana, 'json-float.d.ts'), "export {KananaFloat, kananaFloat} from '../curated/public-types.js';\n")
  ]);
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function record(value: unknown, description: string): RecordValue {
  assert.ok(value !== null && typeof value === 'object' && !Array.isArray(value), `${description} must be an object.`);
  return value as RecordValue;
}

async function readCandidateInputs() {
  const modelSpec = JSON.parse(
    await readFile(join(packageRoot, 'src/catalog/model-spec.json'), 'utf8')
  ) as ModelSpecFile;
  const artifactLock = JSON.parse(
    await readFile(join(packageRoot, 'src/catalog/artifacts.lock.json'), 'utf8')
  ) as ArtifactLock;
  const provenanceManifest = JSON.parse(
    await readFile(join(packageRoot, 'src/catalog/model-provenance.json'), 'utf8')
  ) as {models?: readonly Readonly<{modelId: string; status: string; evidence: {path: string; sha256: string}}>[]};
  const candidateProvenance = provenanceManifest.models?.find((entry) => entry.modelId === modelId);
  assert.equal(candidateProvenance?.status, 'pending', 'Qwen candidate packaging is only for pending provenance.');
  const evidencePath = await resolvePackageFilePath(
    packageRoot,
    candidateProvenance!.evidence.path,
    'Qwen provenance evidence path'
  );
  const evidenceBytes = await readFile(evidencePath);
  assert.equal(sha256(evidenceBytes), candidateProvenance!.evidence.sha256, 'Pending Qwen provenance digest changed.');
  const evidence = JSON.parse(evidenceBytes.toString('utf8')) as RecordValue;
  assert.equal(evidence.status, 'pending', 'Qwen provenance evidence is no longer pending.');

  const spec = modelSpec.models.find((entry) => entry.modelId === modelId);
  const lock = artifactLock.byModelId[modelId];
  assert.ok(spec, 'Qwen model spec entry is missing.');
  assert.ok(lock, 'Qwen artifact lock entry is missing.');
  assert.equal(spec.description.adapter, 'strands-qwen35', 'Qwen candidate requires its dedicated adapter.');

  const manifestPath = await resolvePackageFilePath(packageRoot, lock.manifest.sourcePath, 'Qwen asset manifest path');
  const manifestBytes = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBytes.toString('utf8')) as RecordValue;
  assert.equal(sha256(manifestBytes), lock.manifest.rawSha256, 'Qwen asset manifest raw digest differs from its lock.');
  const canonicalManifestBytes = Buffer.from(`${JSON.stringify(canonical(manifest))}\n`);
  assert.equal(
    canonicalManifestBytes.byteLength,
    lock.manifest.canonicalBytes,
    'Qwen asset manifest size differs from its lock.'
  );
  assert.equal(
    canonicalSha256(manifest),
    lock.manifest.canonicalSha256,
    'Qwen asset manifest canonical digest differs from its lock.'
  );
  const manifestFiles = record(manifest.files, 'Qwen asset manifest files');
  assert.deepEqual(Object.keys(manifestFiles).sort(), Object.keys(lock.manifest.files).sort());
  for (const [name, expectedValue] of Object.entries(lock.manifest.files)) {
    assertSafeRelativePath(name, `Qwen locked asset name ${name}`);
    const expected = record(expectedValue, `Qwen locked file ${name}`) as {bytes: number; sha256: string};
    const assetPath = await resolvePackageFilePath(packageRoot, `assets/qwen35/${name}`, `Qwen locked file ${name}`);
    const bytes = await readFile(assetPath);
    assert.equal(bytes.byteLength, expected.bytes, `Qwen asset length differs for ${name}.`);
    assert.equal(sha256(bytes), expected.sha256, `Qwen asset digest differs for ${name}.`);
  }

  const description = {
    ...spec.description,
    assets: {
      ...record(spec.description.assets, 'Qwen catalog assets'),
      requiredFiles: ['manifest.json', ...Object.keys(lock.manifest.files)],
      manifestSha256: lock.manifest.canonicalSha256
    }
  };
  return {
    entry: {modelId, description},
    modelSpecSha256: sha256(await readFile(join(packageRoot, 'src/catalog/model-spec.json'))),
    artifactLockSha256: sha256(await readFile(join(packageRoot, 'src/catalog/artifacts.lock.json'))),
    manifestSha256: lock.manifest.canonicalSha256,
    manifestRawSha256: lock.manifest.rawSha256
  };
}

function createCandidateAliasPlugin(
  publicEntry: string,
  catalogPath: string,
  adaptersPath: string,
  adapterPath: string
) {
  return {
    name: 'qwen35-pending-candidate-catalog',
    enforce: 'pre' as const,
    resolveId(source: string, importer: string | undefined) {
      if (importer && resolve(importer) === publicEntry && source === '../catalog/public-catalog.generated.ts')
        return catalogPath;
      if (importer && resolve(importer) === publicEntry && source === '../catalog/public-adapters.generated.ts')
        return adaptersPath;
      if (source === '@web-decision/qwen35-candidate-adapter') return adapterPath;
      return null;
    }
  };
}

async function readNpmPackResult(value: string): Promise<{filename: string; files: readonly {path: string}[]}> {
  const parsed = JSON.parse(value) as readonly {filename: string; files: readonly {path: string}[]}[];
  const [result] = parsed;
  assert.ok(
    result && typeof result.filename === 'string' && Array.isArray(result.files),
    'npm pack returned invalid JSON.'
  );
  return result;
}

export async function buildQwen35CandidatePackage(outputDirectory: string) {
  const outputRoot = await validateCandidateOutputDirectory(outputDirectory);
  const inputs = await readCandidateInputs();
  const generatedRoot = join(outputRoot, 'generated');
  const stagedPackage = join(outputRoot, 'package');
  const packDirectory = join(outputRoot, 'pack');
  const consumerRoot = join(outputRoot, 'consumer');
  await Promise.all([mkdir(generatedRoot), mkdir(stagedPackage), mkdir(packDirectory), mkdir(consumerRoot)]);

  const catalogPath = join(generatedRoot, 'public-catalog.generated.ts');
  const adaptersPath = join(generatedRoot, 'public-adapters.generated.ts');
  await Promise.all([
    writeFile(catalogPath, buildCandidateCatalogModule(inputs.entry)),
    writeFile(adaptersPath, buildCandidateAdapterModule())
  ]);

  const publicEntry = resolve(packageRoot, 'src/curated/index.ts');
  const adapterPath = resolve(packageRoot, 'src/curated/adapters/strands-qwen35.ts');
  const configFile = resolve(packageRoot, 'vite.config.ts');
  await build({
    configFile,
    root: packageRoot,
    plugins: [createCandidateAliasPlugin(publicEntry, catalogPath, adaptersPath, adapterPath)],
    build: {
      outDir: join(stagedPackage, 'dist'),
      emptyOutDir: true,
      lib: {entry: publicEntry, formats: ['es'], fileName: 'index'},
      target: 'es2022'
    }
  });

  await stageCandidateDeclarations(join(stagedPackage, 'dist'));
  const sourceMetadata = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8')) as RecordValue;
  const dependencies = record(sourceMetadata.dependencies ?? {}, 'Package runtime dependencies');
  const stagedMetadata = candidatePackageMetadata(dependencies);
  await writeFile(join(stagedPackage, 'package.json'), `${JSON.stringify(stagedMetadata, null, 2)}\n`);

  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const packOutput = execFileSync(npm, ['pack', '--ignore-scripts', '--json', '--pack-destination', packDirectory], {
    cwd: stagedPackage,
    encoding: 'utf8'
  });
  const pack = await readNpmPackResult(packOutput);
  const packedFiles = pack.files.map(({path}) => path);
  assert.ok(packedFiles.includes('package.json'), 'Candidate archive is missing its private package metadata.');
  assert.ok(packedFiles.includes('dist/index.js'), 'Candidate archive is missing the public entrypoint.');
  assert.ok(packedFiles.includes('dist/index.d.ts'), 'Candidate archive is missing its public declarations.');
  assert.ok(
    packedFiles.some((path) => path.endsWith('.wasm')),
    'Candidate archive is missing compiled model graphs.'
  );

  const archivePath = join(packDirectory, pack.filename);
  const archiveMetadata = JSON.parse(
    execFileSync('tar', ['-xOzf', archivePath, 'package/package.json'], {encoding: 'utf8'})
  ) as RecordValue;
  assert.equal(archiveMetadata.private, true, 'Candidate archive must carry npm private:true.');
  assert.equal(archiveMetadata.version, candidateVersion, 'Candidate archive version changed.');
  const archiveExports = record(archiveMetadata.exports, 'Candidate exports');
  const publicExport = record(archiveExports['.'], 'Candidate public export');
  assert.equal(publicExport.import, './dist/index.js', 'Candidate package import entry changed.');
  assert.equal(publicExport.types, archiveMetadata.types, 'Candidate declaration entry differs across resolvers.');

  execFileSync(npm, ['install', archivePath, '--ignore-scripts', '--no-audit', '--no-fund', '--prefix', consumerRoot], {
    stdio: 'inherit'
  });
  const installedPackage = join(consumerRoot, 'node_modules/web-decision');
  const exact = await verifyQwen35ExactPackage(archivePath, await realpath(installedPackage));
  const installedBundlePath = join(installedPackage, 'dist/index.js');
  const installedBundleSha256 = sha256(await readFile(installedBundlePath));
  assert.equal(
    installedBundleSha256,
    exact.archive.bundleSha256,
    'Installed public bundle differs from its archive hash.'
  );

  const evidenceRecord = {
    schemaVersion: 1,
    status: 'unpublishable-pending-qwen-candidate',
    modelId,
    provenanceStatus: 'pending',
    sourcePins: {
      modelSpecSha256: inputs.modelSpecSha256,
      artifactLockSha256: inputs.artifactLockSha256,
      manifestSha256: inputs.manifestSha256,
      manifestRawSha256: inputs.manifestRawSha256
    },
    package: {
      name: exact.archive.packageName,
      version: exact.archive.packageVersion,
      private: archiveMetadata.private,
      archivePath: relative(outputRoot, archivePath),
      archiveSha256: exact.archive.archiveSha256,
      packageTreeSha256: exact.archive.packageTreeSha256,
      bundlePath: relative(outputRoot, installedBundlePath),
      bundleSha256: installedBundleSha256,
      exactInstallMatch: true,
      files: exact.archive.archiveFiles
    }
  };
  await writeFile(join(outputRoot, 'candidate-package.json'), `${JSON.stringify(evidenceRecord, null, 2)}\n`);
  return {
    outputRoot,
    archivePath,
    installedPackagePath: await realpath(installedPackage),
    installedBundlePath: await realpath(installedBundlePath),
    archiveSha256: exact.archive.archiveSha256,
    packageTreeSha256: exact.archive.packageTreeSha256,
    bundleSha256: installedBundleSha256,
    candidateReportPath: join(outputRoot, 'candidate-package.json')
  };
}

function parseOutputDirectory(args: readonly string[]): string {
  if (args.length !== 2 || args[0] !== '--output-dir' || !args[1])
    throw new TypeError(
      'Usage: node --import tsx scripts/strands/qwen35/build_candidate_package.mts --output-dir <existing-empty-directory-outside-repo>'
    );
  return args[1];
}

async function main(): Promise<void> {
  const result = await buildQwen35CandidatePackage(parseOutputDirectory(process.argv.slice(2)));
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'unpublishable-pending-qwen-candidate',
        ...result,
        browserEnvironment: {
          WEB_DECISION_RELEASE_ARCHIVE: result.archivePath,
          WEB_DECISION_INSTALLED_PACKAGE: result.installedPackagePath,
          WEB_DECISION_INSTALLED_BUNDLE: result.installedBundlePath
        }
      },
      null,
      2
    )}\n`
  );
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`Qwen candidate package build failed: ${message}\n`);
    process.exitCode = 1;
  });
}
