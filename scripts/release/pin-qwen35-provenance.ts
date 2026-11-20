import assert from 'node:assert/strict';
import {lstat, readFile, realpath} from 'node:fs/promises';
import {dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {MODEL_PROVENANCE_MANIFEST_PATH, validateCommittedEvidenceReference} from '../ci/model-provenance.ts';
import {
  QWEN35_RELEASE_MODEL,
  assertQwen35EvidenceRecord,
  loadQwen35CandidateInputs,
  verifyQwen35ExactPackage
} from './strands-qwen35-release-evidence.ts';
import {sha256, validateQwen35CandidateArchiveCatalog, type ArchiveInfo} from './verify-release-evidence.ts';

const MODEL_ID = QWEN35_RELEASE_MODEL.catalogModelId;
const PACKAGE_PATH = 'packages/web-decision';
const EVIDENCE_PATH = `src/catalog/provenance/${MODEL_ID}.json`;
const CONVERSION = {
  path: `${PACKAGE_PATH}/src/catalog/provenance/qwen35-conversion.json`,
  sha256: '26101ff70c6f389103a9b744a688374a6ab6798728a460ce5bf04d3eab799dc4'
};
const STRANDS_LICENSE = 'STRANDS-QWEN35-LICENSE.md';
const BASE_LICENSE = 'QWEN-APACHE-LICENSE.txt';
const NOTICE_PATH = 'STRANDS-QWEN35-NOTICE';
const NOTICE_SHA256 = '39744fd3ae377af12b58765603d67e3e15684d62e4cc4de158aa57c401d69861';
const RIGHTS_SOURCES = [
  {
    file: STRANDS_LICENSE,
    sha256: 'c0d777a0fed01cceb107d91f1aef96d51737be04f2ed75b1b933286a170d7411',
    subject: 'Strands checkpoint adapter and pointer head',
    license: 'Apache-2.0 as declared by the pinned model card'
  },
  {
    file: BASE_LICENSE,
    sha256: '50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32',
    subject: 'Qwen base checkpoint including the source of the converted vision weights',
    license: 'Apache-2.0'
  }
] as const;
const NATIVE_TVM_REVISION = '7b5acb32bd271342a24130f4ef439ff6c2e52c6c';
const TEXT_REPORT_SHA256 = '63cf222e1cc6f1ca6ce6c86f0ef188e0d0f3827231652e6ec2cb23bdd651e69c';
const VISION_REPORT_SHA256 = 'fd7f9a1cbc3c2d2cee6d9c9eba3b8d2af47e396139b9fbb77d4390d563371024';
const GRAPHS = {
  text: {
    artifact: 'r19/qwen35-text-pointer.webgpu.wasm',
    report: 'qwen35-gdn-prefix-fp32-candidate.json',
    wasm: 'qwen35-gdn-prefix-fp32-candidate.wasm'
  },
  vision: {
    artifact: 'vision/qwen35-dynamic-vision.webgpu.wasm',
    report: 'build-report.json',
    wasm: 'qwen35-dynamic-vision.webgpu.wasm'
  }
} as const;

type AssetPin = {bytes: number; sha256: string};
type Graph = keyof typeof GRAPHS;

interface Options {
  workspaceRoot: string;
  archivePath: string;
  installedPackagePath: string;
  browserReportPath: string;
  artifactRoot: string;
  textBuildRoot: string;
  visionBuildRoot: string;
}

function json(bytes: Buffer): unknown {
  return JSON.parse(bytes.toString('utf8')) as unknown;
}

function object(value: unknown, description: string): Record<string, unknown> {
  assert.ok(value && typeof value === 'object' && !Array.isArray(value), `Invalid ${description}.`);
  return value as Record<string, unknown>;
}

function within(root: string, path: string): boolean {
  const tail = relative(root, path);
  return tail !== '..' && !tail.startsWith(`..${sep}`) && !isAbsolute(tail);
}

async function readLocalFile(path: string) {
  const info = await lstat(path);
  assert.ok(info.isFile() && !info.isSymbolicLink(), `Expected a regular evidence file: ${path}`);
  assert.equal(await realpath(path), path, `Evidence file must use its canonical path: ${path}`);
  return {info, bytes: await readFile(path)};
}

export function assertQwen35ProductionArchive(archive: ArchiveInfo): void {
  assert.match(
    archive.packageVersion,
    /^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$/u,
    'Promotion requires a stable npm version.'
  );
  assert.ok(archive.archiveFiles.includes('assets/kanana/manifest.json'), 'Archive must also contain Kanana.');
  assert.ok(archive.archiveFiles.includes('assets/qwen35/manifest.json'), 'Archive must also contain Qwen.');
}

export function assertQwen35BrowserArchive(evidence: Record<string, unknown>, archive: ArchiveInfo): void {
  const pin = object(evidence.package, 'Qwen browser archive pin');
  const {version, archiveSha256, packageTreeSha256, bundleSha256} = pin;
  assert.deepEqual(
    {version, archiveSha256, packageTreeSha256, bundleSha256},
    {
      version: archive.packageVersion,
      archiveSha256: archive.archiveSha256,
      packageTreeSha256: archive.packageTreeSha256,
      bundleSha256: archive.bundleSha256
    },
    'Qwen browser evidence was not collected from this exact combined npm archive.'
  );
}

function checkBrowserArtifacts(
  evidence: Record<string, unknown>,
  inputs: Awaited<ReturnType<typeof loadQwen35CandidateInputs>>
) {
  const artifacts = object(evidence.artifacts, 'Qwen browser artifact pins');
  assert.deepEqual(
    artifacts.manifestFiles,
    inputs.manifestFiles,
    'Qwen browser evidence reports different harness artifacts.'
  );
  assert.equal(artifacts.manifestSha256, inputs.manifestSha256, 'Qwen browser harness manifest changed.');
  assert.equal(artifacts.manifestRawSha256, inputs.manifestRawSha256, 'Qwen browser harness raw manifest changed.');
  assert.deepEqual(artifacts.filesSha256, inputs.artifactSha256, 'Qwen browser model artifact bytes changed.');
  assert.deepEqual(artifacts.fixtureSha256, inputs.fixtureSha256, 'Qwen browser reference fixture bytes changed.');
  assert.equal(artifacts.evidenceInputSha256, inputs.evidenceInputSha256, 'Qwen browser evidence inputs changed.');
}

function checkVisionReport(report: Record<string, unknown>, wasmPath: string): void {
  assert.equal(report.wasm, wasmPath, 'Native Qwen vision report points to a different build.');
  const portableReport = {...report, wasm: GRAPHS.vision.wasm};
  assert.equal(
    sha256(`${JSON.stringify(portableReport, null, 2)}\n`),
    VISION_REPORT_SHA256,
    'Pinned native Qwen vision build report changed.'
  );
}

export async function validateQwen35NativeBuild(
  graph: Graph,
  buildRoot: string,
  lockedWasm: AssetPin,
  packagedPaths: readonly string[],
  excludedRoots: readonly string[]
) {
  const root = await realpath(resolve(buildRoot));
  const excluded = await Promise.all(excludedRoots.map((path) => realpath(path)));
  assert.ok(!excluded.some((path) => within(path, root)), 'Use an external Qwen build directory.');
  const recipe = GRAPHS[graph];
  const wasmPath = join(root, recipe.wasm);
  const [{info, bytes}, {bytes: reportBytes}, ...packaged] = await Promise.all([
    readLocalFile(wasmPath),
    readLocalFile(join(root, recipe.report)),
    ...packagedPaths.map((path) => lstat(path))
  ]);
  assert.ok(
    !packaged.some((file) => file.ino === info.ino && file.dev === info.dev),
    'Independent Qwen rebuild must not reuse the packaged artifact file.'
  );
  const wasm = {bytes: bytes.length, sha256: sha256(bytes)};
  assert.deepEqual(wasm, lockedWasm, `Rebuilt Qwen ${graph} WASM differs from its artifact lock.`);
  const reportSha256 = sha256(reportBytes);
  const report = object(json(reportBytes), `Qwen ${graph} build report`);
  assert.equal(report.wasmSha256, wasm.sha256, `Qwen ${graph} build report WASM digest changed.`);
  if (graph === 'text') assert.equal(reportSha256, TEXT_REPORT_SHA256, 'Pinned native Qwen text build report changed.');
  else checkVisionReport(report, wasmPath);
  return {graph, artifact: recipe.artifact, ...wasm, reportSha256};
}

async function currentPins(workspaceRoot: string, archive: ArchiveInfo) {
  const {models, manifests} = await validateQwen35CandidateArchiveCatalog(archive, workspaceRoot);
  const model = models.find(({modelId}) => modelId === MODEL_ID);
  const manifest = manifests.get(MODEL_ID);
  assert.ok(model && manifest, 'Missing Qwen catalog pins.');
  const packageRoot = join(workspaceRoot, PACKAGE_PATH);
  const source = await readFile(join(packageRoot, model.manifestPath));
  const installed = await readFile(join(archive.installedRoot, model.manifestPath));
  assert.deepEqual(installed, source, 'Combined archive Qwen manifest differs from the source manifest bytes.');
  const archiveFiles = await Promise.all(
    model.bundledRuntimeFiles.map(async ({sourcePath, bytes, sha256: digest}) => {
      const file = await readFile(join(archive.installedRoot, sourcePath));
      assert.deepEqual({bytes: file.length, sha256: sha256(file)}, {bytes, sha256: digest}, sourcePath);
      return [sourcePath, digest];
    })
  );
  return {model, manifest, archiveFiles: Object.fromEntries(archiveFiles) as Record<string, string>};
}

async function rightsEvidence(workspaceRoot: string, archive: ArchiveInfo) {
  const notice = await readFile(join(archive.installedRoot, NOTICE_PATH));
  assert.equal(sha256(notice), NOTICE_SHA256, 'The archive changed the Qwen attribution and modification notice.');
  return Promise.all(
    RIGHTS_SOURCES.map(async ({file, sha256: digest, subject, license}) => {
      const evidence = {path: `${PACKAGE_PATH}/${file}`, sha256: digest};
      const source = await validateCommittedEvidenceReference(
        evidence,
        'Qwen redistribution terms',
        workspaceRoot,
        true
      );
      const packaged = await readFile(join(archive.installedRoot, file));
      assert.deepEqual(packaged, source, `Combined archive must include the exact upstream ${file}.`);
      return {
        subject,
        license,
        redistribution: 'conditional',
        conditions: [
          'Include the applicable Apache-2.0 license and retain applicable copyright and attribution notices.',
          'Identify modified files as required by Apache-2.0.'
        ],
        evidence
      };
    })
  );
}

function summarizeExactPackage(evidence: Record<string, unknown>, archive: ArchiveInfo, reportSha256: string) {
  const browser = object(evidence.browser, 'Qwen browser identity');
  const hardware = object(evidence.hardware, 'Qwen hardware identity');
  return {
    version: archive.packageVersion,
    archiveSha256: archive.archiveSha256,
    packageTreeSha256: archive.packageTreeSha256,
    bundleSha256: archive.bundleSha256,
    reportSha256,
    browser: {name: browser.name, version: browser.version, channel: browser.channel},
    hardware: {platform: hardware.platform, arch: hardware.arch, chipType: hardware.chipType}
  };
}

export async function pinQwen35Provenance(options: Options) {
  const workspaceRoot = await realpath(options.workspaceRoot);
  const packageRoot = join(workspaceRoot, PACKAGE_PATH);
  const {archive} = await verifyQwen35ExactPackage(options.archivePath, options.installedPackagePath);
  assertQwen35ProductionArchive(archive);
  const packageMetadata = object(
    json(await readFile(join(archive.installedRoot, 'package.json'))),
    'stable npm metadata'
  );
  assert.ok(packageMetadata.private !== true, 'Promotion requires a publishable npm archive.');
  const [{model, manifest, archiveFiles}, browserBytes, inputs, rights] = await Promise.all([
    currentPins(workspaceRoot, archive),
    readFile(options.browserReportPath),
    loadQwen35CandidateInputs(options.artifactRoot),
    rightsEvidence(workspaceRoot, archive),
    validateCommittedEvidenceReference(CONVERSION, 'Qwen native conversion', workspaceRoot, true)
  ]);
  const browser = assertQwen35EvidenceRecord(json(browserBytes));
  assertQwen35BrowserArchive(browser, archive);
  checkBrowserArtifacts(browser, inputs);
  assert.deepEqual(inputs.artifactSha256, archiveFiles, 'Browser Qwen files differ from the combined archive assets.');
  const builds = await Promise.all(
    (['text', 'vision'] as const).map((graph) => {
      const {artifact} = GRAPHS[graph];
      const wasm = join(dirname(model.manifestPath), artifact);
      const pin = manifest.files[artifact];
      assert.ok(pin, `Combined archive is missing the Qwen ${graph} WASM pin.`);
      return validateQwen35NativeBuild(
        graph,
        graph === 'text' ? options.textBuildRoot : options.visionBuildRoot,
        pin,
        [join(packageRoot, wasm), join(archive.installedRoot, wasm)],
        [workspaceRoot, archive.installedRoot]
      );
    })
  );
  const pending = object(json(await readFile(join(packageRoot, EVIDENCE_PATH))), 'pending Qwen provenance');
  const manifestBytes = await readFile(join(packageRoot, MODEL_PROVENANCE_MANIFEST_PATH));
  const current = object(json(manifestBytes), 'model provenance manifest');
  const finalArchive = await verifyQwen35ExactPackage(options.archivePath, options.installedPackagePath);
  assertQwen35BrowserArchive(browser, finalArchive.archive);
  assert.equal(archive.archiveSha256, finalArchive.archive.archiveSha256, 'Archive changed during verification.');
  return createQwen35ProvenancePins(current, pending, {
    rights,
    builds,
    exactPackage: summarizeExactPackage(browser, archive, sha256(browserBytes))
  });
}

export function createQwen35ProvenancePins(
  current: Record<string, unknown>,
  pending: Record<string, unknown>,
  {
    rights,
    builds,
    exactPackage
  }: {
    rights: Awaited<ReturnType<typeof rightsEvidence>>;
    builds: Awaited<ReturnType<typeof validateQwen35NativeBuild>>[];
    exactPackage: ReturnType<typeof summarizeExactPackage>;
  }
) {
  assert.ok(Array.isArray(current.models), 'Missing current provenance manifest models.');
  const candidate = object(pending.candidate, 'pending Qwen conversion');
  const record = {
    schemaVersion: 1,
    modelId: MODEL_ID,
    status: 'verified',
    sourceLineage: pending.sourceLineage,
    upstreamRuntime: pending.upstreamRuntime,
    conversion: {
      repository: 'https://github.com/apache/tvm',
      revision: NATIVE_TVM_REVISION,
      recipe: CONVERSION,
      config: CONVERSION,
      toolchain: CONVERSION,
      artifactLockSha256: candidate.artifactLockSha256
    },
    rights,
    validation: {
      exactPackageWebgpu: exactPackage,
      rebuiltWasm: builds,
      notice: {path: NOTICE_PATH, sha256: NOTICE_SHA256}
    }
  };
  const recordSha256 = sha256(`${JSON.stringify(record, null, 2)}\n`);
  const models = current.models.map((entry: unknown) => {
    const model = object(entry, 'model provenance entry');
    if (model.modelId !== MODEL_ID) return model;
    assert.equal(model.status, 'pending');
    return {...model, status: 'verified', evidence: {path: EVIDENCE_PATH, sha256: recordSha256}};
  });
  const manifest = {...current, models};
  return {record, recordSha256, manifest, manifestSha256: sha256(`${JSON.stringify(manifest, null, 2)}\n`)};
}

async function main() {
  const {values} = parseArgs({
    options: {
      archive: {type: 'string'},
      'installed-package': {type: 'string'},
      'browser-report': {type: 'string'},
      'artifact-root': {type: 'string'},
      'text-build-root': {type: 'string'},
      'vision-build-root': {type: 'string'}
    },
    strict: true
  });
  function required(key: keyof typeof values) {
    const value = values[key];
    assert.ok(value, `Missing --${key}`);
    return value;
  }
  const pins = await pinQwen35Provenance({
    workspaceRoot: fileURLToPath(new URL('../..', import.meta.url)),
    archivePath: required('archive'),
    installedPackagePath: required('installed-package'),
    browserReportPath: required('browser-report'),
    artifactRoot: required('artifact-root'),
    textBuildRoot: required('text-build-root'),
    visionBuildRoot: required('vision-build-root')
  });
  process.stdout.write(`${JSON.stringify(pins, null, 2)}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
