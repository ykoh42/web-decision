import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {constants, createReadStream} from 'node:fs';
import {copyFile, lstat, mkdir, mkdtemp, readFile, readdir, rename, rm, writeFile} from 'node:fs/promises';
import {basename, dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {assertPinnedWebgpuHash} from '../../../packages/web-decision/vite.config.ts';
import {
  loadQwen35CandidateInputs,
  QWEN35_MIXED_VISION_REFERENCE,
  QWEN35_OFFICIAL_VISION_REFERENCE,
  QWEN35_RELEASE_MODEL
} from '../../release/strands-qwen35-release-evidence.ts';

type JsonObject = Record<string, unknown>;
type FilePin = Readonly<{path: string; bytes: number; sha256: string}>;
type ReferenceSpec = Readonly<{sourcePath: string; artifactPath: string; sourceDirectory: string}>;

const workspaceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const assetRoot = 'packages/web-decision/assets/qwen35';
const textReference: ReferenceSpec = {
  sourcePath: 'source-reference.json',
  artifactPath: 'references/qwen35-system-one-source-reference-v1/source-reference.json',
  sourceDirectory: 'scripts/strands/qwen35/reports/fixtures/qwen35-system-one-source-reference-v1'
};
const visionReference: ReferenceSpec = {
  sourcePath: 'vision-reference.json',
  artifactPath: QWEN35_OFFICIAL_VISION_REFERENCE.path,
  sourceDirectory: 'scripts/strands/qwen35/reports/fixtures/qwen35-official-vision-source-reference-r2-2026-11-18'
};
const mixedVisionReference: ReferenceSpec = {
  sourcePath: 'source-reference.json',
  artifactPath: QWEN35_MIXED_VISION_REFERENCE.path,
  sourceDirectory: 'scripts/strands/qwen35/reports/fixtures/qwen35-image-counts-2-3-source-oracle-r1-2026-11-17'
};
const boundaryReference: ReferenceSpec = {
  sourcePath: 'boundary-reference.json',
  artifactPath: 'references/qwen35-system-one-boundary-source-reference-v1/boundary-reference.json',
  sourceDirectory: 'scripts/strands/qwen35/reports/fixtures/qwen35-system-one-boundary-source-reference-v1'
};
const textBuildReportPath =
  'scripts/strands/qwen35/reports/qwen35-inputs-embeds-packed-bf16-full-tvmjs-batch3-2026-11-19.json';
const fp32TextBuildReportPath = 'scripts/strands/qwen35/reports/qwen35-fp32-decoder-build-2026-11-21.json';
const fp32VisionBuildReportPath = 'scripts/strands/qwen35/reports/qwen35-fp32-vision-build-2026-11-21.json';
const visionInputFixturePath =
  'scripts/strands/qwen35/reports/fixtures/qwen35-image-counts-2-3-source-oracle-r1-2026-11-17/input-fixture.json';
const tvmSourcePath = '.toolchains/apache-tvm-src';
const pinnedTvmRevision = '7b5acb32bd271342a24130f4ef439ff6c2e52c6c';

function object(value: unknown, description: string): JsonObject {
  assert.ok(value !== null && typeof value === 'object' && !Array.isArray(value), `${description} must be an object.`);
  return value as JsonObject;
}

function stringField(value: unknown, description: string): string {
  assert.ok(typeof value === 'string' && value.length > 0, `${description} is required.`);
  return value as string;
}

function safeRelativePath(value: unknown, description: string): string {
  const path = stringField(value, description);
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

async function fileMetadata(path: string): Promise<Readonly<{bytes: number; sha256: string}>> {
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    const data = chunk as Buffer;
    bytes += data.byteLength;
    hash.update(data);
  }
  return {bytes, sha256: hash.digest('hex')};
}

async function readRegularFile(root: string, relativePath: string): Promise<Buffer> {
  const path = resolve(root, ...safeRelativePath(relativePath, 'Input path').split('/'));
  const info = await lstat(path);
  assert.ok(info.isFile() && !info.isSymbolicLink(), `Input must be a regular file: ${relativePath}`);
  assert.ok(path.startsWith(`${resolve(root)}/`), `Input escapes its root: ${relativePath}`);
  return readFile(path);
}

async function readJson(root: string, path: string, description: string): Promise<JsonObject> {
  let parsed: unknown;
  try {
    parsed = JSON.parse((await readRegularFile(root, path)).toString('utf8')) as unknown;
  } catch (cause) {
    throw new Error(`${description} is not valid JSON.`, {cause});
  }
  return object(parsed, description);
}

async function copyVerified(
  sourceRoot: string,
  outputRoot: string,
  sourceRelativePath: string,
  outputRelativePath: string,
  expected?: Readonly<{bytes: number; sha256: string}>
): Promise<Readonly<{bytes: number; sha256: string}>> {
  const sourcePath = resolve(sourceRoot, ...safeRelativePath(sourceRelativePath, 'Source path').split('/'));
  const sourceInfo = await lstat(sourcePath);
  assert.ok(
    sourceInfo.isFile() && !sourceInfo.isSymbolicLink(),
    `Source must be a regular file: ${sourceRelativePath}`
  );
  const sourceMetadata = await fileMetadata(sourcePath);
  if (expected) {
    assert.equal(sourceMetadata.bytes, expected.bytes, `Source size differs from its pin: ${sourceRelativePath}`);
    assert.equal(sourceMetadata.sha256, expected.sha256, `Source hash differs from its pin: ${sourceRelativePath}`);
  }

  const outputPath = resolve(outputRoot, ...safeRelativePath(outputRelativePath, 'Output path').split('/'));
  assert.ok(outputPath.startsWith(`${resolve(outputRoot)}/`), `Output escapes its root: ${outputRelativePath}`);
  await mkdir(dirname(outputPath), {recursive: true});
  try {
    await copyFile(sourcePath, outputPath, constants.COPYFILE_EXCL);
  } catch (error) {
    const outputInfo = await lstat(outputPath).catch(() => null);
    if (!outputInfo?.isFile() || outputInfo.isSymbolicLink()) throw error;
    const outputMetadata = await fileMetadata(outputPath);
    assert.deepEqual(outputMetadata, sourceMetadata, `Conflicting copies target ${outputRelativePath}.`);
  }
  const copied = await fileMetadata(outputPath);
  assert.deepEqual(copied, sourceMetadata, `Copied file differs from source: ${outputRelativePath}`);
  return copied;
}

function collectImagePins(value: unknown): FilePin[] {
  const pins = new Map<string, FilePin>();
  function visit(current: unknown): void {
    if (Array.isArray(current)) {
      for (const item of current) visit(item);
      return;
    }
    if (current === null || typeof current !== 'object') return;
    const record = current as JsonObject;
    if (Array.isArray(record.imageFiles)) {
      for (const raw of record.imageFiles) {
        const image = object(raw, 'Image file reference');
        const pin: FilePin = {
          path: safeRelativePath(image.path, 'Image file path'),
          bytes: Number(image.bytes),
          sha256: stringField(image.sha256, 'Image file SHA-256')
        };
        assert.ok(Number.isSafeInteger(pin.bytes) && pin.bytes > 0, `Invalid image size: ${pin.path}`);
        const previous = pins.get(pin.path);
        assert.ok(
          !previous || (previous.bytes === pin.bytes && previous.sha256 === pin.sha256),
          `Conflicting image pins: ${pin.path}`
        );
        pins.set(pin.path, pin);
      }
    }
    for (const [key, item] of Object.entries(record)) if (key !== 'imageFiles') visit(item);
  }
  visit(value);
  return [...pins.values()].sort((left, right) => left.path.localeCompare(right.path));
}

function parseOutputDir(args: readonly string[]): string {
  if (args.length === 1 && args[0] === '--help') {
    console.log(
      'Usage: node --import tsx scripts/strands/qwen35/stage_release_harness.mts --output-dir <new-directory>'
    );
    process.exit(0);
  }
  assert.equal(args.length, 2, 'Expected exactly --output-dir <new-directory>.');
  assert.equal(args[0], '--output-dir', 'Expected --output-dir <new-directory>.');
  return resolve(args[1]!);
}

async function assertPinnedTvmMetadata(
  packageManifest: JsonObject
): Promise<Readonly<{version: string; revision: string; evidence: JsonObject}>> {
  const lockedFiles = object(packageManifest.files, 'Qwen package asset files');
  const textBuild = await readJson(workspaceRoot, textBuildReportPath, 'Qwen R19 TVM build report');
  const relaxImport = object(textBuild.relaxImport, 'Qwen R19 Relax import report');
  const textExecution = object(textBuild.execution, 'Qwen R19 execution report');
  const webgpuCodegen = object(textBuild.webgpuCodegen, 'Qwen R19 WebGPU codegen report');
  const tvmjsArtifact = object(textBuild.tvmjsArtifact, 'Qwen R19 TVM.js artifact report');
  const version = stringField(relaxImport.tvmVersion, 'Text graph TVM version');
  assert.equal(textBuild.status, 'passed-pinned-full-tvmjs-artifact-export-only');
  assert.equal(textExecution.onnxUsed, false, 'Qwen text graph report used ONNX.');
  assert.equal(textExecution.gpuRuntimePerformed, false, 'Qwen R19 export report unexpectedly includes GPU execution.');
  assert.equal(textExecution.tvmjsWasmExportPerformed, true, 'Qwen R19 TVM.js export did not complete.');
  assert.equal(
    webgpuCodegen.deviceExecutionPerformed,
    false,
    'Qwen R19 export report unexpectedly includes device execution.'
  );
  assert.equal(tvmjsArtifact.status, 'artifact-ready-not-run', 'Qwen R19 artifact report has an unexpected status.');
  assert.equal(tvmjsArtifact.parametersEmbedded, false, 'Qwen R19 artifact unexpectedly embeds checkpoint weights.');
  const r19Manifest = await readJson(workspaceRoot, `${assetRoot}/r19/manifest.json`, 'Qwen R19 artifact manifest');
  const r19Source = object(r19Manifest.source, 'Qwen R19 source pins');
  const reportMetadata = await fileMetadata(resolve(workspaceRoot, textBuildReportPath));
  assert.equal(r19Source.importReport, textBuildReportPath, 'R19 manifest points at a different TVM report.');
  assert.equal(r19Source.importReportSha256, reportMetadata.sha256, 'R19 TVM report differs from its artifact pin.');
  const packedImport = object(relaxImport.packedFullTextBf16, 'Qwen R19 packed BF16 import');
  assert.equal(packedImport.tvmUpstreamPatched, false, 'Qwen R19 report records a patched TVM source.');
  assert.equal(packedImport.onnxUsed, false, 'Qwen R19 report records ONNX use.');
  const r19WasmPin = object(
    object(r19Manifest.files, 'Qwen R19 asset files')['qwen35-text-pointer.webgpu.wasm'],
    'Pinned Qwen R19 text graph WASM'
  );
  const textFixture = await readJson(workspaceRoot, `${assetRoot}/r19/fixture.json`, 'Qwen R19 graph fixture');
  const fixtureSourceEvidence = object(textFixture.sourceEvidence, 'Qwen R19 graph source evidence');
  assert.equal(textFixture.tvmVersion, version, 'R19 fixture TVM version differs from its build report.');
  assert.equal(
    textFixture.savedRelaxSha256,
    r19Source.relaxModuleSha256,
    'R19 fixture differs from its saved Relax graph.'
  );
  assert.equal(fixtureSourceEvidence.relaxScriptSha256, r19Source.relaxModuleSha256, 'R19 fixture graph pin changed.');
  const visionFixture = await readJson(workspaceRoot, `${assetRoot}/vision/fixture.json`, 'Qwen vision graph fixture');
  assert.equal(textFixture.activationPrecision, 'float32', 'The current decoder fixture must use FP32.');
  assert.equal(visionFixture.activationPrecision, 'float32', 'The current vision fixture must use FP32.');
  const build = await readJson(workspaceRoot, fp32TextBuildReportPath, 'Qwen FP32 decoder build');
  const lowering = object(textFixture.modelLowering, 'Qwen R19 model lowering');
  assert.equal(build.activationPrecision, 'fp32');
  assert.equal(build.tvm, version);
  assert.equal(build.sourceGraphSha256, r19Source.relaxModuleSha256);
  assert.equal(build.candidateScriptSha256, lowering.candidateScriptSha256);
  assert.equal(build.mainArguments, textFixture.mainParameterCount);
  assert.deepEqual(build.requestArguments, textFixture.runtimeInputNames);
  assert.equal(build.maxSequence, QWEN35_RELEASE_MODEL.maxLength);
  assert.equal(build.maxChunks, lowering.maxChunks);
  assert.equal(build.wasmSha256, r19WasmPin.sha256, 'FP32 decoder build hash differs from its artifact pin.');

  const visionBuild = await readJson(workspaceRoot, fp32VisionBuildReportPath, 'Qwen FP32 vision TVM build report');
  const toolchain = object(visionBuild.toolchain, 'Vision build toolchain');
  const visionExecution = object(visionBuild.execution, 'Vision build execution');
  const artifact = object(visionBuild.tvmJsArtifact, 'Vision TVM.js artifact');
  const visionWasm = object(artifact.wasm, 'Vision TVM WebAssembly artifact');
  const webgpuBuild = object(visionBuild.tvmWebgpuBuild, 'Vision WebGPU build');
  const visionWasmPin = object(
    lockedFiles['vision/qwen35-dynamic-vision.webgpu.wasm'],
    'Pinned Qwen vision graph WASM'
  );
  assert.equal(toolchain['apache-tvm'], version, 'Text and vision graph TVM versions differ.');
  assert.equal(visionExecution.onnxUsed, false, 'Qwen vision graph report used ONNX.');
  assert.equal(artifact.status, 'passed', 'Qwen vision graph artifact report did not pass.');
  assert.equal(webgpuBuild.status, 'passed', 'Qwen vision WebGPU compilation did not pass.');
  assert.equal(webgpuBuild.deviceRuntimeExecuted, false, 'Vision build report includes an unexpected device run.');
  assert.equal(visionWasm.bytes, visionWasmPin.bytes, 'Vision build report size differs from the package asset lock.');
  assert.equal(
    visionWasm.sha256,
    visionWasmPin.sha256,
    'Vision build report hash differs from the package asset lock.'
  );
  assert.equal(visionBuild.format, 'qwen35-fp32-vision-webgpu-build-v1');
  assert.equal(visionBuild.activationPrecision, 'float32');
  assert.equal(visionBuild.tvmCommit, pinnedTvmRevision);
  assert.equal(visionBuild.replacedKernel, 'round_bf16_webgpu');
  assert.equal(visionBuild.vmSha256, 'be2f5f6209bc1c79b8fd2f1eba540fe0ab21c1afd60460de893e277f54d86a5f');
  assert.equal(visionFixture.tvmSourcePatched, false, 'Vision fixture records a patched TVM source.');
  assert.equal(visionFixture.onnxUsed, false, 'Vision fixture records ONNX use.');
  const buildEnv = (await readRegularFile(workspaceRoot, 'scripts/strands/qwen35/build-env/pyproject.toml')).toString(
    'utf8'
  );
  assert.ok(
    buildEnv.includes(`"apache-tvm==${version}"`),
    'Qwen build environment does not pin the reported TVM version.'
  );

  const tvmRoot = resolve(workspaceRoot, tvmSourcePath);
  const currentRevision = execFileSync('git', ['-C', tvmRoot, 'rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  const sourceChanges = execFileSync('git', ['-C', tvmRoot, 'status', '--porcelain'], {encoding: 'utf8'}).trim();
  assert.equal(currentRevision, pinnedTvmRevision, 'Pinned TVM checkout does not match the graph build report.');
  assert.equal(sourceChanges, '', 'Pinned TVM checkout has local changes.');
  assertPinnedWebgpuHash(sha256(await readRegularFile(tvmRoot, 'web/src/webgpu.ts')));

  return {
    version,
    revision: pinnedTvmRevision,
    evidence: {
      textBuildReportPath: 'evidence/qwen35-r19-tvm-build.json',
      textBuildReportSha256: sha256(await readRegularFile(workspaceRoot, fp32TextBuildReportPath)),
      visionBuildReportPath: 'evidence/qwen35-vision-tvm-build.json',
      visionBuildReportSha256: sha256(await readRegularFile(workspaceRoot, fp32VisionBuildReportPath))
    }
  };
}

async function stageReleaseHarness(outputDir: string): Promise<void> {
  await mkdir(dirname(outputDir), {recursive: true});
  await lstat(outputDir).then(
    () => assert.fail(`Output directory already exists: ${outputDir}`),
    () => undefined
  );
  const stagingDir = await mkdtemp(join(dirname(outputDir), `.${basename(outputDir)}-`));
  let published = false;
  try {
    const packageManifestBytes = await readRegularFile(workspaceRoot, `${assetRoot}/manifest.json`);
    const packageManifest = object(
      JSON.parse(packageManifestBytes.toString('utf8')) as unknown,
      'Qwen package asset lock'
    );
    assert.equal(packageManifest.modelId, QWEN35_RELEASE_MODEL.catalogModelId, 'Qwen asset lock model ID mismatch.');
    assert.equal(packageManifest.repository, QWEN35_RELEASE_MODEL.repository, 'Qwen asset lock repository mismatch.');
    assert.equal(packageManifest.revision, QWEN35_RELEASE_MODEL.revision, 'Qwen asset lock revision mismatch.');
    const assetFiles = object(packageManifest.files, 'Qwen package asset files');
    const stagedAssets: Record<string, Readonly<{bytes: number; sha256: string}>> = {};
    for (const [assetPath, rawPin] of Object.entries(assetFiles).sort(([left], [right]) => left.localeCompare(right))) {
      const pin = object(rawPin, `Qwen asset pin ${assetPath}`);
      const metadata = {
        bytes: Number(pin.bytes),
        sha256: stringField(pin.sha256, `Qwen asset SHA-256 ${assetPath}`)
      };
      assert.ok(Number.isSafeInteger(metadata.bytes) && metadata.bytes >= 0, `Invalid Qwen asset size: ${assetPath}`);
      const outputPath = `assets/qwen35/${safeRelativePath(assetPath, 'Qwen asset path')}`;
      stagedAssets[outputPath] = await copyVerified(
        workspaceRoot,
        stagingDir,
        `${assetRoot}/${assetPath}`,
        outputPath,
        metadata
      );
    }
    const assetLockRawSha256 = sha256(packageManifestBytes);
    await copyVerified(workspaceRoot, stagingDir, `${assetRoot}/manifest.json`, 'assets/qwen35/manifest.json', {
      bytes: packageManifestBytes.byteLength,
      sha256: assetLockRawSha256
    });

    const r19ManifestBytes = await readRegularFile(workspaceRoot, `${assetRoot}/r19/manifest.json`);
    const r19Manifest = object(JSON.parse(r19ManifestBytes.toString('utf8')) as unknown, 'Qwen R19 asset manifest');
    const r19Files = object(r19Manifest.files, 'Qwen R19 asset files');
    for (const [assetPath, rawPin] of Object.entries(r19Files).sort(([left], [right]) => left.localeCompare(right))) {
      const pin = object(rawPin, `Qwen R19 asset pin ${assetPath}`);
      const metadata = {
        bytes: Number(pin.bytes),
        sha256: stringField(pin.sha256, `Qwen R19 asset SHA-256 ${assetPath}`)
      };
      assert.ok(
        Number.isSafeInteger(metadata.bytes) && metadata.bytes >= 0,
        `Invalid Qwen R19 asset size: ${assetPath}`
      );
      const relativePath = `r19/${safeRelativePath(assetPath, 'Qwen R19 asset path')}`;
      const outputPath = `assets/qwen35/${relativePath}`;
      stagedAssets[outputPath] = await copyVerified(
        workspaceRoot,
        stagingDir,
        `${assetRoot}/${relativePath}`,
        outputPath,
        metadata
      );
    }
    const r19ManifestPath = 'assets/qwen35/r19/manifest.json';
    stagedAssets[r19ManifestPath] = await copyVerified(
      workspaceRoot,
      stagingDir,
      `${assetRoot}/r19/manifest.json`,
      r19ManifestPath,
      {bytes: r19ManifestBytes.byteLength, sha256: sha256(r19ManifestBytes)}
    );

    const references = [textReference, visionReference, mixedVisionReference, boundaryReference] as const;
    const referenceValues: JsonObject[] = [];
    const referenceEntries: Record<string, Readonly<{path: string; sha256: string}>> = {};
    for (const [index, spec] of references.entries()) {
      const source = await readRegularFile(workspaceRoot, `${spec.sourceDirectory}/${spec.sourcePath}`);
      const digest = sha256(source);
      if (spec === mixedVisionReference)
        assert.equal(digest, QWEN35_MIXED_VISION_REFERENCE.sha256, 'Pinned mixed-vision reference hash mismatch.');
      if (spec === visionReference)
        assert.equal(
          digest,
          QWEN35_OFFICIAL_VISION_REFERENCE.sha256,
          'Pinned official vision reference hash mismatch.'
        );
      const parsed = object(JSON.parse(source.toString('utf8')) as unknown, `Source reference ${spec.sourcePath}`);
      referenceValues.push(parsed);
      await copyVerified(workspaceRoot, stagingDir, `${spec.sourceDirectory}/${spec.sourcePath}`, spec.artifactPath, {
        bytes: source.byteLength,
        sha256: digest
      });
      const name = ['text', 'vision', 'mixedVision', 'boundaries'][index]!;
      referenceEntries[name] = {path: spec.artifactPath, sha256: digest};
      for (const pin of collectImagePins(parsed))
        await copyVerified(workspaceRoot, stagingDir, `${spec.sourceDirectory}/${pin.path}`, pin.path, pin);
    }

    const boundary = referenceValues[3]!;
    const sourceRuns = object(boundary.sourceRuns, 'Boundary source runs');
    for (const [runName, rawRun] of Object.entries(sourceRuns).sort(([left], [right]) => left.localeCompare(right))) {
      const run = object(rawRun, `Boundary source run ${runName}`);
      for (const [pathKey, shaKey] of [
        ['reportPath', 'reportSha256'],
        ['supervisorPath', 'supervisorSha256']
      ] as const) {
        const runPath = safeRelativePath(run[pathKey], `Boundary ${runName} ${pathKey}`);
        await copyVerified(`${workspaceRoot}/${boundaryReference.sourceDirectory}`, stagingDir, runPath, runPath, {
          bytes: Number((await fileMetadata(resolve(workspaceRoot, boundaryReference.sourceDirectory, runPath))).bytes),
          sha256: stringField(run[shaKey], `Boundary ${runName} ${shaKey}`)
        });
      }
    }

    const mixedInputBytes = await readRegularFile(workspaceRoot, visionInputFixturePath);
    const mixedInputRelative = 'references/qwen35-image-counts-2-3-source-oracle-r1-2026-11-17/input-fixture.json';
    await copyVerified(workspaceRoot, stagingDir, visionInputFixturePath, mixedInputRelative, {
      bytes: mixedInputBytes.byteLength,
      sha256: sha256(mixedInputBytes)
    });

    const tvm = await assertPinnedTvmMetadata(packageManifest);
    await copyVerified(workspaceRoot, stagingDir, fp32TextBuildReportPath, tvm.evidence.textBuildReportPath as string);
    await copyVerified(
      workspaceRoot,
      stagingDir,
      fp32VisionBuildReportPath,
      tvm.evidence.visionBuildReportPath as string
    );
    const manifest = {
      format: 'web-decision-qwen35-generated-artifact-manifest-v1',
      model: QWEN35_RELEASE_MODEL.modelId,
      backend: 'tvm-webgpu',
      runtime: {
        tvm: tvm.version,
        tvmSourceRevision: tvm.revision,
        tvmPatches: [],
        onnxImportOrExportUsed: false
      },
      files: stagedAssets,
      sourceEvidence: {
        assetLockPath: 'assets/qwen35/manifest.json',
        assetLockSha256: assetLockRawSha256,
        assetLockCanonicalSha256: QWEN35_RELEASE_MODEL.assetManifestSha256,
        tvmBuildReports: tvm.evidence,
        mixedVisionInputFixture: {
          path: mixedInputRelative,
          sha256: sha256(mixedInputBytes)
        }
      }
    };
    const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
    await writeFile(join(stagingDir, 'generated-artifact-manifest.json'), manifestBytes, {flag: 'wx'});
    const textSource = object(referenceValues[0]!.source, 'Pinned text source metadata');
    const releaseHarness = {
      schemaVersion: 1,
      modelId: QWEN35_RELEASE_MODEL.modelId,
      source: textSource,
      manifest: {
        path: 'generated-artifact-manifest.json',
        sha256: sha256(manifestBytes)
      },
      references: referenceEntries,
      sourceEvidence: {
        mixedVisionInputFixture: {
          path: mixedInputRelative,
          sha256: sha256(mixedInputBytes)
        }
      }
    };
    await writeFile(join(stagingDir, 'release-harness.json'), `${JSON.stringify(releaseHarness, null, 2)}\n`, {
      flag: 'wx'
    });

    const loaded = await loadQwen35CandidateInputs(stagingDir);
    assert.equal(loaded.visionCases.length > 0, true, 'Staged Qwen vision cases are empty.');
    assert.equal(Object.keys(loaded.manifestFiles).length, Object.keys(stagedAssets).length);
    assert.equal(loaded.manifestRawSha256, sha256(manifestBytes));
    const outputMetadata = await readdir(stagingDir);
    assert.ok(outputMetadata.includes('release-harness.json'));
    await rename(stagingDir, outputDir);
    published = true;
    console.log(
      `QWEN35_RELEASE_HARNESS_READY:${JSON.stringify({
        outputDir,
        modelId: QWEN35_RELEASE_MODEL.catalogModelId,
        manifestSha256: loaded.manifestSha256,
        assetManifestSha256: loaded.assetManifestSha256,
        assets: Object.keys(loaded.manifestFiles).length,
        textCases: loaded.sourceCases.length,
        visionCases: loaded.visionCases.length,
        boundaryCases: Object.keys(loaded.boundaryCases).length,
        sourceRuns: Object.keys(sourceRuns).length
      })}`
    );
  } finally {
    if (!published) await rm(stagingDir, {recursive: true, force: true});
  }
}

const outputDir = parseOutputDir(process.argv.slice(2));
await stageReleaseHarness(outputDir);
