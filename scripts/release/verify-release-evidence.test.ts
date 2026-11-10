import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {copyFile, link, mkdir, mkdtemp, readFile, realpath, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {
  canonicalSha256,
  loadPublicCatalog,
  readArchiveModelManifests,
  BASELINE_REFERENCE,
  BASELINE_EXPECTED_OUTPUT,
  UPSTREAM_REFERENCE,
  RELEASE_EVIDENCE_POLICY_SHA256,
  sha256,
  validateArtifactManifest,
  validatePlaywrightReport,
  validatePolicy,
  validatePolicyBytes,
  verifyArchiveInstall,
  collectReleaseEvidenceInputs,
  collectKananaReleaseEvidenceInputs,
  collectReleaseEvidenceInputsForTestFixture,
  verifyReleaseArtifactsForTestFixture,
  verifyReleaseEvidence
} from './verify-release-evidence.ts';
import {
  validateKananaEvidencePolicy,
  validateKananaManifestBytes,
  validateKananaPlaywrightReport,
  verifyKananaTvmBuildEvidence,
  type KananaReleaseEvidenceInputs,
  type KananaVerificationContext
} from './kanana-release-evidence.ts';
import {
  createExpandedInferenceEvidence,
  loadExpandedInferenceFixture,
  type ExpandedInferenceFixture,
  type ExpandedInferenceObservation,
  type TypedReferenceRun,
  type TypedReferenceAnswer
} from '../../tests/browser/kev06-expanded-inference.ts';
import type {
  KevSystemOneAnswer,
  KevSystemOneQuestion,
  KevSystemOneResult
} from '../../packages/web-decision/src/index.d.ts';

type AnyRecord = Record<string, any>;

const TEST_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const WORKSPACE_ROOT = resolve(process.env.WEB_DECISION_WORKSPACE_ROOT ?? resolve(TEST_DIRECTORY, '../..'));
const POLICY_PATH = join(WORKSPACE_ROOT, 'scripts/release/release-evidence-policy.json');
const POLICY = JSON.parse(await readFile(POLICY_PATH, 'utf8'));
const TAR = process.platform === 'win32' ? 'tar.exe' : 'tar';

function required<T>(value: T | undefined, description: string): T {
  if (value === undefined) throw new Error(`${description} is missing.`);
  return value;
}

function makeDescriptor(modelId: string, salt: string): AnyRecord {
  const manifest = {
    formatVersion: 1,
    model: modelId,
    backend: 'tvm-webgpu',
    files: {
      'model.wasm': {sha256: salt.repeat(64), bytes: 17},
      'weights/model.bin': {sha256: salt.repeat(64), bytes: 23}
    }
  };
  const manifestPath = `assets/${modelId}/manifest.json`;
  const publicDescription = {
    backend: {name: 'tvm-webgpu', requiredFeatures: ['shader-f16'], requiredLimits: {maxBufferSize: 1024}},
    assets: {integrity: 'pinned-manifest', location: 'package-and-pinned-upstream'}
  };
  const artifacts = {
    manifest: {
      sourcePath: manifestPath,
      canonicalSha256: canonicalSha256(manifest),
      files: manifest.files
    },
    bundledRuntimeFiles: [
      {manifestName: 'model.wasm', sourcePath: `assets/${modelId}/model.wasm`, ...manifest.files['model.wasm']}
    ]
  };
  return {
    modelId,
    description: publicDescription,
    artifacts,
    publicDescriptor: {
      modelId,
      ...publicDescription,
      assets: {
        ...publicDescription.assets,
        manifestSha256: artifacts.manifest.canonicalSha256,
        requiredFiles: ['manifest.json', ...Object.keys(manifest.files)]
      }
    },
    manifest
  };
}

function apiSource(descriptors: AnyRecord[]): string {
  const publicDescriptors = descriptors.map(({publicDescriptor}) => publicDescriptor);
  return (
    `const entries = ${JSON.stringify(publicDescriptors)};\n` +
    'export function listModels() { return entries; }\n' +
    'export function describeModel(modelId) { return entries.find((entry) => entry.modelId === modelId); }\n'
  );
}

async function putPackageTree(
  root: string,
  descriptors: AnyRecord[]
): Promise<{packageRoot: string; sourceRoot: string}> {
  const packageRoot = join(root, 'package');
  const sourceRoot = join(root, 'source-package');
  await mkdir(join(packageRoot, 'src', 'catalog'), {recursive: true});
  await mkdir(join(packageRoot, 'dist'), {recursive: true});
  await mkdir(join(sourceRoot, 'src', 'catalog'), {recursive: true});
  await mkdir(join(sourceRoot, 'dist'), {recursive: true});
  const packageJson = {
    name: 'web-decision',
    version: '9.8.7',
    type: 'module',
    exports: {'.': {import: './dist/index.js'}}
  };
  const source = apiSource(descriptors);
  await writeFile(join(packageRoot, 'package.json'), `${JSON.stringify(packageJson)}\n`);
  await writeFile(join(packageRoot, 'dist/index.js'), source);
  await writeFile(join(sourceRoot, 'dist/index.js'), source);
  await writeFile(
    join(sourceRoot, 'src/catalog/model-spec.json'),
    `${JSON.stringify({
      schemaVersion: 1,
      models: descriptors.map(({modelId, description}) => ({modelId, description}))
    })}\n`
  );
  await writeFile(
    join(sourceRoot, 'src/catalog/artifacts.lock.json'),
    `${JSON.stringify({
      schemaVersion: 1,
      byModelId: Object.fromEntries(descriptors.map(({modelId, artifacts}) => [modelId, artifacts]))
    })}\n`
  );
  const provenanceRoot = join(sourceRoot, 'src/catalog/provenance');
  await mkdir(provenanceRoot, {recursive: true});
  await copyFile(
    join(WORKSPACE_ROOT, 'scripts/release/fixtures/model-provenance.json'),
    join(sourceRoot, 'src/catalog/model-provenance.json')
  );
  await copyFile(
    join(WORKSPACE_ROOT, 'scripts/release/fixtures/model-provenance-test-fixture.json'),
    join(provenanceRoot, 'release-test-fixture.json')
  );
  for (const descriptor of descriptors) {
    const manifestPath = join(packageRoot, 'assets', descriptor.modelId, 'manifest.json');
    await mkdir(dirname(manifestPath), {recursive: true});
    await writeFile(manifestPath, `${JSON.stringify(descriptor.manifest, null, 2)}\n`);
  }
  return {packageRoot, sourceRoot};
}

interface Fixture {
  root: string;
  descriptors: AnyRecord[];
  packageRoot: string;
  sourceRoot: string;
  archivePath: string;
}

async function makeFixture(): Promise<Fixture> {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-release-evidence-')));
  const descriptors = [makeDescriptor('kev-0.6b-q4f16', 'a')];
  const {packageRoot, sourceRoot} = await putPackageTree(root, descriptors);
  const representativeFixturePath = join(root, POLICY.expandedInference.fixturePath);
  await mkdir(dirname(representativeFixturePath), {recursive: true});
  await copyFile(join(WORKSPACE_ROOT, POLICY.expandedInference.fixturePath), representativeFixturePath);
  const archivePath = join(root, 'web-decision-9.8.7.tgz');
  execFileSync(TAR, ['-czf', archivePath, '-C', root, 'package']);
  return {root, descriptors, packageRoot, sourceRoot, archivePath};
}

function makeRecord({model, browser, archive, catalogSha256, manifest, expandedInference}: AnyRecord): AnyRecord {
  const asset = validateArtifactManifest(model, manifest);
  const output = {
    selected: BASELINE_EXPECTED_OUTPUT.selected,
    optionOrder: [...BASELINE_EXPECTED_OUTPUT.optionOrder],
    logits: [...BASELINE_EXPECTED_OUTPUT.logits],
    probabilities: [...BASELINE_EXPECTED_OUTPUT.probabilities]
  };
  return {
    schemaVersion: 2,
    testKind: 'real-model-webgpu',
    catalogSha256,
    model: {
      requestedId: model.modelId,
      loadedId: model.modelId,
      describedId: model.modelId,
      assetManifestSha256: model.manifestSha256
    },
    artifact: {
      modelId: model.modelId,
      manifestPath: model.manifestPath,
      manifestSha256: asset.manifestSha256,
      filesSha256: asset.filesSha256,
      verifiedFiles: Object.entries(model.manifestFiles).map(([path, metadata]) => ({
        path,
        ...(metadata as AnyRecord)
      })),
      manifest
    },
    archive: {
      sha256: archive.archiveSha256,
      packageTreeSha256: archive.packageTreeSha256,
      bundleSha256: archive.bundleSha256,
      installedPackagePathSha256: archive.installedPackagePathSha256,
      entryPathSha256: archive.entryPathSha256,
      packageName: archive.packageName,
      packageVersion: archive.packageVersion
    },
    browser: {name: browser.name, version: '1.0.0', userAgent: `${browser.name} real GPU browser`},
    device: {
      manufacturer: 'Example Hardware Vendor',
      model: 'Example Physical GPU Host',
      os: 'Example OS',
      osVersion: '1.0',
      architecture: 'x86_64',
      isVirtual: false,
      evidence: {
        host: {
          source: 'windows-wmi/Win32_ComputerSystem',
          raw: {Manufacturer: 'Example Hardware Vendor', Model: 'Example Physical GPU Host'}
        },
        os: {
          source: 'node:os',
          raw: {version: 'Example OS', release: '1.0', architecture: 'x86_64'}
        }
      }
    },
    gpu: {
      api: 'webgpu',
      isFallbackAdapter: false,
      adapterInfo: {source: 'navigator.gpu.requestAdapter().info', value: {}},
      hardware: {
        manufacturer: 'Example GPU Vendor',
        model: 'Example GPU Device',
        driverVersion: '1.2.3',
        pnpDeviceId: 'PCI\\VEN_0000&DEV_0000'
      },
      hardwareEvidence: {
        source: 'windows-wmi/Win32_VideoController',
        raw: {
          AdapterCompatibility: 'Example GPU Vendor',
          Name: 'Example GPU Device',
          DriverVersion: '1.2.3',
          PNPDeviceID: 'PCI\\VEN_0000&DEV_0000'
        }
      },
      features: ['shader-f16'],
      limits: {maxBufferSize: 2048}
    },
    inference: {
      online: {
        completed: true,
        modelId: model.modelId,
        backend: 'tvm-webgpu',
        assetManifestSha256: model.manifestSha256,
        referenceMatched: true,
        ...structuredClone(output)
      },
      offline: {
        completed: true,
        modelId: model.modelId,
        backend: 'tvm-webgpu',
        assetManifestSha256: model.manifestSha256,
        matchesOnline: true,
        networkAttempts: [],
        ...structuredClone(output)
      },
      reference: BASELINE_REFERENCE,
      upstreamReference: {matched: true, ...UPSTREAM_REFERENCE}
    },
    expandedInference
  };
}

async function prepareValidReport(fixture: Fixture): Promise<AnyRecord> {
  const inputs = await collectReleaseEvidenceInputsForTestFixture({
    archivePath: fixture.archivePath,
    installedPackagePath: fixture.packageRoot,
    sourcePackageRoot: fixture.sourceRoot
  });
  const {archive, models, manifests, catalogSha256} = inputs;
  const expandedPolicy = validatePolicy(POLICY).expandedInference;
  const expanded = loadExpandedInferenceFixture(fixture.root, expandedPolicy);
  const tests = [];

  for (const model of models) {
    for (const browser of POLICY.browsers) {
      const record = makeRecord({
        model,
        browser,
        archive,
        catalogSha256,
        manifest: required(manifests.get(model.modelId), `Manifest for ${model.modelId}`).manifest,
        expandedInference: createExpandedInferenceEvidence(
          expanded.fixture,
          expanded.fixtureSha256,
          model.manifestSha256,
          fakeExpandedInferenceObservation(expanded.fixture)
        )
      });
      tests.push({
        file: POLICY.testFile,
        title: `real model WebGPU ${model.modelId}`,
        tests: [
          {
            expectedStatus: 'passed',
            status: 'expected',
            projectName: browser.project,
            results: [
              {
                status: 'passed',
                errors: [],
                stdout: [`${POLICY.evidencePrefix}${JSON.stringify(record)}\n`]
              }
            ]
          }
        ]
      });
    }
  }

  return {
    archive,
    models,
    catalogSha256,
    manifests,
    report: {
      config: {rootDir: fixture.root},
      errors: [],
      stats: {expected: tests.length, skipped: 0, unexpected: 0, flaky: 0},
      suites: [{title: 'release gpu evidence', specs: tests}]
    }
  };
}

function fakeExpandedInferenceObservation(fixture: ExpandedInferenceFixture): ExpandedInferenceObservation {
  const decisionRuns = fixture.decisionCases.map((item) => {
    const probabilities = item.reference.probabilities;
    const selectedIndex = maximumIndex(probabilities);
    return {
      id: item.id,
      tokenCount: item.tokenCount,
      selected: item.reference.options[selectedIndex]!,
      selectedIndex,
      probabilities: item.reference.options.map((option, index) => ({
        option,
        logit: item.reference.logits[index]!,
        probability: probabilities[index]!
      }))
    };
  });
  const typed = fixture.typedSystemOne;
  const single = Object.fromEntries(
    Object.entries(typed.single).map(([name, reference]) => [
      name,
      fakeTypedRun(reference, {[name]: typed.questions[name]!})
    ])
  );
  return {
    decisionRuns,
    typedSystemOne: {
      packed: fakeTypedRun(typed.packed, typed.questions),
      single
    }
  } as ExpandedInferenceObservation;
}

function fakeTypedRun(
  reference: TypedReferenceRun,
  questions: Readonly<Record<string, KevSystemOneQuestion>>
): KevSystemOneResult<'kev-0.6b-q4f16'> {
  const answerEntries = Object.entries(questions).map(
    ([name, question]) => [name, fakeTypedAnswer(question, reference.answers[name]!)] as const
  );
  return {
    model: 'kev-0.6b-q4f16',
    tokenCount: reference.tokenCount,
    answers: Object.fromEntries(answerEntries),
    answerEntries
  } as KevSystemOneResult<'kev-0.6b-q4f16'>;
}

function fakeTypedAnswer(question: KevSystemOneQuestion, reference: TypedReferenceAnswer): KevSystemOneAnswer {
  if (question.type === 'choice') {
    const keys = Object.keys(question.criteria);
    const entries = keys.map((key, index) => [key, publicRound(reference.probabilities[index]!)] as const);
    return {
      type: 'choice',
      choice: keys[maximumIndex(entries.map(([, probability]) => probability))]!,
      confidence: 0.5,
      probabilities: Object.fromEntries(entries),
      probabilityEntries: entries
    };
  }
  if (question.type === 'noul') {
    return {type: 'noul', noul: publicRound(reference.probabilities[1]!)};
  }
  const entries = question.criteria.map(
    (_, index) => [String(index), publicRound(reference.probabilities[index]!)] as const
  );
  return {
    type: 'score',
    score: 0,
    confidence: 0.5,
    legend: Object.fromEntries(question.criteria.map((label, index) => [String(index), String(label)])),
    probabilities: Object.fromEntries(entries),
    probabilityEntries: entries
  };
}

function publicRound(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

function maximumIndex(values: readonly number[]): number {
  let index = 0;
  for (let candidate = 1; candidate < values.length; candidate++) {
    if (values[candidate]! > values[index]!) index = candidate;
  }
  return index;
}

async function saveReport(fixture: Fixture, report: AnyRecord): Promise<string> {
  const path = join(fixture.root, 'playwright-report.json');
  await writeFile(path, `${JSON.stringify(report)}\n`);
  return path;
}

function updateEvidenceRecord(report: AnyRecord, mutate: (record: AnyRecord) => void): void {
  const result = required(report.suites[0].specs[0].tests[0].results[0], 'Playwright result');
  const stdout = required(result.stdout[0], 'Playwright evidence stdout');
  const record = JSON.parse(stdout.slice(POLICY.evidencePrefix.length));
  mutate(record);
  result.stdout[0] = `${POLICY.evidencePrefix}${JSON.stringify(record)}\n`;
}

async function makeKananaVerificationContext(): Promise<KananaVerificationContext> {
  const policy = validateKananaEvidencePolicy(POLICY.kanana);
  const manifestBytes = await readFile(join(WORKSPACE_ROOT, 'packages/web-decision', policy.model.manifestPath));
  const manifest = {
    path: policy.model.manifestPath,
    ...validateKananaManifestBytes(manifestBytes, policy)
  };
  const fixtureBytes = await readFile(join(WORKSPACE_ROOT, policy.referenceFixture.path));
  assert.equal(sha256(fixtureBytes), policy.referenceFixture.sha256);
  const archive = {
    archivePath: join(tmpdir(), 'pinned-web-decision-0.4.0.tgz'),
    archiveSha256: policy.archive.sha256,
    installedRoot: join(tmpdir(), 'pinned-web-decision-0.4.0'),
    packageName: policy.archive.packageName,
    packageVersion: policy.archive.packageVersion,
    packageTreeSha256: '1'.repeat(64),
    bundlePath: join(tmpdir(), 'pinned-web-decision-0.4.0', 'dist/index.js'),
    bundleSha256: '2'.repeat(64),
    installedPackagePathSha256: '3'.repeat(64),
    entryPathSha256: '4'.repeat(64),
    archiveFiles: [policy.model.manifestPath]
  };
  return {
    archive,
    manifest,
    policy,
    referenceFixture: JSON.parse(fixtureBytes.toString('utf8')),
    workspaceRoot: WORKSPACE_ROOT
  };
}

async function makeKananaBuildFixture() {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-kanana-build-evidence-'));
  const installedRoot = join(root, 'installed');
  const sourceRoot = join(root, 'source');
  const buildRoot = join(root, 'independent-build');
  const evidencePath = join(root, 'evidence', 'tvm-builds.json');
  const artifactInputs = [
    {artifact: 'kanana-text-model.wasm', graph: 'text', bytes: Buffer.from('independent text TVM build')},
    {artifact: 'kanana-vision-model.wasm', graph: 'vision', bytes: Buffer.from('independent vision TVM build')}
  ] as const;
  const files: Record<string, {bytes: number; sha256: string}> = {};
  const builds: Array<Record<string, string>> = [];
  const reports = new Map<string, Record<string, unknown>>();
  for (const item of artifactInputs) {
    const packageArtifact = join(installedRoot, 'assets/kanana', item.artifact);
    const artifactBuildRoot = join(buildRoot, item.graph);
    const wasmPath = join(artifactBuildRoot, 'rebuilt.wasm');
    const reportPath = join(artifactBuildRoot, 'report.json');
    const digest = sha256(item.bytes);
    const report = {
      evidenceSchema: 'web-decision-tvm-build-v1',
      modelId: 'kanana-2-v-decide-3b-2611',
      artifact: item.artifact,
      wasm: 'rebuilt.wasm',
      status: 'success',
      target: 'webgpu / wasm32',
      tvmVersion: '0.27.0',
      embeddedBuildRoot: false,
      wasmBytes: item.bytes.length,
      wasmSha256: digest,
      repository: 'kakaocorp/kanana-2-v-decide-3b-2611',
      modelRevision: '12820d654ed9789a0f7f48daeee1703c3483c464',
      checkpointSha256: 'a'.repeat(64),
      graph: item.graph,
      buildBackend: 'native-tvm',
      weightSource: 'pinned-safetensors',
      onnxImportOrExportUsed: false
    };
    await mkdir(dirname(packageArtifact), {recursive: true});
    await mkdir(artifactBuildRoot, {recursive: true});
    await writeFile(packageArtifact, item.bytes);
    await writeFile(wasmPath, item.bytes);
    await writeFile(reportPath, JSON.stringify(report));
    files[item.artifact] = {bytes: item.bytes.length, sha256: digest};
    reports.set(item.artifact, report);
    builds.push({
      modelId: 'kanana-2-v-decide-3b-2611',
      artifact: item.artifact,
      buildRoot: `../independent-build/${item.graph}`,
      wasm: 'rebuilt.wasm',
      report: 'report.json'
    });
  }
  await mkdir(sourceRoot, {recursive: true});
  await mkdir(dirname(evidencePath), {recursive: true});
  await writeFile(evidencePath, JSON.stringify({schemaVersion: 1, builds}));
  const inputs = {
    archive: {
      archivePath: join(root, 'web-decision-0.4.0.tgz'),
      archiveSha256: 'b'.repeat(64),
      installedRoot,
      packageName: 'web-decision',
      packageVersion: '0.4.0',
      packageTreeSha256: 'c'.repeat(64),
      bundlePath: join(installedRoot, 'dist/index.js'),
      bundleSha256: 'd'.repeat(64),
      installedPackagePathSha256: 'e'.repeat(64),
      entryPathSha256: 'f'.repeat(64),
      archiveFiles: ['assets/kanana/manifest.json']
    },
    policy: {
      model: {
        modelId: 'kanana-2-v-decide-3b-2611',
        repository: 'kakaocorp/kanana-2-v-decide-3b-2611',
        revision: '12820d654ed9789a0f7f48daeee1703c3483c464',
        checkpointSha256: 'a'.repeat(64)
      }
    },
    manifest: {
      path: 'assets/kanana/manifest.json',
      manifest: {runtime: {tvm: '0.27.0'}},
      files
    }
  } as unknown as KananaReleaseEvidenceInputs;
  return {root, sourceRoot, installedRoot, buildRoot, evidencePath, inputs, builds, reports};
}

async function writeKananaBuildEvidence(path: string, builds: Array<Record<string, string>>): Promise<void> {
  await writeFile(path, JSON.stringify({schemaVersion: 1, builds}));
}

function kananaResult(context: KananaVerificationContext): AnyRecord {
  const expected = context.referenceFixture.result.answers.delivery_status;
  if (!expected) throw new Error('Kanana reference fixture is missing delivery_status.');
  return {
    modelId: context.policy.model.modelId,
    descriptionModelId: context.policy.model.modelId,
    checkpointRevision: context.policy.model.revision,
    checkpointSha256: context.policy.model.checkpointSha256,
    backend: 'tvm-webgpu',
    assetManifestSha256: context.manifest.manifestSha256,
    choice: expected.choice,
    confidence: expected.confidence,
    probabilities: structuredClone(expected.probabilities),
    probabilityEntries: Object.entries(expected.probabilities),
    usage: {input_tokens: 10, output_tokens: 0}
  };
}

function makeKananaEvidenceRecord(
  context: KananaVerificationContext,
  kind: keyof KananaVerificationContext['policy']['tests']
): AnyRecord {
  const {archive, manifest, policy} = context;
  const imageProbabilities = {rectangle: 0.9, circle: 0.1};
  const inference =
    kind === 'image'
      ? {
          ...kananaResult(context),
          answer: 'rectangle',
          confidence: imageProbabilities.rectangle - imageProbabilities.circle,
          probabilities: imageProbabilities,
          probabilityEntries: Object.entries(imageProbabilities)
        }
      : kind === 'offline'
        ? {online: kananaResult(context), offline: kananaResult(context)}
        : kananaResult(context);
  return {
    schemaVersion: 1,
    kind: policy.tests[kind].kind,
    archive: {
      sha256: archive.archiveSha256,
      packageTreeSha256: archive.packageTreeSha256,
      bundleSha256: archive.bundleSha256,
      installedPackagePathSha256: archive.installedPackagePathSha256,
      entryPathSha256: archive.entryPathSha256,
      packageName: archive.packageName,
      packageVersion: archive.packageVersion
    },
    artifact: {
      modelId: policy.model.modelId,
      manifestPath: manifest.path,
      manifestRawSha256: manifest.rawSha256,
      manifestSha256: manifest.manifestSha256,
      filesSha256: manifest.filesSha256,
      files: manifest.files,
      runtimeArtifacts: policy.tests[kind].runtimeArtifacts.map((path) => ({path, ...manifest.files[path]}))
    },
    model: {
      requestedId: policy.model.modelId,
      loadedId: policy.model.modelId,
      describedId: policy.model.modelId,
      checkpointRevision: policy.model.revision,
      checkpointSha256: policy.model.checkpointSha256
    },
    browser: {name: policy.browser.name, version: '158.0', userAgent: 'Chromium real WebGPU'},
    gpu: {
      api: 'webgpu',
      isFallbackAdapter: false,
      adapterInfo: {
        source: 'navigator.gpu.requestAdapter() and GPUAdapterInfo',
        value: {vendor: 'apple', architecture: 'apple-gpu', isFallbackAdapter: false}
      },
      features: [],
      limits: structuredClone(policy.gpu.requiredLimits)
    },
    ...(kind === 'text'
      ? {referenceFixtureSha256: policy.referenceFixture.sha256, probabilityTolerance: policy.textProbabilityTolerance}
      : {}),
    ...(kind === 'image'
      ? {input: {kind: 'synthetic-canvas-v1', width: 256, height: 256, sha256: '5'.repeat(64)}}
      : {}),
    ...(kind === 'offline'
      ? {
          referenceFixtureSha256: policy.referenceFixture.sha256,
          probabilityTolerance: policy.textProbabilityTolerance,
          cache: {primedOnline: true, requestedOffline: true, blockedNetwork: true, fetchedRanges: 1, fetchedBytes: 5},
          networkAttempts: []
        }
      : {}),
    inference
  };
}

function makeKananaPlaywrightReport(context: KananaVerificationContext): AnyRecord {
  const specs = Object.entries(context.policy.tests).map(([kind, requirement]) => {
    const record = makeKananaEvidenceRecord(context, kind as keyof KananaVerificationContext['policy']['tests']);
    return {
      file: requirement.file,
      title: requirement.title,
      tests: [
        {
          expectedStatus: 'passed',
          status: 'expected',
          projectName: context.policy.browser.project,
          results: [
            {
              status: 'passed',
              errors: [],
              stdout: [`${context.policy.evidencePrefix}${JSON.stringify(record)}\n`]
            }
          ]
        }
      ]
    };
  });
  return {
    config: {rootDir: context.workspaceRoot},
    errors: [],
    stats: {expected: specs.length, skipped: 0, unexpected: 0, flaky: 0},
    suites: [{title: 'Kanana exact package release evidence', specs}]
  };
}

function updateKananaRecord(
  report: AnyRecord,
  kind: 'text' | 'image' | 'offline',
  mutate: (record: AnyRecord) => void
): void {
  const spec = report.suites[0].specs.find((item: AnyRecord) => item.file === POLICY.kanana.tests[kind].file);
  const result = required(spec?.tests[0]?.results[0], `Kanana ${kind} Playwright result`);
  const stdout = required(result.stdout[0], `Kanana ${kind} evidence stdout`);
  const prefix = POLICY.kanana.evidencePrefix;
  const record = JSON.parse(stdout.slice(prefix.length));
  mutate(record);
  result.stdout[0] = `${prefix}${JSON.stringify(record)}\n`;
}

function maxDifference(left: number[], right: number[]): number {
  return Math.max(...left.map((value, index) => Math.abs(value - required(right[index], 'Reference value'))));
}

test('accepts an exact installed archive with all curated model/browser GPU cases', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const reportPath = await saveReport(fixture, valid.report);

  const result = await verifyReleaseArtifactsForTestFixture({
    archivePath: fixture.archivePath,
    installedPackagePath: fixture.packageRoot,
    playwrightReportPath: reportPath,
    sourcePackageRoot: fixture.sourceRoot,
    workspaceRoot: fixture.root,
    policy: POLICY
  });

  assert.equal(result.packageVersion, '9.8.7');
  assert.deepEqual(result.models, ['kev-0.6b-q4f16']);
  assert.deepEqual(result.browserResults, {modelCount: 1, browserCount: 1, caseCount: 1});
});

test('requires an npm .tgz archive', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const renamedArchive = join(fixture.root, 'web-decision-9.8.7.tar.gz');
  await copyFile(fixture.archivePath, renamedArchive);

  await assert.rejects(verifyArchiveInstall(renamedArchive, fixture.packageRoot), /\.tgz extension/u);
});

test('rejects policy edits that drop browser requirements or leave the matrix empty', async () => {
  const policyBytes = await readFile(POLICY_PATH);
  assert.equal(sha256(policyBytes), RELEASE_EVIDENCE_POLICY_SHA256);
  const digest = RELEASE_EVIDENCE_POLICY_SHA256;
  const reduced = Buffer.from(JSON.stringify({...POLICY, browsers: POLICY.browsers.slice(1)}));
  const empty = Buffer.from(JSON.stringify({...POLICY, browsers: []}));

  assert.throws(() => validatePolicyBytes(reduced, digest), /differs from its committed pin/u);
  assert.throws(() => validatePolicyBytes(empty, digest), /differs from its committed pin/u);
  assert.throws(() => validatePolicy({...POLICY, browsers: []}), /matrix must not be empty/u);
});

test('pins the exact Kanana npm archive, static manifest, and TVM artifact set', async (t) => {
  const policy = validateKananaEvidencePolicy(POLICY.kanana);
  assert.equal(policy.archive.sha256, '2185124f0ffd0f92d8a58eb74f95a8f94d19bf4d12134c3fb0567ba1d313c948');
  assert.equal(policy.archive.packageVersion, '0.4.0');
  const context = await makeKananaVerificationContext();
  assert.equal(context.manifest.manifestSha256, policy.model.manifestSha256);
  assert.deepEqual(context.manifest.files, policy.model.artifactFiles);
  assert.equal(
    context.manifest.files['kanana-text-model.wasm']?.sha256,
    'f0e22c3d113e037675f0cad6dfeefef2ed21045a957b96f15ff270a59add9a09'
  );
  assert.equal(
    context.manifest.files['kanana-vision-model.wasm']?.sha256,
    '931ab048976f6b8e7df272f33b0a65abf253f19adbffb4503b7c3653b94b5e38'
  );

  const invalidManifest = Buffer.concat([
    await readFile(join(WORKSPACE_ROOT, 'packages/web-decision', policy.model.manifestPath)),
    Buffer.from(' ')
  ]);
  assert.throws(() => validateKananaManifestBytes(invalidManifest, policy), /raw artifact manifest/u);

  const missingVision = structuredClone(POLICY.kanana);
  missingVision.tests.image.runtimeArtifacts = ['kanana-text-model.wasm'];
  assert.throws(() => validateKananaEvidencePolicy(missingVision), /runtime artifact set/u);

  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  await assert.rejects(
    collectKananaReleaseEvidenceInputs({
      archivePath: fixture.archivePath,
      installedPackagePath: fixture.packageRoot,
      workspaceRoot: WORKSPACE_ROOT,
      localCandidate: true
    }),
    /pinned npm archive/u
  );
});

test('requires independent native TVM rebuild reports for both Kanana WASM artifacts', async (t) => {
  const fixture = await makeKananaBuildFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const builds = await verifyKananaTvmBuildEvidence(fixture.evidencePath, fixture.inputs, fixture.sourceRoot);
  assert.deepEqual(
    builds.map(({artifact}) => artifact),
    ['kanana-text-model.wasm', 'kanana-vision-model.wasm']
  );

  await writeKananaBuildEvidence(fixture.evidencePath, fixture.builds.slice(0, 1));
  await assert.rejects(
    verifyKananaTvmBuildEvidence(fixture.evidencePath, fixture.inputs, fixture.sourceRoot),
    /Every curated TVM WASM/u
  );
  await writeKananaBuildEvidence(fixture.evidencePath, fixture.builds);

  const visionReportPath = join(fixture.buildRoot, 'vision/report.json');
  const visionReport = fixture.reports.get('kanana-vision-model.wasm');
  assert.ok(visionReport);
  await writeFile(visionReportPath, JSON.stringify({...visionReport, onnxImportOrExportUsed: true}));
  await assert.rejects(
    verifyKananaTvmBuildEvidence(fixture.evidencePath, fixture.inputs, fixture.sourceRoot),
    /onnxImportOrExportUsed differs/u
  );
  await writeFile(visionReportPath, JSON.stringify(visionReport));

  const installedArtifact = join(fixture.installedRoot, 'assets/kanana/kanana-text-model.wasm');
  const rebuiltArtifact = join(fixture.buildRoot, 'text/rebuilt.wasm');
  await rm(rebuiltArtifact);
  await link(installedArtifact, rebuiltArtifact);
  await assert.rejects(
    verifyKananaTvmBuildEvidence(fixture.evidencePath, fixture.inputs, fixture.sourceRoot),
    /packaged source file/u
  );
});

test('requires text, image, and cached-offline Kanana WebGPU evidence bound to the same archive', async () => {
  const context = await makeKananaVerificationContext();
  const valid = makeKananaPlaywrightReport(context);
  assert.deepEqual(validateKananaPlaywrightReport(valid, context), {
    caseCount: 3,
    browser: 'chromium',
    modelId: context.policy.model.modelId
  });

  const skipped = structuredClone(valid);
  skipped.stats.skipped = 1;
  assert.throws(() => validateKananaPlaywrightReport(skipped, context), /Skipped Kanana tests block/u);

  const failed = structuredClone(valid);
  failed.suites[0].specs[1].tests[0].results[0].status = 'failed';
  assert.throws(() => validateKananaPlaywrightReport(failed, context), /did not pass/u);

  const omitted = structuredClone(valid);
  omitted.suites[0].specs.pop();
  assert.throws(() => validateKananaPlaywrightReport(omitted, context), /required text, image, and offline cases/u);

  const wrongArchive = structuredClone(valid);
  updateKananaRecord(wrongArchive, 'text', (record) => {
    record.archive.sha256 = '0'.repeat(64);
  });
  assert.throws(() => validateKananaPlaywrightReport(wrongArchive, context), /different package archive/u);

  const wrongModelArtifact = structuredClone(valid);
  updateKananaRecord(wrongModelArtifact, 'image', (record) => {
    record.artifact.runtimeArtifacts[1].sha256 = '0'.repeat(64);
  });
  assert.throws(() => validateKananaPlaywrightReport(wrongModelArtifact, context), /runtime artifact differs/u);

  const fallback = structuredClone(valid);
  updateKananaRecord(fallback, 'text', (record) => {
    record.gpu.isFallbackAdapter = true;
  });
  assert.throws(() => validateKananaPlaywrightReport(fallback, context), /fallback GPU adapter/u);

  const noAdapterInfo = structuredClone(valid);
  updateKananaRecord(noAdapterInfo, 'text', (record) => {
    record.gpu.adapterInfo.value.vendor = '';
  });
  assert.throws(() => validateKananaPlaywrightReport(noAdapterInfo, context), /adapter vendor/u);

  const wrongTextOutput = structuredClone(valid);
  updateKananaRecord(wrongTextOutput, 'text', (record) => {
    record.inference.choice = 'early';
  });
  assert.throws(() => validateKananaPlaywrightReport(wrongTextOutput, context), /text choice differs/u);

  const wrongImageOutput = structuredClone(valid);
  updateKananaRecord(wrongImageOutput, 'image', (record) => {
    record.inference.answer = 'circle';
  });
  assert.throws(() => validateKananaPlaywrightReport(wrongImageOutput, context), /synthetic-image choice/u);

  const wrongImageConfidence = structuredClone(valid);
  updateKananaRecord(wrongImageConfidence, 'image', (record) => {
    record.inference.confidence = record.inference.probabilities.rectangle;
  });
  assert.throws(() => validateKananaPlaywrightReport(wrongImageConfidence, context), /selected-option margin/u);

  const offlineNetworkAttempt = structuredClone(valid);
  updateKananaRecord(offlineNetworkAttempt, 'offline', (record) => {
    record.networkAttempts.push('https://huggingface.co/example');
  });
  assert.throws(() => validateKananaPlaywrightReport(offlineNetworkAttempt, context), /attempted a network request/u);

  const noFetchedRanges = structuredClone(valid);
  updateKananaRecord(noFetchedRanges, 'offline', (record) => {
    record.cache.fetchedRanges = 0;
  });
  assert.throws(() => validateKananaPlaywrightReport(noFetchedRanges, context), /fetched no verified ranges/u);

  const offlineOutputMismatch = structuredClone(valid);
  updateKananaRecord(offlineOutputMismatch, 'offline', (record) => {
    record.inference.offline.probabilities.late = 0.5;
  });
  assert.throws(() => validateKananaPlaywrightReport(offlineOutputMismatch, context));
});

test('rejects incomplete or non-TVM catalogs and unpinned artifacts', async () => {
  assert.throws(
    () =>
      validateArtifactManifest(
        {modelId: 'missing', manifestSha256: 'a'.repeat(64), requiredFiles: []} as unknown as Parameters<
          typeof validateArtifactManifest
        >[0],
        {}
      ),
    /ID mismatch/u
  );
  assert.throws(
    () =>
      validatePolicy({
        ...POLICY,
        browsers: [
          {name: 'chromium', project: 'chromium'},
          {name: 'chromium', project: 'other'}
        ]
      }),
    /Duplicate browser/u
  );

  const fixture = await makeFixture();
  try {
    const nonTvm = structuredClone(required(fixture.descriptors[0], 'Fixture descriptor'));
    nonTvm.description.backend.name = 'onnxruntime-web-webgpu';
    nonTvm.publicDescriptor.backend.name = 'onnxruntime-web-webgpu';
    const {packageRoot, sourceRoot} = await putPackageTree(fixture.root, [nonTvm]);
    await assert.rejects(loadPublicCatalog(sourceRoot, 'dist/index.js'), /not a TVM WebGPU route/u);
    await writeFile(join(packageRoot, 'dist/index.js'), 'export function listDecisionModels() {}\n');
    await assert.rejects(loadPublicCatalog(packageRoot, 'dist/index.js'), /public listModels\(\)\/describeModel/u);
  } finally {
    await rm(fixture.root, {recursive: true, force: true});
  }
});

test('derives release expectations from curated JSON inputs and never imports the legacy registry', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  await writeFile(
    join(fixture.sourceRoot, 'src/index.js'),
    'throw new Error("Legacy model registry must not be imported by the release gate.");\n'
  );

  const inputs = await collectReleaseEvidenceInputsForTestFixture({
    archivePath: fixture.archivePath,
    installedPackagePath: fixture.packageRoot,
    sourcePackageRoot: fixture.sourceRoot
  });

  assert.deepEqual(
    inputs.models.map(({modelId}) => modelId),
    ['kev-0.6b-q4f16']
  );
  assert.equal(inputs.manifests.get('kev-0.6b-q4f16')?.manifest.model, 'kev-0.6b-q4f16');
});

test('production release verification rejects the explicitly synthetic fixture source root', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  await assert.rejects(
    collectReleaseEvidenceInputs({
      archivePath: fixture.archivePath,
      installedPackagePath: fixture.packageRoot,
      sourcePackageRoot: fixture.sourceRoot
    }),
    /Release verification must use the committed package source root/u
  );
});

test('the release verifier blocks committed pending provenance before checking build inputs', async () => {
  const workingPolicy = await readFile(POLICY_PATH);
  const committedPolicy = execFileSync('git', ['show', 'HEAD:scripts/release/release-evidence-policy.json'], {
    cwd: WORKSPACE_ROOT
  });
  const policyIsCommitted = workingPolicy.equals(committedPolicy);
  await assert.rejects(
    verifyReleaseEvidence({
      archivePath: 'unused.tgz',
      installedPackagePath: 'unused-package',
      playwrightReportPath: 'unused-report.json'
    }),
    policyIsCommitted
      ? /Model kev-0\.6b-q4f16 provenance is pending:/u
      : /Release evidence policy must match the committed version/u
  );
});

test('allows a catalog model with no required features but keeps Kev shader-f16 mandatory', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const featureless = structuredClone(required(fixture.descriptors[0], 'Fixture descriptor'));
  featureless.description.backend.requiredFeatures = [];
  featureless.publicDescriptor.backend.requiredFeatures = [];
  const {sourceRoot} = await putPackageTree(join(fixture.root, 'featureless'), [featureless]);
  const publicCatalog = await loadPublicCatalog(sourceRoot, 'dist/index.js');
  assert.deepEqual(required(publicCatalog[0], 'Public fixture descriptor').backend.requiredFeatures, []);

  const valid = await prepareValidReport(fixture);
  const context = {
    archive: valid.archive,
    catalogSha256: valid.catalogSha256,
    manifests: valid.manifests,
    models: valid.models,
    policy: POLICY,
    workspaceRoot: fixture.root
  };
  required(valid.models[0], 'Curated Kev model').backend.requiredFeatures = [];
  const report = structuredClone(valid.report);
  const stdout = report.suites[0].specs[0].tests[0].results[0].stdout[0];
  const record = JSON.parse(stdout.slice(POLICY.evidencePrefix.length));
  record.gpu.features = [];
  report.suites[0].specs[0].tests[0].results[0].stdout[0] = `${POLICY.evidencePrefix}${JSON.stringify(record)}\n`;
  assert.throws(() => validatePlaywrightReport(report, context), /required feature shader-f16/u);
});

test('derives a non-empty exact asset requirement set from the committed manifest lock', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  for (const model of valid.models) {
    assert.deepEqual(model.requiredFiles, ['manifest.json', ...Object.keys(model.manifestFiles)].sort());
  }

  const modelSpecPath = join(fixture.sourceRoot, 'src/catalog/model-spec.json');
  const modelSpec = JSON.parse(await readFile(modelSpecPath, 'utf8'));
  modelSpec.models = [];
  await writeFile(modelSpecPath, JSON.stringify(modelSpec));
  await assert.rejects(
    collectReleaseEvidenceInputsForTestFixture({
      archivePath: fixture.archivePath,
      installedPackagePath: fixture.packageRoot,
      sourcePackageRoot: fixture.sourceRoot
    }),
    /model spec must not be empty/u
  );

  modelSpec.models = fixture.descriptors.map(({modelId, description}) => ({modelId, description}));
  modelSpec.models[0].description.assets.requiredFiles = ['manifest.json'];
  await writeFile(modelSpecPath, JSON.stringify(modelSpec));
  await assert.rejects(
    collectReleaseEvidenceInputsForTestFixture({
      archivePath: fixture.archivePath,
      installedPackagePath: fixture.packageRoot,
      sourcePackageRoot: fixture.sourceRoot
    }),
    /derive requiredFiles from its pinned manifest/u
  );
});

test('rejects any skip, failure, retry, or omitted required browser case', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const context = {
    archive: valid.archive,
    catalogSha256: valid.catalogSha256,
    manifests: valid.manifests,
    models: valid.models,
    policy: POLICY,
    workspaceRoot: fixture.root
  };

  const skipped = structuredClone(valid.report);
  skipped.stats.skipped = 1;
  assert.throws(() => validatePlaywrightReport(skipped, context), /Skipped tests block/u);

  const failed = structuredClone(valid.report);
  failed.suites[0].specs[0].tests[0].results[0].status = 'failed';
  assert.throws(() => validatePlaywrightReport(failed, context), /did not pass/u);

  const retried = structuredClone(valid.report);
  retried.suites[0].specs[0].tests[0].results.push(structuredClone(retried.suites[0].specs[0].tests[0].results[0]));
  assert.throws(() => validatePlaywrightReport(retried, context), /exactly one execution/u);

  const omitted = structuredClone(valid.report);
  omitted.suites[0].specs.pop();
  omitted.stats.expected -= 1;
  assert.throws(() => validatePlaywrightReport(omitted, context), /pass count differs/u);
});

test('requires pinned runs and reference selections without new numerical tolerances', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const context = {
    archive: valid.archive,
    catalogSha256: valid.catalogSha256,
    manifests: valid.manifests,
    models: valid.models,
    policy: POLICY,
    workspaceRoot: fixture.root
  };
  const corruptions: Array<(record: AnyRecord) => void> = [
    (record) => record.expandedInference.decisionCases.pop(),
    (record) => record.expandedInference.decisionCases.push(structuredClone(record.expandedInference.decisionCases[1])),
    (record) => {
      record.expandedInference.decisionCases[1].request.state = 'different input';
    },
    (record) => {
      record.expandedInference.decisionCases[1].observed.probabilities[0] = 1.25;
    },
    (record) => {
      record.expandedInference.decisionCases[1].observed.selectedIndex = 1;
    },
    (record) => {
      record.expandedInference.typedSystemOne.questions.refund.type = 'score';
    },
    (record) => {
      delete record.expandedInference.typedSystemOne.single.refund;
    },
    (record) => {
      record.expandedInference.typedSystemOne.packed.observed.answerEntries[1][0] = 'team';
    },
    (record) => {
      record.expandedInference.typedSystemOne.packed.observed.answers.refund.noul = 1.1;
    },
    (record) => {
      record.expandedInference.typedSystemOne.packed.observed.answers.sentiment.score = 5;
    },
    (record) => {
      record.expandedInference.typedSystemOne.packed.reference.answers.team.logits[0] = 99;
    },
    (record) => {
      record.expandedInference.typedSystemOne.single.sentiment.comparison.sentiment = 0;
    }
  ];

  for (const mutate of corruptions) {
    const report = structuredClone(valid.report);
    updateEvidenceRecord(report, mutate);
    assert.throws(() => validatePlaywrightReport(report, context));
  }

  const wrongDecision = structuredClone(valid.report);
  updateEvidenceRecord(wrongDecision, (record) => {
    const decision = record.expandedInference.decisionCases[1];
    decision.observed.logits = [0, 3, 0];
    decision.observed.probabilities = [0.1, 0.8, 0.1];
    decision.observed.selected = decision.request.questions[0].options[1];
    decision.observed.selectedIndex = 1;
    decision.comparison.maxAbsoluteLogitDifference = maxDifference(decision.observed.logits, decision.reference.logits);
    decision.comparison.maxAbsoluteProbabilityDifference = maxDifference(
      decision.observed.probabilities,
      decision.reference.probabilities
    );
  });
  assert.throws(() => validatePlaywrightReport(wrongDecision, context), /reference selection/u);

  const wrongChoice = structuredClone(valid.report);
  updateEvidenceRecord(wrongChoice, (record) => {
    const packed = record.expandedInference.typedSystemOne.packed;
    const team = packed.observed.answers.team;
    team.choice = 'sales';
    team.probabilities = {billing: 0.1, support: 0.2, sales: 0.7};
    team.probabilityEntries = [
      ['billing', 0.1],
      ['support', 0.2],
      ['sales', 0.7]
    ];
    packed.observed.answerEntries[0][1] = structuredClone(team);
    packed.comparison.team = maxDifference(
      team.probabilityEntries.map((entry: [string, number]) => entry[1]),
      packed.reference.answers.team.probabilities
    );
  });
  assert.throws(() => validatePlaywrightReport(wrongChoice, context), /reference selection/u);

  const wrongNoul = structuredClone(valid.report);
  updateEvidenceRecord(wrongNoul, (record) => {
    const single = record.expandedInference.typedSystemOne.single.refund;
    single.observed.answers.refund.noul = 0.1;
    single.observed.answerEntries[0][1] = structuredClone(single.observed.answers.refund);
    single.comparison.refund = maxDifference([0.9, 0.1], single.reference.answers.refund.probabilities);
  });
  assert.throws(() => validatePlaywrightReport(wrongNoul, context), /reference selection/u);

  const wrongScore = structuredClone(valid.report);
  updateEvidenceRecord(wrongScore, (record) => {
    const packed = record.expandedInference.typedSystemOne.packed;
    const answer = packed.observed.answers.sentiment;
    answer.score = 3.9;
    answer.probabilities = {'0': 0.01, '1': 0.01, '2': 0.01, '3': 0.01, '4': 0.96};
    answer.probabilityEntries = [
      ['0', 0.01],
      ['1', 0.01],
      ['2', 0.01],
      ['3', 0.01],
      ['4', 0.96]
    ];
    packed.observed.answerEntries[2][1] = structuredClone(answer);
    packed.comparison.sentiment = maxDifference(
      answer.probabilityEntries.map((entry: [string, number]) => entry[1]),
      packed.reference.answers.sentiment.probabilities
    );
  });
  assert.throws(() => validatePlaywrightReport(wrongScore, context), /reference selection/u);

  const observational = structuredClone(valid.report);
  updateEvidenceRecord(observational, (record) => {
    const expanded = record.expandedInference;
    const decision = expanded.decisionCases[1];
    decision.observed.logits = [3, 0, 0];
    decision.observed.probabilities = [0.8, 0.1, 0.1];
    decision.observed.selected = decision.request.questions[0].options[0];
    decision.observed.selectedIndex = 0;
    decision.comparison.maxAbsoluteLogitDifference = maxDifference(decision.observed.logits, decision.reference.logits);
    decision.comparison.maxAbsoluteProbabilityDifference = maxDifference(
      decision.observed.probabilities,
      decision.reference.probabilities
    );

    const team = expanded.typedSystemOne.packed.observed.answers.team;
    team.choice = 'billing';
    team.probabilities = {billing: 0.7, support: 0.2, sales: 0.1};
    team.probabilityEntries = [
      ['billing', 0.7],
      ['support', 0.2],
      ['sales', 0.1]
    ];
    expanded.typedSystemOne.packed.observed.answerEntries[0][1] = structuredClone(team);
    expanded.typedSystemOne.packed.comparison.team = maxDifference(
      team.probabilityEntries.map((entry: [string, number]) => entry[1]),
      expanded.typedSystemOne.packed.reference.answers.team.probabilities
    );
  });
  assert.doesNotThrow(() => validatePlaywrightReport(observational, context));
});

test('rejects changes to the pinned expanded-input fixture bytes', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const path = join(fixture.root, POLICY.expandedInference.fixturePath);
  const bytes = await readFile(path, 'utf8');
  await writeFile(path, bytes.replace('A customer was charged twice.', 'A customer has two charges.'));
  assert.throws(
    () => loadExpandedInferenceFixture(fixture.root, validatePolicy(POLICY).expandedInference),
    /fixture digest mismatch/u
  );
});

test('rejects evidence detached from the archive, install, model manifest, or physical GPU', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const context = {
    archive: valid.archive,
    catalogSha256: valid.catalogSha256,
    manifests: valid.manifests,
    models: valid.models,
    policy: POLICY,
    workspaceRoot: fixture.root
  };
  const mutations: Array<(record: AnyRecord) => void> = [
    (record) => {
      record.archive.sha256 = '0'.repeat(64);
    },
    (record) => {
      record.archive.bundleSha256 = '0'.repeat(64);
    },
    (record) => {
      record.archive.installedPackagePathSha256 = '0'.repeat(64);
    },
    (record) => {
      record.archive.entryPathSha256 = '0'.repeat(64);
    },
    (record) => {
      record.model.loadedId = 'other-model';
    },
    (record) => {
      record.model.assetManifestSha256 = '0'.repeat(64);
    },
    (record) => {
      record.inference.reference.fixtureSha256 = '0'.repeat(64);
    },
    (record) => {
      record.inference.reference.packageSha256 = '0'.repeat(64);
    },
    (record) => {
      record.inference.upstreamReference.fixtureSha256 = '0'.repeat(64);
    },
    (record) => {
      record.inference.upstreamReference.modelId = 'other-model';
    },
    (record) => {
      record.inference.online.referenceMatched = false;
    },
    (record) => {
      record.inference.online.optionOrder.reverse();
    },
    (record) => {
      record.inference.online.logits[0] = 10;
    },
    (record) => {
      record.inference.online.probabilities[0] = 0;
    },
    (record) => {
      record.inference.online.optionIndices = [0, 1];
    },
    (record) => {
      record.inference.offline.logits[0] += 0.01;
    },
    (record) => {
      record.inference.offline.networkAttempts.push('https://example.invalid/model');
    },
    (record) => {
      record.artifact.modelId = 'other-model';
    },
    (record) => {
      record.artifact.manifestPath = 'assets/other/manifest.json';
    },
    (record) => {
      record.artifact.manifest.files['model.wasm'].sha256 = '0'.repeat(64);
    },
    (record) => {
      record.artifact.verifiedFiles.pop();
    },
    (record) => {
      record.artifact.verifiedFiles[0].sha256 = '0'.repeat(64);
    },
    (record) => {
      record.device.isVirtual = true;
    },
    (record) => {
      record.device.model = 'Microsoft Virtual Machine';
      record.device.evidence.host.raw.Model = 'Microsoft Virtual Machine';
    },
    (record) => {
      record.gpu.isFallbackAdapter = true;
    },
    (record) => {
      record.gpu.features = [];
    },
    (record) => {
      record.browser.version = '';
    }
  ];

  for (const mutate of mutations) {
    const report = structuredClone(valid.report);
    const stdout = report.suites[0].specs[0].tests[0].results[0].stdout[0];
    const record = JSON.parse(stdout.slice(POLICY.evidencePrefix.length));
    mutate(record);
    report.suites[0].specs[0].tests[0].results[0].stdout[0] = `${POLICY.evidencePrefix}${JSON.stringify(record)}\n`;
    assert.throws(() => validatePlaywrightReport(report, context));
  }
});

test('rejects archive manifest changes and installed-package drift', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const changed = join(
    fixture.packageRoot,
    'assets',
    required(fixture.descriptors[0], 'Fixture descriptor').modelId,
    'manifest.json'
  );
  const models = valid.models;
  const originalArchive = await verifyArchiveInstall(fixture.archivePath, fixture.packageRoot);
  assert.equal((await readArchiveModelManifests(originalArchive, models)).size, 1);

  await writeFile(changed, '{"model":"kev-0.6b-q4f16","files":{}}\n');
  await assert.rejects(
    verifyArchiveInstall(fixture.archivePath, fixture.packageRoot),
    /differs from the release archive/u
  );
  const changedArchivePath = join(fixture.root, 'changed-manifest.tgz');
  execFileSync(TAR, ['-czf', changedArchivePath, '-C', dirname(fixture.packageRoot), 'package']);
  const changedArchive = await verifyArchiveInstall(changedArchivePath, fixture.packageRoot);
  await assert.rejects(readArchiveModelManifests(changedArchive, models), /Artifact manifest SHA differs/u);
});

test('resolves Playwright test paths from its test root and rejects a different root', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const context = {
    archive: valid.archive,
    catalogSha256: valid.catalogSha256,
    manifests: valid.manifests,
    models: valid.models,
    policy: POLICY,
    workspaceRoot: fixture.root
  };
  const report = structuredClone(valid.report);
  report.config.rootDir = join(fixture.root, 'tests/browser');
  report.suites[0].specs[0].file = 'kev06-default-source-real-webgpu.spec.ts';
  assert.deepEqual(validatePlaywrightReport(report, context), {modelCount: 1, browserCount: 1, caseCount: 1});
  report.config.rootDir = join(fixture.root, 'other');
  assert.throws(() => validatePlaywrightReport(report, context), /Only the dedicated release GPU test/u);
  delete report.config.rootDir;
  assert.throws(() => validatePlaywrightReport(report, context), /absolute test root/u);
});

test('rejects old browser test files even when their results passed', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const context = {
    archive: valid.archive,
    catalogSha256: valid.catalogSha256,
    manifests: valid.manifests,
    models: valid.models,
    policy: POLICY,
    workspaceRoot: fixture.root
  };
  const oldTest = structuredClone(valid.report);
  oldTest.suites[0].specs[0].file = 'tests/browser/kev06-real-webgpu.spec.mjs';

  assert.throws(() => validatePlaywrightReport(oldTest, context), /Only the dedicated release GPU test/u);
});

test('rejects untracked archive members and unknown GPU model manifests', async (t) => {
  const fixture = await makeFixture();
  t.after(() => rm(fixture.root, {recursive: true, force: true}));
  const valid = await prepareValidReport(fixture);
  const unknownManifest = {
    model: 'not-in-catalog',
    backend: 'tvm-webgpu',
    files: {'model.wasm': {sha256: 'c'.repeat(64), bytes: 19}}
  };
  const path = join(fixture.packageRoot, 'assets', 'uncurated', 'manifest.json');
  await mkdir(dirname(path), {recursive: true});
  await writeFile(path, JSON.stringify(unknownManifest));
  const archiveRoot = dirname(fixture.packageRoot);
  const replacementArchive = join(fixture.root, 'with-extra.tgz');
  execFileSync(TAR, ['-czf', replacementArchive, '-C', archiveRoot, 'package']);
  const installed = await verifyArchiveInstall(replacementArchive, fixture.packageRoot);
  const models = valid.models;

  await assert.rejects(readArchiveModelManifests(installed, models), /uncurated TVM manifest/u);
});
