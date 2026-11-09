import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {copyFile, mkdir, mkdtemp, readFile, realpath, rm, writeFile} from 'node:fs/promises';
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
  collectReleaseEvidenceInputsForTestFixture,
  verifyReleaseArtifactsForTestFixture,
  verifyReleaseEvidence
} from './verify-release-evidence.ts';
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
  await assert.rejects(
    verifyReleaseEvidence({
      archivePath: 'unused.tgz',
      installedPackagePath: 'unused-package',
      playwrightReportPath: 'unused-report.json'
    }),
    /Model kev-0\.6b-q4f16 provenance is pending:/u
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
