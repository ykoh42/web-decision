import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {
  MODEL_PROVENANCE_EVIDENCE_ROOT,
  MODEL_PROVENANCE_MANIFEST_PATH,
  MODEL_PROVENANCE_MANIFEST_SHA256,
  RELEASE_TEST_PROVENANCE_MANIFEST_SHA256,
  RELEASE_TEST_PROVENANCE_MANIFEST_PATH,
  sha256,
  validateModelProvenance,
  validateLockedUpstream
} from './model-provenance.ts';

const WORKSPACE_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PACKAGE_ROOT = join(WORKSPACE_ROOT, 'packages/web-decision');
const MODEL_SPEC_PATH = join(PACKAGE_ROOT, 'src/catalog/model-spec.json');
const ARTIFACT_LOCK_PATH = join(PACKAGE_ROOT, 'src/catalog/artifacts.lock.json');
const TEST_EVIDENCE_PATH = join(WORKSPACE_ROOT, 'scripts/release/fixtures/model-provenance-test-fixture.json');

async function readProductionInputs(): Promise<{
  modelSpec: unknown;
  artifactLock: unknown;
  artifactLockBytes: Buffer;
  manifestBytes: Buffer;
}> {
  const [specBytes, lockBytes, manifestBytes] = await Promise.all([
    readFile(MODEL_SPEC_PATH),
    readFile(ARTIFACT_LOCK_PATH),
    readFile(join(PACKAGE_ROOT, MODEL_PROVENANCE_MANIFEST_PATH))
  ]);
  return {
    modelSpec: JSON.parse(specBytes.toString('utf8')) as unknown,
    artifactLock: JSON.parse(lockBytes.toString('utf8')) as unknown,
    artifactLockBytes: lockBytes,
    manifestBytes
  };
}

async function validateTemporaryManifest(
  manifestBytes: Buffer,
  options: {expectedSha256?: string; evidenceBytes?: Buffer; allowPending?: boolean} = {}
): Promise<{pendingModelIds: string[]}> {
  const temporaryRoot = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-model-provenance-')));
  try {
    const packageRoot = join(temporaryRoot, 'package');
    const evidenceRoot = join(packageRoot, MODEL_PROVENANCE_EVIDENCE_ROOT);
    await mkdir(evidenceRoot, {recursive: true});
    const sourceEvidenceRoot = join(PACKAGE_ROOT, MODEL_PROVENANCE_EVIDENCE_ROOT);
    for (const name of await readdir(sourceEvidenceRoot)) {
      const source = await readFile(join(sourceEvidenceRoot, name));
      const evidence = name === 'kanana-2-v-decide-3b-2611.json' ? (options.evidenceBytes ?? source) : source;
      await writeFile(join(evidenceRoot, name), evidence);
    }
    const inputs = await readProductionInputs();
    return await validateModelProvenance({
      packageRoot,
      workspaceRoot: WORKSPACE_ROOT,
      modelSpec: inputs.modelSpec,
      artifactLock: inputs.artifactLock,
      artifactLockBytes: inputs.artifactLockBytes,
      manifestBytes,
      expectedManifestSha256: options.expectedSha256 ?? MODEL_PROVENANCE_MANIFEST_SHA256,
      expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
      allowTestFixture: false,
      requireCommitted: false,
      ...(options.allowPending === undefined ? {} : {allowPending: options.allowPending})
    });
  } finally {
    await rm(temporaryRoot, {recursive: true, force: true});
  }
}

test('the pinned public-model record has verified source and conditional-rights evidence', async () => {
  const {manifestBytes, modelSpec, artifactLock} = await readProductionInputs();
  const catalogIds = (modelSpec as {models: Array<{modelId: string}>}).models.map(({modelId}) => modelId).sort();
  const lockIds = Object.keys((artifactLock as {byModelId: Record<string, unknown>}).byModelId).sort();
  const manifestIds = (JSON.parse(manifestBytes.toString('utf8')) as {models: Array<{modelId: string}>}).models
    .map(({modelId}) => modelId)
    .sort();
  assert.deepEqual(catalogIds, ['kanana-2-v-decide-3b-2611']);
  assert.deepEqual(manifestIds, catalogIds);
  assert.deepEqual(lockIds, catalogIds);
  assert.equal(sha256(manifestBytes), MODEL_PROVENANCE_MANIFEST_SHA256);
  const result = await validateTemporaryManifest(manifestBytes);
  assert.deepEqual(result.pendingModelIds, []);
});

test('verified records pass the provenance gate without a project license approval record', async () => {
  const {manifestBytes} = await readProductionInputs();
  const result = await validateTemporaryManifest(manifestBytes, {allowPending: true});

  assert.deepEqual(result.pendingModelIds, []);
});

test('Kanana provenance pins the license conditions and captured build environment', async () => {
  const evidenceBytes = await readFile(
    join(PACKAGE_ROOT, MODEL_PROVENANCE_EVIDENCE_ROOT, 'kanana-2-v-decide-3b-2611.json')
  );
  const evidence = JSON.parse(evidenceBytes.toString('utf8')) as {
    status: string;
    knownReferences: Array<{url: string; bytes: number; sha256: string; scope: string}>;
    rights: Array<{license: string; redistribution: string; conditions: string[]; evidence: {sha256: string}}>;
    conversion: {
      artifactLockSha256: string;
      recipe: {path: string; sha256: string};
      config: {path: string; sha256: string};
      toolchain: {path: string; sha256: string};
    };
  };
  const licenseBytes = await readFile(join(PACKAGE_ROOT, 'KANANA-OPEN-LICENSE.txt'));
  const toolchainBytes = await readFile(join(WORKSPACE_ROOT, evidence.conversion.toolchain.path));
  const toolchain = JSON.parse(toolchainBytes.toString('utf8')) as {
    observedBuildEnvironment: {
      python: {version: string; executableSha256: string};
      pythonBuild: {lockfile: string; lockfileSha256: string; packages: Record<string, string>};
      host: {system: string; build: string; machine: string};
    };
  };
  const licenseReference = evidence.knownReferences.find(({url}) => url.endsWith('/LICENSE'));
  const rights = evidence.rights.find(({license}) => license === 'KananaOpenLicense');

  assert.equal(evidence.status, 'verified');
  assert.ok(licenseReference);
  assert.ok(rights);
  assert.equal(licenseReference.bytes, licenseBytes.byteLength);
  assert.equal(licenseReference.sha256, sha256(licenseBytes));
  assert.match(licenseReference.scope, /Section 4\.1\(iii\)/u);
  assert.equal(rights.redistribution, 'conditional');
  assert.equal(rights.evidence.sha256, sha256(licenseBytes));
  assert.ok(rights.conditions.some((condition) => condition.includes('on-device domains')));
  assert.equal(toolchain.observedBuildEnvironment.python.version, '3.11.11');
  assert.match(toolchain.observedBuildEnvironment.python.executableSha256, /^[a-f0-9]{64}$/u);
  assert.equal(toolchain.observedBuildEnvironment.pythonBuild.lockfile, 'scripts/kanana/build-env/uv.lock');
  assert.equal(
    toolchain.observedBuildEnvironment.pythonBuild.lockfileSha256,
    sha256(await readFile(join(WORKSPACE_ROOT, toolchain.observedBuildEnvironment.pythonBuild.lockfile)))
  );
  assert.equal(toolchain.observedBuildEnvironment.host.system, 'Darwin');
  assert.equal(toolchain.observedBuildEnvironment.host.build, '26B123');
  assert.equal(toolchain.observedBuildEnvironment.host.machine, 'arm64');
  assert.equal(evidence.conversion.artifactLockSha256, sha256(await readFile(ARTIFACT_LOCK_PATH)));
  for (const reference of [evidence.conversion.recipe, evidence.conversion.config, evidence.conversion.toolchain]) {
    const bytes = await readFile(join(WORKSPACE_ROOT, reference.path));
    assert.equal(sha256(bytes), reference.sha256);
  }
});

test('changing only verified to pending cannot bypass the fixed provenance hash', async () => {
  const {manifestBytes} = await readProductionInputs();
  const manifest = JSON.parse(manifestBytes.toString('utf8')) as {
    models: Array<{status: string}>;
  };
  const first = manifest.models[0];
  assert.ok(first);
  first.status = 'pending';
  await assert.rejects(
    validateTemporaryManifest(Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`)),
    /fixed SHA-256 pin/u
  );
});

test('verified provenance requires immutable lineage, pinned conversion, and explicit rights conditions', async (t) => {
  const temporaryRoot = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-verified-provenance-')));
  t.after(() => rm(temporaryRoot, {recursive: true, force: true}));
  const packageRoot = join(temporaryRoot, 'packages/web-decision');
  const provenanceRoot = join(packageRoot, MODEL_PROVENANCE_EVIDENCE_ROOT);
  const docsRoot = join(temporaryRoot, 'docs');
  await mkdir(provenanceRoot, {recursive: true});
  await mkdir(docsRoot, {recursive: true});

  const evidenceFiles = new Map([
    ['recipe.json', Buffer.from('{"recipe":"pinned"}\n')],
    ['config.json', Buffer.from('{"conversion":"fixed"}\n')],
    ['toolchain.json', Buffer.from('{"tools":"pinned"}\n')],
    ['rights.md', Buffer.from('Conditional rights terms are documented.\n')]
  ]);
  for (const [name, bytes] of evidenceFiles) await writeFile(join(docsRoot, name), bytes);
  const reference = (name: string) => ({
    path: `docs/${name}`,
    sha256: sha256(requiredEvidence(evidenceFiles.get(name)))
  });
  const inputs = await readProductionInputs();
  const sourceSpec = inputs.modelSpec as {
    schemaVersion: number;
    models: Array<{modelId: string; description: {licenses: {modelTerms: string}}}>;
  };
  const modelId = 'kanana-2-v-decide-3b-2611';
  const sourceModel = sourceSpec.models.find(({modelId: id}) => id === modelId);
  assert.ok(sourceModel);
  const verifiedSpec = {schemaVersion: sourceSpec.schemaVersion, models: [structuredClone(sourceModel)]};
  const sourceLock = inputs.artifactLock as {schemaVersion: number; byModelId: Record<string, unknown>};
  const modelLockEntry = sourceLock.byModelId[modelId];
  assert.ok(modelLockEntry);
  const artifactLock = {
    schemaVersion: sourceLock.schemaVersion,
    byModelId: {[modelId]: modelLockEntry}
  };
  const artifactLockBytes = Buffer.from(`${JSON.stringify(artifactLock, null, 2)}\n`);
  const specModel = verifiedSpec.models[0];
  assert.ok(specModel);
  const artifactLockSha256 = sha256(artifactLockBytes);
  const modelLock = modelLockEntry as {
    upstream: {repository: string; revision: string; sourcePath: string; sourceBytes: number; sourceSha256: string};
  };
  const verifiedEvidence = {
    schemaVersion: 1,
    modelId,
    status: 'verified',
    sourceLineage: [
      {
        repository: modelLock.upstream.repository,
        revision: modelLock.upstream.revision,
        path: modelLock.upstream.sourcePath,
        bytes: modelLock.upstream.sourceBytes,
        sha256: modelLock.upstream.sourceSha256
      }
    ],
    conversion: {
      repository: 'example/converter',
      revision: 'c'.repeat(40),
      recipe: reference('recipe.json'),
      config: reference('config.json'),
      toolchain: reference('toolchain.json'),
      artifactLockSha256
    },
    rights: [
      {
        subject: 'converted model weights',
        license: 'KananaOpenLicense',
        redistribution: 'conditional',
        conditions: ['Include the agreement copy and required attribution.'],
        evidence: reference('rights.md')
      }
    ]
  };
  const evidenceBytes = Buffer.from(`${JSON.stringify(verifiedEvidence, null, 2)}\n`);
  await writeFile(join(provenanceRoot, `${modelId}.json`), evidenceBytes);
  const manifestBytes = Buffer.from(
    `${JSON.stringify(
      {
        schemaVersion: 1,
        models: [
          {
            modelId,
            status: 'verified',
            artifactLockSha256,
            evidence: {
              path: `src/catalog/provenance/${modelId}.json`,
              sha256: sha256(evidenceBytes)
            }
          }
        ]
      },
      null,
      2
    )}\n`
  );
  const checked = await validateModelProvenance({
    packageRoot,
    workspaceRoot: temporaryRoot,
    modelSpec: verifiedSpec,
    artifactLock,
    artifactLockBytes,
    manifestBytes,
    expectedManifestSha256: sha256(manifestBytes),
    expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
    allowTestFixture: false,
    requireCommitted: false
  });
  assert.deepEqual(checked.modelIds, [modelId]);
  assert.equal(checked.artifactLockSha256, artifactLockSha256);

  const missingConditions = structuredClone(verifiedEvidence);
  missingConditions.rights[0]!.conditions = [];
  const missingConditionsBytes = Buffer.from(`${JSON.stringify(missingConditions, null, 2)}\n`);
  await writeFile(join(provenanceRoot, `${modelId}.json`), missingConditionsBytes);
  const missingConditionsManifest = JSON.parse(manifestBytes.toString('utf8')) as {
    models: Array<{evidence: {sha256: string}}>;
  };
  missingConditionsManifest.models[0]!.evidence.sha256 = sha256(missingConditionsBytes);
  const missingConditionsManifestBytes = Buffer.from(`${JSON.stringify(missingConditionsManifest, null, 2)}\n`);
  await assert.rejects(
    validateModelProvenance({
      packageRoot,
      workspaceRoot: temporaryRoot,
      modelSpec: verifiedSpec,
      artifactLock,
      artifactLockBytes,
      manifestBytes: missingConditionsManifestBytes,
      expectedManifestSha256: sha256(missingConditionsManifestBytes),
      expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
      allowTestFixture: false,
      requireCommitted: false
    }),
    /must list their conditions/u
  );

  await writeFile(join(provenanceRoot, `${modelId}.json`), evidenceBytes);
  const mismatchedEvidence = structuredClone(verifiedEvidence);
  mismatchedEvidence.sourceLineage[0]!.sha256 = 'd'.repeat(64);
  const mismatchedEvidenceBytes = Buffer.from(`${JSON.stringify(mismatchedEvidence, null, 2)}\n`);
  await writeFile(join(provenanceRoot, `${modelId}.json`), mismatchedEvidenceBytes);
  const mismatchedManifest = JSON.parse(manifestBytes.toString('utf8')) as {
    models: Array<{evidence: {sha256: string}}>;
  };
  mismatchedManifest.models[0]!.evidence.sha256 = sha256(mismatchedEvidenceBytes);
  const mismatchedManifestBytes = Buffer.from(`${JSON.stringify(mismatchedManifest, null, 2)}\n`);
  await assert.rejects(
    validateModelProvenance({
      packageRoot,
      workspaceRoot: temporaryRoot,
      modelSpec: verifiedSpec,
      artifactLock,
      artifactLockBytes,
      manifestBytes: mismatchedManifestBytes,
      expectedManifestSha256: sha256(mismatchedManifestBytes),
      expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
      allowTestFixture: false,
      requireCommitted: false
    }),
    /does not match its locked upstream artifact/u
  );
  await writeFile(join(provenanceRoot, `${modelId}.json`), evidenceBytes);

  await rm(join(docsRoot, 'toolchain.json'));
  await assert.rejects(
    validateModelProvenance({
      packageRoot,
      workspaceRoot: temporaryRoot,
      modelSpec: verifiedSpec,
      artifactLock,
      artifactLockBytes,
      manifestBytes,
      expectedManifestSha256: sha256(manifestBytes),
      expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
      allowTestFixture: false,
      requireCommitted: false
    }),
    /ENOENT/u
  );
  await writeFile(join(docsRoot, 'toolchain.json'), requiredEvidence(evidenceFiles.get('toolchain.json')));

  const unresolvedSpec = structuredClone(verifiedSpec);
  const unresolvedModel = unresolvedSpec.models[0];
  assert.ok(unresolvedModel);
  unresolvedModel.description.licenses.modelTerms = 'separate-review-required ';
  await assert.rejects(
    validateModelProvenance({
      packageRoot,
      workspaceRoot: temporaryRoot,
      modelSpec: unresolvedSpec,
      artifactLock,
      artifactLockBytes,
      manifestBytes,
      expectedManifestSha256: sha256(manifestBytes),
      expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
      allowTestFixture: false,
      requireCommitted: false
    }),
    /catalog license terms remain unresolved/u
  );
});

test('a verified model cannot omit its locked upstream source', () => {
  assert.throws(() => validateLockedUpstream([], {}, 'fixture-model'), /Locked upstream.*must be an object/u);
});

test('tampered committed evidence cannot match the manifest evidence digest', async () => {
  const {manifestBytes} = await readProductionInputs();
  await assert.rejects(
    validateTemporaryManifest(manifestBytes, {evidenceBytes: Buffer.from('{"status":"verified"}\n')}),
    /Provenance evidence SHA-256 mismatch/u
  );
});

test('an evidence file must match its committed HEAD bytes even when its hash is updated', async (t) => {
  const workspaceRoot = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-committed-provenance-')));
  t.after(() => rm(workspaceRoot, {recursive: true, force: true}));
  const packageRoot = join(workspaceRoot, 'packages/web-decision');
  const catalogRoot = join(packageRoot, 'src/catalog');
  const evidenceRoot = join(catalogRoot, 'provenance');
  await mkdir(evidenceRoot, {recursive: true});

  const modelId = 'fixture-model';
  const modelSpec = {schemaVersion: 1, models: [{modelId}]};
  const artifactLock = {schemaVersion: 1, byModelId: {[modelId]: {}}};
  const artifactLockBytes = Buffer.from(`${JSON.stringify(artifactLock)}\n`);
  const artifactLockSha256 = sha256(artifactLockBytes);
  const evidencePath = 'src/catalog/provenance/fixture-model.json';
  const evidenceAbsolute = join(packageRoot, evidencePath);
  const manifestAbsolute = join(catalogRoot, 'model-provenance.json');
  const evidence = (detail: string) =>
    Buffer.from(`${JSON.stringify({schemaVersion: 1, modelId, status: 'pending', unresolved: [detail]}, null, 2)}\n`);
  const oldEvidenceBytes = evidence('Old review note.');
  await writeFile(join(catalogRoot, 'model-spec.json'), `${JSON.stringify(modelSpec)}\n`);
  await writeFile(join(catalogRoot, 'artifacts.lock.json'), artifactLockBytes);
  await writeFile(evidenceAbsolute, oldEvidenceBytes);
  const manifestFor = (evidenceBytes: Buffer) =>
    Buffer.from(
      `${JSON.stringify(
        {
          schemaVersion: 1,
          models: [
            {
              modelId,
              status: 'pending',
              artifactLockSha256,
              evidence: {path: evidencePath, sha256: sha256(evidenceBytes)}
            }
          ]
        },
        null,
        2
      )}\n`
    );
  await writeFile(manifestAbsolute, manifestFor(oldEvidenceBytes));
  execFileSync('git', ['init', '--initial-branch=main'], {cwd: workspaceRoot, stdio: 'ignore'});
  execFileSync('git', ['add', '.'], {cwd: workspaceRoot, stdio: 'ignore'});
  execFileSync(
    'git',
    ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture base'],
    {cwd: workspaceRoot, stdio: 'ignore'}
  );

  const newEvidenceBytes = evidence('Updated review note.');
  const manifestBytes = manifestFor(newEvidenceBytes);
  await writeFile(evidenceAbsolute, newEvidenceBytes);
  await writeFile(manifestAbsolute, manifestBytes);
  execFileSync('git', ['add', 'packages/web-decision/src/catalog/model-provenance.json'], {
    cwd: workspaceRoot,
    stdio: 'ignore'
  });
  execFileSync(
    'git',
    [
      '-c',
      'user.name=Fixture',
      '-c',
      'user.email=fixture@example.invalid',
      'commit',
      '-m',
      'update the evidence pin only'
    ],
    {cwd: workspaceRoot, stdio: 'ignore'}
  );

  await assert.rejects(
    validateModelProvenance({
      packageRoot,
      workspaceRoot,
      modelSpec,
      artifactLock,
      artifactLockBytes,
      manifestBytes,
      expectedManifestSha256: sha256(manifestBytes),
      expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
      allowTestFixture: false,
      requireCommitted: true
    }),
    /Provenance evidence for fixture-model must match its committed bytes/u
  );
});

test('model IDs and artifact-lock bytes are bound to provenance', async () => {
  const inputs = await readProductionInputs();
  const missingModel = JSON.parse(inputs.manifestBytes.toString('utf8')) as {models: unknown[]};
  missingModel.models = [];
  await assert.rejects(
    validateTemporaryManifest(Buffer.from(`${JSON.stringify(missingModel, null, 2)}\n`), {
      expectedSha256: sha256(Buffer.from(`${JSON.stringify(missingModel, null, 2)}\n`))
    }),
    /Provenance model IDs must exactly match/u
  );

  const changedLock = JSON.parse(inputs.manifestBytes.toString('utf8')) as {
    models: Array<{artifactLockSha256: string}>;
  };
  const first = changedLock.models[0];
  assert.ok(first);
  first.artifactLockSha256 = '0'.repeat(64);
  const changedBytes = Buffer.from(`${JSON.stringify(changedLock, null, 2)}\n`);
  await assert.rejects(
    validateTemporaryManifest(changedBytes, {expectedSha256: sha256(changedBytes)}),
    /Artifact lock digest differs from provenance/u
  );

  const unsafePath = JSON.parse(inputs.manifestBytes.toString('utf8')) as {
    models: Array<{evidence: {path: string}}>;
  };
  const unsafeEntry = unsafePath.models[0];
  assert.ok(unsafeEntry);
  unsafeEntry.evidence.path = '../../outside.json';
  const unsafeBytes = Buffer.from(`${JSON.stringify(unsafePath, null, 2)}\n`);
  await assert.rejects(validateTemporaryManifest(unsafeBytes, {expectedSha256: sha256(unsafeBytes)}), /unsafe path/u);
});

test('the release test fixture is explicitly synthetic and never an approval', async () => {
  const [manifestBytes, evidenceBytes] = await Promise.all([
    readFile(join(WORKSPACE_ROOT, RELEASE_TEST_PROVENANCE_MANIFEST_PATH)),
    readFile(TEST_EVIDENCE_PATH)
  ]);
  assert.equal(sha256(manifestBytes), RELEASE_TEST_PROVENANCE_MANIFEST_SHA256);
  const evidence = JSON.parse(evidenceBytes.toString('utf8')) as Record<string, unknown>;
  assert.equal(evidence.status, 'test-fixture');
  assert.equal(evidence.notAnApproval, true);
  assert.match(String(evidence.purpose), /no source, rights, or release status is asserted/u);
});

function requiredEvidence(value: Buffer | undefined): Buffer {
  if (!value) throw new Error('Synthetic provenance evidence fixture is missing.');
  return value;
}
