import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdir, mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
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
  validateModelProvenance
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
  options: {expectedSha256?: string; evidenceBytes?: Buffer} = {}
): Promise<void> {
  const temporaryRoot = await mkdtemp(join(tmpdir(), 'web-decision-model-provenance-'));
  try {
    const packageRoot = join(temporaryRoot, 'package');
    const evidenceRoot = join(packageRoot, MODEL_PROVENANCE_EVIDENCE_ROOT);
    await mkdir(evidenceRoot, {recursive: true});
    const source = options.evidenceBytes
      ? undefined
      : await readFile(join(PACKAGE_ROOT, MODEL_PROVENANCE_EVIDENCE_ROOT, 'kev-0.6b-q4f16.json'));
    await writeFile(join(evidenceRoot, 'kev-0.6b-q4f16.json'), options.evidenceBytes ?? source ?? Buffer.alloc(0));
    const inputs = await readProductionInputs();
    await validateModelProvenance({
      packageRoot,
      workspaceRoot: WORKSPACE_ROOT,
      modelSpec: inputs.modelSpec,
      artifactLock: inputs.artifactLock,
      artifactLockBytes: inputs.artifactLockBytes,
      manifestBytes,
      expectedManifestSha256: options.expectedSha256 ?? MODEL_PROVENANCE_MANIFEST_SHA256,
      expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
      allowTestFixture: false,
      requireCommitted: false
    });
  } finally {
    await rm(temporaryRoot, {recursive: true, force: true});
  }
}

test('the pinned public-model record remains pending with explicit unresolved lineage and rights', async () => {
  const {manifestBytes} = await readProductionInputs();
  assert.equal(sha256(manifestBytes), MODEL_PROVENANCE_MANIFEST_SHA256);
  await assert.rejects(
    validateTemporaryManifest(manifestBytes),
    /Model kev-0\.6b-q4f16 provenance is pending: .*Exact Kev adapter source revision.*Exact Qwen3-0\.6B base revision.*Redistribution rights/u
  );
});

test('changing only pending to approved cannot bypass the fixed provenance hash', async () => {
  const {manifestBytes} = await readProductionInputs();
  const manifest = JSON.parse(manifestBytes.toString('utf8')) as {
    models: Array<{status: string}>;
  };
  const first = manifest.models[0];
  assert.ok(first);
  first.status = 'approved';
  await assert.rejects(
    validateTemporaryManifest(Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`)),
    /fixed SHA-256 pin/u
  );
});

test('approved provenance requires immutable lineage, pinned conversion, rights, and review evidence', async (t) => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), 'web-decision-approved-provenance-'));
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
    ['rights.md', Buffer.from('Rights reviewed for redistribution.\n')],
    ['approval.md', Buffer.from('Approval record.\n')]
  ]);
  for (const [name, bytes] of evidenceFiles) await writeFile(join(docsRoot, name), bytes);
  const reference = (name: string) => ({
    path: `docs/${name}`,
    sha256: sha256(requiredEvidence(evidenceFiles.get(name)))
  });
  const {artifactLock, artifactLockBytes, modelSpec} = await readProductionInputs();
  const approvedSpec = structuredClone(modelSpec) as {
    models: Array<{description: {licenses: {modelTerms: string}}}>;
  };
  const specModel = approvedSpec.models[0];
  assert.ok(specModel);
  specModel.description.licenses.modelTerms = 'Apache-2.0';
  const artifactLockSha256 = sha256(artifactLockBytes);
  const approvalEvidence = {
    schemaVersion: 1,
    modelId: 'kev-0.6b-q4f16',
    status: 'approved',
    sourceLineage: [
      {
        repository: 'example/model',
        revision: 'a'.repeat(40),
        path: 'weights/model.bin',
        sha256: 'b'.repeat(64)
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
        license: 'Apache-2.0',
        redistribution: 'permitted',
        evidence: reference('rights.md')
      }
    ],
    approval: {
      decision: 'approved',
      reviewer: 'authorized-reviewer',
      reviewedAt: '2026-11-09T00:00:00Z',
      evidence: reference('approval.md')
    }
  };
  const evidenceBytes = Buffer.from(`${JSON.stringify(approvalEvidence, null, 2)}\n`);
  await writeFile(join(provenanceRoot, 'kev-0.6b-q4f16.json'), evidenceBytes);
  const manifestBytes = Buffer.from(
    `${JSON.stringify(
      {
        schemaVersion: 1,
        models: [
          {
            modelId: 'kev-0.6b-q4f16',
            status: 'approved',
            artifactLockSha256,
            evidence: {
              path: 'src/catalog/provenance/kev-0.6b-q4f16.json',
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
    modelSpec: approvedSpec,
    artifactLock,
    artifactLockBytes,
    manifestBytes,
    expectedManifestSha256: sha256(manifestBytes),
    expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
    allowTestFixture: false,
    requireCommitted: false
  });
  assert.deepEqual(checked.modelIds, ['kev-0.6b-q4f16']);
  assert.equal(checked.artifactLockSha256, artifactLockSha256);

  await rm(join(docsRoot, 'toolchain.json'));
  await assert.rejects(
    validateModelProvenance({
      packageRoot,
      workspaceRoot: temporaryRoot,
      modelSpec: approvedSpec,
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

  const unresolvedSpec = structuredClone(approvedSpec);
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

test('tampered committed evidence cannot match the manifest evidence digest', async () => {
  const {manifestBytes} = await readProductionInputs();
  await assert.rejects(
    validateTemporaryManifest(manifestBytes, {evidenceBytes: Buffer.from('{"status":"approved"}\n')}),
    /Provenance evidence SHA-256 mismatch/u
  );
});

test('an evidence file must match its committed HEAD bytes even when its hash is updated', async (t) => {
  const workspaceRoot = await mkdtemp(join(tmpdir(), 'web-decision-committed-provenance-'));
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
