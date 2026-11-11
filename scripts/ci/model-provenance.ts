import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {lstat, readFile, realpath} from 'node:fs/promises';
import {isAbsolute, join, relative, resolve, sep} from 'node:path';

export const MODEL_PROVENANCE_MANIFEST_PATH = 'src/catalog/model-provenance.json';
export const MODEL_PROVENANCE_MANIFEST_SHA256 = 'ac4d4836dfdc59dbd9adba9e53065e411824a1fb62e54439e6cbe81b82bdb74c';
export const MODEL_PROVENANCE_EVIDENCE_ROOT = 'src/catalog/provenance';
export const RELEASE_TEST_PROVENANCE_MANIFEST_SHA256 =
  '2f709fc940e00bcfe15f4e9f49c49c067d9654affa0d17d5997c42677e277870';
export const RELEASE_TEST_PROVENANCE_MANIFEST_PATH = 'scripts/release/fixtures/model-provenance.json';

const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const IMMUTABLE_REVISION_PATTERN = /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/u;
const UNKNOWN_TERMS = new Set(['', 'pending', 'unknown', 'unresolved', 'separate-review-required', 'not-reviewed']);

interface ModelProvenanceEntry {
  modelId: string;
  status: 'pending' | 'approved' | 'test-fixture';
  artifactLockSha256: string;
  evidence: {path: string; sha256: string};
}

interface ProvenanceManifest {
  schemaVersion: 1;
  models: ModelProvenanceEntry[];
}

interface ProvenanceValidationOptions {
  packageRoot: string;
  workspaceRoot: string;
  modelSpec: unknown;
  artifactLock: unknown;
  artifactLockBytes: Uint8Array;
  manifestBytes: Uint8Array;
  expectedManifestSha256: string;
  expectedEvidenceRoot: string;
  allowTestFixture: boolean;
  requireCommitted: boolean;
  allowPending?: boolean;
}

export function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function failUnless(condition: unknown, message: string): asserts condition {
  assert.ok(condition, message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isResolvedTerm(value: unknown): value is string {
  return typeof value === 'string' && !UNKNOWN_TERMS.has(value.trim().toLowerCase());
}

function isPathWithin(root: string, target: string): boolean {
  const path = relative(resolve(root), resolve(target));
  return path === '' || (path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}

function safeRelativePath(value: unknown, description: string): string {
  failUnless(typeof value === 'string' && value.length > 0, `${description} must be a non-empty relative path.`);
  failUnless(!value.includes('\\') && !isAbsolute(value), `${description} has an unsafe path: ${value}`);
  const parts = value.split('/');
  failUnless(
    parts.every((part) => part.length > 0 && part !== '.' && part !== '..'),
    `${description} has an unsafe path: ${value}`
  );
  return parts.join('/');
}

function modelIds(spec: unknown): string[] {
  failUnless(isRecord(spec) && spec.schemaVersion === 1 && Array.isArray(spec.models), 'Invalid model spec.');
  const ids = spec.models.map((model: unknown) => {
    failUnless(isRecord(model) && typeof model.modelId === 'string', 'Every model spec entry needs a modelId.');
    return model.modelId;
  });
  failUnless(new Set(ids).size === ids.length, 'Model spec contains duplicate model IDs.');
  return ids.sort();
}

function lockModelIds(lock: unknown): string[] {
  failUnless(
    isRecord(lock) && lock.schemaVersion === 1 && isRecord(lock.byModelId),
    'Artifact lock has an unsupported schema.'
  );
  return Object.keys(lock.byModelId).sort();
}

function parseManifest(bytes: Uint8Array): ProvenanceManifest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.from(bytes).toString('utf8')) as unknown;
  } catch (error) {
    throw new Error('Model provenance manifest is not valid JSON.', {cause: error});
  }
  failUnless(
    isRecord(parsed) && parsed.schemaVersion === 1 && Array.isArray(parsed.models),
    'Model provenance manifest has an unsupported schema.'
  );
  return parsed as unknown as ProvenanceManifest;
}

async function validateReviewEvidence(
  value: unknown,
  modelId: string,
  artifactEntry: unknown,
  workspaceRoot: string,
  requireCommitted: boolean
): Promise<void> {
  failUnless(isRecord(value), `Provenance evidence for ${modelId} must be an object.`);
  failUnless(
    value.schemaVersion === 1 && value.modelId === modelId,
    `Provenance evidence model ID mismatch for ${modelId}.`
  );
  failUnless(value.status === 'approved', `Provenance evidence for ${modelId} is not approved.`);
  validateSourceLineage(value.sourceLineage, modelId);
  validateLockedUpstream(value.sourceLineage, artifactEntry, modelId);
  await validateConversionEvidence(value.conversion, modelId, workspaceRoot, requireCommitted);
  await validateRightsEvidence(value.rights, modelId, workspaceRoot, requireCommitted);
  await validateApprovalEvidence(value.approval, modelId, workspaceRoot, requireCommitted);
}

function validateSourceLineage(sourceLineage: unknown, modelId: string): void {
  failUnless(
    Array.isArray(sourceLineage) && sourceLineage.length > 0,
    `Approved provenance for ${modelId} needs source lineage.`
  );
  for (const [index, item] of sourceLineage.entries()) {
    failUnless(isRecord(item), `Source lineage ${index} for ${modelId} must be an object.`);
    failUnless(
      typeof item.repository === 'string' && item.repository.length > 0,
      `Source repository ${index} for ${modelId} is required.`
    );
    failUnless(
      typeof item.revision === 'string' && IMMUTABLE_REVISION_PATTERN.test(item.revision),
      `Source revision ${index} for ${modelId} must be immutable.`
    );
    safeRelativePath(item.path, `Source path ${index} for ${modelId}`);
    failUnless(
      Number.isSafeInteger(item.bytes) && (item.bytes as number) >= 0,
      `Source byte length ${index} for ${modelId} is invalid.`
    );
    failUnless(
      typeof item.sha256 === 'string' && SHA256_PATTERN.test(item.sha256),
      `Source SHA-256 ${index} for ${modelId} is invalid.`
    );
  }
}

interface LockedUpstream {
  readonly role?: string;
  readonly repository: string;
  readonly revision: string;
  readonly sourcePath: string;
  readonly sourceBytes: number;
  readonly sourceSha256: string;
}

function isLockedUpstream(value: unknown): value is LockedUpstream {
  return (
    isRecord(value) &&
    (value.role === undefined || (typeof value.role === 'string' && value.role.length > 0)) &&
    typeof value.repository === 'string' &&
    typeof value.revision === 'string' &&
    typeof value.sourcePath === 'string' &&
    Number.isSafeInteger(value.sourceBytes) &&
    (value.sourceBytes as number) >= 0 &&
    typeof value.sourceSha256 === 'string' &&
    SHA256_PATTERN.test(value.sourceSha256)
  );
}

function matchesLockedUpstream(item: unknown, upstream: LockedUpstream, requireRole: boolean): boolean {
  return (
    isRecord(item) &&
    item.repository === upstream.repository &&
    item.revision === upstream.revision &&
    item.path === upstream.sourcePath &&
    item.sha256 === upstream.sourceSha256 &&
    item.bytes === upstream.sourceBytes &&
    (!requireRole || item.role === upstream.role)
  );
}

export function validateLockedUpstream(sourceLineage: unknown, artifactEntry: unknown, modelId: string): void {
  failUnless(isRecord(artifactEntry), `Artifact lock entry for ${modelId} must be an object.`);
  const requiresExactSources = Object.hasOwn(artifactEntry, 'upstreams');
  if (!requiresExactSources)
    failUnless(isRecord(artifactEntry.upstream), `Locked upstream for ${modelId} must be an object.`);
  const upstreams = requiresExactSources ? artifactEntry.upstreams : [artifactEntry.upstream];
  failUnless(
    Array.isArray(upstreams) && upstreams.length > 0,
    `Locked upstream for ${modelId} must contain source records.`
  );
  failUnless(upstreams.every(isLockedUpstream), `Locked upstream for ${modelId} is incomplete.`);
  failUnless(
    !requiresExactSources || upstreams.every((item) => typeof item.role === 'string'),
    `Every locked source for ${modelId} needs a role.`
  );
  const roles = requiresExactSources ? upstreams.map((item) => item.role) : [];
  failUnless(
    !requiresExactSources || new Set(roles).size === roles.length,
    `Locked source roles for ${modelId} must be unique.`
  );
  failUnless(
    Array.isArray(sourceLineage) &&
      (!requiresExactSources || sourceLineage.length === upstreams.length) &&
      upstreams.every((upstream) =>
        sourceLineage.some((item: unknown) => matchesLockedUpstream(item, upstream, requiresExactSources))
      ),
    `Approved source lineage for ${modelId} does not match its locked upstream artifact.`
  );
}

async function validateConversionEvidence(
  conversion: unknown,
  modelId: string,
  workspaceRoot: string,
  requireCommitted: boolean
): Promise<void> {
  failUnless(isRecord(conversion), `Approved provenance for ${modelId} needs conversion evidence.`);
  failUnless(
    typeof conversion.repository === 'string' && conversion.repository.length > 0,
    `Conversion repository for ${modelId} is required.`
  );
  failUnless(
    typeof conversion.revision === 'string' && IMMUTABLE_REVISION_PATTERN.test(conversion.revision),
    `Conversion revision for ${modelId} must be immutable.`
  );
  for (const field of ['recipe', 'config', 'toolchain']) {
    const reference = conversion[field];
    safeRelativePath(isRecord(reference) ? reference.path : undefined, `Conversion ${field} path for ${modelId}`);
    await validateCommittedEvidenceReference(
      reference,
      `Conversion ${field} for ${modelId}`,
      workspaceRoot,
      requireCommitted
    );
  }
  failUnless(
    typeof conversion.artifactLockSha256 === 'string' && SHA256_PATTERN.test(conversion.artifactLockSha256),
    `Conversion artifact-lock SHA-256 for ${modelId} is invalid.`
  );
}

async function validateRightsEvidence(
  rights: unknown,
  modelId: string,
  workspaceRoot: string,
  requireCommitted: boolean
): Promise<void> {
  failUnless(Array.isArray(rights) && rights.length > 0, `Approved provenance for ${modelId} needs rights evidence.`);
  for (const [index, item] of rights.entries()) {
    failUnless(isRecord(item), `Rights evidence ${index} for ${modelId} must be an object.`);
    failUnless(
      typeof item.subject === 'string' && item.subject.length > 0,
      `Rights subject ${index} for ${modelId} is required.`
    );
    failUnless(isResolvedTerm(item.license), `Rights license ${index} for ${modelId} is unresolved.`);
    failUnless(item.redistribution === 'permitted', `Redistribution rights ${index} for ${modelId} are not confirmed.`);
    await validateCommittedEvidenceReference(
      item.evidence,
      `Rights evidence ${index} for ${modelId}`,
      workspaceRoot,
      requireCommitted
    );
  }
}

async function validateApprovalEvidence(
  approval: unknown,
  modelId: string,
  workspaceRoot: string,
  requireCommitted: boolean
): Promise<void> {
  failUnless(
    isRecord(approval) && approval.decision === 'approved',
    `Approved provenance for ${modelId} needs a review decision.`
  );
  failUnless(
    typeof approval.reviewer === 'string' && approval.reviewer.trim().length > 0,
    `Approval reviewer for ${modelId} is required.`
  );
  failUnless(
    typeof approval.reviewedAt === 'string' && !Number.isNaN(Date.parse(approval.reviewedAt)),
    `Approval date for ${modelId} is invalid.`
  );
  await validateCommittedEvidenceReference(
    approval.evidence,
    `Approval evidence for ${modelId}`,
    workspaceRoot,
    requireCommitted
  );
}

async function validateCommittedEvidenceReference(
  value: unknown,
  description: string,
  workspaceRoot: string,
  requireCommitted: boolean
): Promise<void> {
  failUnless(isRecord(value), `${description} must include a committed file reference.`);
  const path = safeRelativePath(value.path, `${description} path`);
  failUnless(
    typeof value.sha256 === 'string' && SHA256_PATTERN.test(value.sha256),
    `${description} SHA-256 is invalid.`
  );
  failUnless(
    path.startsWith('docs/') || path.startsWith('packages/web-decision/src/catalog/provenance/'),
    `${description} must be under docs/ or the catalog provenance directory.`
  );
  const absolute = resolve(workspaceRoot, path);
  failUnless(isPathWithin(workspaceRoot, absolute), `${description} escapes the workspace.`);
  const info = await lstat(absolute);
  failUnless(info.isFile() && !info.isSymbolicLink(), `${description} must be a regular committed file.`);
  assert.equal(await realpath(absolute), absolute, `${description} must use its canonical path.`);
  const bytes = await readFile(absolute);
  assert.equal(sha256(bytes), value.sha256, `${description} SHA-256 mismatch.`);
  if (requireCommitted) assertCommittedFile(workspaceRoot, path, bytes, description);
}

async function validatePendingProvenance(
  evidence: unknown,
  modelId: string,
  artifactLockSha256: string,
  workspaceRoot: string,
  requireCommitted: boolean,
  allowPending: boolean
): Promise<void> {
  failUnless(
    isRecord(evidence) &&
      evidence.status === 'pending' &&
      evidence.modelId === modelId &&
      Array.isArray(evidence.unresolved) &&
      evidence.unresolved.length > 0 &&
      evidence.unresolved.every((item) => typeof item === 'string' && item.length > 0),
    `Pending provenance for ${modelId} must list unresolved evidence.`
  );
  if (isRecord(evidence.conversion)) {
    await validateConversionEvidence(evidence.conversion, modelId, workspaceRoot, requireCommitted);
    assert.equal(
      evidence.conversion.artifactLockSha256,
      artifactLockSha256,
      `Pending conversion artifact-lock digest differs for ${modelId}.`
    );
  }
  if (allowPending) return;
  throw new Error(`Model ${modelId} provenance is pending: ${(evidence.unresolved as string[]).join('; ')}`);
}

export async function validateModelProvenance(options: ProvenanceValidationOptions): Promise<{
  manifestSha256: string;
  artifactLockSha256: string;
  modelIds: string[];
  pendingModelIds: string[];
}> {
  const manifestSha256 = sha256(options.manifestBytes);
  assert.equal(
    manifestSha256,
    options.expectedManifestSha256,
    'Model provenance manifest differs from its fixed SHA-256 pin.'
  );
  if (options.requireCommitted) {
    const manifestAbsolute = resolve(options.packageRoot, MODEL_PROVENANCE_MANIFEST_PATH);
    const manifestRelative = relative(options.workspaceRoot, manifestAbsolute).split(sep).join('/');
    assertCommittedFile(
      options.workspaceRoot,
      manifestRelative,
      Buffer.from(options.manifestBytes),
      'Model provenance manifest'
    );
  }
  const artifactLockSha256 = sha256(options.artifactLockBytes);
  const ids = modelIds(options.modelSpec);
  assert.deepEqual(
    lockModelIds(options.artifactLock),
    ids,
    'Artifact lock model IDs must exactly match the model spec.'
  );
  const manifest = parseManifest(options.manifestBytes);
  const pendingModelIds: string[] = [];
  const entryIds = manifest.models.map((entry) => entry?.modelId);
  failUnless(
    entryIds.every((id) => typeof id === 'string'),
    'Every provenance record needs a modelId.'
  );
  assert.deepEqual([...entryIds].sort(), ids, 'Provenance model IDs must exactly match the public model spec.');
  failUnless(new Set(entryIds).size === entryIds.length, 'Provenance manifest contains duplicate model IDs.');

  for (const entry of manifest.models) {
    failUnless(
      entry.modelId in (options.artifactLock as {byModelId: Record<string, unknown>}).byModelId,
      `Artifact lock is missing ${entry.modelId}.`
    );
    assert.equal(
      entry.artifactLockSha256,
      artifactLockSha256,
      `Artifact lock digest differs from provenance for ${entry.modelId}.`
    );
    failUnless(
      ['pending', 'approved', 'test-fixture'].includes(entry.status),
      `Invalid provenance status for ${entry.modelId}.`
    );
    const evidencePath = safeRelativePath(entry.evidence?.path, `Provenance evidence path for ${entry.modelId}`);
    failUnless(
      evidencePath.startsWith(options.expectedEvidenceRoot + '/'),
      `Provenance evidence for ${entry.modelId} is outside the approved evidence directory.`
    );
    failUnless(
      SHA256_PATTERN.test(entry.evidence?.sha256 ?? ''),
      `Provenance evidence hash for ${entry.modelId} is invalid.`
    );
    const evidenceAbsolute = resolve(options.packageRoot, evidencePath);
    failUnless(
      isPathWithin(options.packageRoot, evidenceAbsolute),
      `Provenance evidence escapes the package root for ${entry.modelId}.`
    );
    const evidenceStat = await lstat(evidenceAbsolute);
    failUnless(
      evidenceStat.isFile() && !evidenceStat.isSymbolicLink(),
      `Provenance evidence must be a regular file for ${entry.modelId}.`
    );
    assert.equal(
      await realpath(evidenceAbsolute),
      evidenceAbsolute,
      `Provenance evidence must use its canonical path for ${entry.modelId}.`
    );
    const evidenceBytes = await readFile(evidenceAbsolute);
    assert.equal(
      sha256(evidenceBytes),
      entry.evidence.sha256,
      `Provenance evidence SHA-256 mismatch for ${entry.modelId}.`
    );
    const evidenceValue = JSON.parse(evidenceBytes.toString('utf8')) as unknown;

    if (entry.status === 'test-fixture') {
      failUnless(options.allowTestFixture, `Test-fixture provenance is not allowed for ${entry.modelId}.`);
      failUnless(
        isRecord(evidenceValue) &&
          evidenceValue.status === 'test-fixture' &&
          evidenceValue.modelId === entry.modelId &&
          evidenceValue.notAnApproval === true,
        `Release test fixture evidence is not explicit for ${entry.modelId}.`
      );
      continue;
    }

    if (options.requireCommitted) {
      const relativePath = relative(options.workspaceRoot, evidenceAbsolute).split(sep).join('/');
      assertCommittedFile(
        options.workspaceRoot,
        relativePath,
        evidenceBytes,
        `Provenance evidence for ${entry.modelId}`
      );
    }

    if (entry.status === 'pending') {
      await validatePendingProvenance(
        evidenceValue,
        entry.modelId,
        artifactLockSha256,
        options.workspaceRoot,
        options.requireCommitted,
        options.allowPending ?? false
      );
      pendingModelIds.push(entry.modelId);
      continue;
    }

    await validateReviewEvidence(
      evidenceValue,
      entry.modelId,
      (options.artifactLock as {byModelId: Record<string, unknown>}).byModelId[entry.modelId],
      options.workspaceRoot,
      options.requireCommitted
    );
    const specModels = (options.modelSpec as {models: Array<{modelId: string; description: Record<string, unknown>}>})
      .models;
    const spec = specModels.find((item) => item.modelId === entry.modelId);
    const licenses = spec?.description.licenses;
    failUnless(isRecord(licenses), `Approved model ${entry.modelId} must declare reviewed license terms.`);
    failUnless(isResolvedTerm(licenses.modelTerms), `Model ${entry.modelId} catalog license terms remain unresolved.`);
    failUnless(isRecord(evidenceValue), `Approved provenance for ${entry.modelId} must be an object.`);
    failUnless(
      Array.isArray(evidenceValue.rights) &&
        evidenceValue.rights.some((item) => isRecord(item) && item.license === licenses.modelTerms),
      `Catalog model terms for ${entry.modelId} do not match reviewed rights evidence.`
    );
    const conversion = evidenceValue.conversion;
    failUnless(isRecord(conversion), `Approved model ${entry.modelId} has no conversion evidence.`);
    assert.equal(
      conversion.artifactLockSha256,
      artifactLockSha256,
      `Approved conversion artifact-lock digest differs for ${entry.modelId}.`
    );
  }

  return {manifestSha256, artifactLockSha256, modelIds: ids, pendingModelIds};
}

function assertCommittedFile(workspaceRoot: string, path: string, bytes: Uint8Array, description: string): void {
  try {
    execFileSync('git', ['ls-files', '--error-unmatch', path], {cwd: workspaceRoot, stdio: 'ignore'});
  } catch (error) {
    throw new Error(`${description} must be tracked in Git: ${path}`, {cause: error});
  }
  const committed = execFileSync('git', ['show', `HEAD:${path}`], {cwd: workspaceRoot});
  assert.deepEqual(Buffer.from(bytes), committed, `${description} must match its committed bytes.`);
}

export async function assertCommittedPackageProvenance(
  workspaceRoot: string,
  options: {allowPending?: boolean} = {}
): Promise<{
  manifestSha256: string;
  artifactLockSha256: string;
  modelIds: string[];
  pendingModelIds: string[];
}> {
  const root = resolve(workspaceRoot);
  const packageRoot = resolve(root, 'packages/web-decision');
  assert.equal(await realpath(packageRoot), packageRoot, 'Model package root must use its canonical path.');
  const manifestPath = resolve(packageRoot, MODEL_PROVENANCE_MANIFEST_PATH);
  const manifestBytes = await readFile(manifestPath);
  const modelSpecPath = join(packageRoot, 'src/catalog/model-spec.json');
  const artifactLockPath = join(packageRoot, 'src/catalog/artifacts.lock.json');
  const [modelSpecBytes, artifactLockBytes] = await Promise.all([readFile(modelSpecPath), readFile(artifactLockPath)]);
  assertCommittedFile(root, 'packages/web-decision/src/catalog/model-spec.json', modelSpecBytes, 'Model spec');
  assertCommittedFile(
    root,
    'packages/web-decision/src/catalog/artifacts.lock.json',
    artifactLockBytes,
    'Artifact lock'
  );
  const result = await validateModelProvenance({
    packageRoot,
    workspaceRoot: root,
    modelSpec: JSON.parse(modelSpecBytes.toString('utf8')) as unknown,
    artifactLock: JSON.parse(artifactLockBytes.toString('utf8')) as unknown,
    artifactLockBytes,
    manifestBytes,
    expectedManifestSha256: MODEL_PROVENANCE_MANIFEST_SHA256,
    expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
    allowTestFixture: false,
    requireCommitted: true,
    ...(options.allowPending === undefined ? {} : {allowPending: options.allowPending})
  });
  return result;
}

export async function assertTestFixtureProvenance(
  packageRoot: string,
  workspaceRoot: string,
  modelSpec: unknown,
  artifactLock: unknown,
  artifactLockBytes: Uint8Array
): Promise<{manifestSha256: string; artifactLockSha256: string; modelIds: string[]}> {
  const manifestBytes = await readFile(join(packageRoot, MODEL_PROVENANCE_MANIFEST_PATH));
  return validateModelProvenance({
    packageRoot,
    workspaceRoot,
    modelSpec,
    artifactLock,
    artifactLockBytes,
    manifestBytes,
    expectedManifestSha256: RELEASE_TEST_PROVENANCE_MANIFEST_SHA256,
    expectedEvidenceRoot: MODEL_PROVENANCE_EVIDENCE_ROOT,
    allowTestFixture: true,
    requireCommitted: false
  });
}

export function assertProductionPackageRoot(packageRoot: string, workspaceRoot: string): void {
  assert.equal(
    resolve(packageRoot),
    resolve(workspaceRoot, 'packages/web-decision'),
    'Release verification must use the committed package source root.'
  );
}
