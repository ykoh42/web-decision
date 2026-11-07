import assert from 'node:assert/strict';
import {Buffer} from 'node:buffer';
import {createHash, randomUUID} from 'node:crypto';
import {link, lstat, open, readFile, realpath, unlink} from 'node:fs/promises';
import {isAbsolute, relative, resolve, sep} from 'node:path';

export const CURRENT_PREVIEW_VERSION = '0.0.0-preview.43-local.1';
export const CURRENT_PREVIEW_RUN_TOKEN_ENV = 'WEB_DECISION_KEV_PREVIEW_RUN_TOKEN';

export function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

export function isPathWithin(root, target) {
  const relativePath = relative(resolve(root), resolve(target));
  return (
    relativePath === '' || (relativePath !== '..' && !relativePath.startsWith(`..${sep}`) && !isAbsolute(relativePath))
  );
}

export function assertCandidateDoesNotOverlapHistory(candidateRoot, historicalRoot) {
  if (isPathWithin(candidateRoot, historicalRoot) || isPathWithin(historicalRoot, candidateRoot)) {
    throw new Error('The current-source candidate path overlaps the historical preview path.');
  }
}

async function getCanonicalRootInfo(root) {
  const info = await lstat(root);
  if (!info.isDirectory() || info.isSymbolicLink() || (await realpath(root)) !== root) {
    throw new Error(`Workspace root must be a canonical directory: ${root}`);
  }
  return info;
}

async function inspectPathFromRoot(root, target, rootInfo, label) {
  let current = root;
  let finalInfo = rootInfo;
  const parts = relative(root, target).split(sep).filter(Boolean);
  for (let index = 0; index < parts.length; index++) {
    current = resolve(current, parts[index]);
    let info;
    try {
      info = await lstat(current);
    } catch (cause) {
      if (cause?.code === 'ENOENT') return undefined;
      throw cause;
    }
    if (info.isSymbolicLink() || (await realpath(current)) !== current) {
      throw new Error(`${label} path contains a symbolic link: ${current}`);
    }
    if (index < parts.length - 1 && !info.isDirectory()) {
      throw new Error(`${label} path ancestor is not a directory: ${current}`);
    }
    finalInfo = info;
  }
  return finalInfo;
}

async function assertHistoricalPathIsCanonical(root, historicalRoot, rootInfo) {
  const history = resolve(historicalRoot);
  if (!isPathWithin(root, history)) throw new Error('The historical preview path is outside the workspace.');
  const info = await inspectPathFromRoot(root, history, rootInfo, 'Historical preview');
  if (info && !info.isDirectory()) throw new Error(`Historical preview path is not a directory: ${history}`);
}

export async function assertSafeCandidatePath({
  workspaceRoot,
  candidateRoot,
  targetPath,
  historicalRoot,
  mustExist = false,
  mustBeAbsent = false,
  expectedType
}) {
  const root = resolve(workspaceRoot);
  const candidate = resolve(candidateRoot);
  const target = resolve(targetPath);
  if (!isPathWithin(root, candidate) || !isPathWithin(candidate, target)) {
    throw new Error(`Candidate path escapes its allowed root: ${target}`);
  }
  const rootInfo = await getCanonicalRootInfo(root);
  if (historicalRoot) {
    assertCandidateDoesNotOverlapHistory(candidate, resolve(historicalRoot));
    await assertHistoricalPathIsCanonical(root, historicalRoot, rootInfo);
  }
  const finalInfo = await inspectPathFromRoot(root, target, rootInfo, 'Candidate');

  if (mustExist && !finalInfo) throw new Error(`Required candidate path does not exist: ${target}`);
  if (mustBeAbsent && finalInfo) throw new Error(`Candidate output already exists: ${target}`);
  if (expectedType === 'directory' && finalInfo && !finalInfo.isDirectory()) {
    throw new Error(`Expected a candidate directory: ${target}`);
  }
  if (expectedType === 'file' && finalInfo && !finalInfo.isFile()) {
    throw new Error(`Expected a candidate file: ${target}`);
  }
  return target;
}

export async function assertFreshOutputPaths({workspaceRoot, candidateRoot, historicalRoot, outputPaths}) {
  for (const targetPath of outputPaths) {
    await assertSafeCandidatePath({
      workspaceRoot,
      candidateRoot,
      historicalRoot,
      targetPath,
      mustBeAbsent: true
    });
  }
}

export async function acquireCandidateRunLock({workspaceRoot, candidateRoot, historicalRoot, lockPath}) {
  await assertSafeCandidatePath({
    workspaceRoot,
    candidateRoot,
    historicalRoot,
    targetPath: lockPath,
    mustBeAbsent: true
  });

  const lock = {schemaVersion: 1, token: randomUUID(), processId: process.pid};
  const handle = await open(lockPath, 'wx', 0o600);
  try {
    await handle.writeFile(`${JSON.stringify(lock)}\n`);
    await handle.sync();
    await handle.close();
  } catch (cause) {
    await handle.close().catch(() => {});
    await unlink(lockPath).catch(() => {});
    throw cause;
  }
  return lock;
}

export async function assertCandidateRunLock({workspaceRoot, candidateRoot, historicalRoot, lockPath, token}) {
  if (typeof token !== 'string' || token.length === 0) {
    throw new Error('The current-source preview run lock token is required.');
  }
  await assertSafeCandidatePath({
    workspaceRoot,
    candidateRoot,
    historicalRoot,
    targetPath: lockPath,
    mustExist: true,
    expectedType: 'file'
  });
  const lock = JSON.parse(await readFile(lockPath, 'utf8'));
  assert.equal(lock.schemaVersion, 1, 'The current-source preview run lock schema is not supported.');
  assert.equal(lock.token, token, 'The current-source preview run lock token does not match.');
  return lock;
}

export async function releaseCandidateRunLock({workspaceRoot, candidateRoot, historicalRoot, lockPath, token}) {
  await assertCandidateRunLock({workspaceRoot, candidateRoot, historicalRoot, lockPath, token});
  await unlink(lockPath);
}

function fingerprintEntries(entries) {
  return sha256(Buffer.from(JSON.stringify(entries)));
}

export async function createBuildManifest({
  workspaceRoot,
  sourcePaths,
  bundleBytes,
  loadedSourceHashes,
  version = CURRENT_PREVIEW_VERSION
}) {
  const root = resolve(workspaceRoot);
  const paths = [...new Set(sourcePaths.map((path) => resolve(path)))].sort();
  const sourceInputs = [];
  for (const path of paths) {
    await assertSafeCandidatePath({
      workspaceRoot: root,
      candidateRoot: root,
      targetPath: path,
      mustExist: true,
      expectedType: 'file'
    });
    if (path === root || !isPathWithin(root, path)) throw new Error(`Source input is outside the workspace: ${path}`);
    const bytes = await readFile(path);
    const sourceHash = sha256(bytes);
    const loadedHash = loadedSourceHashes?.get(path);
    if (loadedHash !== undefined) {
      assert.equal(sourceHash, loadedHash, `A source changed after Rollup loaded it: ${path}`);
    }
    sourceInputs.push({path: relative(root, path).split(sep).join('/'), bytes: bytes.length, sha256: sourceHash});
  }
  sourceInputs.sort((left, right) => left.path.localeCompare(right.path));
  return {
    schemaVersion: 1,
    packageVersion: version,
    sourceFingerprint: fingerprintEntries(sourceInputs),
    sourceInputs,
    bundle: {bytes: bundleBytes.length, sha256: sha256(bundleBytes)}
  };
}

export async function verifyBuildManifest({manifest, workspaceRoot, bundleBytes, version = CURRENT_PREVIEW_VERSION}) {
  assert.equal(manifest?.schemaVersion, 1, 'The build manifest schema is not supported.');
  assert.equal(manifest.packageVersion, version, 'The build manifest belongs to a different local candidate.');
  assert.equal(manifest.bundle?.bytes, bundleBytes.length, 'The build bundle size differs from its manifest.');
  assert.equal(manifest.bundle?.sha256, sha256(bundleBytes), 'The build bundle hash differs from its manifest.');
  assert.ok(
    Array.isArray(manifest.sourceInputs) && manifest.sourceInputs.length > 0,
    'No build source inputs were recorded.'
  );

  const root = resolve(workspaceRoot);
  const sourceInputs = [...manifest.sourceInputs];
  if (sourceInputs.some((input) => typeof input?.path !== 'string')) {
    throw new Error('Invalid source path in the build manifest.');
  }
  sourceInputs.sort((left, right) => left.path.localeCompare(right.path));
  const seenPaths = new Set();
  for (const input of sourceInputs) {
    if (
      typeof input?.path !== 'string' ||
      !input.path ||
      isAbsolute(input.path) ||
      input.path.includes('\\') ||
      input.path.split('/').some((part) => part === '' || part === '.' || part === '..') ||
      seenPaths.has(input.path)
    ) {
      throw new Error('Invalid source path in the build manifest.');
    }
    seenPaths.add(input.path);
    const path = resolve(root, input.path);
    if (!isPathWithin(root, path)) throw new Error('A build source path escapes the workspace.');
    if (relative(root, path).split(sep).join('/') !== input.path) {
      throw new Error('A build source path is not normalized.');
    }
    if (!Number.isSafeInteger(input.bytes) || input.bytes < 0 || !/^[a-f0-9]{64}$/.test(input.sha256)) {
      throw new Error(`Invalid source fingerprint entry: ${input.path}`);
    }
    await assertSafeCandidatePath({
      workspaceRoot: root,
      candidateRoot: root,
      targetPath: path,
      mustExist: true,
      expectedType: 'file'
    });
    const bytes = await readFile(path);
    assert.equal(input.bytes, bytes.length, `A build source size changed: ${input.path}`);
    assert.equal(input.sha256, sha256(bytes), `A build source changed after the build: ${input.path}`);
  }
  assert.equal(
    manifest.sourceFingerprint,
    fingerprintEntries(sourceInputs),
    'The build source fingerprint is inconsistent.'
  );
  return manifest;
}

async function writeEvidenceContents(handle, contents) {
  await handle.writeFile(contents);
  await handle.sync();
}

export async function writeEvidenceOnce({
  workspaceRoot,
  candidateRoot,
  historicalRoot,
  evidencePath,
  evidence,
  writeTemporaryContents = writeEvidenceContents,
  beforePromote = async () => {}
}) {
  await assertSafeCandidatePath({
    workspaceRoot,
    candidateRoot,
    historicalRoot,
    targetPath: evidencePath,
    mustBeAbsent: true
  });
  const temporaryPath = `${evidencePath}.${randomUUID()}.tmp`;
  const contents = Buffer.from(`${JSON.stringify(evidence, null, 2)}\n`);
  let handle;
  let createdTemporaryFile = false;
  try {
    handle = await open(temporaryPath, 'wx', 0o600);
    createdTemporaryFile = true;
    await writeTemporaryContents(handle, contents);
    await handle.close();
    handle = undefined;
    await beforePromote();
    await link(temporaryPath, evidencePath);
  } finally {
    try {
      if (handle) await handle.close();
    } finally {
      if (createdTemporaryFile) {
        await unlink(temporaryPath).catch((cause) => {
          if (cause?.code !== 'ENOENT') throw cause;
        });
      }
    }
  }
}
