import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {lstat, readFile, realpath} from 'node:fs/promises';
import {dirname, isAbsolute, relative, resolve, sep} from 'node:path';

export function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

export function isPathWithin(root, target) {
  const path = relative(resolve(root), resolve(target));
  return path === '' || (path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}

export function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical(value[key])])
    );
  }
  return value;
}

export function canonicalSha256(value) {
  return sha256(Buffer.from(`${JSON.stringify(canonical(value))}\n`));
}

export async function assertRegularFile(path) {
  const absolute = resolve(path);
  const info = await lstat(absolute);
  assert.ok(info.isFile() && !info.isSymbolicLink(), `Expected a regular file: ${absolute}`);
  assert.equal(await realpath(absolute), absolute, `File path is not canonical: ${absolute}`);
  return absolute;
}

export async function assertFreshOutputDirectory(parent, output, label) {
  const absoluteParent = resolve(parent);
  const absoluteOutput = resolve(output);
  assert.equal(dirname(absoluteOutput), absoluteParent, `${label} must be a direct child of its parent.`);

  const parentInfo = await lstat(absoluteParent);
  assert.ok(parentInfo.isDirectory() && !parentInfo.isSymbolicLink(), `${label} parent must be a canonical directory.`);
  assert.equal(await realpath(absoluteParent), absoluteParent, `${label} parent must be a canonical directory.`);

  try {
    await lstat(absoluteOutput);
  } catch (error) {
    if (error.code === 'ENOENT') return absoluteOutput;
    throw error;
  }

  throw new Error(`Refusing to overwrite ${label}: ${absoluteOutput}`);
}

export async function verifySourceLock({workspaceRoot, lockPath}) {
  const root = resolve(workspaceRoot);
  const lock = JSON.parse(await readFile(await assertRegularFile(lockPath), 'utf8'));
  assert.equal(lock.schemaVersion, 1, 'Unsupported release source-lock schema.');
  assert.ok(Array.isArray(lock.files) && lock.files.length > 0, 'Release source-lock is empty.');
  const seen = new Set();

  for (const entry of lock.files) {
    assert.equal(typeof entry.path, 'string');
    assert.ok(!isAbsolute(entry.path), `Absolute path in release source-lock: ${entry.path}`);
    assert.ok(!seen.has(entry.path), `Duplicate path in release source-lock: ${entry.path}`);
    seen.add(entry.path);
    const path = resolve(root, entry.path);
    assert.ok(isPathWithin(root, path), `Release source-lock path escapes the checkout: ${entry.path}`);
    await assertRegularFile(path);
    const bytes = await readFile(path);
    assert.equal(bytes.length, entry.bytes, `Unexpected file size: ${entry.path}`);
    assert.equal(sha256(bytes), entry.sha256, `Unexpected file hash: ${entry.path}`);
  }

  return lock;
}
