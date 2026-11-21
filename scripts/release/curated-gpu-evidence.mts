import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import type {ArchiveInfo} from './verify-release-evidence.ts';

export function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

export function cleanSourceCommit(workspaceRoot: string): string {
  const status = execFileSync('git', ['status', '--porcelain'], {cwd: workspaceRoot, encoding: 'utf8'});
  assert.equal(status, '', 'GPU release evidence requires a clean source checkout.');
  return execFileSync('git', ['rev-parse', 'HEAD'], {cwd: workspaceRoot, encoding: 'utf8'}).trim();
}

export function record(value: unknown, label: string): Record<string, unknown> {
  assert.ok(value && typeof value === 'object' && !Array.isArray(value), label);
  return value as Record<string, unknown>;
}

export function archiveIdentity(archive: ArchiveInfo) {
  return {
    sha256: archive.archiveSha256,
    packageTreeSha256: archive.packageTreeSha256,
    bundleSha256: archive.bundleSha256,
    packageName: archive.packageName,
    packageVersion: archive.packageVersion
  };
}

export function assertEvidenceArchive(value: unknown, archive: ArchiveInfo): void {
  assert.deepEqual(value, archiveIdentity(archive), 'GPU evidence belongs to a different npm archive or installation.');
}
