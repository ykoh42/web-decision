import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile, realpath} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {verifyArchiveInstall} from '../../../release/verify-release-evidence.ts';

type Report = {
  status: string;
  sourceRevision: string;
  sourceSnapshotSha256: string;
  sourceSnapshot: Array<{file: string; sha256: string}>;
  archive: string;
  archiveSha256: string;
  installed: string;
  installedBundle: string;
  tokenizerPath: string;
  tokenizerSha256: string;
  consumerLockSha256: string;
};
const packageRoot = fileURLToPath(new URL('../../../../packages/web-decision/', import.meta.url));
const hash = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex');

/** Require the exact private archive, installed bytes, dependency, and unchanged source snapshot. */
export async function verifyGemmaV2Candidate(reportPath: string, selectedBundle: string): Promise<Report> {
  const report = JSON.parse(await readFile(reportPath, 'utf8')) as Report;
  assert.equal(report.status, 'private-unreleased-test-candidate');
  assert.equal(await realpath(report.installedBundle), await realpath(selectedBundle));
  assert.equal(hash(await readFile(report.archive)), report.archiveSha256);
  const tokenizer = resolve(report.installed, '../../node_modules/@huggingface/tokenizers/dist/tokenizers.mjs');
  assert.equal(await realpath(report.tokenizerPath), await realpath(tokenizer));
  assert.equal(hash(await readFile(tokenizer)), report.tokenizerSha256);
  assert.equal(hash(await readFile(resolve(report.installed, '../../package-lock.json'))), report.consumerLockSha256);
  assert.equal(hash(JSON.stringify(report.sourceSnapshot)), report.sourceSnapshotSha256);
  assert.equal(
    execFileSync('git', ['rev-parse', 'HEAD'], {cwd: packageRoot, encoding: 'utf8'}).trim(),
    report.sourceRevision
  );
  for (const item of report.sourceSnapshot)
    assert.equal(
      hash(await readFile(resolve(packageRoot, item.file))),
      item.sha256,
      `Source changed after candidate build: ${item.file}`
    );
  await verifyArchiveInstall(report.archive, report.installed);
  return report;
}

/** Pin the report and test files while an installed-candidate test runs. */
export async function gemmaV2EvidenceFingerprint(reportPath: string, testPath: string): Promise<string> {
  const files = [
    reportPath,
    fileURLToPath(import.meta.url),
    fileURLToPath(new URL('./gemma4-v2-cache-evidence.mts', import.meta.url)),
    testPath
  ];
  const hashes = await Promise.all(files.map(async (file) => hash(await readFile(file))));
  return hash(JSON.stringify(hashes));
}
