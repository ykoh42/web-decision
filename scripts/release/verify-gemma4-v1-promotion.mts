import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {lstat, readFile, realpath} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import artifactLock from '../../packages/web-decision/src/catalog/artifacts.lock.json' with {type: 'json'};
import {gitEnvironmentFor} from '../git-worktree-environment.mjs';
import {record, sha256} from './curated-gpu-evidence.mjs';

export const reportPath = 'scripts/release/evidence/gemma4-v1-promotion-0.8.0.json';
const tool = 'scripts/release/verify-gemma4-v1-promotion.mts';
const test = 'scripts/release/verify-gemma4-v1-promotion.test.mts';
export const promotion = {
  sourceCommit: '0ddee0f7560015c1d19dc462c9ebfa774afb98df',
  archive: {
    sha256: '68e75c7495ff00df3c45c540801780ed03f2be087eac7af91d9d988ee2bc7101',
    packageTreeSha256: '0a1a06872a9ce328a6b84304f2a491b634e2bacdbaa58b46ad9c06b7aa6805ea',
    bundleSha256: '0d0aa4970fc7241caa2871fd3d28ba5ca28d416ec2e75ae9ee89a21f262c1152',
    packageName: 'web-decision',
    packageVersion: '0.8.0'
  },
  catalogSha256: '825e4e24777cb3b12615e0379d9cd14c0da040ac70f0be27ee9a7c704632486f',
  nativeTvmRevision: '7b5acb32bd271342a24130f4ef439ff6c2e52c6c'
} as const;

export const requiredReports = {
  'kanana-2-v-decide-3b-2611': ['browserMatrix', 'maximumContext'],
  'mapika-decider-2b-v12': ['playwright', 'model'],
  'strands-decider-2b-qwen3.5-v1-2610': ['playwright', 'model'],
  'strands-decider-e2b-gemma4-v1-2610': ['rawSmoke', 'online', 'offlineRestart', 'boundaries', 'corpus'],
  'strands-decider-e2b-gemma4-v2-2611': ['rawCorpus', 'context32k', 'viteReplay', 'online', 'offlineRestart']
} as const;

const releaseFiles = new Set([
  '.github/RELEASING.md',
  '.github/workflows/release-checks.yml',
  'scripts/release/tsconfig.json',
  reportPath,
  tool,
  test
]);

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string) {
  assert.deepEqual(Object.keys(value).sort(), [...keys].sort(), label);
}

function digest(value: unknown, label: string) {
  assert.ok(typeof value === 'string' && /^[a-f0-9]{64}$/u.test(value) && !/^0+$/u.test(value), label);
}

type ModelId = keyof typeof requiredReports;

export function assertPromotionReceipt(value: unknown) {
  const receipt = record(value, 'Expected a reviewed promotion receipt.');
  exactKeys(
    receipt,
    [
      'schemaVersion',
      'status',
      'sourceCommit',
      'archive',
      'catalogSha256',
      'nativeTvmRevision',
      'kananaEvidenceCommit',
      'localVerificationSha256',
      'models'
    ],
    'Unexpected promotion receipt fields.'
  );
  assert.equal(receipt.schemaVersion, 1);
  assert.equal(receipt.status, 'locally-verified-exact-package');
  assert.equal(receipt.sourceCommit, promotion.sourceCommit, 'Promotion evidence belongs to another tested source.');
  assert.deepEqual(receipt.archive, promotion.archive, 'Promotion evidence belongs to another package.');
  assert.equal(receipt.catalogSha256, promotion.catalogSha256, 'Promotion catalog differs.');
  assert.equal(receipt.nativeTvmRevision, promotion.nativeTvmRevision, 'Promotion TVM revision differs.');
  assert.ok(typeof receipt.kananaEvidenceCommit === 'string' && /^[a-f0-9]{40}$/u.test(receipt.kananaEvidenceCommit));
  digest(receipt.localVerificationSha256, 'Missing original local verification digest.');
  const models = record(receipt.models, 'Expected exact model evidence.');
  const ids = Object.keys(requiredReports) as ModelId[];
  exactKeys(models, ids, 'Every published model requires exact evidence.');
  for (const id of ids) assertModelEvidence(id, models[id]);
}

function assertModelEvidence(id: ModelId, value: unknown) {
  const model = record(value, 'Missing verified model evidence: ' + id);
  exactKeys(model, ['reports', 'native'], 'Missing verified model evidence: ' + id);
  const reports = record(model.reports, 'Missing GPU report pins: ' + id);
  exactKeys(reports, requiredReports[id], 'Missing GPU report pins: ' + id);
  for (const [key, hash] of Object.entries(reports)) digest(hash, 'Invalid report digest: ' + id + '/' + key);
  const native = record(model.native, 'Missing independent native builds: ' + id);
  const artifacts = Object.entries(artifactLock.byModelId[id].manifest.files).filter(([name]) =>
    name.endsWith('.wasm')
  );
  exactKeys(
    native,
    artifacts.map(([name]) => name),
    'Missing independent native builds: ' + id
  );
  for (const [name, {sha256: pin}] of artifacts) {
    const build = record(native[name], 'Missing native build: ' + id + '/' + name);
    exactKeys(build, ['wasmSha256', 'reportSha256'], 'Incomplete native build: ' + id + '/' + name);
    assert.equal(build.wasmSha256, pin, 'Native build differs from the committed artifact: ' + id + '/' + name);
    digest(build.reportSha256, 'Invalid native report digest: ' + id + '/' + name);
  }
}

function testCommandChange(original: string) {
  const previous = JSON.parse(original).scripts['test:quality:release'];
  assert.equal(typeof previous, 'string');
  const oldField = JSON.stringify('test:quality:release') + ': ' + JSON.stringify(previous);
  assert.equal(original.split(oldField).length, 2);
  assert.ok(!previous.includes(test));
  const newField = JSON.stringify('test:quality:release') + ': ' + JSON.stringify(previous + ' ' + test);
  return original.replace(oldField, newField);
}

export function assertPromotionSource(root: string, testedSource: string = promotion.sourceCommit) {
  assert.match(testedSource, /^[a-f0-9]{40}$/u);
  const git = (...args: string[]) =>
    execFileSync('git', ['--no-optional-locks', '--no-replace-objects', ...args], {
      cwd: root,
      env: gitEnvironmentFor(root),
      encoding: 'utf8'
    });
  const verifierCommit = git('rev-parse', 'HEAD').trim();
  assert.equal(git('rev-parse', '--show-toplevel').trim(), root);
  git('merge-base', '--is-ancestor', testedSource, verifierCommit);
  const files = git('diff', '--name-only', '--no-renames', '-z', testedSource, verifierCommit)
    .split('\0')
    .filter(Boolean);
  for (const file of files)
    assert.ok(file === 'package.json' || releaseFiles.has(file), 'Change outside release verification: ' + file);
  if (files.includes('package.json'))
    assert.equal(
      git('show', verifierCommit + ':package.json'),
      testCommandChange(git('show', testedSource + ':package.json'))
    );
  return verifierCommit;
}

export async function committedReceipt(root: string) {
  const path = join(root, reportPath);
  const info = await lstat(path);
  assert.ok(info.isFile() && !info.isSymbolicLink(), 'The promotion receipt must be a committed regular file.');
  const bytes = await readFile(path);
  const options = {cwd: root, env: gitEnvironmentFor(root)};
  const head = execFileSync(
    'git',
    ['--no-optional-locks', '--no-replace-objects', 'show', 'HEAD:' + reportPath],
    options
  );
  assert.deepEqual(bytes, head, 'The promotion receipt differs from the reviewed Git blob.');
  return bytes;
}

export async function verifyPromotionArchive(archive: string) {
  const info = await lstat(archive);
  assert.ok(info.isFile() && !info.isSymbolicLink() && archive.endsWith('.tgz'));
  const actual = sha256(await readFile(archive));
  assert.equal(actual, promotion.archive.sha256, 'Packed npm archive differs from the exact GPU-tested archive.');
}

export async function verifyPromotion(root: string, archive: string) {
  const source = await realpath(root);
  const releaseCommit = assertPromotionSource(source);
  const receipt = await committedReceipt(source);
  assertPromotionReceipt(JSON.parse(receipt.toString('utf8')));
  await verifyPromotionArchive(archive);
  return {
    status: 'verified-committed-promotion-receipt-and-archive',
    releaseCommit,
    archive: promotion.archive,
    modelIds: Object.keys(requiredReports),
    receiptSha256: sha256(receipt)
  };
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  const [archive, ...extra] = process.argv.slice(2);
  assert.ok(archive && extra.length === 0, 'Supply the exact release .tgz.');
  const workspace = fileURLToPath(new URL('../../', import.meta.url));
  console.log(JSON.stringify(await verifyPromotion(workspace, resolve(archive)), null, 2));
}
