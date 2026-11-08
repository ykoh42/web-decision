import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {access, mkdtemp, mkdir, readFile, rm, symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

const packager = fileURLToPath(new URL('../../../scripts/prepare_web_decision_assets.py', import.meta.url));
const provenancePath = fileURLToPath(new URL('../../../scripts/asset-notices/PROVENANCE.md', import.meta.url));
const workspaceRoot = fileURLToPath(new URL('../../../', import.meta.url));
const modelMetadataPath = fileURLToPath(new URL('../src/kev-preview-model.json', import.meta.url));
const modelLicensePath = fileURLToPath(
  new URL('../assets/experimental-kev-upstream/notices/MODEL-APACHE-2.0.txt', import.meta.url)
);

test('Kev asset packager help loads without a release-only helper module', () => {
  const python = process.env.PYTHON ?? 'python3';
  const result = spawnSync(python, [packager, '--help'], {encoding: 'utf8'});

  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /--output/);
});

test('Kev asset packager refuses an output path under a symlinked parent', async () => {
  const python = process.env.PYTHON ?? 'python3';
  const tempRoot = await mkdtemp(join(workspaceRoot, '.kev06-packager-test-'));
  const realParent = join(tempRoot, 'real-parent');
  const linkedParent = join(tempRoot, 'linked-parent');
  const output = join(linkedParent, 'assets');

  try {
    await mkdir(realParent);
    await symlink(realParent, linkedParent, 'dir');
    const result = spawnSync(python, [packager, '--output', output], {encoding: 'utf8'});

    assert.equal(result.status, 1, result.stderr || result.stdout);
    assert.match(result.stderr, /symlink component/);
    await assert.rejects(access(join(realParent, 'assets')));
  } finally {
    await rm(tempRoot, {recursive: true, force: true});
  }
});

test('Kev asset packager uses the current provenance and unchanged pinned model license', async () => {
  const [provenance, modelMetadataText, modelLicense] = await Promise.all([
    readFile(provenancePath, 'utf8'),
    readFile(modelMetadataPath, 'utf8'),
    readFile(modelLicensePath)
  ]);
  const modelMetadata = JSON.parse(modelMetadataText);
  const provenanceSha256 = createHash('sha256').update(provenance).digest('hex');
  const modelLicenseSha256 = createHash('sha256').update(modelLicense).digest('hex');

  assert.ok(provenance.includes(modelMetadata.checkpoint.revision));
  assert.ok(provenance.includes(modelMetadata.assets.manifestSha256));
  assert.doesNotMatch(provenance, /q2-k128|experimental-kev-0\.6b-q4f16-upstream|78026118/);
  assert.equal(provenanceSha256, 'b63d3b71202bff7f75d2e4a28240728e8fbf0135e1fc2d9cd5bd601ba01782e7');
  assert.equal(modelLicenseSha256, '832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e');
});
