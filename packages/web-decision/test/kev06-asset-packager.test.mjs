import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {access, mkdtemp, mkdir, rm, symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

const packager = fileURLToPath(new URL('../../../scripts/prepare_web_decision_assets.py', import.meta.url));
const workspaceRoot = fileURLToPath(new URL('../../../', import.meta.url));

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
