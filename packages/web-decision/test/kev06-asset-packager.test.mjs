import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

const packager = fileURLToPath(new URL('../../../scripts/prepare_web_decision_assets.py', import.meta.url));

test('Kev asset packager help loads without a release-only helper module', () => {
  const python = process.env.PYTHON ?? 'python3';
  const result = spawnSync(python, [packager, '--help'], {encoding: 'utf8'});

  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /--output/);
});
