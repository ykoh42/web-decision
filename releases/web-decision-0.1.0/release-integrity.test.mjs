import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {assertFreshOutputDirectory} from './release-integrity.mjs';

test('accepts a fresh output below a canonical parent', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-release-path-')));
  try {
    const parent = join(root, 'build');
    await mkdir(parent);

    assert.equal(
      await assertFreshOutputDirectory(parent, join(parent, 'dist'), 'Build output'),
      join(parent, 'dist')
    );
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('rejects a symlinked output parent', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-release-path-')));
  try {
    const actual = join(root, 'actual');
    const alias = join(root, 'build');
    await mkdir(actual);
    await symlink(actual, alias, 'dir');

    await assert.rejects(
      assertFreshOutputDirectory(alias, join(alias, 'dist'), 'Build output'),
      /canonical directory/
    );
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('refuses any existing output without changing it', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-release-path-')));
  try {
    const parent = join(root, 'build');
    const output = join(parent, 'dist');
    await mkdir(output, {recursive: true});
    await writeFile(join(output, 'keep.txt'), 'keep');

    await assert.rejects(
      assertFreshOutputDirectory(parent, output, 'Build output'),
      /Refusing to overwrite/
    );
    assert.equal(await readFile(join(output, 'keep.txt'), 'utf8'), 'keep');
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});
