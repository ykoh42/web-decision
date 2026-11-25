import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtemp, mkdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {assertSplitVerifierSource} from './strands-gemma4-v2-split-source.mts';

test('only permitted release verifier changes may follow the tested source', async () => {
  const root = await mkdtemp(join(tmpdir(), 'release-source-test-'));
  const git = (...args: string[]) =>
    execFileSync('git', args, {
      cwd: root,
      env: {...process.env, GIT_CONFIG_NOSYSTEM: '1'},
      stdio: ['ignore', 'pipe', 'pipe']
    })
      .toString()
      .trim();
  try {
    git('init');
    git('config', 'user.email', 'test@example.invalid');
    git('config', 'user.name', 'test');
    await writeFile(join(root, 'model.ts'), 'original');
    const manifest = {
      version: '0.1.0',
      scripts: {
        'test:quality:release': 'node --test scripts/release/strands-gemma4-v2-release-evidence.test.mts old.test.mts',
        'build:package': 'pnpm --filter web-decision build'
      }
    };
    const writePackage = (value: typeof manifest) =>
      writeFile(join(root, 'package.json'), JSON.stringify(value, null, 2) + '\n');
    await writePackage(manifest);
    git('add', '.');
    git('commit', '-m', 'source');
    const tested = git('rev-parse', 'HEAD');
    await mkdir(join(root, 'scripts/release'), {recursive: true});
    await writeFile(join(root, 'scripts/release/strands-gemma4-v2-split-source.mts'), 'verification only');
    git('add', '.');
    git('commit', '-m', 'verifier');
    const verifier = git('rev-parse', 'HEAD');
    assert.deepEqual(assertSplitVerifierSource(root, tested, verifier).verificationFiles, [
      'scripts/release/strands-gemma4-v2-split-source.mts'
    ]);
    const wired = structuredClone(manifest);
    wired.scripts['test:quality:release'] =
      'node --test scripts/release/strands-gemma4-v2-release-evidence.test.mts ' +
      'scripts/release/strands-gemma4-v2-split-evidence.test.mts ' +
      'scripts/release/strands-gemma4-v2-split-source.test.mts old.test.mts';
    await writePackage(wired);
    git('add', '.');
    git('commit', '-m', 'wire split release regression tests');
    assert.deepEqual(assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')).verificationFiles, [
      'package.json',
      'scripts/release/strands-gemma4-v2-split-source.mts'
    ]);
    for (const changed of [
      {...wired, version: '0.2.0'},
      {...wired, scripts: {...wired.scripts, 'build:package': 'different command'}},
      {
        ...wired,
        scripts: {...wired.scripts, 'test:quality:release': wired.scripts['test:quality:release'] + ' extra.test.mts'}
      }
    ]) {
      await writePackage(changed);
      git('add', '.');
      git('commit', '-m', 'unexpected root package edit');
      assert.throws(
        () => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')),
        /Only the two split release tests/
      );
    }
    await writePackage(wired);
    await writeFile(join(root, 'model.ts'), 'changed');
    git('add', '.');
    git('commit', '-m', 'model');
    assert.throws(
      () => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')),
      /outside release verification/
    );
    assert.throws(() => assertSplitVerifierSource(root, verifier, tested));
    assert.throws(() => assertSplitVerifierSource(root, 'invalid', verifier));
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('combined Mapika and Gemma verifier changes leave the tested source intact', async () => {
  const root = await mkdtemp(join(tmpdir(), 'combined-release-source-'));
  const git = (...args: string[]) =>
    execFileSync('git', args, {cwd: root, stdio: ['ignore', 'pipe', 'pipe']})
      .toString()
      .trim();
  try {
    git('init');
    git('config', 'user.email', 'test@example.invalid');
    git('config', 'user.name', 'test');
    await writeFile(join(root, 'source.txt'), 'tested source');
    git('add', '.');
    git('commit', '-m', 'source');
    const tested = git('rev-parse', 'HEAD');
    await mkdir(join(root, 'scripts/release'), {recursive: true});
    for (const file of [
      'mapika-current-evidence.mts',
      'mapika-release-evidence.mts',
      'mapika-release-evidence.test.mts',
      'strands-gemma4-v2-split-source.mts'
    ])
      await writeFile(join(root, 'scripts/release', file), 'verification code');
    git('add', '.');
    git('commit', '-m', 'combined verification');
    const valid = assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD'));
    assert.equal(valid.verificationFiles.length, 4);
    await mkdir(join(root, 'tests/browser'), {recursive: true});
    await writeFile(join(root, 'tests/browser/vite.config.ts'), 'altered');
    git('add', '.');
    git('commit', '-m', 'tamper tested configuration');
    assert.throws(
      () => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')),
      /outside release verification/
    );
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});
