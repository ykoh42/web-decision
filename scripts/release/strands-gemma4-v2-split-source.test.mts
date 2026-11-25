import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {assertSplitVerifierSource} from './strands-gemma4-v2-split-source.mts';

test('only the exact pinned Kanana evidence file may be added', async () => {
  const root = await mkdtemp(join(tmpdir(), 'kanana-evidence-source-'));
  const git = (...args: string[]) =>
    execFileSync('git', args, {cwd: root, stdio: ['ignore', 'pipe', 'pipe']})
      .toString()
      .trim();
  const path =
    'scripts/release/evidence/kanana-current/bfa9922b25a32e37d09783b67ad26b4c9284432095c5feb307d243857c0c768a.json';
  try {
    git('init');
    git('config', 'user.email', 'test@example.invalid');
    git('config', 'user.name', 'test');
    await writeFile(join(root, 'source.txt'), 'tested');
    git('add', '.');
    git('commit', '-m', 'tested');
    const tested = git('rev-parse', 'HEAD');
    await mkdir(join(root, 'scripts/release/evidence/kanana-current'), {recursive: true});
    const evidence = await readFile(
      new URL(
        './evidence/kanana-current/bfa9922b25a32e37d09783b67ad26b4c9284432095c5feb307d243857c0c768a.json',
        import.meta.url
      )
    );
    await writeFile(join(root, path), evidence);
    git('add', '.');
    git('commit', '-m', 'pinned evidence');
    assert.deepEqual(assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')).verificationFiles, [path]);
    await writeFile(join(root, path), Buffer.concat([evidence, Buffer.from(' ')]));
    git('add', '.');
    git('commit', '-m', 'alter evidence');
    assert.throws(
      () => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')),
      /Unexpected Kanana evidence bytes/
    );
    await writeFile(join(root, path), evidence);
    await writeFile(join(root, 'scripts/release/evidence/kanana-current/other.json'), evidence);
    git('add', '.');
    git('commit', '-m', 'add other evidence');
    assert.throws(
      () => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')),
      /outside release verification/
    );
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

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

test('both Python workflow changes must be the exact pinned SHA correction', async () => {
  const root = await mkdtemp(join(tmpdir(), 'python-pin-source-'));
  const git = (...args: string[]) =>
    execFileSync('git', args, {cwd: root, stdio: ['ignore', 'pipe', 'pipe']})
      .toString()
      .trim();
  const old = 'actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97b';
  const correct = 'actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97';
  const paths = ['.github/workflows/python-quality.yml', '.github/workflows/release-checks.yml'];
  try {
    git('init');
    git('config', 'user.email', 'test@example.invalid');
    git('config', 'user.name', 'test');
    await mkdir(join(root, '.github/workflows'), {recursive: true});
    for (const path of paths) await writeFile(join(root, path), `- uses: ${old}\n`);
    git('add', '.');
    git('commit', '-m', 'tested');
    const tested = git('rev-parse', 'HEAD');
    await writeFile(join(root, paths[0]!), `- uses: ${correct}\n`);
    git('add', '.');
    git('commit', '-m', 'one correction');
    assert.throws(() => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')), /Both Python workflow pins/);
    await writeFile(join(root, paths[1]!), `- uses: ${correct}\n`);
    git('add', '.');
    git('commit', '-m', 'both corrections');
    assert.deepEqual(assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')).verificationFiles, paths);
    await writeFile(join(root, paths[0]!), `- uses: ${correct}\nextra: changed\n`);
    git('add', '.');
    git('commit', '-m', 'unrelated change');
    assert.throws(
      () => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')),
      /Unexpected Python workflow change/
    );
    await writeFile(join(root, paths[0]!), '- uses: actions/setup-python@0000000000000000000000000000000000000000\n');
    git('add', '.');
    git('commit', '-m', 'wrong pin');
    assert.throws(
      () => assertSplitVerifierSource(root, tested, git('rev-parse', 'HEAD')),
      /Unexpected Python workflow change/
    );
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});
