import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {gitEnvironmentFor} from '../git-worktree-environment.mjs';

const releaseTest = 'test:quality:release';
const existingTest = ' scripts/release/strands-gemma4-v2-release-evidence.test.mts ';
// These are the only root package arguments a verification-only commit may add.
const splitTests =
  'scripts/release/strands-gemma4-v2-split-evidence.test.mts ' +
  'scripts/release/strands-gemma4-v2-split-source.test.mts ';
const pythonWorkflows = ['.github/workflows/python-quality.yml', '.github/workflows/release-checks.yml'];
const invalidPythonPin = 'actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97b';
const correctedPythonPin = 'actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97';

// Only release verification code and its tests may differ from the tested package source.
const allowed = new Set([
  'scripts/release/strands-gemma4-v2-release-evidence.mts',
  'scripts/release/strands-gemma4-v2-release-evidence.test.mts',
  'scripts/release/RELEASE-EVIDENCE.md',
  'scripts/release/strands-gemma4-v2-vite-replay.mts',
  'scripts/release/strands-gemma4-v2-local-checkpoint.mts',
  'scripts/release/strands-gemma4-v2-persistent-variant.mts',
  'scripts/release/strands-gemma4-v2-persistent-trace.mts',
  'scripts/release/strands-gemma4-v2-warm-seed.mts',
  'scripts/release/strands-gemma4-v2-split-candidate.mts',
  'scripts/release/strands-gemma4-v2-split-evidence.mts',
  'scripts/release/strands-gemma4-v2-split-evidence.test.mts',
  'scripts/release/strands-gemma4-v2-split-producer.mts',
  'scripts/release/strands-gemma4-v2-split-source.mts',
  'scripts/release/strands-gemma4-v2-split-source.test.mts',
  'scripts/release/mapika-current-evidence.mts',
  'scripts/release/mapika-release-evidence.mts',
  'scripts/release/mapika-release-evidence.test.mts'
]);

function assertSplitReleaseTestCommand(before: string, after: string) {
  const original = JSON.parse(before).scripts?.[releaseTest];
  assert.ok(typeof original === 'string' && original.split(existingTest).length === 2);
  const entry = JSON.stringify(releaseTest) + ': ' + JSON.stringify(original);
  assert.equal(before.split(entry).length, 2);
  const updated =
    JSON.stringify(releaseTest) + ': ' + JSON.stringify(original.replace(existingTest, existingTest + splitTests));
  assert.equal(
    after,
    before.replace(entry, updated),
    'Only the two split release tests may be added to root package.json.'
  );
}

export function assertSplitVerifierSource(root: string, testedCommit: string, verifierCommit: string) {
  assert.match(testedCommit, /^[a-f0-9]{40}$/u);
  assert.match(verifierCommit, /^[a-f0-9]{40}$/u);
  const options = {cwd: root, env: gitEnvironmentFor(root)};
  execFileSync(
    'git',
    ['--no-optional-locks', '--no-replace-objects', 'merge-base', '--is-ancestor', testedCommit, verifierCommit],
    options
  );
  const changes = execFileSync(
    'git',
    [
      '--no-optional-locks',
      '--no-replace-objects',
      'diff',
      '--no-renames',
      '--name-only',
      '-z',
      testedCommit,
      verifierCommit
    ],
    options
  )
    .toString('utf8')
    .split('\0')
    .filter(Boolean);
  for (const path of changes)
    assert.ok(
      path === 'package.json' || allowed.has(path) || pythonWorkflows.includes(path),
      `Tested source differs outside release verification: ${path}`
    );
  const changedWorkflows = pythonWorkflows.filter((path) => changes.includes(path));
  if (changedWorkflows.length) {
    assert.deepEqual(changedWorkflows, pythonWorkflows, 'Both Python workflow pins must be corrected together.');
    for (const path of changedWorkflows) {
      const show = (commit: string) =>
        execFileSync(
          'git',
          ['--no-optional-locks', '--no-replace-objects', 'show', `${commit}:${path}`],
          options
        ).toString('utf8');
      const before = show(testedCommit);
      const after = show(verifierCommit);
      assert.equal(before.split(invalidPythonPin).length, 2, `Unexpected original Python pin: ${path}`);
      assert.equal(
        after,
        before.replace(invalidPythonPin, correctedPythonPin),
        `Unexpected Python workflow change: ${path}`
      );
    }
  }
  if (changes.includes('package.json')) {
    const read = (commit: string) =>
      execFileSync(
        'git',
        ['--no-optional-locks', '--no-replace-objects', 'show', commit + ':package.json'],
        options
      ).toString('utf8');
    assertSplitReleaseTestCommand(read(testedCommit), read(verifierCommit));
  }
  return {testedCommit, verifierCommit, verificationFiles: changes};
}
