import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync, mkdirSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {test} from 'node:test';
import {
  discoverChangedJavaScriptPaths,
  discoverJavaScriptPaths,
  parseTypeScriptMigrationPolicy,
  validateTypeScriptMigrationPolicy,
  type TypeScriptMigrationPolicy
} from './check-typescript-policy.mts';

const initialInventoryCommit = 'baa3c144d23e1882ba29084c9f95897b8c56accb';
const historicalPath = 'releases/web-decision-0.3.0/test/index.mjs';
const historicalReason =
  'Immutable historical release 0.3.0; preserve the published JavaScript test snapshot byte for byte.';

type GitFixture = {
  root: string;
  baseCommit: string;
  seedJavaScriptPaths: Set<string>;
  policy: TypeScriptMigrationPolicy;
};

function git(root: string, args: string[]): string {
  return execFileSync('git', args, {cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
}

function write(root: string, path: string, contents: string): void {
  const fullPath = join(root, path);
  mkdirSync(dirname(fullPath), {recursive: true});
  writeFileSync(fullPath, contents);
}

function createGitFixture(): GitFixture {
  const root = mkdtempSync(join(tmpdir(), 'web-decision-typescript-policy-'));
  git(root, ['init', '-q']);
  git(root, ['config', 'user.name', 'TypeScript policy test']);
  git(root, ['config', 'user.email', 'typescript-policy@example.invalid']);
  write(root, '.gitignore', 'ignored.js\nignored/**\n');
  write(root, 'src/legacy.mjs', 'export const legacy = true;\n');
  write(root, historicalPath, 'export const historical = true;\n');
  git(root, ['add', '--all']);
  git(root, ['commit', '-m', 'Create TypeScript policy fixture']);
  const baseCommit = git(root, ['rev-parse', 'HEAD']).trim();
  const policy: TypeScriptMigrationPolicy = {
    schemaVersion: 1,
    baselineCommit: initialInventoryCommit,
    managedJavaScript: ['src/legacy.mjs'],
    exceptions: [{path: historicalPath, reason: historicalReason}]
  };
  git(root, ['commit', '--allow-empty', '-m', 'Reserve policy snapshot commit']);

  return {
    root,
    baseCommit,
    seedJavaScriptPaths: new Set(['src/legacy.mjs', historicalPath]),
    policy
  };
}

function check(fixture: GitFixture, overrides: Partial<Parameters<typeof validateTypeScriptMigrationPolicy>[1]> = {}) {
  const previousPolicy = fixture.policy;
  return validateTypeScriptMigrationPolicy(fixture.policy, {
    currentJavaScriptPaths: discoverJavaScriptPaths(fixture.root),
    seedJavaScriptPaths: fixture.seedJavaScriptPaths,
    previousPolicy,
    changedJavaScriptPaths: discoverChangedJavaScriptPaths(fixture.root, fixture.baseCommit),
    requireComplete: false,
    ...overrides
  });
}

test('discovers tracked and nonignored untracked JavaScript while honoring git ignores', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  write(fixture.root, 'src/new.jsx', 'export const added = true;\n');
  write(fixture.root, 'ignored.js', 'export const ignored = true;\n');
  write(fixture.root, 'ignored/build.cjs', 'module.exports = true;\n');

  assert.deepEqual(discoverJavaScriptPaths(fixture.root), [historicalPath, 'src/legacy.mjs', 'src/new.jsx']);
  const result = check(fixture);
  assert.ok(result.problems.some((problem) => problem.includes('src/new.jsx')));
});

test('rejects edits to an existing managed JavaScript file until it is converted', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  write(fixture.root, 'src/legacy.mjs', 'export const legacy = false;\n');

  const result = check(fixture);
  assert.ok(result.problems.some((problem) => problem.includes('modified while it remains JavaScript')));
});

test('rejects edits to exact historical exception files', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  write(fixture.root, historicalPath, 'export const historical = false;\n');

  const result = check(fixture);
  assert.ok(result.problems.some((problem) => problem.includes('modified while it remains JavaScript')));
});

test('allows generated exception outputs to change after they are recorded', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  const generatedPath = 'src/catalog.generated.js';
  const generatedReason =
    'Generated from the checked-in catalog specification by the catalog generator; edit the specification instead.';
  write(fixture.root, generatedPath, 'export const generated = false;\n');
  const policy: TypeScriptMigrationPolicy = {
    ...fixture.policy,
    exceptions: [...fixture.policy.exceptions, {path: generatedPath, reason: generatedReason}].sort((left, right) =>
      left.path.localeCompare(right.path)
    )
  };
  const result = validateTypeScriptMigrationPolicy(policy, {
    currentJavaScriptPaths: discoverJavaScriptPaths(fixture.root),
    seedJavaScriptPaths: fixture.seedJavaScriptPaths,
    previousPolicy: policy,
    changedJavaScriptPaths: new Set([generatedPath]),
    requireComplete: false
  });

  assert.deepEqual(result.problems, []);
});

test('allows the managed baseline to shrink to zero and enforces the completion gate', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  rmSync(join(fixture.root, 'src/legacy.mjs'));
  write(fixture.root, 'src/legacy.mts', 'export const legacy: boolean = true;\n');
  const completedPolicy: TypeScriptMigrationPolicy = {
    ...fixture.policy,
    managedJavaScript: []
  };
  const result = validateTypeScriptMigrationPolicy(completedPolicy, {
    currentJavaScriptPaths: discoverJavaScriptPaths(fixture.root),
    seedJavaScriptPaths: fixture.seedJavaScriptPaths,
    previousPolicy: fixture.policy,
    changedJavaScriptPaths: discoverChangedJavaScriptPaths(fixture.root, fixture.baseCommit),
    requireComplete: true
  });

  assert.deepEqual(result.problems, []);
  assert.equal(result.managedCount, 0);
});

test('rejects baseline growth even when the new JavaScript file is listed', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  write(fixture.root, 'src/new.mjs', 'export const added = true;\n');
  const expandedPolicy: TypeScriptMigrationPolicy = {
    ...fixture.policy,
    managedJavaScript: ['src/legacy.mjs', 'src/new.mjs']
  };
  const result = validateTypeScriptMigrationPolicy(expandedPolicy, {
    currentJavaScriptPaths: discoverJavaScriptPaths(fixture.root),
    seedJavaScriptPaths: fixture.seedJavaScriptPaths,
    previousPolicy: fixture.policy,
    changedJavaScriptPaths: new Set(),
    requireComplete: false
  });

  assert.ok(result.problems.some((problem) => problem.includes('grows managedJavaScript')));
  assert.ok(result.problems.some((problem) => problem.includes('new to the migration baseline')));
});

test('accepts a new generated-code exception only with its exact path and reason', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  const generatedPath = 'src/catalog/index.generated.js';
  write(fixture.root, generatedPath, 'export const generated = true;\n');
  const generatedReason =
    'Generated from the checked-in model catalog specification by the TypeScript catalog generator; do not edit this output.';
  const policy = parseTypeScriptMigrationPolicy(
    {
      ...fixture.policy,
      exceptions: [...fixture.policy.exceptions, {path: generatedPath, reason: generatedReason}].sort((left, right) =>
        left.path.localeCompare(right.path)
      )
    },
    'test policy'
  );
  const result = validateTypeScriptMigrationPolicy(policy, {
    currentJavaScriptPaths: discoverJavaScriptPaths(fixture.root),
    seedJavaScriptPaths: fixture.seedJavaScriptPaths,
    previousPolicy: fixture.policy,
    changedJavaScriptPaths: new Set(),
    requireComplete: false
  });

  assert.deepEqual(result.problems, []);
  assert.equal(result.exceptionCount, 2);
  assert.throws(
    () =>
      parseTypeScriptMigrationPolicy(
        {
          ...fixture.policy,
          exceptions: [...fixture.policy.exceptions, {path: 'src/catalog.js', reason: generatedReason}].sort(
            (left, right) => left.path.localeCompare(right.path)
          )
        },
        'test policy'
      ),
    /does not use the \.generated JavaScript suffix/u
  );
});

test('rejects moving a seeded migration file into the exception list', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  const generatedReason =
    'Generated from the checked-in catalog specification by the catalog generator; edit the specification instead.';
  const policy: TypeScriptMigrationPolicy = {
    ...fixture.policy,
    managedJavaScript: [],
    exceptions: [...fixture.policy.exceptions, {path: 'src/legacy.mjs', reason: generatedReason}].sort((left, right) =>
      left.path.localeCompare(right.path)
    )
  };
  const result = validateTypeScriptMigrationPolicy(policy, {
    currentJavaScriptPaths: discoverJavaScriptPaths(fixture.root),
    seedJavaScriptPaths: fixture.seedJavaScriptPaths,
    previousPolicy: fixture.policy,
    changedJavaScriptPaths: new Set(),
    requireComplete: false
  });

  assert.ok(result.problems.some((problem) => problem.includes('cannot be newly exempted from migration')));
});

test('rejects glob paths and changed reasons in the exact exception list', (context) => {
  const fixture = createGitFixture();
  context.after(() => rmSync(fixture.root, {recursive: true, force: true}));
  assert.throws(
    () =>
      parseTypeScriptMigrationPolicy(
        {
          ...fixture.policy,
          exceptions: [{path: 'releases/**/*.mjs', reason: historicalReason}]
        },
        'test policy'
      ),
    /exact JavaScript file path/u
  );

  const changedPolicy: TypeScriptMigrationPolicy = {
    ...fixture.policy,
    exceptions: [{path: historicalPath, reason: historicalReason.replace('preserve', 'keep')}]
  };
  const result = validateTypeScriptMigrationPolicy(changedPolicy, {
    currentJavaScriptPaths: discoverJavaScriptPaths(fixture.root),
    seedJavaScriptPaths: fixture.seedJavaScriptPaths,
    previousPolicy: fixture.policy,
    changedJavaScriptPaths: new Set(),
    requireComplete: false
  });
  assert.ok(result.problems.some((problem) => problem.includes('changed its recorded exception reason')));
});
