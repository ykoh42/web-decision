import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import type {CoverageFile, CoverageGroupConfiguration} from './coverage-groups.ts';
import {checkCoverageGroups} from './coverage-groups.ts';

const sourceRoot = '/workspace/src';
const configuredGroups: unknown = JSON.parse(readFileSync(new URL('./coverage-groups.json', import.meta.url), 'utf8'));

function file(linesCovered: number, linesTotal: number, branchesCovered: number, branchesTotal: number): CoverageFile {
  return {
    lines: {covered: linesCovered, total: linesTotal},
    branches: {covered: branchesCovered, total: branchesTotal}
  };
}

test('keeps the source-wide coverage thresholds at 85% lines and 80% branches', () => {
  assert.deepEqual(configuredGroups, {
    groups: [{name: 'all source files', prefixes: [''], lines: 85, branches: 80}]
  });
});

test('checks weighted line and branch coverage for each matching group', () => {
  const report = {
    [`${sourceRoot}/core/loader.mjs`]: file(8, 10, 3, 4),
    [`${sourceRoot}/core/protocol.mjs`]: file(9, 10, 4, 4),
    [`${sourceRoot}/other/runner.mjs`]: file(1, 1, 1, 1)
  };
  const config: CoverageGroupConfiguration = {
    groups: [
      {name: 'all', prefixes: [''], lines: 85, branches: 80},
      {name: 'core', prefixes: ['core/'], lines: 80, branches: 80}
    ]
  };

  assert.deepEqual(checkCoverageGroups(report, sourceRoot, config), []);
});

test('reports line and branch thresholds separately', () => {
  const report = {
    [`${sourceRoot}/core/loader.mjs`]: file(7, 10, 2, 4)
  };
  const errors = checkCoverageGroups(report, sourceRoot, {
    groups: [{name: 'core', prefixes: ['core/'], lines: 80, branches: 80}]
  });

  assert.deepEqual(errors, [
    'Coverage group core has 70.00% line coverage; 80% required across 1 files.',
    'Coverage group core has 50.00% branch coverage; 80% required across 1 files.'
  ]);
});

test('matches directory prefixes without matching similarly named siblings', () => {
  const report = {
    [`${sourceRoot}/core/loader.mjs`]: file(10, 10, 4, 4),
    [`${sourceRoot}/core-extra/loader.mjs`]: file(0, 10, 0, 4)
  };
  const config: CoverageGroupConfiguration = {
    groups: [{name: 'core', prefixes: ['core'], lines: 100, branches: 100}]
  };

  assert.deepEqual(checkCoverageGroups(report, sourceRoot, config), []);
});

test('treats groups with no measurable lines or branches as fully covered', () => {
  const report = {[`${sourceRoot}/core/empty.mjs`]: file(0, 0, 0, 0)};
  const config: CoverageGroupConfiguration = {
    groups: [{name: 'core', prefixes: ['core/'], lines: 100, branches: 100}]
  };

  assert.deepEqual(checkCoverageGroups(report, sourceRoot, config), []);
});

test('reports groups that match no source files', () => {
  const errors = checkCoverageGroups({}, sourceRoot, {
    groups: [{name: 'missing', prefixes: ['missing/'], lines: 85, branches: 80}]
  });

  assert.deepEqual(errors, ['Coverage group missing matched no source files.']);
});

test('rejects coverage entries outside the source root', () => {
  assert.throws(
    () =>
      checkCoverageGroups({'/workspace/other/file.mjs': file(1, 1, 1, 1)}, sourceRoot, {
        groups: [{name: 'all', prefixes: [''], lines: 85, branches: 80}]
      }),
    /outside the source directory/
  );
});

test('rejects invalid prefixes, duplicate names, thresholds, and coverage counts', () => {
  assert.throws(
    () => checkCoverageGroups({}, sourceRoot, {groups: [{name: 'bad', prefixes: ['../src'], lines: 85, branches: 80}]}),
    /invalid path prefix/
  );
  assert.throws(
    () => checkCoverageGroups({}, sourceRoot, {groups: [{name: 'bad', prefixes: ['src/'], lines: 101, branches: 80}]}),
    /percentages from 0 to 100/
  );
  assert.throws(
    () =>
      checkCoverageGroups({}, sourceRoot, {
        groups: [
          {name: 'duplicate', prefixes: [''], lines: 85, branches: 80},
          {name: 'duplicate', prefixes: ['src/'], lines: 85, branches: 80}
        ]
      }),
    /unique, non-empty names/
  );
  assert.throws(
    () =>
      checkCoverageGroups({[`${sourceRoot}/core/file.mjs`]: file(2, 1, 1, 1)}, sourceRoot, {
        groups: [{name: 'all', prefixes: [''], lines: 85, branches: 80}]
      }),
    /Invalid lines coverage/
  );
});
