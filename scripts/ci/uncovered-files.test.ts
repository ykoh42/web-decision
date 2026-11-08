import assert from 'node:assert/strict';
import {test} from 'node:test';
import {compareUncoveredFiles, findUncoveredFiles} from './uncovered-files.ts';

const sourceRoot = '/workspace/src';

test('finds uncovered source files in stable path order and ignores empty totals', () => {
  const summary = {
    total: {lines: {total: 100, covered: 90}},
    [`${sourceRoot}/z/never-run.mjs`]: {lines: {total: 4, covered: 0}},
    [`${sourceRoot}/a/empty.mjs`]: {lines: {total: 0, covered: 0}},
    [`${sourceRoot}/b/partial.mjs`]: {lines: {total: 5, covered: 1}},
    [`${sourceRoot}/a/never-run.mjs`]: {lines: {total: 3, covered: 0}}
  };

  assert.deepEqual(findUncoveredFiles(summary, sourceRoot), ['a/never-run.mjs', 'z/never-run.mjs']);
});

test('reports newly uncovered and resolved paths while preserving baseline order', () => {
  assert.deepEqual(compareUncoveredFiles(['src/new.mjs', 'src/shared.mjs'], ['src/old.mjs', 'src/shared.mjs']), {
    added: ['src/new.mjs'],
    resolved: ['src/old.mjs']
  });
});

test('rejects coverage paths outside the source root and invalid line counts', () => {
  assert.throws(
    () => findUncoveredFiles({'/workspace/other/file.mjs': {lines: {total: 1, covered: 0}}}, sourceRoot),
    /outside the source directory/
  );
  assert.throws(
    () => findUncoveredFiles({[`${sourceRoot}/bad.mjs`]: {lines: {total: '1', covered: 0}}}, sourceRoot),
    /Invalid line coverage/
  );
});
