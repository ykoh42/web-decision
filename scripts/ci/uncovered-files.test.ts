import assert from 'node:assert/strict';
import {mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {
  compareUncoveredFiles,
  findCoverageInventoryMismatches,
  findUncoveredFiles,
  listRuntimeSourceFiles,
  RUNTIME_SOURCE_EXTENSIONS
} from './uncovered-files.ts';

const sourceRoot = '/workspace/src';

function isWindowsSymlinkPermissionError(error: unknown): boolean {
  if (process.platform !== 'win32' || typeof error !== 'object' || error === null || !('code' in error)) return false;
  return typeof error.code === 'string' && ['EACCES', 'EPERM', 'ENOTSUP', 'UNKNOWN'].includes(error.code);
}

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

test('discovers each executable source extension and skips declarations', () => {
  const temporaryRoot = mkdtempSync(join(tmpdir(), 'coverage-inventory-'));
  const sourceDirectory = join(temporaryRoot, 'src');
  const nestedDirectory = join(sourceDirectory, 'nested');
  const correctionsDirectory = join(sourceDirectory, 'tinydecide');
  mkdirSync(nestedDirectory, {recursive: true});
  mkdirSync(correctionsDirectory, {recursive: true});

  try {
    for (const extension of RUNTIME_SOURCE_EXTENSIONS) {
      writeFileSync(join(nestedDirectory, `runtime${extension}`), '');
    }
    for (const extension of ['.d.ts', '.d.mts', '.d.cts']) {
      writeFileSync(join(nestedDirectory, `declaration${extension}`), '');
    }
    writeFileSync(join(correctionsDirectory, 'corrections.ts'), '');

    assert.deepEqual(
      listRuntimeSourceFiles(sourceDirectory),
      [
        ...RUNTIME_SOURCE_EXTENSIONS.map((extension) => `nested/runtime${extension}`),
        'tinydecide/corrections.ts'
      ].sort()
    );
  } finally {
    rmSync(temporaryRoot, {recursive: true, force: true});
  }
});

test('rejects symbolic links in the runtime source directory', (context) => {
  const sourceDirectory = mkdtempSync(join(tmpdir(), 'coverage-symlink-'));

  try {
    writeFileSync(join(sourceDirectory, 'target.mts'), '');
    try {
      symlinkSync('target.mts', join(sourceDirectory, 'linked.mts'), 'file');
    } catch (error) {
      if (isWindowsSymlinkPermissionError(error)) {
        context.skip('Windows did not grant permission to create a file symlink.');
        return;
      }
      throw error;
    }

    assert.throws(() => listRuntimeSourceFiles(sourceDirectory), /symbolic link: linked\.mts/);
  } finally {
    rmSync(sourceDirectory, {recursive: true, force: true});
  }
});

test('rejects a source directory with no executable JavaScript or TypeScript files', () => {
  const sourceDirectory = mkdtempSync(join(tmpdir(), 'coverage-empty-'));

  try {
    assert.throws(() => listRuntimeSourceFiles(sourceDirectory), /no executable JavaScript or TypeScript files/);
  } finally {
    rmSync(sourceDirectory, {recursive: true, force: true});
  }
});

test('fails inventory reconciliation when an .mts or .cts file is absent from the coverage report', () => {
  const sourceFiles = ['runtime.cts', 'runtime.mts', 'runtime.js'];
  const summary = {
    total: {lines: {total: 30, covered: 30}},
    [`${sourceRoot}/runtime.cts`]: {lines: {total: 10, covered: 10}},
    [`${sourceRoot}/runtime.js`]: {lines: {total: 10, covered: 10}},
    [`${sourceRoot}/extra.mjs`]: {lines: {total: 10, covered: 10}}
  };

  assert.deepEqual(findCoverageInventoryMismatches(summary, sourceRoot, sourceFiles), {
    missing: ['runtime.mts'],
    unexpected: ['extra.mjs']
  });
});

test('keeps c8 configured for the complete executable source extension set and excludes declarations', () => {
  const packageConfiguration = JSON.parse(
    readFileSync(new URL('../../packages/web-decision/package.json', import.meta.url), 'utf8')
  ) as {scripts: {'test:coverage': string}};
  const script = packageConfiguration.scripts['test:coverage'];
  const configuredExtensions = [...script.matchAll(/--extension\s+(\S+)/g)].map((match) => match[1]);

  assert.deepEqual(configuredExtensions, RUNTIME_SOURCE_EXTENSIONS);
  assert.match(script, /--exclude=src\/\*\.d\.mts/);
  assert.match(script, /--exclude=src\/\*\*\/\*\.d\.mts/);
  assert.match(script, /--exclude=src\/\*\.d\.cts/);
  assert.match(script, /--exclude=src\/\*\*\/\*\.d\.cts/);
  assert.doesNotMatch(script, /--exclude=src\/tinydecide\/corrections\.(?:js|ts)/);
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
