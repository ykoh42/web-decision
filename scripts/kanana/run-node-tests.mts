import {execFileSync} from 'node:child_process';
import {copyFile, mkdir, mkdtemp, readdir, rm, symlink, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const packageRoot = fileURLToPath(new URL('../../packages/web-decision/', import.meta.url));
const packageRequire = createRequire(join(packageRoot, 'package.json'));
const compiler = join(dirname(packageRequire.resolve('typescript/package.json')), 'bin/tsc');
const testDirectory = join(packageRoot, 'test');
const allTests = (await readdir(testDirectory))
  .filter((file) => /\.test\.(?:mjs|mts|ts)$/u.test(file))
  .sort()
  .map((file) => join('test', file));
const coreTests = [
  'test/kanana-protocol.test.mjs',
  'test/kanana-scoring.test.mts',
  'test/kanana-candidates.test.mjs',
  'test/kanana-chat-template.test.mts',
  'test/kanana-truncation.test.mjs',
  'test/kanana-tokenizer.test.mjs',
  'test/kanana-image-ingress.test.mjs',
  'test/kanana-encoded-image.test.mjs',
  'test/kanana-image-processing.test.mjs',
  'test/kanana-magick-image.test.mjs',
  'test/kanana-pnm-image.test.mjs',
  'test/kanana-tiff16-image.test.mts'
];
const tests = process.argv.includes('--core') ? coreTests : allTests;
// These tests import emitted JS with external TypeScript source maps. Keep tsx
// off these files so its JS transform does not replace two maps with identity maps.
const compiledSourceMapTests = new Set(coreTests);
const coverageDirectory = join(packageRoot, 'coverage');
await mkdir(coverageDirectory, {recursive: true});
const outputDirectory = await mkdtemp(join(coverageDirectory, 'kanana-tests-'));

try {
  await writeFile(join(outputDirectory, 'package.json'), '{"type":"module"}\n');
  await symlink(join(packageRoot, 'node_modules'), join(outputDirectory, 'node_modules'), 'junction');
  execFileSync(
    process.execPath,
    [
      compiler,
      '--project',
      join(packageRoot, 'tsconfig.kanana-core.json'),
      '--noEmit',
      'false',
      '--outDir',
      outputDirectory
    ],
    {cwd: packageRoot, stdio: 'inherit'}
  );
  // Emit TypeScript core tests separately so tsx never rewrites the compiled
  // Kanana source maps used by the coverage gate.
  const emittedTests = join(outputDirectory, 'core-tests');
  execFileSync(
    process.execPath,
    [compiler, '--project', join(packageRoot, 'tsconfig.kanana-tests.json'), '--outDir', emittedTests],
    {
      cwd: packageRoot,
      stdio: 'inherit'
    }
  );
  await copyFile(join(testDirectory, 'kanana-core-modules.mjs'), join(emittedTests, 'test', 'kanana-core-modules.mjs'));
  await symlink(join(testDirectory, 'fixtures'), join(emittedTests, 'test', 'fixtures'), 'junction');
  const runTests = (files: string[], useTsx: boolean) => {
    const testArguments = [...(useTsx ? ['--import', 'tsx'] : []), '--test', ...files];
    if (process.env.NODE_V8_COVERAGE) testArguments.unshift('--enable-source-maps');
    execFileSync(process.execPath, testArguments, {
      cwd: packageRoot,
      env: {...process.env, KANANA_CORE_DIST: outputDirectory},
      stdio: 'inherit'
    });
  };
  const compiledTests = tests.filter((file) => compiledSourceMapTests.has(file));
  const tsxTests = tests.filter((file) => !compiledSourceMapTests.has(file));
  if (compiledTests.length > 0)
    runTests(
      compiledTests.map((file) => (file.endsWith('.mts') ? join(emittedTests, file.replace(/\.mts$/u, '.mjs')) : file)),
      false
    );
  if (tsxTests.length > 0) runTests(tsxTests, true);
} finally {
  await rm(outputDirectory, {recursive: true, force: true});
}
