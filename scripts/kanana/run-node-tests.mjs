import {execFileSync} from 'node:child_process';
import {mkdir, mkdtemp, readdir, rm, symlink, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const packageRoot = fileURLToPath(new URL('../../packages/web-decision/', import.meta.url));
const packageRequire = createRequire(join(packageRoot, 'package.json'));
const compiler = join(dirname(packageRequire.resolve('typescript/package.json')), 'bin/tsc');
const testDirectory = join(packageRoot, 'test');
const allTests = (await readdir(testDirectory))
  .filter((file) => file.endsWith('.test.mjs'))
  .sort()
  .map((file) => join('test', file));
const coreTests = [
  'test/kanana-protocol.test.mjs',
  'test/kanana-scoring.test.mjs',
  'test/kanana-candidates.test.mjs',
  'test/kanana-chat-template.test.mjs',
  'test/kanana-truncation.test.mjs',
  'test/kanana-tokenizer.test.mjs',
  'test/kanana-image-ingress.test.mjs',
  'test/kanana-encoded-image.test.mjs',
  'test/kanana-image-processing.test.mjs',
  'test/kanana-magick-image.test.mjs',
  'test/kanana-pnm-image.test.mjs',
  'test/kanana-tiff16-image.test.mjs'
];
const tests = process.argv.includes('--core') ? coreTests : allTests;
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
  const testArguments = ['--test', ...tests];
  if (process.env.NODE_V8_COVERAGE) testArguments.unshift('--enable-source-maps');
  execFileSync(process.execPath, testArguments, {
    cwd: packageRoot,
    env: {...process.env, KANANA_CORE_DIST: outputDirectory},
    stdio: 'inherit'
  });
} finally {
  await rm(outputDirectory, {recursive: true, force: true});
}
