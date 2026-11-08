import {cp, copyFile, mkdtemp, mkdir, readFile, rm} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const packageDirectory = resolve(root, 'releases/web-decision-0.3.0/package');
const fixtureDirectory = resolve(root, 'tests/types');
const temporaryDirectory = await mkdtemp(join(tmpdir(), 'web-decision-0.3.0-types-'));

try {
  const packageJson = JSON.parse(await readFile(resolve(packageDirectory, 'package.json'), 'utf8'));
  if (packageJson.name !== 'web-decision' || packageJson.version !== '0.3.0') {
    throw new Error('Type fixture requires the exact web-decision@0.3.0 package.');
  }

  const dependencyDirectory = resolve(temporaryDirectory, 'node_modules/web-decision');
  await mkdir(dirname(dependencyDirectory), {recursive: true});
  await cp(packageDirectory, dependencyDirectory, {recursive: true});
  await copyFile(resolve(fixtureDirectory, 'consumer.mts'), resolve(temporaryDirectory, 'consumer.mts'));
  await copyFile(resolve(fixtureDirectory, 'tsconfig.json'), resolve(temporaryDirectory, 'tsconfig.json'));

  const compiler = spawnSync(
    process.execPath,
    [resolve(root, 'node_modules/typescript/bin/tsc'), '--project', resolve(temporaryDirectory, 'tsconfig.json')],
    {
      cwd: temporaryDirectory,
      encoding: 'utf8',
      env: {...process.env, GOMAXPROCS: '1', GOMEMLIMIT: '256MiB', NODE_OPTIONS: '--max-old-space-size=256'},
      maxBuffer: 1024 * 1024,
      timeout: 30_000
    }
  );

  if (compiler.error) throw new Error('TypeScript consumer check could not complete.', {cause: compiler.error});
  if (compiler.status !== 0) {
    const details = `${compiler.stdout}${compiler.stderr}`.trim();
    throw new Error(`TypeScript consumer check failed with exit status ${compiler.status}.\n${details}`);
  }

  console.log('TypeScript consumer check passed against the web-decision@0.3.0 package.');
} finally {
  await rm(temporaryDirectory, {force: true, recursive: true});
}
