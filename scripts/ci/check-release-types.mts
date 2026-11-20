import {cp, copyFile, mkdtemp, mkdir, readFile, rm, symlink, writeFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {checkConsumerTypes} from './check-types.mts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const target = process.argv[2] ?? '0.3.0';
const workspacePackage = ['--current', '0.4.0', '0.5.0'].includes(target);
if (target !== '0.3.0' && !workspacePackage) {
  throw new Error(`Unsupported type-check package version: ${target}`);
}
const packageDirectory = resolve(
  root,
  workspacePackage ? 'packages/web-decision' : 'releases/web-decision-0.3.0/package'
);
const packageJson = JSON.parse(await readFile(resolve(packageDirectory, 'package.json'), 'utf8')) as {
  name: string;
  version: string;
};
const version = target === '--current' ? packageJson.version : target;
if (packageJson.name !== 'web-decision' || packageJson.version !== version) {
  throw new Error(`Type fixture requires the exact web-decision@${version} package.`);
}
const fixtureDirectory = resolve(root, 'tests/types');
const temporaryDirectory = await mkdtemp(join(tmpdir(), `web-decision-${version}-types-`));

try {
  const dependencyDirectory = resolve(temporaryDirectory, 'node_modules/web-decision');
  await mkdir(dirname(dependencyDirectory), {recursive: true});
  if (workspacePackage) {
    await symlink(packageDirectory, dependencyDirectory, 'dir');
    await writeFile(resolve(temporaryDirectory, 'package.json'), '{"private":true,"type":"module"}\n');
    checkConsumerTypes(temporaryDirectory);
  } else {
    await cp(packageDirectory, dependencyDirectory, {recursive: true});
    await Promise.all(
      ['consumer.mts', 'tsconfig.json'].map((name) =>
        copyFile(resolve(fixtureDirectory, name), resolve(temporaryDirectory, name))
      )
    );
    checkHistoricalConsumer(temporaryDirectory);
  }

  console.log(`TypeScript consumer check passed against the web-decision@${version} package.`);
} finally {
  await rm(temporaryDirectory, {force: true, recursive: true});
}

function checkHistoricalConsumer(directory: string): void {
  const compiler = spawnSync(
    process.execPath,
    [resolve(root, 'node_modules/typescript/bin/tsc'), '--project', resolve(directory, 'tsconfig.json')],
    {
      cwd: directory,
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
}
