import {cp, copyFile, mkdtemp, mkdir, readFile, rm, symlink} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const version = process.argv[2] ?? '0.3.0';
const currentPackage = version === '0.4.0';
if (version !== '0.3.0' && !currentPackage) {
  throw new Error(`Unsupported type-check package version: ${version}`);
}
const packageDirectory = resolve(
  root,
  currentPackage ? 'packages/web-decision' : 'releases/web-decision-0.3.0/package'
);
const fixtureDirectory = resolve(root, 'tests/types');
const temporaryDirectory = await mkdtemp(join(tmpdir(), `web-decision-${version}-types-`));

try {
  const packageJson = JSON.parse(await readFile(resolve(packageDirectory, 'package.json'), 'utf8'));
  if (packageJson.name !== 'web-decision' || packageJson.version !== version) {
    throw new Error(`Type fixture requires the exact web-decision@${version} package.`);
  }

  const dependencyDirectory = resolve(temporaryDirectory, 'node_modules/web-decision');
  await mkdir(dirname(dependencyDirectory), {recursive: true});
  const projects = currentPackage ? ['tsconfig.bundler.json', 'tsconfig.nodenext.json'] : ['tsconfig.json'];
  const fixtureFiles = currentPackage
    ? ['consumer.ts', 'kanana-consumer.ts', 'tsconfig.base.json', ...projects]
    : ['consumer.mts', ...projects];
  if (currentPackage) {
    await symlink(packageDirectory, dependencyDirectory, 'dir');
    await copyFile(resolve(packageDirectory, 'package.json'), resolve(temporaryDirectory, 'package.json'));
  } else {
    await cp(packageDirectory, dependencyDirectory, {recursive: true});
  }
  await Promise.all(
    fixtureFiles.map((name) => copyFile(resolve(fixtureDirectory, name), resolve(temporaryDirectory, name)))
  );

  for (const project of projects) {
    const compiler = spawnSync(
      process.execPath,
      [
        resolve(root, 'node_modules/typescript/bin/tsc'),
        '--project',
        resolve(temporaryDirectory, project),
        ...(currentPackage ? ['--exactOptionalPropertyTypes', 'true'] : [])
      ],
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
  }

  console.log(`TypeScript consumer check passed against the web-decision@${version} package.`);
} finally {
  await rm(temporaryDirectory, {force: true, recursive: true});
}
