import {mkdtemp, mkdir, readFile, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {checkConsumerTypes} from './check-types.mts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const target = process.argv[2] ?? '--current';
if (!['--current', '0.4.0', '0.5.0'].includes(target)) {
  throw new Error(`Unsupported type-check package version: ${target}`);
}
const packageDirectory = resolve(root, 'packages/web-decision');
const packageJson = JSON.parse(await readFile(resolve(packageDirectory, 'package.json'), 'utf8')) as {
  name: string;
  version: string;
};
const version = target === '--current' ? packageJson.version : target;
if (packageJson.name !== 'web-decision' || packageJson.version !== version) {
  throw new Error(`Type fixture requires the exact web-decision@${version} package.`);
}
const temporaryDirectory = await mkdtemp(join(tmpdir(), `web-decision-${version}-types-`));

try {
  const dependencyDirectory = resolve(temporaryDirectory, 'node_modules/web-decision');
  await mkdir(dirname(dependencyDirectory), {recursive: true});
  await symlink(packageDirectory, dependencyDirectory, 'dir');
  await writeFile(resolve(temporaryDirectory, 'package.json'), '{"private":true,"type":"module"}\n');
  checkConsumerTypes(temporaryDirectory);

  console.log(`TypeScript consumer check passed against the web-decision@${version} package.`);
} finally {
  await rm(temporaryDirectory, {force: true, recursive: true});
}
