import assert from 'node:assert/strict';
import {lstat, readFile, readdir} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const source = resolve('packages/web-decision');
const installed = resolve('examples/basic/node_modules/web-decision');
assert.equal(
  (await lstat(installed)).isSymbolicLink(),
  false,
  'The example should install a package copy, not link to the source checkout'
);
interface PackageMetadata {
  name: string;
  version: string;
  license: string;
}

const metadata = async (path: string): Promise<PackageMetadata> =>
  JSON.parse(await readFile(resolve(path, 'package.json'), 'utf8'));
const [original, copy] = await Promise.all([metadata(source), metadata(installed)]);
assert.equal(copy.version, original.version);
assert.equal(copy.license, original.license);
const [installedEntry, sourceEntry] = await Promise.all([
  readFile(resolve(installed, 'dist/index.js')),
  readFile(resolve(source, 'dist/index.js'))
]);
assert.ok(installedEntry.equals(sourceEntry), 'Installed package entry must match the source package entry');
const library = await import(pathToFileURL(resolve(installed, 'dist/index.js')).href);
assert.equal(typeof library.loadDecisionModel, 'function');
const build = resolve('examples/basic/dist');
async function files(directory: string, prefix = ''): Promise<string[]> {
  const result: string[] = [];
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) result.push(...(await files(join(directory, entry.name), name)));
    else if (entry.isFile()) result.push(name);
    else throw Error(`Unexpected entry in example build: ${name}`);
  }
  return result;
}
const builtFiles = await files(build);
assert.ok(builtFiles.includes('index.html'), 'Built example must include its HTML entry');
assert.ok(
  builtFiles.some((path) => path.startsWith('assets/') && path.endsWith('.js')),
  'Built example must include its JavaScript'
);
assert.ok(
  !builtFiles.some((path) => /(^|\/)(model\.wasm|params_shard_\d+\.bin)$/.test(path)),
  'Model assets must remain separate from the example build'
);
console.log(`Example has an installed ${copy.name}@${copy.version} package with resolvable dependencies.`);
console.log(`Example build has ${builtFiles.length} files and no model WASM or weight shards.`);
