import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {copyFile, mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'vite';
import {verifyArchiveInstall} from '../../../release/verify-release-evidence.ts';
import {catalogEntry, curatedCandidate, modelId} from './catalog.mjs';

const [graphDirectory, inventoryPath, official, output] = process.argv.slice(2).map((path) => resolve(path));
assert.ok(
  graphDirectory && inventoryPath && official && output,
  'Pass <graph dir> <source inventory> <official asset dir> <new output>.'
);
const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../../../../packages/web-decision');
const pkg = join(output, 'package');
const dist = join(pkg, 'dist');
const git = (...args: string[]) => execFileSync('git', args, {cwd: root, encoding: 'utf8'}).trim();
const source = {commit: git('rev-parse', 'HEAD'), clean: !git('status', '--porcelain')};
assert.equal(source.clean, true, 'Build the private archive from a clean source revision.');

const expected = {
  'graph.wasm': '7a1a2f32f080011334f628714de8dbc69c6fc46b660c16c78737998b008cff3c',
  'source-inventory.json': 'eb953da8adbbddce431abf40bbc3f2fe6f217a83255ee4c41001d760b9a98c87',
  'tokenizer.json': '30383f3ce90a169cbabe9eb3da7bb77988bf906aad2e59184b378f5470ec4dbd',
  'tokenizer_config.json': '3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999'
};
const digest = (value: Uint8Array) => createHash('sha256').update(value).digest('hex');
const files = {
  'graph.wasm': join(graphDirectory, 'graph.wasm'),
  'graph.json': join(graphDirectory, 'graph.json'),
  'source-inventory.json': inventoryPath,
  'tokenizer.json': join(official, 'tokenizer.json'),
  'tokenizer_config.json': join(official, 'tokenizer_config.json')
};
const assetBytes = Object.fromEntries(
  await Promise.all(Object.entries(files).map(async ([name, path]) => [name, await readFile(path)] as const))
);
const assets = Object.fromEntries(
  Object.entries(assetBytes).map(([name, bytes]) => [name, {bytes: bytes.length, sha256: digest(bytes)}])
);
for (const [name, pin] of Object.entries(expected)) assert.equal(assets[name]?.sha256, pin, name);
const graph = JSON.parse(await readFile(files['graph.json'], 'utf8'));
assert.equal(graph.format, 'web-decision-gemma4-e2b-v1-graph-v1');
assert.equal(graph.archiveApproved, false);
assert.equal(graph.target, 'webgpu-wasm');
assert.equal(graph.tvmCommit, '7b5acb32bd271342a24130f4ef439ff6c2e52c6c');
assert.equal(graph.graphSha256, assets['graph.wasm']?.sha256);
assert.equal(graph.inventorySha256, assets['source-inventory.json']?.sha256);
assert.equal(graph.layers.length, 35);
assert.equal(graph.maxLength, 131_072);
assert.deepEqual(graph.source, source);
const license = await readFile(join(official, 'LICENSE.md'));
assert.equal(digest(license), '0a526e4695eb2b9e8cd57655446ed77c3d4a35e55165f434ee5479071ce44ca0');
await mkdir(output);
await mkdir(pkg);
const inventory = JSON.parse(assetBytes['source-inventory.json']!.toString('utf8'));
const upstreamPin = ({repository, revision, file, bytes, sha256}: typeof inventory.base) => ({
  repository,
  revision,
  file,
  bytes,
  sha256
});
const manifest = {
  format: 'web-decision-strands-gemma4-e2b-v1-assets-v1',
  modelId,
  base: upstreamPin(inventory.base),
  adapter: upstreamPin(inventory.adapter),
  pointer: {
    repository: inventory.adapter.repository,
    revision: inventory.adapter.revision,
    file: 'head.safetensors',
    bytes: 9_456_104,
    sha256: '31220ff165a8bb6e48b87a85de11cbfc8660b7047049ea2f91872e1b6ddb45a9'
  },
  files: assets
};
const manifestBytes = Buffer.from(JSON.stringify(manifest) + '\n');
const manifestSha256 = digest(manifestBytes);
const entry = catalogEntry(manifest, manifestSha256);
await build({
  configFile: join(root, 'vite.config.ts'),
  root,
  plugins: [curatedCandidate(root, entry, {...assetBytes, 'manifest.json': manifestBytes}, manifest)],
  build: {outDir: dist, emptyOutDir: true}
});
await copyFile(join(here, 'index.d.ts'), join(dist, 'index.d.ts'));
for (const [name, bytes] of Object.entries({...assetBytes, 'manifest.json': manifestBytes})) {
  assert.equal(digest(await readFile(join(dist, 'gemma4-v1', name))), digest(bytes), name);
}
const version = '0.0.0-gemma4-e2b-v1-private-test';
const standardPackage = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));
const publicFiles: string[] = standardPackage.files.filter((name: string) => name !== 'dist');
for (const name of publicFiles) {
  await mkdir(dirname(join(pkg, name)), {recursive: true});
  await copyFile(join(root, name), join(pkg, name));
}
await writeFile(join(pkg, 'STRANDS-GEMMA4-V1-LICENSE.md'), license, {flag: 'wx'});
await copyFile(join(here, 'NOTICE'), join(pkg, 'STRANDS-GEMMA4-V1-NOTICE'));
await writeFile(
  join(pkg, 'package.json'),
  JSON.stringify(
    {
      name: 'web-decision',
      private: true,
      version,
      type: 'module',
      license: 'Apache-2.0',
      description: 'UNPUBLISHABLE internal candidate including the pending Gemma 4 E2B v1 model.',
      sideEffects: false,
      files: ['dist', ...publicFiles, 'STRANDS-GEMMA4-V1-LICENSE.md', 'STRANDS-GEMMA4-V1-NOTICE'],
      exports: {'.': {types: './dist/index.d.ts', import: './dist/index.js'}},
      dependencies: standardPackage.dependencies
    },
    null,
    2
  ) + '\n'
);
const run = (cwd: string, ...args: string[]) => execFileSync('npm', args, {cwd, encoding: 'utf8'});
const packed = JSON.parse(run(pkg, 'pack', '--ignore-scripts', '--json', '--pack-destination', output));
const archivePath = join(output, packed[0].filename);
const consumer = join(output, 'consumer');
await mkdir(consumer);
await writeFile(join(consumer, 'package.json'), '{"type":"module","private":true}\n');
run(consumer, 'install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', archivePath);
const archive = await verifyArchiveInstall(archivePath, join(consumer, 'node_modules/web-decision'));
const checked = JSON.parse(await readFile(join(archive.installedRoot, 'package.json'), 'utf8'));
assert.equal(checked.private, true);
assert.equal(checked.version, version);
assert.deepEqual({commit: git('rev-parse', 'HEAD'), clean: !git('status', '--porcelain')}, source);
const report = {
  releaseApproved: false,
  source,
  modelId,
  manifestSha256,
  assets,
  archive
};
await writeFile(join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify({archive: archivePath, sha256: archive.archiveSha256, releaseApproved: false}));
