import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {copyFile, mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'vite';
import {verifyArchiveInstall} from '../../../release/verify-release-evidence.ts';

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
const assets = Object.fromEntries(
  await Promise.all(
    Object.entries(files).map(async ([name, path]) => {
      const bytes = await readFile(path);
      return [name, {bytes: bytes.length, sha256: digest(bytes)}] as const;
    })
  )
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
await build({
  configFile: join(root, 'vite.config.ts'),
  root,
  resolve: {preserveSymlinks: true},
  define: {__GEMMA_V1_PRIVATE_PINS__: JSON.stringify({files: assets})},
  build: {
    outDir: dist,
    emptyOutDir: true,
    rollupOptions: {output: {codeSplitting: false}},
    lib: {entry: join(here, 'entry.mts'), formats: ['es'], fileName: () => 'index.js'}
  }
});
await copyFile(join(here, 'index.d.ts'), join(dist, 'index.d.ts'));
await mkdir(join(dist, 'gemma4-v1'));
for (const [name, file] of Object.entries(files)) {
  const target = join(dist, 'gemma4-v1', name);
  await copyFile(file, target);
  assert.equal(digest(await readFile(target)), assets[name]?.sha256);
}
for (const file of ['TVM-LICENSE', 'TVM-NOTICE', 'LICENSE']) await copyFile(join(root, file), join(pkg, file));
await writeFile(join(pkg, 'STRANDS-GEMMA4-V1-LICENSE.md'), license, {flag: 'wx'});
const version = '0.0.0-gemma4-e2b-v1-private-test';
const dependencies = JSON.parse(await readFile(join(root, 'package.json'), 'utf8')).dependencies;
await writeFile(
  join(pkg, 'package.json'),
  JSON.stringify(
    {
      name: 'web-decision',
      private: true,
      version,
      type: 'module',
      license: 'Apache-2.0',
      description: 'Private Gemma 4 E2B v1 validation package.',
      files: ['dist', 'TVM-LICENSE', 'TVM-NOTICE', 'STRANDS-GEMMA4-V1-LICENSE.md'],
      exports: {'.': {types: './dist/index.d.ts', import: './dist/index.js'}},
      dependencies
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
  modelId: 'strands-decider-e2b-gemma4-v1-2610',
  assets,
  archive
};
await writeFile(join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify({archive: archivePath, sha256: archive.archiveSha256, releaseApproved: false}));
