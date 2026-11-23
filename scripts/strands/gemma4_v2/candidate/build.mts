import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {copyFile, mkdir, mkdtemp, readFile, realpath, writeFile} from 'node:fs/promises';
import {existsSync, readFileSync, statSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {createRequire} from 'node:module';
import {isAbsolute, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build, type Plugin} from 'vite';
import {verifyArchiveInstall} from '../../../release/verify-release-evidence.ts';

const packageRoot = fileURLToPath(new URL('../../../../packages/web-decision/', import.meta.url));
const entry = fileURLToPath(new URL('./entry.mts', import.meta.url));
const root = resolve(process.argv[2] ?? '');
const wasmPath = resolve(process.argv[3] ?? '');
if (!process.argv[2] || !process.argv[3]) throw new Error('Usage: build.mts <fixture-root> <graph-wasm>');
const inputs = {
  'graph.wasm': wasmPath,
  'local.bin': join(root, 'e2b-v2-positions/local.bin'),
  'global.bin': join(root, 'e2b-v2-positions/global.bin'),
  'scoring.bin': join(packageRoot, 'assets/strands-gemma4-v2/scoring.bin')
};
const transformedFiles = new Map<string, string>();
const snapshotPlugin: Plugin = {
  name: 'private-candidate-source-snapshot',
  enforce: 'pre',
  transform(_code, id) {
    const file = id.split('?')[0]!;
    if (!isAbsolute(file) || !existsSync(file) || !statSync(file).isFile()) return;
    const sha256 = createHash('sha256').update(readFileSync(file)).digest('hex');
    const previous = transformedFiles.get(file);
    if (previous && previous !== sha256) throw new Error(`Source changed while building: ${file}`);
    transformedFiles.set(file, sha256);
  }
};
const pins: Record<string, {bytes: number; sha256: string}> = {};
for (const [name, path] of Object.entries(inputs)) {
  const bytes = await readFile(path);
  pins[name] = {bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')};
}
assert.equal(pins['local.bin']!.bytes, 512);
assert.equal(pins['global.bin']!.bytes, 1024);
assert.deepEqual(pins['scoring.bin'], {
  bytes: 746272,
  sha256: '8976812e52bc591dee1a1b891d427b16eb0bf641ea61299a518276d35e6da3ad'
});
const staging = await mkdtemp(join(tmpdir(), 'web-decision-gemma4-v2-private-candidate-'));
const candidate = join(staging, 'package');
const dist = join(candidate, 'dist');
await mkdir(candidate);
await build({
  configFile: join(packageRoot, 'vite.config.ts'),
  root: packageRoot,
  plugins: [snapshotPlugin],
  resolve: {
    preserveSymlinks: true,
    alias: [{find: 'fast-png', replacement: createRequire(join(packageRoot, 'package.json')).resolve('fast-png')}]
  },
  define: {__GEMMA_V2_CANDIDATE_PINS__: JSON.stringify(pins)},
  build: {
    outDir: dist,
    emptyOutDir: true,
    rollupOptions: {output: {codeSplitting: false}},
    lib: {entry, formats: ['es'], fileName: () => 'index.js'}
  }
});
await copyFile(fileURLToPath(new URL('./index.d.ts', import.meta.url)), join(dist, 'index.d.ts'));
const artifactRoot = join(dist, 'gemma4-v2');
await mkdir(artifactRoot);
for (const [name, path] of Object.entries(inputs)) await copyFile(path, join(artifactRoot, name));
const {dependencies} = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8')) as {
  dependencies: Record<string, string>;
};
const version = '0.0.0-gemma4-v2-private-test';
await writeFile(
  join(candidate, 'package.json'),
  JSON.stringify(
    {
      name: 'web-decision',
      version,
      private: true,
      description: 'Private one-off Gemma 4 v2 candidate; not a public model release.',
      type: 'module',
      exports: {'.': {types: './dist/index.d.ts', import: './dist/index.js'}},
      types: './dist/index.d.ts',
      files: ['dist'],
      dependencies
    },
    null,
    2
  ) + '\n'
);
const pack = JSON.parse(
  execFileSync('npm', ['pack', '--json', '--ignore-scripts', '--pack-destination', staging], {
    cwd: candidate,
    encoding: 'utf8'
  })
) as Array<{filename: string}>;
const archive = join(staging, pack[0]!.filename);
const consumer = join(staging, 'consumer');
await mkdir(consumer);
execFileSync(
  'npm',
  ['install', archive, '--offline', '--ignore-scripts', '--no-audit', '--no-fund', '--prefix', consumer],
  {stdio: 'inherit'}
);
const installed = await realpath(join(consumer, 'node_modules/web-decision'));
for (const [name, pin] of Object.entries(pins)) {
  const bytes = await readFile(join(installed, 'dist/gemma4-v2', name));
  assert.equal(bytes.length, pin.bytes);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), pin.sha256);
}
await verifyArchiveInstall(archive, installed);
for (const [file, sha256] of transformedFiles) {
  assert.equal(
    createHash('sha256')
      .update(await readFile(file))
      .digest('hex'),
    sha256,
    `Source changed during build: ${file}`
  );
}
const snapshot = [...transformedFiles]
  .map(([file, sha256]) => ({file: relative(packageRoot, file), sha256}))
  .sort((a, b) => a.file.localeCompare(b.file));
assert.ok(
  snapshot.some((item) => item.file.includes('gemma4-v2-runner.mts')),
  'Candidate bundle did not include the runner source.'
);
const sourceSnapshotSha256 = createHash('sha256').update(JSON.stringify(snapshot)).digest('hex');
const tokenizerPath = join(consumer, 'node_modules/@huggingface/tokenizers/dist/tokenizers.mjs');
const tokenizerSha256 = createHash('sha256')
  .update(await readFile(tokenizerPath))
  .digest('hex');
const lockSha256 = createHash('sha256')
  .update(await readFile(join(consumer, 'package-lock.json')))
  .digest('hex');
const archiveBytes = await readFile(archive);
const report = {
  status: 'private-unreleased-test-candidate',
  sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], {cwd: packageRoot, encoding: 'utf8'}).trim(),
  sourceSnapshotSha256,
  sourceSnapshot: snapshot,
  nodeVersion: process.version,
  tokenizerPath,
  tokenizerSha256,
  consumerLockSha256: lockSha256,
  archive,
  archiveSha256: createHash('sha256').update(archiveBytes).digest('hex'),
  installed,
  installedBundle: join(installed, 'dist/index.js'),
  pins
};
await writeFile(join(staging, 'report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
