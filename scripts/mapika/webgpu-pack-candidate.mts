import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {copyFile, mkdir, mkdtemp, readFile, realpath, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'vite';
import {verifyArchiveInstall} from '../release/verify-release-evidence.ts';

export const mapikaQaVersion = '0.0.0-mapika2b-candidate';
const packageRoot = fileURLToPath(new URL('../../packages/web-decision/', import.meta.url));
const candidateEntry = fileURLToPath(new URL('webgpu-candidate-entry.mts', import.meta.url));
const assetFiles = ['text.wasm', 'text-bindings.json'] as const;

function digest(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

async function readTextPins(stage: string) {
  const [wasm, manifest] = await Promise.all(assetFiles.map((name) => readFile(join(stage, name))));
  const {wasmSha256, bindings} = JSON.parse(manifest!.toString('utf8')) as {
    wasmSha256?: string;
    bindings?: unknown[];
  };
  assert.equal(wasmSha256, digest(wasm!), 'Staged Mapika text Wasm differs from its compiled manifest.');
  assert.ok(Array.isArray(bindings) && bindings.length === 319, 'Staged Mapika manifest needs 319 decoder bindings.');
  return {wasmSha256, bindingsSha256: digest(manifest!)};
}

async function buildPackage(root: string, stage: string): Promise<void> {
  const dist = join(root, 'dist');
  const require = createRequire(join(packageRoot, 'package.json'));
  await build({
    configFile: join(packageRoot, 'vite.config.ts'),
    root: packageRoot,
    resolve: {
      preserveSymlinks: true,
      alias: [
        {find: 'fast-png', replacement: require.resolve('fast-png')},
        {
          find: '@imagemagick/magick-wasm/magick.wasm?url',
          replacement: '@imagemagick/magick-wasm/magick.wasm?url&no-inline'
        }
      ]
    },
    build: {
      outDir: dist,
      emptyOutDir: true,
      rollupOptions: {output: {codeSplitting: false}},
      lib: {entry: candidateEntry, formats: ['es'], fileName: () => 'index.js'}
    }
  });
  const packagedAssets = join(dist, 'mapika2b');
  await mkdir(packagedAssets);
  await Promise.all(assetFiles.map((name) => copyFile(join(stage, name), join(packagedAssets, name))));
  const {dependencies} = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8')) as {
    dependencies: Record<string, string>;
  };
  const metadata = {
    name: 'web-decision',
    version: mapikaQaVersion,
    private: true,
    description: 'Private Mapika 2B v12 runtime candidate for archive verification.',
    type: 'module',
    exports: {'.': {import: './dist/index.js'}},
    files: ['dist'],
    dependencies
  };
  await writeFile(join(root, 'package.json'), JSON.stringify(metadata, null, 2) + '\n');
}

async function packAndInstall(root: string, candidate: string) {
  const consumer = join(root, 'consumer');
  await mkdir(consumer);
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const pack = execFileSync(npm, ['pack', '--json', '--ignore-scripts', '--pack-destination', root], {
    cwd: candidate,
    encoding: 'utf8'
  });
  const [entry] = JSON.parse(pack) as {filename: string}[];
  assert.equal(entry?.filename, `web-decision-${mapikaQaVersion}.tgz`, 'Mapika candidate archive name changed.');
  const archivePath = join(root, entry!.filename);
  execFileSync(npm, ['install', archivePath, '--ignore-scripts', '--no-audit', '--no-fund', '--prefix', consumer], {
    stdio: 'inherit'
  });
  const installed = await realpath(join(consumer, 'node_modules', 'web-decision'));
  return verifyArchiveInstall(archivePath, installed);
}

async function main(args: readonly string[]) {
  if (args.length !== 1)
    throw new Error('Usage: node --import tsx scripts/mapika/webgpu-pack-candidate.mts <asset-root>');
  const stage = resolve(args[0]!, 'mapika2b');
  const artifacts = await readTextPins(stage);
  const root = await mkdtemp(join(tmpdir(), 'web-decision-mapika2b-qa-'));
  const candidate = join(root, 'package');
  await mkdir(candidate);
  await buildPackage(candidate, stage);
  const archive = await packAndInstall(root, candidate);
  const {private: isPrivate, version} = JSON.parse(await readFile(join(archive.installedRoot, 'package.json'), 'utf8'));
  assert.equal(isPrivate, true, 'Mapika QA candidate must use private npm metadata.');
  assert.equal(version, mapikaQaVersion, 'Mapika candidate version changed.');
  assert.deepEqual(
    archive.archiveFiles.filter((name) => name.endsWith('.js')),
    ['dist/index.js']
  );
  assert.deepEqual(await readTextPins(join(archive.installedRoot, 'dist', 'mapika2b')), artifacts);
  const evidence = {
    status: 'private-mapika2b-qa-candidate',
    artifacts,
    archive,
    browserEnvironment: {
      WEB_DECISION_MAPIKA2B_ARCHIVE: archive.archivePath,
      WEB_DECISION_MAPIKA2B_INSTALLED_PACKAGE: archive.installedRoot,
      WEB_DECISION_MAPIKA2B_INSTALLED_BUNDLE: archive.bundlePath
    }
  };
  const report = JSON.stringify(evidence, null, 2) + '\n';
  await writeFile(join(root, 'candidate-package.json'), report);
  process.stdout.write(report);
}

await main(process.argv.slice(2));
