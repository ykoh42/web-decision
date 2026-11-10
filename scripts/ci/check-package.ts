import assert from 'node:assert/strict';
import {execFileSync, type ExecFileSyncOptions} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {checkConsumerTypes} from './check-types.mts';
import {findPackedContentIssues} from './packed-content.ts';
import {withTemporaryDirectories} from './temporary-directories.ts';

const kananaModelId = 'kanana-2-v-decide-3b-2611';
const kananaManifestSha256 = 'd4e6d1c735b1e61c8e8c941730886e0914f6820d5cc9025fd4ef3de087689eb5';
const kananaLicenseSha256 = '3c13f0cf41c4d49f6d3885bffb27c39431826f9886cacac75349b4ffb58361c4';
const kananaRequiredNotice =
  'Kanana is licensed in accordance with the Kanana Open License Agreement. Copyright © KAKAO Corp. All Rights Reserved.';
const kananaWasmFiles = ['kanana-text-model.wasm', 'kanana-vision-model.wasm'] as const;
type ManifestFile = Readonly<{sha256: string; bytes: number}>;
type KananaManifest = Readonly<{files: Readonly<Record<string, ManifestFile>>}>;
type PackageMetadata = Readonly<{name: string; version: string; license: string; private?: boolean}>;
type PackArchive = Readonly<{filename: string; files: readonly Readonly<{path: string}>[]}>;

const packagedSourceFiles = [
  'LICENSE',
  'KANANA-NOTICE',
  'KANANA-OPEN-LICENSE.txt',
  'EMSCRIPTEN-LICENSE',
  'ROLLDOWN-LICENSE',
  'ROLLDOWN-THIRD-PARTY-LICENSE',
  'VITE-LICENSE.md',
  'TVM-LICENSE',
  'TVM-NOTICE',
  'README.md',
  'assets/kanana/manifest.json',
  'assets/kanana/kanana-text-model.wasm',
  'assets/kanana/text-weight-manifest.json',
  'assets/kanana/kanana-vision-model.wasm',
  'assets/kanana/vision-weight-manifest.json',
  'assets/kanana/safetensors-range-manifest.json',
  'src/catalog/README.md',
  'src/catalog/types.generated.d.ts',
  'src/curated/index.d.ts',
  'src/curated/types.d.ts'
];

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical((value as Record<string, unknown>)[key])])
    );
  return value;
}

function runInstalledApiSmoke(consumer: string): void {
  const smokeTest = [
    "import assert from 'node:assert/strict';",
    "import * as library from 'web-decision';",
    "assert.equal(typeof library.listModels, 'function');",
    "assert.equal(typeof library.describeModel, 'function');",
    "assert.equal(typeof library.loadDecisionModel, 'function');",
    "assert.equal(typeof library.clearModelCache, 'function');",
    "assert.equal(typeof library.WebDecisionError, 'function');",
    'const models = library.listModels();',
    "assert.deepEqual(models.map(({modelId}) => modelId), ['" + kananaModelId + "']);",
    "assert.ok(models.every(({backend}) => backend?.name === 'tvm-webgpu'));",
    "const kanana = library.describeModel('" + kananaModelId + "');",
    "assert.equal(kanana.modelId, '" + kananaModelId + "');",
    "assert.equal(kanana.backend.name, 'tvm-webgpu');",
    "assert.equal(kanana.assets.manifestSha256, '" + kananaManifestSha256 + "');",
    "assert.equal(kanana.verification.currentBundleGpu, 'verified-m4-pro-chromium-157');",
    'assert.deepEqual([...kanana.assets.requiredFiles].sort(), ' +
      JSON.stringify(
        [
          'manifest.json',
          'kanana-text-model.wasm',
          'text-weight-manifest.json',
          'kanana-vision-model.wasm',
          'vision-weight-manifest.json',
          'safetensors-range-manifest.json'
        ].sort()
      ) +
      ');',
    "const invalidInput = (error) => error instanceof library.WebDecisionError && error.code === 'INVALID_INPUT';",
    'const originalFetch = globalThis.fetch;',
    'let networkRequested = false;',
    "globalThis.fetch = async () => { networkRequested = true; throw new Error('unexpected network access'); };",
    'try {',
    "  for (const option of ['maxPrefetchBytes', 'maxInputTokens', 'maxInputCharacters']) {",
    "    await assert.rejects(library.loadDecisionModel({modelId: '" +
      kananaModelId +
      "', [option]: 128}), invalidInput);",
    '  }',
    '  assert.equal(networkRequested, false);',
    '} finally {',
    '  globalThis.fetch = originalFetch;',
    '}',
    "assert.throws(() => library.describeModel('unknown-model'), invalidInput);",
    'assert.throws(() => library.describeModel(null), invalidInput);',
    "await assert.rejects(library.loadDecisionModel({modelId: 'unknown-model'}), invalidInput);",
    "await assert.rejects(library.loadDecisionModel({modelId: '" +
      kananaModelId +
      "', assetsUrl: 'https://example.invalid/'}), invalidInput);",
    "await assert.rejects(library.loadDecisionModel({modelId: '" +
      kananaModelId +
      "', sourceUrl: 'https://example.invalid/'}), invalidInput);",
    "await assert.rejects(library.loadDecisionModel({model: '" + kananaModelId + "'}), invalidInput);"
  ].join('\n');
  execFileSync(process.execPath, ['--input-type=module', '-e', smokeTest], {cwd: consumer, stdio: 'inherit'});
}

function runCli(command: string, args: readonly string[], options: ExecFileSyncOptions): string | Buffer {
  return execFileSync(command, args, {
    ...options,
    shell: process.platform === 'win32'
  });
}

function assertConsumerBytes(consumer: string, files: readonly string[], distFiles: readonly string[]): void {
  const installedPackage = join(consumer, 'node_modules/web-decision');
  for (const file of packagedSourceFiles) {
    assert.deepEqual(
      readFileSync(join(installedPackage, file)),
      readFileSync(join(pkg, file)),
      'The tarball must preserve ' + file + '.'
    );
  }
  for (const name of distFiles) {
    assert.deepEqual(
      readFileSync(join(installedPackage, 'dist', name)),
      readFileSync(join(pkg, 'dist', name)),
      'The tarball must preserve dist/' + name + '.'
    );
  }
  for (const file of files) {
    assert.deepEqual(
      findPackedContentIssues(readFileSync(join(installedPackage, file))),
      [],
      'Potential private content in packed file: ' + file
    );
  }
}

function expectedPackageFiles(distFiles: readonly string[]): string[] {
  return [...packagedSourceFiles, ...distFiles.map((name) => 'dist/' + name), 'package.json'].sort();
}

function verifyKananaArtifacts(distDirectory: string, distFiles: readonly string[]): void {
  const sourceDirectory = join(pkg, 'assets/kanana');
  const manifest = JSON.parse(readFileSync(join(sourceDirectory, 'manifest.json'), 'utf8')) as KananaManifest & {
    modelId: string;
  };
  assert.equal(manifest.modelId, kananaModelId);
  assert.equal(
    createHash('sha256')
      .update(JSON.stringify(canonical(manifest)) + '\n')
      .digest('hex'),
    kananaManifestSha256,
    'The canonical Kanana manifest differs from the public catalog.'
  );
  for (const [name, metadata] of Object.entries(manifest.files)) {
    const source = readFileSync(join(sourceDirectory, name));
    assert.equal(source.byteLength, metadata.bytes, 'The Kanana source size differs from its manifest: ' + name + '.');
    assert.equal(
      createHash('sha256').update(source).digest('hex'),
      metadata.sha256,
      'The Kanana source differs from its manifest: ' + name + '.'
    );
    if (!(kananaWasmFiles as readonly string[]).includes(name)) continue;
    assert.ok(distFiles.includes(name), 'The package build must emit dist/' + name + '.');
    const built = readFileSync(join(distDirectory, name));
    assert.equal(built.byteLength, metadata.bytes, 'The Kanana runtime size differs from its manifest: ' + name + '.');
    assert.equal(
      createHash('sha256').update(built).digest('hex'),
      metadata.sha256,
      'The Kanana runtime differs from its manifest: ' + name + '.'
    );
  }
  assert.deepEqual(
    Object.keys(manifest.files).sort(),
    [
      'kanana-text-model.wasm',
      'text-weight-manifest.json',
      'kanana-vision-model.wasm',
      'vision-weight-manifest.json',
      'safetensors-range-manifest.json'
    ].sort()
  );
}

function verifyKananaLicenseNotice(): void {
  const licenseBytes = readFileSync(join(pkg, 'KANANA-OPEN-LICENSE.txt'));
  assert.equal(licenseBytes.byteLength, 11_867, 'The pinned Kanana license size changed.');
  assert.equal(createHash('sha256').update(licenseBytes).digest('hex'), kananaLicenseSha256);
  const notice = readFileSync(join(pkg, 'KANANA-NOTICE'), 'utf8');
  assert.ok(notice.startsWith(kananaRequiredNotice), 'The Kanana notice must include Kakao’s required attribution.');
  const readme = readFileSync(join(pkg, 'README.md'), 'utf8');
  for (const term of ['Guidelines For Responsible AI', 'copy of the agreement', 'modified files', 'on-device domains'])
    assert.ok(readme.includes(term), 'The package README must pass on Kanana license requirements: ' + term + '.');
}

const root = fileURLToPath(new URL('../../', import.meta.url));
const pkg = join(root, 'packages/web-decision');
withTemporaryDirectories((makeTempDirectory: (prefix: string) => string) => {
  const npmConsumer = makeTempDirectory('web-decision-ci-npm-consumer-');
  const pnpmConsumer = makeTempDirectory('web-decision-ci-pnpm-consumer-');
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const pnpm = process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm';
  const metadata = JSON.parse(readFileSync(join(pkg, 'package.json'), 'utf8')) as PackageMetadata;
  assert.notEqual(metadata.private, true);
  assert.equal(metadata.license, 'Apache-2.0');

  const output = runCli(npm, ['pack', '--json', '--pack-destination', npmConsumer], {
    cwd: pkg,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit']
  });
  const [archive] = JSON.parse(output as string) as PackArchive[];
  assert.ok(archive, 'npm pack did not return an archive.');
  const packageFiles = archive.files.map((file) => file.path).sort();
  const distDirectory = join(pkg, 'dist');
  const distFiles = readdirSync(distDirectory).sort();
  verifyKananaArtifacts(distDirectory, distFiles);
  verifyKananaLicenseNotice();
  const expectedFiles = expectedPackageFiles(distFiles);
  assert.deepEqual(packageFiles, expectedFiles);
  assert.deepEqual(
    packageFiles.filter((file) => file.endsWith('.wasm')).sort(),
    [
      'assets/kanana/kanana-text-model.wasm',
      'assets/kanana/kanana-vision-model.wasm',
      'dist/kanana-text-model.wasm',
      'dist/kanana-vision-model.wasm',
      'dist/model.wasm'
    ].sort()
  );

  const allDistChunks = distFiles
    .filter((name) => name.endsWith('.js'))
    .map((name) => ({name, text: readFileSync(join(distDirectory, name), 'utf8')}));
  for (const {name, text} of allDistChunks) {
    assert.doesNotMatch(
      text,
      /\b(?:from\s*|import\s*\(\s*)["'][^"']*\?url(?:&|["'])/u,
      'The published chunk dist/' + name + ' must not depend on consumer asset plugins.'
    );
  }

  const packagePath = resolve(npmConsumer, archive.filename);
  writeFileSync(join(npmConsumer, 'package.json'), '{"private":true,"type":"module"}\n');
  runCli(npm, ['install', '--ignore-scripts', '--no-audit', '--no-fund', packagePath], {
    cwd: npmConsumer,
    stdio: 'inherit'
  });
  runCli(npm, ['audit', '--omit=dev'], {cwd: npmConsumer, stdio: 'inherit'});
  const installed = JSON.parse(
    readFileSync(join(npmConsumer, 'node_modules/web-decision/package.json'), 'utf8')
  ) as PackageMetadata;
  assert.equal(installed.version, metadata.version);
  assert.equal(installed.license, metadata.license);
  assertConsumerBytes(npmConsumer, packageFiles, distFiles);
  runInstalledApiSmoke(npmConsumer);
  checkConsumerTypes(npmConsumer);

  const packageManager = (
    JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')) as {
      packageManager: string;
    }
  ).packageManager;
  writeFileSync(
    join(pnpmConsumer, 'package.json'),
    JSON.stringify({
      private: true,
      type: 'module',
      packageManager
    }) + '\n'
  );
  runCli(pnpm, ['add', '--ignore-scripts', packagePath], {cwd: pnpmConsumer, stdio: 'inherit'});
  const pnpmInstalled = JSON.parse(
    readFileSync(join(pnpmConsumer, 'node_modules/web-decision/package.json'), 'utf8')
  ) as PackageMetadata;
  assert.equal(pnpmInstalled.version, metadata.version);
  assert.equal(pnpmInstalled.license, metadata.license);
  assertConsumerBytes(pnpmConsumer, packageFiles, distFiles);
  runInstalledApiSmoke(pnpmConsumer);
  checkConsumerTypes(pnpmConsumer);

  console.log(
    'Installed ' + metadata.name + '@' + metadata.version + '; Kanana manifest SHA-256: ' + kananaManifestSha256
  );
  console.log('The archive installs, imports, and type-checks in clean npm and pnpm consumers.');
  console.log('This check imports the package in Node; it does not run browser inference.');
});
