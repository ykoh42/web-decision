import assert from 'node:assert/strict';
import {execFileSync, type ExecFileSyncOptions} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {hasQuotedAssetKey} from './asset-key.ts';
import {checkConsumerTypes} from './check-types.mts';
import {findPackedContentIssues} from './packed-content.ts';
import {withTemporaryDirectories} from './temporary-directories.ts';

const stableKevModelId = 'kev-0.6b-q4f16';
const stableKevManifestSha256 = '57c554e757a96093669c1555e9e2564780c7eaba5e102b4fcd9ad0c535878fe2';
type BundleChunk = Readonly<{name: string; text: string}>;
type AssetInput = readonly [name: string, sourcePath: string];
type ManifestFile = Readonly<{sha256: string; bytes: number}>;
type StableManifest = Readonly<{files: Readonly<Record<string, ManifestFile>>}>;
type PackageMetadata = Readonly<{name: string; version: string; license: string; private?: boolean}>;
type PackArchive = Readonly<{filename: string; files: readonly Readonly<{path: string}>[]}>;

const stableAssetInputs: readonly AssetInput[] = [
  ['model.wasm', 'assets/kev06-stable/model.wasm'],
  ['tokenizer.json', 'assets/experimental-kev-upstream/tokenizer.json'],
  ['tokenizer_config.json', 'assets/experimental-kev-upstream/tokenizer_config.json'],
  ['model_config.json', 'assets/experimental-kev-upstream/model_config.json'],
  ['weights/tensor-cache.json', 'assets/experimental-kev-upstream/weights/tensor-cache.json']
];
const packagedSourceFiles = [
  'LICENSE',
  'KEV06-NOTICE',
  'KEV-NOTICE',
  'EMSCRIPTEN-LICENSE',
  'ROLLDOWN-LICENSE',
  'ROLLDOWN-THIRD-PARTY-LICENSE',
  'VITE-LICENSE.md',
  'TVM-LICENSE',
  'TVM-NOTICE',
  'README.md',
  'assets/kev06-stable/manifest.json',
  'assets/kev06-stable/PROVENANCE.md',
  'assets/kev06-stable/notices/MODEL-APACHE-2.0.txt',
  'assets/experimental-kev-upstream/notices/COMPILER-RT-HEADER-NOTICE.txt',
  'assets/experimental-kev-upstream/notices/COMPILER-RT-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/DLPACK-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/EMSCRIPTEN-COMPILER-RT-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/EMSCRIPTEN-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/LIBCXX-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/LIBCXXABI-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/MODEL-APACHE-2.0.txt',
  'assets/experimental-kev-upstream/notices/MUSL-COPYRIGHT.txt',
  'assets/experimental-kev-upstream/notices/NOTICE-SOURCES.md',
  'assets/experimental-kev-upstream/notices/PROVENANCE.md',
  'assets/experimental-kev-upstream/notices/TVM-FFI-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/TVM-FFI-NOTICE.txt',
  'assets/experimental-kev-upstream/notices/TVM-LICENSE.txt',
  'assets/experimental-kev-upstream/notices/TVM-NOTICE.txt',
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

function localChunkImports(text: string): Set<string> {
  const imports = new Set<string>();
  const pattern = /\b(?:from\s*|import\s*(?:\(\s*)?)["']\.\/([^"']+\.js)["']/gu;
  for (const match of text.matchAll(pattern)) {
    const imported = match[1];
    if (imported) imports.add(imported);
  }
  return imports;
}

function dynamicChunkImports(text: string): Set<string> {
  const imports = new Set<string>();
  for (const match of text.matchAll(/\bimport\s*\(\s*["']\.\/([^"']+\.js)["']\s*\)/gu)) {
    const imported = match[1];
    if (imported) imports.add(imported);
  }
  return imports;
}

function collectChunkGraph(directory: string, entryFiles: Iterable<string>): BundleChunk[] {
  const pending = [...entryFiles];
  const visited = new Set<string>();
  const chunks: BundleChunk[] = [];
  while (pending.length) {
    const name = pending.pop();
    if (!name || visited.has(name)) continue;
    visited.add(name);
    const text = readFileSync(join(directory, name), 'utf8');
    chunks.push({name, text});
    for (const imported of localChunkImports(text)) {
      if (!visited.has(imported)) pending.push(imported);
    }
  }
  return chunks;
}

function inlineDataHashes(chunks: readonly BundleChunk[]): Set<string> {
  const hashes = new Set<string>();
  for (const {text} of chunks) {
    for (const match of text.matchAll(/["'](data:[^"']+)["']/gu)) {
      const dataUrl = match[1];
      if (!dataUrl) continue;
      const comma = dataUrl.indexOf(',');
      if (comma <= 5) continue;
      const metadata = dataUrl.slice(5, comma);
      const payload = dataUrl.slice(comma + 1);
      try {
        const bytes = metadata.includes(';base64')
          ? Buffer.from(payload, 'base64')
          : Buffer.from(decodeURIComponent(payload));
        hashes.add(createHash('sha256').update(bytes).digest('hex'));
      } catch {
        continue;
      }
    }
  }
  return hashes;
}

function verifyInlineAssets(chunks: readonly BundleChunk[], assets: readonly AssetInput[], description: string): void {
  const dataHashes = inlineDataHashes(chunks);
  for (const [name, sourcePath] of assets) {
    assert.ok(
      chunks.some(({text}) => hasQuotedAssetKey(text, name)),
      'The ' + description + ' lazy chunk is missing the asset key ' + name + '.'
    );
    const source = readFileSync(join(pkg, sourcePath));
    const sourceHash = createHash('sha256').update(source).digest('hex');
    assert.ok(dataHashes.has(sourceHash), 'The ' + description + ' lazy chunk changed the pinned asset ' + name + '.');
  }
}

function verifyStableManifest(
  manifest: StableManifest,
  packageEntry: string,
  reachableChunks: readonly BundleChunk[]
): string {
  const manifestHash = createHash('sha256')
    .update(JSON.stringify(canonical(manifest)) + '\n')
    .digest('hex');
  assert.equal(manifestHash, stableKevManifestSha256, 'The bundled stable manifest must match its released pin.');
  const pinnedText = reachableChunks.map(({text}) => text).join('\n');
  assert.ok(packageEntry.includes(manifestHash), 'The package entry must retain the stable manifest pin.');
  for (const metadata of Object.values(manifest.files)) {
    assert.ok(pinnedText.includes(metadata.sha256), 'The package must retain every stable manifest file pin.');
  }
  for (const [name, sourcePath] of stableAssetInputs) {
    const source = readFileSync(join(pkg, sourcePath));
    const metadata = manifest.files[name];
    assert.ok(metadata, 'The stable manifest is missing ' + name + '.');
    assert.equal(
      source.byteLength,
      metadata.bytes,
      'The stable source asset size differs from its manifest: ' + name + '.'
    );
    assert.equal(
      createHash('sha256').update(source).digest('hex'),
      metadata.sha256,
      'The stable source asset differs from its manifest: ' + name + '.'
    );
  }
  return manifestHash;
}

function candidateWasmHash(): string | undefined {
  const wasmPath = join(pkg, 'assets/experimental-kev-upstream/model.wasm');
  if (existsSync(wasmPath)) return createHash('sha256').update(readFileSync(wasmPath)).digest('hex');

  const manifestPath = join(pkg, 'assets/experimental-kev-upstream/manifest.json');
  if (!existsSync(manifestPath)) return undefined;
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as StableManifest;
  return manifest.files?.['model.wasm']?.sha256;
}

function stableInlineHashes(manifest: StableManifest): Record<string, string> {
  return Object.fromEntries(
    stableAssetInputs.map(([name]) => {
      const metadata = manifest.files[name];
      assert.ok(metadata, 'The stable manifest is missing ' + name + '.');
      return [name, metadata.sha256];
    })
  );
}

function runInstalledApiSmoke(consumer: string, requiredFiles: readonly string[]): void {
  const smokeTest = [
    "import assert from 'node:assert/strict';",
    "import * as library from 'web-decision';",
    "assert.equal(typeof library.listModels, 'function');",
    "assert.equal(typeof library.describeModel, 'function');",
    "assert.equal(typeof library.loadDecisionModel, 'function');",
    "assert.equal(typeof library.clearModelCache, 'function');",
    "assert.equal(typeof library.WebDecisionError, 'function');",
    'const models = library.listModels();',
    "assert.deepEqual(models.map(({modelId}) => modelId), ['" + stableKevModelId + "']);",
    "assert.ok(models.every(({backend}) => backend?.name === 'tvm-webgpu'));",
    "const description = library.describeModel('" + stableKevModelId + "');",
    "assert.equal(description.modelId, '" + stableKevModelId + "');",
    "assert.equal(description.backend.name, 'tvm-webgpu');",
    "assert.equal(description.assets.manifestSha256, '" + stableKevManifestSha256 + "');",
    'assert.deepEqual([...description.assets.requiredFiles].sort(), ' +
      JSON.stringify([...requiredFiles].sort()) +
      ');',
    "const invalidInput = (error) => error instanceof library.WebDecisionError && error.code === 'INVALID_INPUT';",
    "assert.throws(() => library.describeModel('unknown-model'), invalidInput);",
    'assert.throws(() => library.describeModel(null), invalidInput);',
    "await assert.rejects(library.loadDecisionModel({modelId: 'unknown-model'}), invalidInput);",
    "await assert.rejects(library.loadDecisionModel({modelId: '" +
      stableKevModelId +
      "', assetsUrl: 'https://example.invalid/'}), invalidInput);",
    "await assert.rejects(library.loadDecisionModel({modelId: '" +
      stableKevModelId +
      "', sourceUrl: 'https://example.invalid/'}), invalidInput);",
    "await assert.rejects(library.loadDecisionModel({model: '" + stableKevModelId + "'}), invalidInput);"
  ].join('\n');
  execFileSync(process.execPath, ['--input-type=module', '-e', smokeTest], {cwd: consumer, stdio: 'inherit'});
}

function runInstalledAssetSmoke(
  consumer: string,
  chunkNames: readonly string[],
  expectedInlineHashes: Readonly<Record<string, string>>
): void {
  const smokeTest = [
    "import assert from 'node:assert/strict';",
    "import {createHash} from 'node:crypto';",
    "const entry = import.meta.resolve('web-decision');",
    'const modules = await Promise.all(' +
      JSON.stringify(chunkNames) +
      ".map((name) => import(new URL('./' + name, entry))));",
    'const exports = modules.flatMap((module) => Object.entries(module));',
    "const assets = exports.find(([name]) => name === 'stableKevBundledUrls')?.[1];",
    "assert.ok(assets && typeof assets === 'object', 'The stable lazy chunk must export its bundled asset map.');",
    'const expected = ' + JSON.stringify(expectedInlineHashes) + ';',
    'const hashDataUrl = (dataUrl) => {',
    "  assert.equal(typeof dataUrl, 'string');",
    "  assert.ok(dataUrl.startsWith('data:'), 'The stable asset must be inline in its lazy chunk.');",
    "  const comma = dataUrl.indexOf(',');",
    '  const metadata = dataUrl.slice(5, comma);',
    '  const payload = dataUrl.slice(comma + 1);',
    "  const bytes = metadata.includes(';base64') ? Buffer.from(payload, 'base64') : Buffer.from(decodeURIComponent(payload));",
    "  return createHash('sha256').update(bytes).digest('hex');",
    '};',
    'for (const [name, sha256] of Object.entries(expected)) {',
    "  assert.equal(hashDataUrl(assets[name]), sha256, 'The installed stable asset hash changed: ' + name + '.');",
    '}'
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
  const expectedFiles = expectedPackageFiles(distFiles);
  assert.deepEqual(packageFiles, expectedFiles);
  assert.ok(
    packageFiles.every((file) => !file.endsWith('.wasm')),
    'WASM assets must be inline in the stable lazy chunk, not separate package files.'
  );

  const packageEntry = readFileSync(join(distDirectory, 'index.js'), 'utf8');
  const stableManifest = JSON.parse(
    readFileSync(join(pkg, 'assets/kev06-stable/manifest.json'), 'utf8')
  ) as StableManifest;
  const expectedInlineHashes = stableInlineHashes(stableManifest);
  const reachableChunks = collectChunkGraph(distDirectory, ['index.js']);
  const stableAssetChunks = collectChunkGraph(distDirectory, dynamicChunkImports(packageEntry)).filter(({text}) =>
    text.includes('stableKevBundledUrls')
  );
  assert.ok(stableAssetChunks.length > 0, 'The package must emit a lazy stable Kev asset chunk.');
  const manifestHash = verifyStableManifest(stableManifest, packageEntry, reachableChunks);
  verifyInlineAssets(stableAssetChunks, stableAssetInputs, 'stable Kev');

  const allDistChunks = distFiles
    .filter((name) => name.endsWith('.js'))
    .map((name) => ({name, text: readFileSync(join(distDirectory, name), 'utf8')}));
  const unverifiedWasmHash = candidateWasmHash();
  if (unverifiedWasmHash) {
    assert.ok(
      !inlineDataHashes(allDistChunks).has(unverifiedWasmHash),
      'The unverified candidate WASM must not be embedded in the package.'
    );
  }
  for (const {name, text} of allDistChunks) {
    assert.doesNotMatch(
      text,
      /\?url(?:&|["'])/u,
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
  runInstalledApiSmoke(npmConsumer, ['manifest.json', ...Object.keys(stableManifest.files)]);
  runInstalledAssetSmoke(
    npmConsumer,
    stableAssetChunks.map(({name}) => name),
    expectedInlineHashes
  );
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
  runInstalledApiSmoke(pnpmConsumer, ['manifest.json', ...Object.keys(stableManifest.files)]);
  runInstalledAssetSmoke(
    pnpmConsumer,
    stableAssetChunks.map(({name}) => name),
    expectedInlineHashes
  );
  checkConsumerTypes(pnpmConsumer);

  console.log(
    'Installed ' +
      metadata.name +
      '@' +
      metadata.version +
      '; stable Kev manifest SHA-256: ' +
      manifestHash +
      '; lazy asset chunk: ' +
      stableAssetChunks.map(({name}) => name).join(', ')
  );
  console.log('The archive installs, imports, and type-checks in clean npm and pnpm consumers.');
  console.log('This check imports the package in Node; it does not run browser inference.');
});
