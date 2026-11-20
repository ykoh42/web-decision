import assert from 'node:assert/strict';
import {execFileSync, type ExecFileSyncOptions} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {basename, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {APPROVED_MODEL_IDS} from '../../packages/web-decision/src/catalog/public-catalog.generated.ts';
import {hasQuotedAssetKey} from './asset-key.ts';
import {checkConsumerTypes} from './check-types.mts';
import {findPackedContentIssues} from './packed-content.ts';
import {withTemporaryDirectories} from './temporary-directories.ts';

const stableKevModelId = 'kev-0.6b-q4f16';
const stableKevManifestSha256 = '57c554e757a96093669c1555e9e2564780c7eaba5e102b4fcd9ad0c535878fe2';
const kananaModelId = 'kanana-2-v-decide-3b-2611';
const kananaManifestSha256 = 'd4e6d1c735b1e61c8e8c941730886e0914f6820d5cc9025fd4ef3de087689eb5';
const qwen35ModelId = 'strands-decider-2b-qwen3.5-v1-2610';
const kananaLicenseSha256 = '3c13f0cf41c4d49f6d3885bffb27c39431826f9886cacac75349b4ffb58361c4';
const kananaRequiredNotice =
  'Kanana is licensed in accordance with the Kanana Open License Agreement. Copyright © KAKAO Corp. All Rights Reserved.';
const kananaWasmFiles = ['kanana-text-model.wasm', 'kanana-vision-model.wasm'] as const;
type BundleChunk = Readonly<{name: string; text: string}>;
type AssetInput = readonly [name: string, sourcePath: string];
type ManifestFile = Readonly<{sha256: string; bytes: number}>;
type StableManifest = Readonly<{files: Readonly<Record<string, ManifestFile>>}>;
type PackageMetadata = Readonly<{name: string; version: string; license: string; private?: boolean}>;
type LockedModelAsset = ManifestFile & Readonly<{sourcePath: string; manifestName: string}>;
type BundledArtifactLock = Readonly<{
  manifest: StableManifest & Readonly<{sourcePath: string; rawSha256: string; canonicalSha256: string}>;
  bundledRuntimeFiles: readonly LockedModelAsset[];
}>;
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
  'KANANA-NOTICE',
  'KANANA-OPEN-LICENSE.txt',
  'KEV06-NOTICE',
  'KEV-NOTICE',
  'EMSCRIPTEN-LICENSE',
  'ROLLDOWN-LICENSE',
  'ROLLDOWN-THIRD-PARTY-LICENSE',
  'VITE-LICENSE.md',
  'TVM-LICENSE',
  'TVM-NOTICE',
  'README.md',
  'src/curated/public-index.d.ts',
  'src/curated/public-types.d.ts',
  'src/curated/public-model-id.generated.d.ts'
];
const qwen35LicensePins = {
  'QWEN-APACHE-LICENSE.txt': '50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32',
  'STRANDS-QWEN35-LICENSE.md': 'c0d777a0fed01cceb107d91f1aef96d51737be04f2ed75b1b933286a170d7411'
};
const qwen35PackageFiles = [
  ...Object.keys(qwen35LicensePins),
  'STRANDS-QWEN35-NOTICE',
  'src/curated/strands-qwen35-candidate-types.d.ts',
  'src/kanana/json-float.d.ts'
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

function runInstalledApiSmoke(
  consumer: string,
  requiredFiles: readonly string[],
  qwen35Manifest?: BundledArtifactLock['manifest']
): void {
  const approvedModelIds = new Set<string>(APPROVED_MODEL_IDS);
  const unavailableCandidateIds = [stableKevModelId, kananaModelId, qwen35ModelId].filter(
    (modelId) => !approvedModelIds.has(modelId)
  );
  const smokeTest = [
    "import assert from 'node:assert/strict';",
    "import * as library from 'web-decision';",
    "assert.equal(typeof library.listModels, 'function');",
    "assert.equal(typeof library.describeModel, 'function');",
    "assert.equal(typeof library.loadDecisionModel, 'function');",
    "assert.equal(typeof library.clearModelCache, 'function');",
    "assert.equal(typeof library.WebDecisionError, 'function');",
    'const models = library.listModels();',
    'assert.deepEqual(models.map(({modelId}) => modelId), ' + JSON.stringify(APPROVED_MODEL_IDS) + ');',
    "assert.ok(models.every(({backend}) => backend?.name === 'tvm-webgpu'));",
    ...(approvedModelIds.has(stableKevModelId)
      ? [
          "const description = library.describeModel('" + stableKevModelId + "');",
          "assert.equal(description.modelId, '" + stableKevModelId + "');",
          "assert.equal(description.backend.name, 'tvm-webgpu');",
          "assert.equal(description.assets.manifestSha256, '" + stableKevManifestSha256 + "');",
          'assert.deepEqual([...description.assets.requiredFiles].sort(), ' +
            JSON.stringify([...requiredFiles].sort()) +
            ');'
        ]
      : []),
    ...(approvedModelIds.has(kananaModelId)
      ? [
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
            ');'
        ]
      : []),
    ...(qwen35Manifest
      ? [
          'const qwen35 = library.describeModel(' + JSON.stringify(qwen35ModelId) + ');',
          'assert.equal(qwen35.assets.manifestSha256, ' + JSON.stringify(qwen35Manifest.canonicalSha256) + ');',
          'assert.deepEqual([...qwen35.assets.requiredFiles].sort(), ' +
            JSON.stringify(['manifest.json', ...Object.keys(qwen35Manifest.files)].sort()) +
            ');'
        ]
      : []),
    "const invalidInput = (error) => error instanceof library.WebDecisionError && error.code === 'INVALID_INPUT';",
    'const originalFetch = globalThis.fetch;',
    'let networkRequested = false;',
    "globalThis.fetch = async () => { networkRequested = true; throw new Error('unexpected network access'); };",
    'try {',
    '  for (const {modelId} of models) {',
    "    await assert.rejects(library.loadDecisionModel({modelId, assetsUrl: 'https://example.invalid/'}), invalidInput);",
    '  }',
    ...(approvedModelIds.has(kananaModelId)
      ? [
          "  for (const option of ['maxPrefetchBytes', 'maxInputTokens', 'maxInputCharacters']) {",
          "    await assert.rejects(library.loadDecisionModel({modelId: '" +
            kananaModelId +
            "', [option]: 128}), invalidInput);",
          '  }'
        ]
      : []),
    '  for (const modelId of ' + JSON.stringify(unavailableCandidateIds) + ') {',
    '    assert.throws(() => library.describeModel(modelId), invalidInput);',
    '    await assert.rejects(library.loadDecisionModel({modelId}), invalidInput);',
    '  }',
    '  assert.equal(networkRequested, false);',
    '} finally {',
    '  globalThis.fetch = originalFetch;',
    '}',
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

function assertConsumerBytes(consumer: string, files: readonly string[]): void {
  const installedPackage = join(consumer, 'node_modules/web-decision');
  for (const file of files) {
    const bytes = readFileSync(join(installedPackage, file));
    assert.ok(bytes.equals(readFileSync(join(pkg, file))), 'The tarball must preserve ' + file + '.');
    assert.deepEqual(findPackedContentIssues(bytes), [], 'Potential private content in packed file: ' + file);
  }
}

function listDistFiles(directory: string): string[] {
  return readdirSync(directory, {recursive: true, withFileTypes: true})
    .filter((entry) => !entry.isDirectory())
    .map((entry) => {
      assert.ok(entry.isFile(), `The build emitted a non-regular file: ${entry.name}.`);
      return relative(directory, join(entry.parentPath, entry.name)).split(sep).join('/');
    })
    .sort();
}

function readBundledArtifactLock(modelId: string): BundledArtifactLock {
  const lock = JSON.parse(readFileSync(join(pkg, 'src/catalog/artifacts.lock.json'), 'utf8')) as {
    byModelId: Readonly<Record<string, BundledArtifactLock>>;
  };
  const artifact = lock.byModelId[modelId];
  assert.ok(artifact, 'Missing artifact lock for ' + modelId + '.');
  return artifact;
}

function matchesAsset(path: string, metadata: ManifestFile): boolean {
  const bytes = readFileSync(path);
  return bytes.byteLength === metadata.bytes && createHash('sha256').update(bytes).digest('hex') === metadata.sha256;
}

function verifyCanonicalAssets({manifest, bundledRuntimeFiles}: BundledArtifactLock): string[] {
  const manifestBytes = readFileSync(join(pkg, manifest.sourcePath));
  assert.equal(createHash('sha256').update(manifestBytes).digest('hex'), manifest.rawSha256, manifest.sourcePath);
  for (const pin of bundledRuntimeFiles) {
    assert.ok(matchesAsset(join(pkg, pin.sourcePath), pin), 'Source asset differs from its lock: ' + pin.sourcePath);
  }
  return [manifest.sourcePath, ...bundledRuntimeFiles.map(({sourcePath}) => sourcePath)];
}

function verifyQwen35Artifacts(
  distDirectory: string,
  distFiles: readonly string[],
  {bundledRuntimeFiles}: BundledArtifactLock
): string[] {
  const pending = new Set(distFiles.filter((name) => name.startsWith('qwen35/')));
  const wasm: string[] = [];
  for (const pin of bundledRuntimeFiles) {
    // These two JSON imports are compiled into the model chunk by Vite.
    if (['pinned-word-assets-manifest.json', 'source-range-inventory.json'].includes(pin.manifestName)) continue;
    const matches = [...pending].filter(
      (name) => basename(name) === basename(pin.manifestName) && matchesAsset(join(distDirectory, name), pin)
    );
    assert.equal(matches.length, 1, 'Expected one exact bundled Qwen3.5 asset: ' + pin.manifestName);
    const [name] = matches as [string];
    pending.delete(name);
    if (name.endsWith('.wasm')) wasm.push('dist/' + name);
  }
  assert.deepEqual([...pending], [], 'The package emitted unexpected Qwen3.5 assets.');
  return wasm;
}

function verifyKananaArtifacts(distDirectory: string, distFiles: readonly string[]): void {
  const sourceDirectory = join(pkg, 'assets/kanana');
  const manifest = JSON.parse(readFileSync(join(sourceDirectory, 'manifest.json'), 'utf8')) as {
    modelId: string;
    files: Readonly<Record<string, ManifestFile>>;
  };
  assert.equal(manifest.modelId, kananaModelId);
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

function verifyQwen35LicenseNotice(): void {
  for (const [name, pin] of Object.entries(qwen35LicensePins)) {
    assert.equal(
      createHash('sha256')
        .update(readFileSync(join(pkg, name)))
        .digest('hex'),
      pin,
      name
    );
  }
  const notice = readFileSync(join(pkg, 'STRANDS-QWEN35-NOTICE'), 'utf8');
  for (const term of ['Strands Decider 2B Qwen3.5', 'Copyright 2026 Alibaba Cloud', 'web-decision adapted']) {
    assert.ok(notice.includes(term), 'The Qwen3.5 notice must preserve ' + term + '.');
  }
}

function verifyPackageInventory(
  packageFiles: readonly string[],
  distDirectory: string,
  distFiles: readonly string[],
  kananaApproved: boolean,
  qwen35?: BundledArtifactLock
): void {
  const assetFiles = [
    ...(kananaApproved && qwen35 ? verifyCanonicalAssets(readBundledArtifactLock(kananaModelId)) : []),
    ...(qwen35 ? verifyCanonicalAssets(qwen35) : [])
  ];
  const files = [
    ...packagedSourceFiles,
    ...(qwen35 ? qwen35PackageFiles : []),
    ...assetFiles,
    ...distFiles.map((name) => 'dist/' + name),
    'package.json'
  ];
  assert.deepEqual(packageFiles, files.sort());
  const wasm = [
    ...(kananaApproved ? kananaWasmFiles.map((name) => 'dist/' + name) : []),
    ...(qwen35 ? verifyQwen35Artifacts(distDirectory, distFiles, qwen35) : []),
    ...assetFiles.filter((name) => name.endsWith('.wasm'))
  ];
  assert.deepEqual(
    packageFiles.filter((file) => file.endsWith('.wasm')),
    wasm.sort()
  );
}

function selectArchive(packedPath: string, requestedPath?: string): string {
  if (!requestedPath) return packedPath;
  const archivePath = resolve(requestedPath);
  assert.ok(
    readFileSync(archivePath).equals(readFileSync(packedPath)),
    'Release archive differs from npm pack output.'
  );
  return archivePath;
}

function verifyBuiltChunks(
  distDirectory: string,
  distFiles: readonly string[],
  approvedModelIds: ReadonlySet<string>
): void {
  const chunks = distFiles
    .filter((name) => name.endsWith('.js'))
    .map((name) => ({name, text: readFileSync(join(distDirectory, name), 'utf8')}));
  const unverifiedWasmHash = candidateWasmHash();
  if (unverifiedWasmHash) {
    assert.ok(
      !inlineDataHashes(chunks).has(unverifiedWasmHash),
      'The unverified candidate WASM must not be embedded in the package.'
    );
  }
  for (const {name, text} of chunks) {
    if (!approvedModelIds.has(qwen35ModelId)) {
      assert.doesNotMatch(
        text,
        /strands-decider-2b-qwen3\.5-v1-2610|qwen35-(?:dynamic-vision|text-pointer)/u,
        'Pending Qwen3.5 IDs or graph assets leaked into dist/' + name + '.'
      );
    }
    if (!approvedModelIds.has(stableKevModelId) && !approvedModelIds.has(kananaModelId)) {
      assert.doesNotMatch(
        text,
        /kev-0\.6b-q4f16|kanana-2-v-decide-3b-2611/u,
        'Pending model IDs leaked into dist/' + name + '.'
      );
    }
    assert.doesNotMatch(
      text,
      /\b(?:from\s*|import\s*\(\s*)["'][^"']*\?url(?:&|["'])/u,
      'The published chunk dist/' + name + ' must not depend on consumer asset plugins.'
    );
  }
}

const root = fileURLToPath(new URL('../../', import.meta.url));
const pkg = join(root, 'packages/web-decision');
assert.ok(process.argv.length <= 3, 'Expected at most one release archive path.');
withTemporaryDirectories((makeTempDirectory: (prefix: string) => string) => {
  const npmConsumer = makeTempDirectory('web-decision-ci-npm-consumer-');
  const pnpmConsumer = makeTempDirectory('web-decision-ci-pnpm-consumer-');
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const pnpm = process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm';
  const metadata = JSON.parse(readFileSync(join(pkg, 'package.json'), 'utf8')) as PackageMetadata;
  assert.notEqual(metadata.private, true);
  assert.equal(metadata.license, 'Apache-2.0');
  const approvedModelIds = new Set<string>(APPROVED_MODEL_IDS);
  const kevApproved = approvedModelIds.has(stableKevModelId);
  const kananaApproved = approvedModelIds.has(kananaModelId);

  const output = runCli(npm, ['pack', '--ignore-scripts', '--json', '--pack-destination', npmConsumer], {
    cwd: pkg,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit']
  });
  const [archive] = JSON.parse(output as string) as PackArchive[];
  assert.ok(archive, 'npm pack did not return an archive.');
  const packageFiles = archive.files.map((file) => file.path).sort();
  const distDirectory = join(pkg, 'dist');
  const distFiles = listDistFiles(distDirectory);
  const qwen35 = approvedModelIds.has(qwen35ModelId) ? readBundledArtifactLock(qwen35ModelId) : undefined;
  if (kananaApproved) {
    verifyKananaArtifacts(distDirectory, distFiles);
    verifyKananaLicenseNotice();
  }
  if (qwen35) verifyQwen35LicenseNotice();
  verifyPackageInventory(packageFiles, distDirectory, distFiles, kananaApproved, qwen35);
  if (!approvedModelIds.has(qwen35ModelId)) {
    assert.ok(
      packageFiles.every((file) => !file.startsWith('dist/') || !/qwen35|strands-qwen/iu.test(file)),
      'Pending Qwen3.5 assets and candidate modules must not enter the npm archive.'
    );
  }
  const packageEntry = readFileSync(join(distDirectory, 'index.js'), 'utf8');
  let stableAssetChunks: BundleChunk[] = [];
  let expectedInlineHashes: Readonly<Record<string, string>> = {};
  let requiredKevFiles: string[] = [];
  let manifestHash = 'not-approved';
  if (kevApproved) {
    const stableManifest = JSON.parse(
      readFileSync(join(pkg, 'assets/kev06-stable/manifest.json'), 'utf8')
    ) as StableManifest;
    expectedInlineHashes = stableInlineHashes(stableManifest);
    requiredKevFiles = ['manifest.json', ...Object.keys(stableManifest.files)];
    const reachableChunks = collectChunkGraph(distDirectory, ['index.js']);
    stableAssetChunks = collectChunkGraph(distDirectory, dynamicChunkImports(packageEntry)).filter(({text}) =>
      text.includes('stableKevBundledUrls')
    );
    assert.ok(stableAssetChunks.length > 0, 'The package must emit a lazy stable Kev asset chunk.');
    manifestHash = verifyStableManifest(stableManifest, packageEntry, reachableChunks);
    verifyInlineAssets(stableAssetChunks, stableAssetInputs, 'stable Kev');
  }

  verifyBuiltChunks(distDirectory, distFiles, approvedModelIds);

  const packagePath = selectArchive(resolve(npmConsumer, archive.filename), process.argv[2]);
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
  assertConsumerBytes(npmConsumer, packageFiles);
  runInstalledApiSmoke(npmConsumer, requiredKevFiles, qwen35?.manifest);
  if (kevApproved)
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
  assertConsumerBytes(pnpmConsumer, packageFiles);
  runInstalledApiSmoke(pnpmConsumer, requiredKevFiles, qwen35?.manifest);
  if (kevApproved)
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
      '; approved model IDs: ' +
      JSON.stringify(APPROVED_MODEL_IDS) +
      '; stable Kev manifest SHA-256: ' +
      manifestHash +
      '; lazy asset chunk: ' +
      stableAssetChunks.map(({name}) => name).join(', ')
  );
  console.log('The archive installs, imports, and type-checks in clean npm and pnpm consumers.');
  console.log('This check imports the package in Node; it does not run browser inference.');
});
