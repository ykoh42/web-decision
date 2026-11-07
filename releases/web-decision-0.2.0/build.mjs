import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, realpathSync} from 'node:fs';
import {mkdir, readFile, readdir} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {build} from 'vite';
import {
  assertFreshOutputDirectory,
  canonicalSha256,
  isPathWithin,
  sha256,
  verifySourceLock
} from './release-integrity.mjs';

const RELEASE_ROOT = dirname(fileURLToPath(import.meta.url));
const WORKSPACE_ROOT = resolve(RELEASE_ROOT, '../..');
const SOURCE_ROOT = join(RELEASE_ROOT, 'source/overlay/packages/web-decision/src');
const TVM_SOURCE_ROOT = join(RELEASE_ROOT, 'source/overlay/.toolchains/apache-tvm-src/web/src');
const PACKAGE_ROOT = join(RELEASE_ROOT, 'package');
const BUILD_PARENT = join(RELEASE_ROOT, 'build');
const BUILD_ROOT = join(RELEASE_ROOT, 'build/dist');
const sourceLockPath = join(RELEASE_ROOT, 'source-lock.json');
const pins = JSON.parse(await readFile(join(RELEASE_ROOT, 'pins.json'), 'utf8'));
const VERSION = pins.package.version;
const MANIFEST_SHA256 = pins.package.assetManifestCanonicalSha256;
const BUNDLE_SHA256 = pins.package.bundleSha256;
assert.equal(pins.package.version, VERSION);
assert.equal(VERSION, '0.2.0');
assert.equal(MANIFEST_SHA256, 'bc79c92a7011d2aa1aff4ede66c06cc365de81e744507256afb5c0449870716a');
if (BUNDLE_SHA256) assert.match(BUNDLE_SHA256, /^[0-9a-f]{64}$/u);
await verifySourceLock({workspaceRoot: WORKSPACE_ROOT, lockPath: sourceLockPath});

try {
  await mkdir(BUILD_PARENT);
} catch (error) {
  if (error.code !== 'EEXIST') throw error;
}
await assertFreshOutputDirectory(BUILD_PARENT, BUILD_ROOT, 'build output');

async function filesUnder(directory) {
  const result = [];
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) result.push(...(await filesUnder(path)));
    else if (entry.isFile()) result.push(path);
    else throw new Error(`Unexpected source entry: ${path}`);
  }
  return result;
}

const sourceModel = JSON.parse(await readFile(join(SOURCE_ROOT, 'kev-preview-model.json'), 'utf8'));
const sourceContract = await readFile(join(SOURCE_ROOT, 'kev06/contract.js'), 'utf8');
const sourceManifestTool = await readFile(join(RELEASE_ROOT, 'source/scripts/kev06_asset_manifest.py'), 'utf8');
const assetManifest = JSON.parse(await readFile(join(RELEASE_ROOT, pins.assets.manifestPath), 'utf8'));
assert.match(sourceContract, new RegExp(MANIFEST_SHA256));
assert.equal(sourceModel.assets.manifestSha256, MANIFEST_SHA256);
assert.match(sourceManifestTool, new RegExp(MANIFEST_SHA256));
assert.equal(canonicalSha256(assetManifest), MANIFEST_SHA256);
assert.equal(assetManifest.files[pins.assets.wasmPath].bytes, pins.assets.wasmBytes);
assert.equal(assetManifest.files[pins.assets.wasmPath].sha256, pins.assets.wasmSha256);

const safety = await import(
  pathToFileURL(join(WORKSPACE_ROOT, 'scripts/release/kev-current-source-preview-safety.mjs'))
);
const viteHelpers = await import(pathToFileURL(join(WORKSPACE_ROOT, 'packages/web-decision/vite.config.mjs')));
const tvmWebgpuPath = join(TVM_SOURCE_ROOT, 'webgpu.ts');
viteHelpers.assertPreviewWebgpuHash(sha256(await readFile(tvmWebgpuPath)));

const loadedSourceHashes = new Map();
function sourcePathFromModuleId(id) {
  if (id.startsWith('\0')) return null;
  let path = resolve(id.split('?')[0]);
  try {
    path = realpathSync(path);
  } catch {
    return null;
  }
  return isPathWithin(SOURCE_ROOT, path) || isPathWithin(TVM_SOURCE_ROOT, path) ? path : null;
}

const packageFiles = (await filesUnder(PACKAGE_ROOT)).filter((path) => !path.endsWith('/dist/index.js'));
const sourcePaths = [
  fileURLToPath(import.meta.url),
  join(RELEASE_ROOT, 'README.md'),
  join(RELEASE_ROOT, 'pins.json'),
  sourceLockPath,
  join(RELEASE_ROOT, 'source/scripts/kev06_asset_manifest.py'),
  join(RELEASE_ROOT, pins.assets.manifestPath),
  join(WORKSPACE_ROOT, 'package.json'),
  join(WORKSPACE_ROOT, 'pnpm-lock.yaml'),
  join(WORKSPACE_ROOT, 'pnpm-workspace.yaml'),
  join(WORKSPACE_ROOT, 'packages/web-decision/package.json'),
  join(WORKSPACE_ROOT, 'packages/web-decision/vite.config.mjs'),
  join(WORKSPACE_ROOT, 'scripts/release/kev-current-source-preview-safety.mjs'),
  join(WORKSPACE_ROOT, 'scripts/release/kev-current-source-preview-package-policy.mjs'),
  ...packageFiles
];

await build({
  configFile: false,
  root: RELEASE_ROOT,
  define: {__filename: 'undefined', __dirname: 'undefined'},
  plugins: [
    {
      name: 'capture-release-source-inputs',
      enforce: 'pre',
      transform(code, id) {
        const sourcePath = sourcePathFromModuleId(id);
        if (!sourcePath) return null;
        const sourceHash = createHash('sha256').update(Buffer.from(code)).digest('hex');
        const previousHash = loadedSourceHashes.get(sourcePath);
        if (previousHash && previousHash !== sourceHash) {
          throw new Error(`Rollup loaded changing source contents: ${sourcePath}`);
        }
        loadedSourceHashes.set(sourcePath, sourceHash);
        return null;
      }
    },
    viteHelpers.browserOnlyTvmWasiExport(),
    {
      name: 'verify-release-webgpu-source',
      buildStart() {
        viteHelpers.assertPreviewWebgpuHash(sha256(requireRead(tvmWebgpuPath)));
      }
    },
    {
      name: 'normalize-tvm-source-markers',
      generateBundle(_options, bundle) {
        for (const item of Object.values(bundle)) {
          if (item.type !== 'chunk') continue;
          const marker = /^\/\/ (?:.*\/)?\.toolchains\/apache-tvm-src\/web\/src\//gm;
          if (!marker.test(item.code)) throw new Error('Expected TVM source markers in the bundle');
          item.code = item.code.replace(marker, '// [TVM]/');
          const packageMarker = /^\/\/ (?:.*\/)?source\/overlay\/packages\/web-decision\/src\//gm;
          if (!packageMarker.test(item.code)) throw new Error('Expected package source markers in the bundle');
          item.code = item.code.replace(packageMarker, '// source/overlay/packages/web-decision/src/');
        }
      }
    },
    {
      name: 'record-release-build-manifest',
      generateBundle: {
        order: 'post',
        async handler(_options, bundle) {
          const chunks = Object.values(bundle).filter((item) => item.type === 'chunk');
          assert.equal(chunks.length, 1);
          assert.equal(chunks[0].fileName, 'index.js');
          const moduleSources = [...new Set([...this.getModuleIds()].map(sourcePathFromModuleId).filter(Boolean))];
          for (const sourcePath of moduleSources) {
            assert.ok(loadedSourceHashes.has(sourcePath), `Uncaptured source module: ${sourcePath}`);
          }
          const manifest = await safety.createBuildManifest({
            workspaceRoot: WORKSPACE_ROOT,
            sourcePaths: [...new Set([...moduleSources, ...sourcePaths])],
            bundleBytes: Buffer.from(chunks[0].code),
            loadedSourceHashes,
            version: VERSION
          });
          this.emitFile({
            type: 'asset',
            fileName: 'current-source-manifest.json',
            source: `${JSON.stringify(manifest, null, 2)}\n`
          });
          if (BUNDLE_SHA256)
            assert.equal(manifest.bundle.sha256, BUNDLE_SHA256, 'The package bundle does not match the release pin.');
        }
      }
    }
  ],
  resolve: {alias: [{find: 'ws', replacement: join(SOURCE_ROOT, 'browser-only-ws.js')}]},
  build: {
    outDir: BUILD_ROOT,
    emptyOutDir: false,
    lib: {
      entry: join(SOURCE_ROOT, 'kev-preview-entry.js'),
      formats: ['es'],
      fileName: () => 'index.js'
    },
    target: 'es2022',
    minify: false,
    rollupOptions: {external: ['@huggingface/tokenizers', 'onnxruntime-web/wasm']}
  }
});

function requireRead(path) {
  return readFileSync(path);
}
