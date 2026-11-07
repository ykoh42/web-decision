import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdir, readFile, readdir, writeFile} from 'node:fs/promises';
import {basename, dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {assertFreshOutputDirectory, sha256, verifySourceLock} from './release-integrity.mjs';

const RELEASE_ROOT = dirname(fileURLToPath(import.meta.url));
const WORKSPACE_ROOT = resolve(RELEASE_ROOT, '../..');
const PACKAGE_ROOT = join(RELEASE_ROOT, 'package');
const BUILD_ROOT = join(RELEASE_ROOT, 'build');
const DIST_ROOT = join(BUILD_ROOT, 'dist');
const RELEASE_OUTPUT = join(BUILD_ROOT, 'release');
const VERSION = '0.1.0';
const pins = JSON.parse(await readFile(join(RELEASE_ROOT, 'pins.json'), 'utf8'));
const sourceLock = await verifySourceLock({
  workspaceRoot: WORKSPACE_ROOT,
  lockPath: join(RELEASE_ROOT, 'source-lock.json')
});
assert.equal(pins.package.version, VERSION);

const safety = await import(
  pathToFileURL(join(WORKSPACE_ROOT, 'scripts/release/kev-current-source-preview-safety.mjs'))
);
const policy = await import(
  pathToFileURL(join(WORKSPACE_ROOT, 'scripts/release/kev-current-source-preview-package-policy.mjs'))
);
const NOTICE_FILES = [
  'EMSCRIPTEN-LICENSE',
  'KEV-NOTICE',
  'KEV06-NOTICE',
  'LICENSE',
  'ROLLDOWN-LICENSE',
  'ROLLDOWN-THIRD-PARTY-LICENSE',
  'TVM-LICENSE',
  'TVM-NOTICE',
  'VITE-LICENSE.md'
];
const EXPECTED_FILES = ['README.md', 'dist/index.js', 'index.d.ts', 'package.json', ...NOTICE_FILES].sort();

async function listFiles(directory, prefix = '') {
  const result = [];
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const path = join(directory, entry.name);
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) result.push(...(await listFiles(path, name)));
    else if (entry.isFile()) result.push(name);
    else throw new Error(`Unexpected package entry: ${name}`);
  }
  return result.sort();
}

function tarMember(archivePath, name) {
  return execFileSync('tar', ['-xOf', archivePath, `package/${name}`], {maxBuffer: 8_000_000});
}

const bundleBytes = await readFile(join(DIST_ROOT, 'index.js'));
const buildManifest = JSON.parse(await readFile(join(DIST_ROOT, 'current-source-manifest.json'), 'utf8'));
await safety.verifyBuildManifest({
  manifest: buildManifest,
  workspaceRoot: WORKSPACE_ROOT,
  bundleBytes,
  version: VERSION
});
assert.equal(sha256(bundleBytes), pins.package.bundleSha256, 'The built bundle does not match its pin.');
assert.ok(bundleBytes.toString('utf8').includes(pins.package.assetManifestCanonicalSha256));
assert.ok(sourceLock.files.length > 0);

const packageJson = JSON.parse(await readFile(join(PACKAGE_ROOT, 'package.json'), 'utf8'));
assert.equal(packageJson.name, pins.package.name);
assert.equal(packageJson.version, VERSION);
policy.assertKevQwen3PackagePolicy({
  noticeFiles: NOTICE_FILES,
  packageFiles: packageJson.files,
  routeNotice: await readFile(join(PACKAGE_ROOT, 'KEV06-NOTICE'), 'utf8'),
  packageReadme: await readFile(join(PACKAGE_ROOT, 'README.md'), 'utf8')
});

const packageDist = join(PACKAGE_ROOT, 'dist');
const outputInfo = await listFiles(PACKAGE_ROOT);
assert.ok(!outputInfo.includes('dist/index.js'), 'Refusing to overwrite the package bundle.');
await mkdir(packageDist, {recursive: false});
await writeFile(join(packageDist, 'index.js'), bundleBytes, {flag: 'wx'});
assert.deepEqual(await listFiles(PACKAGE_ROOT), EXPECTED_FILES);

execFileSync('pnpm', ['exec', 'publint', 'run', PACKAGE_ROOT, '--strict'], {
  cwd: WORKSPACE_ROOT,
  stdio: 'inherit'
});
execFileSync('pnpm', ['exec', 'attw', '--pack', '--profile', 'esm-only', '--no-definitely-typed', PACKAGE_ROOT], {
  cwd: WORKSPACE_ROOT,
  stdio: 'inherit'
});

await assertFreshOutputDirectory(BUILD_ROOT, RELEASE_OUTPUT, 'release archive output');
await mkdir(RELEASE_OUTPUT, {recursive: false});
const packed = JSON.parse(
  execFileSync('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', RELEASE_OUTPUT], {
    cwd: PACKAGE_ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit']
  })
);
assert.equal(packed.length, 1);
assert.deepEqual(packed[0].files.map((file) => file.path).sort(), EXPECTED_FILES);

const archivePath = join(RELEASE_OUTPUT, packed[0].filename);
const archiveBytes = await readFile(archivePath);
const packageFiles = {};
for (const name of EXPECTED_FILES) {
  const member = tarMember(archivePath, name);
  assert.deepEqual(member, await readFile(join(PACKAGE_ROOT, name)));
  packageFiles[name] = {bytes: member.length, sha256: sha256(member)};
}

const evidence = {
  state: 'code-only-package-verified',
  package: {name: packageJson.name, version: packageJson.version},
  archive: {
    file: basename(archivePath),
    path: relative(WORKSPACE_ROOT, archivePath),
    bytes: archiveBytes.length,
    sha256: sha256(archiveBytes),
    fileCount: EXPECTED_FILES.length
  },
  bundle: packageFiles['dist/index.js'],
  files: packageFiles,
  sourceFingerprint: buildManifest.sourceFingerprint,
  sourceInputCount: buildManifest.sourceInputs.length
};
await writeFile(join(BUILD_ROOT, 'package-verification.json'), `${JSON.stringify(evidence, null, 2)}\n`, {flag: 'wx'});
console.log(JSON.stringify(evidence, null, 2));
