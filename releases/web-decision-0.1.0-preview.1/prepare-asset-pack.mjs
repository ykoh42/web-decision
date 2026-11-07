import assert from 'node:assert/strict';
import {createReadStream} from 'node:fs';
import {copyFile, lstat, mkdir, mkdtemp, readFile, readdir, realpath, rename, rm, writeFile} from 'node:fs/promises';
import {basename, dirname, isAbsolute, join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {canonicalSha256, isPathWithin, sha256} from './release-integrity.mjs';

async function hashFile(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function safeAssetPath(root, name) {
  assert.equal(typeof name, 'string');
  assert.ok(name.length > 0 && !isAbsolute(name), `Asset path must be relative: ${name}`);
  const parts = name.split(/[\\/]/);
  assert.ok(
    parts.every((part) => part && part !== '.' && part !== '..'),
    `Unsafe asset path: ${name}`
  );
  const path = resolve(root, ...parts);
  assert.ok(isPathWithin(root, path) && path !== resolve(root), `Asset path escapes its root: ${name}`);
  return path;
}

async function assertCanonicalDirectory(path, label) {
  const absolute = resolve(path);
  const info = await lstat(absolute);
  assert.ok(info.isDirectory() && !info.isSymbolicLink(), `${label} must be a directory: ${absolute}`);
  return realpath(absolute);
}

async function assertRegularSource(path, label) {
  const absolute = resolve(path);
  const info = await lstat(absolute);
  assert.ok(info.isFile() && !info.isSymbolicLink(), `${label} must be a regular file: ${absolute}`);
  return realpath(absolute);
}

async function assertFilePin(path, expected, label, allowedRoot) {
  const source = await assertRegularSource(path, label);
  if (allowedRoot) {
    assert.ok(isPathWithin(allowedRoot, source), `${label} resolves outside its input directory.`);
  }
  const info = await lstat(source);
  assert.equal(info.size, expected.bytes, `${label} has an unexpected byte size.`);
  assert.equal(await hashFile(source), expected.sha256, `${label} does not match its SHA-256 pin.`);
  return source;
}

async function filesUnder(directory, prefix = '') {
  const files = [];
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const path = join(directory, entry.name);
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...(await filesUnder(path, name)));
    else if (entry.isFile()) files.push(name);
    else throw new Error(`Unexpected asset-pack entry: ${name}`);
  }
  return files.sort();
}

export async function prepareAssetPack({
  baselineDirectory,
  wasmPath,
  outputDirectory,
  manifest,
  manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`),
  expectedManifestSha256,
  expectedWasmSha256,
  expectedWasmBytes,
  expectedNotices = []
}) {
  const baseline = await assertCanonicalDirectory(baselineDirectory, 'Baseline asset directory');
  const wasm = await assertFilePin(
    wasmPath,
    {
      sha256: expectedWasmSha256,
      bytes: expectedWasmBytes
    },
    'q2-k128 WASM'
  );
  assert.equal(canonicalSha256(manifest), expectedManifestSha256, 'Asset manifest does not match its canonical pin.');
  assert.ok(manifest.files && typeof manifest.files === 'object' && !Array.isArray(manifest.files));
  assert.equal(manifest.files['model.wasm']?.sha256, expectedWasmSha256);
  assert.equal(manifest.files['model.wasm']?.bytes, expectedWasmBytes);

  const assets = Object.entries(manifest.files);
  assert.ok(assets.length > 0, 'Asset manifest contains no files.');
  const assetNames = new Set();
  for (const [name] of assets) {
    assert.ok(!assetNames.has(name), `Duplicate asset path: ${name}`);
    assert.ok(name !== 'manifest.json', 'The asset manifest cannot include itself as a file.');
    assetNames.add(name);
  }
  const sourcePaths = new Map();
  for (const [name, expected] of assets) {
    if (name === 'model.wasm') {
      sourcePaths.set(name, wasm);
    } else {
      const source = safeAssetPath(baseline, name);
      sourcePaths.set(name, await assertFilePin(source, expected, `Asset ${name}`, baseline));
    }
  }
  const noticesRoot = await assertCanonicalDirectory(join(baseline, 'notices'), 'Baseline notices directory');
  const noticeNames = expectedNotices.map((entry) => entry.path).sort();
  assert.equal(new Set(noticeNames).size, noticeNames.length, 'Duplicate notice path.');
  assert.deepEqual(await filesUnder(noticesRoot), noticeNames, 'Baseline notices differ from their pinned file set.');
  for (const notice of expectedNotices) {
    const source = safeAssetPath(noticesRoot, notice.path);
    sourcePaths.set(
      `notices/${notice.path}`,
      await assertFilePin(source, notice, `Notice ${notice.path}`, noticesRoot)
    );
  }

  const outputInput = resolve(outputDirectory);
  const outputParent = await assertCanonicalDirectory(dirname(outputInput), 'Output parent');
  const output = join(outputParent, basename(outputInput));
  assert.ok(
    isPathWithin(outputParent, output) && output !== outputParent,
    'Output must be a child of its parent directory.'
  );
  assert.ok(
    !isPathWithin(output, baseline) && !isPathWithin(baseline, output),
    'Output and baseline directories must not overlap.'
  );
  assert.ok(!isPathWithin(output, wasm) && !isPathWithin(wasm, output), 'Output and q2-k128 WASM must not overlap.');
  try {
    await lstat(output);
    throw new Error(`Refusing to overwrite existing asset output: ${output}`);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }

  const staging = await mkdtemp(join(outputParent, '.web-decision-asset-pack-'));
  try {
    for (const [name, expected] of assets) {
      const destination = safeAssetPath(staging, name);
      await mkdir(dirname(destination), {recursive: true});
      await copyFile(sourcePaths.get(name), destination);
      await assertFilePin(destination, expected, `Staged asset ${name}`);
    }
    for (const notice of expectedNotices) {
      const name = `notices/${notice.path}`;
      const destination = safeAssetPath(staging, name);
      await mkdir(dirname(destination), {recursive: true});
      await copyFile(sourcePaths.get(name), destination);
      await assertFilePin(destination, notice, `Staged notice ${notice.path}`);
    }
    const manifestPath = join(staging, 'manifest.json');
    await writeFile(manifestPath, manifestBytes, {flag: 'wx'});
    assert.equal(await hashFile(manifestPath), sha256(manifestBytes));
    assert.equal(canonicalSha256(JSON.parse(await readFile(manifestPath, 'utf8'))), expectedManifestSha256);
    const expectedFiles = [...assetNames, 'manifest.json', ...noticeNames.map((name) => `notices/${name}`)].sort();
    assert.deepEqual(await filesUnder(staging), expectedFiles, 'Staged pack contains unexpected files.');
    await rename(staging, output);
  } catch (error) {
    await rm(staging, {recursive: true, force: true});
    throw error;
  }

  return {
    output,
    fileCount: assets.length,
    totalBytes: assets.reduce((total, [, entry]) => total + entry.bytes, 0),
    noticeFileCount: expectedNotices.length,
    noticeBytes: expectedNotices.reduce((total, entry) => total + entry.bytes, 0),
    manifestCanonicalSha256: expectedManifestSha256,
    wasm: {bytes: expectedWasmBytes, sha256: expectedWasmSha256}
  };
}

function parseArguments(args) {
  if (args.length === 1 && args[0] === '--help') return null;
  const result = {};
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const value = args[index + 1];
    assert.ok(['--baseline-dir', '--wasm', '--output'].includes(name), `Unknown option: ${name}`);
    assert.ok(value && !value.startsWith('--'), `Missing value for ${name}`);
    assert.ok(!result[name], `Duplicate option: ${name}`);
    result[name] = value;
  }
  for (const name of ['--baseline-dir', '--wasm', '--output']) {
    assert.ok(result[name], `Required option is missing: ${name}`);
  }
  return result;
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args) {
    console.log('Usage: node prepare-asset-pack.mjs --baseline-dir DIR --wasm FILE --output NEW_DIR');
    return;
  }
  const releaseRoot = dirname(fileURLToPath(import.meta.url));
  const pins = JSON.parse(await readFile(join(releaseRoot, 'pins.json'), 'utf8'));
  const manifestBytes = await readFile(join(releaseRoot, pins.assets.manifestPath));
  const manifest = JSON.parse(manifestBytes);
  const result = await prepareAssetPack({
    baselineDirectory: args['--baseline-dir'],
    wasmPath: args['--wasm'],
    outputDirectory: args['--output'],
    manifest,
    manifestBytes,
    expectedManifestSha256: pins.assets.manifestCanonicalSha256,
    expectedWasmSha256: pins.assets.wasmSha256,
    expectedWasmBytes: pins.assets.wasmBytes,
    expectedNotices: pins.notices
  });
  console.log(JSON.stringify(result, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
