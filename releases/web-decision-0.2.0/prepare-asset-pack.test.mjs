import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {canonicalSha256, sha256} from './release-integrity.mjs';
import {prepareAssetPack} from './prepare-asset-pack.mjs';

async function writePinnedFile(path, value) {
  const bytes = Buffer.from(value);
  await mkdir(path.slice(0, path.lastIndexOf('/')), {recursive: true});
  await writeFile(path, bytes);
  return {bytes: bytes.length, sha256: sha256(bytes)};
}

test('prepares an exact asset pack with the q2 WASM and leaves baseline WASM unused', async () => {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-assets-'));
  try {
    const baseline = join(root, 'baseline');
    const outputParent = join(root, 'outputs');
    const wasmPath = join(root, 'q2', 'model.wasm');
    const output = join(outputParent, 'pack');
    await mkdir(baseline);
    await mkdir(outputParent);
    await mkdir(join(baseline, 'notices'));
    const wasm = await writePinnedFile(wasmPath, 'q2-wasm');
    const source = await writePinnedFile(join(baseline, 'weights', 'params.bin'), 'baseline-weights');
    const notice = await writePinnedFile(join(baseline, 'notices', 'LICENSE.txt'), 'license-note');
    await writeFile(join(baseline, 'model.wasm'), 'wrong baseline wasm');
    const manifest = {
      formatVersion: 1,
      model: 'test-model',
      backend: 'tvm-webgpu',
      files: {'model.wasm': wasm, 'weights/params.bin': source}
    };
    const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
    const result = await prepareAssetPack({
      baselineDirectory: baseline,
      wasmPath,
      outputDirectory: output,
      manifest,
      manifestBytes,
      expectedManifestSha256: canonicalSha256(manifest),
      expectedWasmSha256: wasm.sha256,
      expectedWasmBytes: wasm.bytes,
      expectedNotices: [{path: 'LICENSE.txt', ...notice}]
    });
    assert.equal(result.fileCount, 2);
    assert.equal(result.noticeFileCount, 1);
    assert.deepEqual(
      (await readdir(output, {recursive: true})).sort(),
      ['manifest.json', 'model.wasm', 'notices', 'notices/LICENSE.txt', 'weights', 'weights/params.bin'].sort()
    );
    assert.equal(await readFile(join(output, 'model.wasm'), 'utf8'), 'q2-wasm');
    assert.deepEqual(await readFile(join(output, 'manifest.json')), manifestBytes);
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('rejects a WASM pin mismatch without creating output', async () => {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-assets-'));
  try {
    const baseline = join(root, 'baseline');
    const outputParent = join(root, 'outputs');
    const wasmPath = join(root, 'q2', 'model.wasm');
    const output = join(outputParent, 'pack');
    await mkdir(baseline);
    await mkdir(outputParent);
    const wasm = await writePinnedFile(wasmPath, 'q2-wasm');
    const source = await writePinnedFile(join(baseline, 'tokenizer.json'), 'tokenizer');
    const manifest = {
      formatVersion: 1,
      model: 'test-model',
      backend: 'tvm-webgpu',
      files: {'model.wasm': wasm, 'tokenizer.json': source}
    };
    await assert.rejects(
      prepareAssetPack({
        baselineDirectory: baseline,
        wasmPath,
        outputDirectory: output,
        manifest,
        expectedManifestSha256: canonicalSha256(manifest),
        expectedWasmSha256: '0'.repeat(64),
        expectedWasmBytes: wasm.bytes
      }),
      /SHA-256 pin/
    );
    await assert.rejects(readFile(output), {code: 'ENOENT'});
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('rejects traversal paths in the manifest', async () => {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-assets-'));
  try {
    const baseline = join(root, 'baseline');
    const outputParent = join(root, 'outputs');
    const wasmPath = join(root, 'q2', 'model.wasm');
    const output = join(outputParent, 'pack');
    await mkdir(baseline);
    await mkdir(outputParent);
    await mkdir(join(baseline, 'notices'));
    const wasm = await writePinnedFile(wasmPath, 'q2-wasm');
    const manifest = {
      formatVersion: 1,
      model: 'test-model',
      backend: 'tvm-webgpu',
      files: {'model.wasm': wasm, '../outside.bin': {bytes: 1, sha256: createHash('sha256').update('x').digest('hex')}}
    };
    await assert.rejects(
      prepareAssetPack({
        baselineDirectory: baseline,
        wasmPath,
        outputDirectory: output,
        manifest,
        expectedManifestSha256: canonicalSha256(manifest),
        expectedWasmSha256: wasm.sha256,
        expectedWasmBytes: wasm.bytes
      }),
      /Unsafe asset path/
    );
    await assert.rejects(readFile(output), {code: 'ENOENT'});
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('rejects baseline asset symlinks that resolve outside the input directory', async () => {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-assets-'));
  try {
    const baseline = join(root, 'baseline');
    const outputParent = join(root, 'outputs');
    const wasmPath = join(root, 'q2', 'model.wasm');
    const outside = join(root, 'outside');
    const output = join(outputParent, 'pack');
    await mkdir(baseline);
    await mkdir(outputParent);
    await mkdir(outside);
    await mkdir(join(baseline, 'notices'));
    const wasm = await writePinnedFile(wasmPath, 'q2-wasm');
    const data = await writePinnedFile(join(outside, 'params.bin'), 'outside data');
    await symlink(outside, join(baseline, 'weights'));
    const manifest = {
      formatVersion: 1,
      model: 'test-model',
      backend: 'tvm-webgpu',
      files: {'model.wasm': wasm, 'weights/params.bin': data}
    };
    await assert.rejects(
      prepareAssetPack({
        baselineDirectory: baseline,
        wasmPath,
        outputDirectory: output,
        manifest,
        expectedManifestSha256: canonicalSha256(manifest),
        expectedWasmSha256: wasm.sha256,
        expectedWasmBytes: wasm.bytes
      }),
      /outside its input directory/
    );
    await assert.rejects(readFile(output), {code: 'ENOENT'});
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});
