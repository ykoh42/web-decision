import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readFile, readdir, realpath, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {build} from 'vite';
import type {AssetCache} from '../src/assets.js';
import artifactLock from '../src/catalog/artifacts.lock.json' with {type: 'json'};
import {QWEN35_R19_TEXT} from '../src/strands/qwen35-r19-text-contract.mts';
import {importBuiltModule} from './fixtures/import-built-module.mts';

const FILES = ['manifest.json', 'fixture.json', 'qwen35-text-pointer.webgpu.wasm', 'weight-bindings.json'] as const;
const lockedFiles = artifactLock.byModelId['strands-decider-2b-qwen3.5-v1-2610'].manifest.files;

function hash(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

test('opens the exact packed R19 graph and weight bindings from offline verified cache', async (t) => {
  const outDir = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-qwen35-r19-')));
  t.after(() => rm(outDir, {recursive: true, force: true}));

  await build({
    configFile: false,
    base: './',
    root: fileURLToPath(new URL('..', import.meta.url)),
    build: {
      outDir,
      emptyOutDir: true,
      lib: {
        entry: fileURLToPath(new URL('../src/strands/qwen35-r19-artifacts.mts', import.meta.url)),
        formats: ['es'],
        fileName: 'qwen35-r19-artifacts.js'
      },
      target: 'es2022',
      sourcemap: true
    }
  });

  const files = await readdir(outDir, {recursive: true});
  const jsName = files.find((name) => String(name).endsWith('.js') || String(name).endsWith('.mjs'));
  assert.ok(jsName);
  const jsPath = join(outDir, jsName);
  const js = await readFile(jsPath, 'utf8');
  assert.ok((await readFile(jsPath)).byteLength < 32_768);
  const cachedByHash = new Map<string, Uint8Array>();
  for (const name of FILES) {
    const metadata = lockedFiles[`r19/${name}`];
    assert.ok(files.includes(name));
    const bytes = new Uint8Array(await readFile(join(outDir, name)));
    assert.equal(bytes.byteLength, metadata.bytes);
    assert.equal(hash(bytes), metadata.sha256);
    assert.match(js, new RegExp(`new URL\\("${name.replaceAll('.', '\\.')}", import\\.meta\\.url\\)`));
    cachedByHash.set(metadata.sha256, bytes);
  }

  const cache: AssetCache = {
    async match(key) {
      const sha256 = new URL(key).searchParams.get('digest');
      const bytes = sha256 ? cachedByHash.get(sha256) : undefined;
      return bytes ? new Response(bytes.slice()) : null;
    },
    async put() {}
  };
  const module = (await importBuiltModule(jsPath)) as typeof import('../src/strands/qwen35-r19-artifacts.mts');
  const artifacts = await module.openPinnedQwen35R19Artifacts({cache, offline: true});

  assert.equal(artifacts.manifest.generation, 'R19');
  const names = [...QWEN35_R19_TEXT.inputNames, ...artifacts.bindings.map(({relaxArgumentName}) => relaxArgumentName)];
  assert.deepEqual(artifacts.manifest.abi.runtimeInputNames, QWEN35_R19_TEXT.inputNames);
  assert.equal(artifacts.manifest.abi.mainParameterCount, names.length);
  assert.equal(artifacts.manifest.abi.fullParameterNamesSha256, hash(new TextEncoder().encode(names.join('\n'))));
  assert.equal(artifacts.manifest.abi.externalWeightCount, 697);
  assert.equal(artifacts.bindings.length, 697);
  assert.equal(artifacts.bindings[0]?.relaxDtype, 'int32');
  assert.equal(artifacts.bindings[0]?.shape[0], 16);
  assert.equal(artifacts.bindings[0]?.relaxShape[0], 8);
});
