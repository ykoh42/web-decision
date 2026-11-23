import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readFile, rm, symlink} from 'node:fs/promises';
import * as nodeModule from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {build} from 'vite';
import manifest from '../assets/strands-gemma4-v2/manifest.json' with {type: 'json'};

type Resolve = (specifier: string, context: Record<string, unknown>) => {url: string; [key: string]: unknown};
type RegisterHooks = (options: {
  resolve: (specifier: string, context: Record<string, unknown>, nextResolve: Resolve) => ReturnType<Resolve>;
}) => {deregister(): void};
const registerHooks = (nodeModule as unknown as {registerHooks?: RegisterHooks}).registerHooks;

test('the dormant packaged runtime binds the four emitted assets to their pinned bytes', async () => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const outDir = await mkdtemp(join(tmpdir(), 'web-decision-gemma-assets-'));
  try {
    await build({
      root,
      configFile: join(root, 'vite.config.ts'),
      logLevel: 'silent',
      build: {
        outDir,
        emptyOutDir: true,
        assetsInlineLimit: 0,
        lib: {entry: join(root, 'src/strands/gemma4-v2-packaged.mts'), formats: ['es'], fileName: () => 'index.mjs'}
      }
    });
    await symlink(join(root, 'node_modules'), join(outDir, 'node_modules'), 'dir');
    const {gemma4V2PackagedAssets} = (await import(pathToFileURL(join(outDir, 'index.mjs')).href)) as {
      gemma4V2PackagedAssets(): Record<string, {base: URL; name: string; metadata: {bytes: number; sha256: string}}>;
    };
    const assets = gemma4V2PackagedAssets();
    assert.deepEqual(Object.keys(assets).sort(), ['globalFrequency', 'localFrequency', 'scoring', 'wasm']);
    for (const {base, name, metadata} of Object.values(assets)) {
      const bytes = await readFile(fileURLToPath(new URL(name, base)));
      assert.equal(bytes.length, metadata.bytes);
      assert.equal(createHash('sha256').update(bytes).digest('hex'), metadata.sha256);
    }
  } finally {
    await rm(outDir, {recursive: true, force: true});
  }
});

test(
  'the source adapter uses the packaged Gemma pins and reports unavailable assets before creating a GPU runner',
  {skip: !registerHooks},
  async (t) => {
    const base = new URL('http://localhost/gemma4-v2/');
    const hooks = registerHooks!({
      resolve(specifier, context, nextResolve) {
        if (!specifier.includes('/assets/strands-gemma4-v2/') || !specifier.endsWith('?url&no-inline'))
          return nextResolve(specifier, context);
        const name = specifier.split('/').at(-1)!.split('?')[0]!;
        const source = 'export default ' + JSON.stringify(new URL(name, base).href);
        return {url: 'data:text/javascript,' + encodeURIComponent(source), shortCircuit: true};
      }
    });
    try {
      const packaged = await import('../src/strands/gemma4-v2-packaged.mts');
      const {loadModel} = await import('../src/curated/adapters/strands-gemma4-v2.ts');
      const assets = packaged.gemma4V2PackagedAssets();
      for (const [key, name] of [
        ['wasm', 'graph.wasm'],
        ['localFrequency', 'local.bin'],
        ['globalFrequency', 'global.bin'],
        ['scoring', 'scoring.bin']
      ] as const)
        assert.deepEqual(assets[key], {base, name, metadata: manifest.files[name]});

      const scoring = new Uint8Array(
        await readFile(new URL('../assets/strands-gemma4-v2/scoring.bin', import.meta.url))
      );
      const scoringUrl = new URL('scoring.bin', base).href;
      const fetch = t.mock.method(globalThis, 'fetch', async (input: string | URL) =>
        String(input) === scoringUrl ? new Response(scoring) : new Response(null, {status: 404})
      );
      const descriptor = {modelId: manifest.modelId};
      await assert.rejects(loadModel({offline: true, cache: false}, descriptor), {code: 'INVALID_INPUT'});
      assert.equal(fetch.mock.callCount(), 0);
      await assert.rejects(loadModel({cache: false}, descriptor), {
        code: 'ASSET_LOAD_FAILED',
        message: /HTTP 404/u
      });
      const requests = fetch.mock.calls.map(({arguments: [input]}) => String(input));
      const checkpoint = `https://huggingface.co/${manifest.repository}/resolve/${manifest.revision}/`;
      assert.ok(requests.includes(new URL('config.json', checkpoint).href));
      assert.ok(requests.includes(scoringUrl));
    } finally {
      hooks.deregister();
    }
  }
);
