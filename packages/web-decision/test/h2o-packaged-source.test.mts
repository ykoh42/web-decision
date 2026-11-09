import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {mkdtemp, mkdir, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {Tokenizer} from '@huggingface/tokenizers';
import {pathFreeSourceDependencies as sourcePathFreeSourceDependencies} from '../src/h2o/loader.mjs';

const pkg = fileURLToPath(new URL('../', import.meta.url));
const model = new URL('./fixtures/h2o/', import.meta.url);
const inventoryDir = new URL('h2o-inventory-31d7b5f89e2c41a6/', model);
const inventoryPath = new URL('inventory.pending.json', inventoryDir);
const publicMetadata = new URL('./fixtures/h2o/h2o-public-metadata.pending.json', import.meta.url);
const originalStage = new URL('./fixtures/h2o/h2o-full-split-stage-20261030-v2.log', import.meta.url);
const assets = new Map([
  ['evidence/inventory.json', inventoryPath],
  ['h2o-public-metadata.json', publicMetadata],
  ['manifest.pending.json', new URL('h2o-run-c476df0e9ff43007/candidate/manifest.pending.json', model)],
  ['model.pending.wasm', new URL('full-split-webgpu-20261030-v2/model.pending.wasm', model)],
  ['tokenizer/tokenizer.json', new URL('checkpoint/tokenizer/tokenizer.json', model)],
  ['tokenizer/tokenizer_config.json', new URL('checkpoint/tokenizer/tokenizer_config.json', model)],
  ['tokenizer_contract.json', new URL('checkpoint/tokenizer_contract.json', model)],
  ['tensor-0415-retained.pending', new URL('h2o-run-c476df0e9ff43007/candidate/tensor-0415-retained.pending', model)]
]);
const requiredFixtures = [inventoryPath, publicMetadata, originalStage, ...assets.values()];
const fixtureSkip = requiredFixtures.every(existsSync)
  ? false
  : 'Local H2O model metadata and graph fixtures are not included.';

function assertNoAbsolutePaths(value: unknown): void {
  if (typeof value === 'string') assert.doesNotMatch(value, /^(?:\/|[A-Za-z]:\\)/);
  else if (value && typeof value === 'object') {
    for (const [key, item] of Object.entries(value)) {
      assertNoAbsolutePaths(key);
      assertNoAbsolutePaths(item);
    }
  }
}

test(
  'installed internal H2O source verifies pinned evidence and a bounded asset range',
  {skip: fixtureSkip},
  async () => {
    const temp = await mkdtemp(join(tmpdir(), 'web-decision-h2o-package-'));
    const originalFetch = globalThis.fetch;
    try {
      const output = JSON.parse(
        execFileSync('npm', ['pack', '--json', '--ignore-scripts', '--pack-destination', temp], {
          cwd: pkg,
          env: {...process.env, npm_config_cache: join(temp, 'npm-cache')},
          encoding: 'utf8'
        })
      );
      const installed = join(temp, 'node_modules', 'web-decision');
      await mkdir(installed, {recursive: true});
      execFileSync('tar', ['-xzf', join(temp, output[0].filename), '-C', installed, '--strip-components=1']);
      const manifest = JSON.parse(await readFile(join(installed, 'package.json'), 'utf8'));
      assert.deepEqual(Object.keys(manifest.exports), ['.']);
      const {openH2oAssetSource} = await import(pathToFileURL(join(installed, 'src/h2o/source-entry.mjs')).href);
      const inventory = new Uint8Array(await readFile(inventoryPath));
      const {prepareCandidatePlan, preparePublicCandidatePlan} = await import(
        pathToFileURL(join(installed, 'src/h2o/plan.mjs')).href
      );
      const manifestBytes = new Uint8Array(await readFile(assets.get('manifest.pending.json')!));
      const publicBytes = new Uint8Array(await readFile(publicMetadata));
      const oldPlan = await prepareCandidatePlan(new Uint8Array(await readFile(originalStage)), manifestBytes);
      assert.deepEqual(await preparePublicCandidatePlan(publicBytes, manifestBytes), oldPlan);
      const publicText = new TextDecoder().decode(publicBytes);
      assert.doesNotMatch(publicText, /(?:\/Users\/|\/home\/|[A-Za-z]:\\)/);
      assertNoAbsolutePaths(JSON.parse(publicText));
      assertNoAbsolutePaths(JSON.parse(new TextDecoder().decode(inventory)));
      let calls = 0,
        tamperWeight = false,
        tamperEvidence = false,
        tamperMetadata = false;
      globalThis.fetch = async (url, options) => {
        calls++;
        const pathname = new URL(url instanceof Request ? url.url : url).pathname;
        assert.ok(pathname.startsWith('/assets/'));
        const name = pathname.slice('/assets/'.length);
        const path = assets.get(name);
        assert.ok(path, `unexpected asset: ${name}`);
        const data = await readFile(path);
        const match = /^bytes=(\d+)-(\d+)$/.exec(new Headers(options?.headers).get('Range') ?? '');
        assert.ok(match);
        const start = Number(match[1]),
          end = Number(match[2]);
        const range = Uint8Array.from(data.subarray(start, end + 1));
        if (tamperWeight && name === 'tensor-0415-retained.pending') range[0] = range[0]! ^ 1;
        if (tamperEvidence && name === 'evidence/inventory.json') range[0] = range[0]! ^ 1;
        if (tamperMetadata && name === 'h2o-public-metadata.json') range[0] = range[0]! ^ 1;
        return new Response(range, {
          status: 206,
          headers: {
            'Content-Range': `bytes ${start}-${end}/${data.length}`,
            'Content-Length': String(range.length)
          }
        });
      };
      const changed = Uint8Array.from(inventory);
      changed[0] = changed[0]! ^ 1;
      await assert.rejects(
        openH2oAssetSource({assetsUrl: 'https://example.invalid/assets/', inventoryBytes: changed}),
        {
          code: 'INVALID_ASSET'
        }
      );
      assert.equal(calls, 0);
      tamperEvidence = true;
      await assert.rejects(openH2oAssetSource({assetsUrl: 'https://example.invalid/assets/'}), {code: 'INVALID_ASSET'});
      assert.equal(calls, 1);
      tamperEvidence = false;
      tamperMetadata = true;
      await assert.rejects(openH2oAssetSource({assetsUrl: 'https://example.invalid/assets/'}), {code: 'INVALID_ASSET'});
      tamperMetadata = false;
      const source = await openH2oAssetSource({assetsUrl: 'https://example.invalid/assets/'});
      const path = 'tensor-0415-retained.pending';
      const range = {path, totalBytes: 64, ...source.inventory[path][0]};
      const actual = await source.readRange('https://example.invalid/assets/', range);
      assert.deepEqual(new Uint8Array(actual.bytes), new Uint8Array(await readFile(assets.get(path)!)));
      await assert.rejects(source.readRange('https://example.invalid/assets/', {...range, sha256: '0'.repeat(64)}), {
        code: 'INVALID_ASSET'
      });
      tamperWeight = true;
      await assert.rejects(source.readRange('https://example.invalid/assets/', range), {code: 'INVALID_ASSET'});
      source.close();
      await assert.rejects(source.readRange('https://example.invalid/assets/', range), {code: 'INVALID_ASSET'});
      tamperWeight = false;

      // Exercise the real source, plan, tokenizer and session in the installed
      // loader. Only the device/runtime, upload and model runner are simulated.
      const loader = await import(pathToFileURL(join(installed, 'src/h2o/loader.mjs')).href);
      await assert.rejects(
        loader.loadH2oCandidate(
          {assetsUrl: 'https://example.invalid/assets/'},
          {runtimeBindings: {Tokenizer, instantiate: async () => ({}), createPolyfillWASI: () => ({})}}
        ),
        {code: 'UNSUPPORTED_ENVIRONMENT'}
      );
      const events: string[] = [];
      const fakeDevice = {
        lost: new Promise(() => {}),
        destroy() {
          events.push('destroy');
        }
      };
      const fakeGpu = {
        async requestAdapter() {
          return {
            features: new Set(['shader-f16']),
            limits: loader.PRELIMINARY_LIMITS,
            async requestDevice() {
              events.push('device');
              return fakeDevice;
            }
          };
        }
      };
      const runner = {
        async run(inputs: unknown[]) {
          events.push('run');
          assert.equal(inputs.length, 5);
          return new Float32Array(255);
        },
        async releaseCachedMemory() {
          events.push('release');
        },
        async dispose() {
          events.push('dispose');
        }
      };
      const realFactories = loader.createH2oBrowserFactories({
        Tokenizer,
        instantiate: async () => ({}),
        createPolyfillWASI: () => ({})
      });
      const installedPathFree = loader.pathFreeSourceDependencies();
      const sourcePathFree = sourcePathFreeSourceDependencies();
      assert.deepEqual(Object.keys(installedPathFree).sort(), Object.keys(sourcePathFree).sort());
      const deps = {
        ...realFactories,
        ...loader.pathFreeSourceDependencies(),
        fake: true,
        gpu: fakeGpu,
        beforeDevice: () => {
          events.push('guard');
        },
        initialize: () => ({}),
        async upload({
          plan,
          inventory,
          base,
          readRange,
          makeHasher
        }: {
          plan: {parameters: unknown[]};
          inventory: Record<string, {offset: number; length: number; sha256: string}[]>;
          base: string;
          readRange: (
            base: string,
            spec: {path: string; totalBytes: number; offset: number; length: number; sha256: string}
          ) => Promise<{bytes: ArrayBuffer}>;
          makeHasher: () => {update(bytes: Uint8Array): void; digest(): string};
        }) {
          assert.equal(plan.parameters.length, 682);
          const spec = {path, totalBytes: 64, ...inventory[path]![0]!};
          const bytes = new Uint8Array((await readRange(base, spec)).bytes);
          const hash = makeHasher();
          hash.update(bytes);
          assert.equal(hash.digest(), spec.sha256);
          events.push('upload');
          return [];
        },
        attach: () => runner
      };
      const session = await loader.loadFakeH2oCandidate(
        {model: loader.H2O_CANDIDATE_MODEL_ID, assetsUrl: 'https://example.invalid/assets', cache: false},
        deps
      );
      const result = await session.decide({
        state: 'record',
        questions: [{instructions: 'Choose', options: ['first', 'second']}]
      });
      assert.equal(result.answers.length, 1);
      await session.releaseCachedMemory();
      await session.dispose();
      assert.deepEqual(events, ['guard', 'device', 'upload', 'run', 'release', 'dispose']);
    } finally {
      globalThis.fetch = originalFetch;
      await rm(temp, {recursive: true, force: true});
    }
  }
);
