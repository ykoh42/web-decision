import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import * as nodeModule from 'node:module';
import test from 'node:test';
import manifest from '../assets/strands-gemma4-v1/manifest.json' with {type: 'json'};

type Resolve = (specifier: string, context: Record<string, unknown>) => {url: string; [key: string]: unknown};
type RegisterHooks = (options: {
  resolve: (specifier: string, context: Record<string, unknown>, nextResolve: Resolve) => ReturnType<Resolve>;
}) => {deregister(): void};
const registerHooks = (nodeModule as unknown as {registerHooks?: RegisterHooks}).registerHooks;

test(
  'the E2B v1 curated loader validates options and verifies its pinned asset responses',
  {skip: !registerHooks},
  async (t) => {
    const base = new URL('http://localhost/gemma4-v1/');
    const hooks = registerHooks!({
      resolve(specifier, context, nextResolve) {
        if (!specifier.includes('/assets/strands-gemma4-v1/') || !specifier.endsWith('?url&no-inline'))
          return nextResolve(specifier, context);
        const name = specifier.split('/').at(-1)!.split('?')[0]!;
        const source = 'export default ' + JSON.stringify(new URL(name, base).href);
        return {url: 'data:text/javascript,' + encodeURIComponent(source), shortCircuit: true};
      }
    });
    try {
      const {loadModel} = await import('../src/curated/adapters/strands-gemma4-v1.ts');
      const descriptor = {modelId: manifest.modelId};
      let corrupt = false;
      const bytes = new Uint8Array(
        await readFile(new URL('../assets/strands-gemma4-v1/tokenizer_config.json', import.meta.url))
      );
      bytes[0] = bytes[0]! ^ 1;
      const fetch = t.mock.method(globalThis, 'fetch', async (input: string | URL) => {
        if (!corrupt) return new Response(null, {status: 404});
        if (String(input) === new URL('tokenizer_config.json', base).href) return new Response(bytes);
        return new Promise<Response>(() => {});
      });
      await assert.rejects(loadModel({}, {modelId: 'different'}), {code: 'INVALID_ASSET'});
      for (const options of [
        {cache: 'yes'},
        {offline: 'yes'},
        {offline: true, cache: false},
        {onProgress: 'not a function'}
      ])
        await assert.rejects(loadModel(options, descriptor), {code: 'INVALID_INPUT'});
      assert.equal(fetch.mock.callCount(), 0);
      await assert.rejects(loadModel({cache: false}, descriptor), {code: 'ASSET_LOAD_FAILED', message: /HTTP 404/u});
      const expected = Object.keys(manifest.files)
        .map((name) => new URL(name, base).href)
        .sort();
      assert.deepEqual(fetch.mock.calls.map(({arguments: [input]}) => String(input)).sort(), expected);
      corrupt = true;
      await assert.rejects(loadModel({cache: false}, descriptor), {
        code: 'INVALID_ASSET',
        message: /Integrity check failed for tokenizer_config\.json/u
      });
      assert.deepEqual(
        fetch.mock.calls
          .slice(5)
          .map(({arguments: [input]}) => String(input))
          .sort(),
        expected
      );
    } finally {
      hooks.deregister();
    }
  }
);
