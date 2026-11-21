import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import * as nodeModule from 'node:module';
import test from 'node:test';

type Resolve = (specifier: string, context: Record<string, unknown>) => {url: string; [key: string]: unknown};
type RegisterHooks = (options: {
  resolve: (specifier: string, context: Record<string, unknown>, nextResolve: Resolve) => ReturnType<Resolve>;
}) => {deregister(): void};
const registerHooks = (nodeModule as unknown as {registerHooks?: RegisterHooks}).registerHooks;

test(
  'packaged Mapika graph is verified and adapter options are checked before loading',
  {skip: !registerHooks},
  async () => {
    const hooks = registerHooks!({
      resolve(specifier, context, nextResolve) {
        if (!specifier.endsWith('?url&no-inline')) return nextResolve(specifier, context);
        const name = specifier.split('/').at(-1)!.split('?')[0];
        const source = 'export default ' + JSON.stringify('http://localhost/' + name);
        return {url: 'data:text/javascript,' + encodeURIComponent(source), shortCircuit: true};
      }
    });
    const originalFetch = globalThis.fetch;
    try {
      const packaged = await import('../src/mapika2b/packaged.mts');
      const adapter = await import('../src/curated/adapters/mapika2b.ts');
      const descriptor = {modelId: 'mapika-decider-2b-v12'} as Parameters<typeof adapter.loadModel>[1];
      await assert.rejects(adapter.loadModel({offline: true, cache: false}, descriptor), /offline requires/u);
      await assert.rejects(adapter.loadModel({cache: 'yes'}, descriptor), /cache must be a boolean/u);
      await assert.rejects(adapter.loadModel({}, {modelId: 'other'} as never), /different model descriptor/u);

      const graph = await readFile(new URL('../assets/mapika2b/text.wasm', import.meta.url));
      const bindings = await readFile(new URL('../assets/mapika2b/text-bindings.json', import.meta.url));
      let corrupt = false;
      globalThis.fetch = async (url) => {
        const name = new URL(String(url)).pathname.split('/').at(-1);
        const data = name === 'text.wasm' ? (corrupt ? Buffer.from('modified') : graph) : bindings;
        return new Response(data);
      };
      const artifacts = await packaged.openPackagedMapika2bText();
      assert.equal(artifacts.textWasm.byteLength, graph.byteLength);
      assert.equal(artifacts.textBindings.length, 319);
      const statuses: boolean[] = [];
      await packaged.openPackagedMapika2bText({
        cache: {
          async match() {
            return undefined;
          },
          async put() {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
          }
        },
        onCacheStatus: (cached) => statuses.push(cached)
      });
      assert.deepEqual(statuses, [false, false]);
      corrupt = true;
      await assert.rejects(packaged.openPackagedMapika2bText(), /integrity check failed/iu);
    } finally {
      globalThis.fetch = originalFetch;
      hooks.deregister();
    }
  }
);
