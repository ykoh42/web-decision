import test from 'node:test';
import assert from 'node:assert/strict';
import config, {
  assertPinnedWebgpuHash,
  browserOnlyTvmWasiExport,
  stripNodeOnlyWasiExport,
  tvmRuntimeImportWithoutTypeMarker
} from '../vite.config.ts';
// These hooks are exercised with the minimal bundle and context the test needs.
type Chunk = {type: 'chunk'; code: string};
type NormalizeHook = {generateBundle: (options: object, bundle: Record<string, Chunk>) => void};
type WasiHooks = {
  buildStart: () => void;
  transform: (source: string, id: string) => {code: string; map: null} | null;
  generateBundle: () => void;
};
const plugins = config.plugins ?? [];
const plugin = plugins.find(
  (candidate) =>
    candidate && !Array.isArray(candidate) && 'name' in candidate && candidate.name === 'normalize-tvm-source-markers'
) as unknown as NormalizeHook;
assert.ok(plugin);

test('TVM type boundary only rewrites the two pinned runtime imports', () => {
  const prefix = '../../../../.toolchains/apache-tvm-src/web/src/';
  for (const file of ['runtime.ts', 'compact.ts']) {
    assert.equal(tvmRuntimeImportWithoutTypeMarker(`${prefix}${file}?web-decision-runtime`), `${prefix}${file}`);
  }
  assert.equal(tvmRuntimeImportWithoutTypeMarker(`${prefix}other.ts?web-decision-runtime`), null);
  assert.equal(tvmRuntimeImportWithoutTypeMarker(`${prefix}runtime.ts`), null);
  assert.equal(tvmRuntimeImportWithoutTypeMarker('/other/runtime.ts?web-decision-runtime'), null);
});

test('TVM source marker normalization removes checkout-depth differences', () => {
  const build = (path: string): string => {
    const bundle = {
      entry: {
        type: 'chunk' as const,
        code: `// ${path}.toolchains/apache-tvm-src/web/src/runtime.ts\nconst text = "../.toolchains/apache-tvm-src/web/src/runtime.ts";\n`
      }
    };
    plugin.generateBundle({}, bundle);
    return bundle.entry.code;
  };
  assert.equal(build(''), build('../../'));
  assert.equal(build('../../'), build('../../../../'));
  assert.match(build('../../'), /^\/\/ \[TVM\]\/runtime\.ts/m);
  assert.match(build('../../'), /const text = "\.\.\/\.toolchains/);
});

test('build fails when the expected TVM marker format is missing', () => {
  assert.throws(
    () => plugin.generateBundle({}, {entry: {type: 'chunk' as const, code: 'const value = 1;'}}),
    /Expected TVM source markers/
  );
});

test('the package build rejects a changed TVM WebGPU source', () => {
  assert.doesNotThrow(() => assertPinnedWebgpuHash('01e67c6e6f0dd887bd12dac03619866456e2a9caf72e8c8a9dc249ed3bcf83e1'));
  assert.throws(
    () => assertPinnedWebgpuHash('b8f58d5abafc2d32d5462bdfbaa4b4b53abf0fd5b3c0a09dbafd9d8ff58502b7'),
    /differs from the verified patch set/
  );
  assert.throws(
    () => assertPinnedWebgpuHash('92ba506841dc0eddedc8b1ea1ea64e37ff7dfce2d2f3fae297df6f9ec282bc5c'),
    /differs from the verified patch set/
  );
});

test('browser-only WASI transform removes only the exact Node export', () => {
  const nodeExport = 'if(typeof module!="undefined"){module["exports"]=Module}';
  const source = `before();${nodeExport};after();`;
  assert.equal(stripNodeOnlyWasiExport(source), 'before();;after();');
  assert.throws(() => stripNodeOnlyWasiExport('before();after();'), /exactly one/);
  assert.throws(() => stripNodeOnlyWasiExport(source + nodeExport), /exactly one/);
});

test('browser-only WASI transform targets the TVM glue and fails if missing', () => {
  const plugin = browserOnlyTvmWasiExport() as unknown as WasiHooks;
  plugin.buildStart();
  assert.equal(plugin.transform('no edit', '/other/wasi.js'), null);
  assert.throws(() => plugin.generateBundle(), /Expected TVM WASI glue/);
  const source = 'if(typeof module!="undefined"){module["exports"]=Module}';
  assert.deepEqual(
    plugin.transform(source, 'C:\\work\\.toolchains\\apache-tvm-src\\web\\src\\tvmjs_runtime_wasi.js?test'),
    {code: '', map: null}
  );
  assert.doesNotThrow(() => plugin.generateBundle());
  plugin.buildStart();
  assert.throws(() => plugin.generateBundle(), /Expected TVM WASI glue/);
});
