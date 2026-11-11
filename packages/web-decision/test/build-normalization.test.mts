import test from 'node:test';
import assert from 'node:assert/strict';
import config, {
  assertPinnedWebgpuHash,
  createTvmSourceMarkerPlugin,
  tvmRuntimeImportWithoutTypeMarker
} from '../vite.config.ts';
// These hooks are exercised with the minimal bundle and context the test needs.
type Chunk = {type: 'chunk'; code: string};
type Bundle = Record<string, Chunk>;
type GenerateBundle =
  | ((options: object, bundle: Bundle) => void)
  | {handler: (options: object, bundle: Bundle) => void};
type NormalizeHook = {generateBundle: GenerateBundle};
function generateBundle(plugin: NormalizeHook, bundle: Bundle) {
  const hook = plugin.generateBundle;
  if (typeof hook === 'function') hook({}, bundle);
  else hook.handler({}, bundle);
}
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
    generateBundle(plugin, bundle);
    return bundle.entry.code;
  };
  assert.equal(build(''), build('../../'));
  assert.equal(build('../../'), build('../../../../'));
  assert.match(build('../../'), /^\/\/ \[TVM\]\/runtime\.ts/m);
  assert.match(build('../../'), /const text = "\.\.\/\.toolchains/);
});

test('an empty approved catalog does not require TVM source markers', () => {
  assert.doesNotThrow(() => generateBundle(plugin, {entry: {type: 'chunk' as const, code: 'const value = 1;'}}));
});

test('build fails when the approved model bundle has no TVM markers', () => {
  const approvedPlugin = createTvmSourceMarkerPlugin(['kev-0.6b-q4f16']);
  assert.throws(
    () =>
      generateBundle(approvedPlugin as unknown as NormalizeHook, {
        entry: {type: 'chunk' as const, code: 'const value = 1;'}
      }),
    /Expected TVM source markers/
  );
});

test('the package build rejects a changed TVM WebGPU source', () => {
  assert.doesNotThrow(() => assertPinnedWebgpuHash('77003ab2bc6c7ae103b38fb52c12a5a7a3d49597260b43f010deaacc5d65b7c1'));
  assert.throws(
    () => assertPinnedWebgpuHash('b8f58d5abafc2d32d5462bdfbaa4b4b53abf0fd5b3c0a09dbafd9d8ff58502b7'),
    /differs from the pinned upstream revision/
  );
  assert.throws(
    () => assertPinnedWebgpuHash('92ba506841dc0eddedc8b1ea1ea64e37ff7dfce2d2f3fae297df6f9ec282bc5c'),
    /differs from the pinned upstream revision/
  );
});
