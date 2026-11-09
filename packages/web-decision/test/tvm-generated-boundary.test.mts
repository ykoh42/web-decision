import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';

const runtime = new URL('../../../.toolchains/apache-tvm-src/web/src/runtime.ts', import.meta.url);
const compact = new URL('../../../.toolchains/apache-tvm-src/web/src/compact.ts', import.meta.url);
const hasPinnedSource = existsSync(fileURLToPath(runtime)) && existsSync(fileURLToPath(compact));

test('generated boundary preserves the pinned TVM module identities', {skip: !hasPinnedSource}, async () => {
  const bridge = await import('../src/runtime/tvm.generated.mjs');
  const directRuntime = await import(runtime.href);
  const directCompact = await import(compact.href);
  assert.equal(bridge.instantiate, directRuntime.instantiate);
  assert.equal(bridge.createPolyfillWASI, directCompact.createPolyfillWASI);
});
