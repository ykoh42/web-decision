import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {fileURLToPath, pathToFileURL} from 'node:url';

const runtime = new URL('../../../.toolchains/apache-tvm-src/web/src/runtime.ts', import.meta.url);
const compact = new URL('../../../.toolchains/apache-tvm-src/web/src/compact.ts', import.meta.url);
const hasPinnedSource = existsSync(fileURLToPath(runtime)) && existsSync(fileURLToPath(compact));

test('generated boundary preserves the pinned TVM module identities', {skip: !hasPinnedSource}, async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'web-decision-tvm-identity-'));
  context.after(() => rm(directory, {recursive: true, force: true}));
  const referenceModule = join(directory, 'tvm-reference.mjs');
  // Node 22.13's tsx loader pre-parses a dynamic file-URL import into a
  // separate CommonJS module; compare equivalent static ESM re-exports.
  await writeFile(
    referenceModule,
    `export {instantiate} from ${JSON.stringify(runtime.href)};\n` +
      `export {createPolyfillWASI} from ${JSON.stringify(compact.href)};\n`
  );

  const bridge = await import('../src/runtime/tvm.generated.mjs');
  const direct = await import(pathToFileURL(referenceModule).href);
  assert.equal(bridge.instantiate, direct.instantiate);
  assert.equal(bridge.createPolyfillWASI, direct.createPolyfillWASI);
});
