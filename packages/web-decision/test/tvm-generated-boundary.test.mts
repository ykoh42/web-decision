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

test(
  'generated and TypeScript boundaries preserve their pinned TVM module identities',
  {skip: !hasPinnedSource},
  async (context) => {
    const directory = await mkdtemp(join(tmpdir(), 'web-decision-tvm-identity-'));
    context.after(() => rm(directory, {recursive: true, force: true}));
    const referenceModule = join(directory, 'tvm-reference.mjs');
    const markedRuntime = `${runtime.href}?web-decision-runtime`;
    const markedCompact = `${compact.href}?web-decision-runtime`;
    // Keep marker and markerless references separate because they are distinct
    // Node specifiers; the marker only keeps upstream sources out of strict tsc.
    // Node 22.13's tsx loader pre-parses a dynamic file-URL import into a
    // separate CommonJS module; compare equivalent static ESM re-exports.
    await writeFile(
      referenceModule,
      `export {instantiate} from ${JSON.stringify(runtime.href)};\n` +
        `export {createPolyfillWASI} from ${JSON.stringify(compact.href)};\n` +
        `export {instantiate as markedInstantiate} from ${JSON.stringify(markedRuntime)};\n` +
        `export {createPolyfillWASI as markedCreatePolyfillWASI} from ${JSON.stringify(markedCompact)};\n`
    );

    const bridge = await import('../src/runtime/tvm.generated.mjs');
    const sourceBoundary = await import('../src/runtime/tvm.ts');
    const direct = await import(pathToFileURL(referenceModule).href);
    assert.equal(bridge.instantiate, direct.instantiate);
    assert.equal(bridge.createPolyfillWASI, direct.createPolyfillWASI);
    assert.equal(sourceBoundary.instantiate, direct.markedInstantiate);
    assert.equal(sourceBoundary.createPolyfillWASI, direct.markedCreatePolyfillWASI);
    assert.deepEqual(Object.keys(sourceBoundary).sort(), ['createPolyfillWASI', 'instantiate']);
  }
);
