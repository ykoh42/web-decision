import assert from 'node:assert/strict';
import test from 'node:test';
import {loadMapika2bCandidate} from '../src/mapika2b/loader.mts';

const artifacts = {textWasm: new Uint8Array(), textWasmSha256: '', textBindings: []};

test('Mapika rejects invalid configuration before requesting model assets', async () => {
  await assert.rejects(
    loadMapika2bCandidate({artifacts, temperature: 0}),
    (error: unknown) => error instanceof Error && /temperatures must be finite positive/u.test(error.message)
  );
  await assert.rejects(
    loadMapika2bCandidate({artifacts, limits: {maxRows: -1}}),
    (error: unknown) => error instanceof Error && /limit/u.test(error.message)
  );
});

test('Mapika preserves cancellation when loading is already aborted', async () => {
  const controller = new AbortController();
  const reason = new Error('cancelled');
  controller.abort(reason);
  await assert.rejects(loadMapika2bCandidate({artifacts, signal: controller.signal}), (error) => error === reason);
});
