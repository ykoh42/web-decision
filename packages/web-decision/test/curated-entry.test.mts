import assert from 'node:assert/strict';
import {test} from 'node:test';
import * as library from '../src/curated/index.ts';

test('exposes only the curated public entry points', () => {
  assert.deepEqual(Object.keys(library).sort(), [
    'WebDecisionError',
    'clearModelCache',
    'describeModel',
    'listModels',
    'loadDecisionModel'
  ]);
});

test('the catalog only lists the pinned TVM model and rejects unknown IDs', () => {
  const models = library.listModels();
  assert.ok(Object.isFrozen(models));
  assert.deepEqual(
    models.map((entry) => entry.modelId),
    ['kev-0.6b-q4f16']
  );
  assert.equal(library.describeModel('kev-0.6b-q4f16'), models[0]);
  // @ts-expect-error Runtime validation must reject unavailable IDs too.
  assert.throws(() => library.describeModel('tamev-micro-minilm-int8'), {code: 'INVALID_INPUT'});
  // @ts-expect-error Prototype names are not model IDs.
  assert.throws(() => library.describeModel('__proto__'), {code: 'INVALID_INPUT'});
});

test('invalid or caller-selected sources fail before making a network request', async () => {
  const oldFetch = globalThis.fetch;
  let called = false;
  globalThis.fetch = () => {
    called = true;
    throw new Error('unexpected fetch');
  };
  try {
    for (const options of [
      {},
      {model: 'kev-0.6b-q4f16'},
      {modelId: 'tamev-micro-minilm-int8'},
      {modelId: 'kev-0.6b-q4f16', assetsUrl: 'https://example.org/'}
    ]) {
      await assert.rejects(library.loadDecisionModel(options), {code: 'INVALID_INPUT'});
    }
    assert.equal(called, false);
  } finally {
    globalThis.fetch = oldFetch;
  }
});
