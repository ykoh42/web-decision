import assert from 'node:assert/strict';
import {test} from 'node:test';
import modelProvenance from '../src/catalog/model-provenance.json' with {type: 'json'};
import {APPROVED_MODEL_IDS} from '../src/catalog/public-catalog.generated.ts';
import * as library from '../src/curated/index.ts';
import type {ModelId} from '../src/catalog/types.generated.js';

test('exposes only the curated public entry points', () => {
  assert.deepEqual(Object.keys(library).sort(), [
    'KananaFloat',
    'WebDecisionError',
    'clearModelCache',
    'describeModel',
    'kananaFloat',
    'listModels',
    'loadDecisionModel'
  ]);
  assert.ok(library.kananaFloat(2) instanceof library.KananaFloat);
});

test('the public catalog lists only provenance-verified TVM models', () => {
  const models = library.listModels();
  assert.ok(Object.isFrozen(models));
  const approvedIds = modelProvenance.models.filter(({status}) => status === 'verified').map(({modelId}) => modelId);
  assert.deepEqual(APPROVED_MODEL_IDS, approvedIds);
  assert.deepEqual(
    models.map((entry) => entry.modelId),
    approvedIds
  );
  for (const model of models) assert.equal(library.describeModel(model.modelId), model);
  for (const {modelId, status} of modelProvenance.models) {
    if (status !== 'verified') assert.throws(() => library.describeModel(modelId as ModelId), {code: 'INVALID_INPUT'});
  }
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
    for (const {modelId, status} of modelProvenance.models) {
      if (status !== 'verified') {
        await assert.rejects(library.loadDecisionModel({modelId}), {code: 'INVALID_INPUT'});
      }
    }
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

test(
  'Kev-only input controls are rejected for approved Kanana before loading assets',
  {skip: !library.listModels().some(({modelId}) => modelId === 'kanana-2-v-decide-3b-2611')},
  async () => {
    const oldFetch = globalThis.fetch;
    let called = false;
    globalThis.fetch = () => {
      called = true;
      throw new Error('unexpected fetch');
    };
    try {
      for (const option of ['maxPrefetchBytes', 'maxInputTokens', 'maxInputCharacters']) {
        await assert.rejects(library.loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611', [option]: 128}), {
          code: 'INVALID_INPUT'
        });
      }
      assert.equal(called, false);
    } finally {
      globalThis.fetch = oldFetch;
    }
  }
);

test(
  'a cancelled Kanana load preserves its reason without requesting assets',
  {skip: !library.listModels().some(({modelId}) => modelId === 'kanana-2-v-decide-3b-2611')},
  async () => {
    const oldFetch = globalThis.fetch;
    const reason = new Error('load cancelled');
    const controller = new AbortController();
    controller.abort(reason);
    globalThis.fetch = () => {
      throw new Error('unexpected fetch');
    };
    try {
      await assert.rejects(
        library.loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611', signal: controller.signal}),
        (error) => {
          assert.equal(error, reason);
          return true;
        }
      );
    } finally {
      globalThis.fetch = oldFetch;
    }
  }
);
