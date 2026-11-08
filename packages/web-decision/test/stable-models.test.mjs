import assert from 'node:assert/strict';
import test from 'node:test';
import {
  describeStableModel,
  isStableModelId,
  listStableModelIds,
  normalizeStableModelOptions
} from '../src/stable-models.js';

test('stable IDs use the published benchmark order and omit preview route names', () => {
  assert.deepEqual(listStableModelIds(), ['tamev-micro-minilm-int8', 'tamev-nano-tinybert-int8', 'kev-0.6b-q4f16']);
  assert.equal(Object.isFrozen(listStableModelIds()), true);
  assert.equal(isStableModelId('tamev-micro-minilm-int8'), true);
  assert.equal(isStableModelId('experimental-tamev-micro-minilm-int8'), false);
  assert.equal(isStableModelId('unknown-model'), false);
});

test('stable descriptions expose only the public ID', () => {
  for (const modelId of listStableModelIds()) {
    const description = describeStableModel(modelId);
    assert.equal(description.model, modelId);
    assert.equal(description.model.startsWith('experimental-'), false);
    assert.equal(Object.isFrozen(description), true);
  }
  assert.throws(() => describeStableModel('experimental-tamev-micro-minilm-int8'), {
    code: 'INVALID_INPUT'
  });
});

test('stable options replace an undefined legacy selector with the pinned route', () => {
  const request = normalizeStableModelOptions({
    modelId: 'tamev-micro-minilm-int8',
    model: undefined,
    assetsUrl: '/mirror/'
  });

  assert.equal(request.modelId, 'tamev-micro-minilm-int8');
  assert.equal(request.options.model, 'experimental-tamev-micro-minilm-int8');
  assert.equal(request.options.assetsUrl, '/mirror/');
  assert.equal(Object.hasOwn(request.options, 'modelId'), false);
  assert.throws(
    () =>
      normalizeStableModelOptions({modelId: 'tamev-micro-minilm-int8', model: 'experimental-tamev-micro-minilm-int8'}),
    {code: 'INVALID_INPUT'}
  );
});
