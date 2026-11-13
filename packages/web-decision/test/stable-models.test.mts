import assert from 'node:assert/strict';
import test from 'node:test';
import {
  describeStableModel,
  isStableModelId,
  listStableModelIds,
  normalizeStableModelOptions
} from '../src/stable-models.js';

test('only the pinned Kev model is exposed through the stable ID API', () => {
  assert.deepEqual(listStableModelIds(), ['kev-0.6b-q4f16']);
  assert.equal(Object.isFrozen(listStableModelIds()), true);
  assert.equal(isStableModelId('kev-0.6b-q4f16'), true);
  assert.equal(isStableModelId('experimental-strands-qwen-2b-fp16'), false);
  assert.equal(isStableModelId('unknown-model'), false);
});

test('stable descriptions expose only the public ID', () => {
  for (const modelId of listStableModelIds()) {
    const description = describeStableModel(modelId);
    assert.equal(description.model, modelId);
    assert.equal(description.model.startsWith('experimental-'), false);
    assert.equal(Object.isFrozen(description), true);
  }
  assert.throws(() => describeStableModel('experimental-strands-qwen-2b-fp16'), {code: 'INVALID_INPUT'});
});

test('stable options replace an undefined legacy selector with the pinned route', () => {
  const request = normalizeStableModelOptions({
    modelId: 'kev-0.6b-q4f16',
    model: undefined,
    assetsUrl: '/mirror/'
  });

  assert.equal(request.modelId, 'kev-0.6b-q4f16');
  assert.equal(request.options.model, 'kev-0.6b-q4f16');
  assert.equal(request.options.assetsUrl, '/mirror/');
  assert.equal(Object.hasOwn(request.options, 'modelId'), false);
  assert.throws(() => normalizeStableModelOptions({modelId: 'kev-0.6b-q4f16', model: 'other-model'}), {
    code: 'INVALID_INPUT'
  });
});
