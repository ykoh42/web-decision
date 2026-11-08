import assert from 'node:assert/strict';
import test from 'node:test';
import {describeModel, isDecisionModel, listDecisionModels} from '../src/kev-preview-registry.js';

test('the release entry exposes only Kev 0.6B Q4F16', () => {
  assert.deepEqual(listDecisionModels(), ['kev-0.6b-q4f16']);
  assert.equal(Object.isFrozen(listDecisionModels()), true);
  assert.equal(isDecisionModel('kev-0.6b-q4f16'), true);
  assert.equal(isDecisionModel('experimental-strands-qwen-2b-fp16'), false);
  assert.equal(describeModel('kev-0.6b-q4f16').model, 'kev-0.6b-q4f16');
  assert.deepEqual(describeModel('kev-0.6b-q4f16').features.typedTextQuestionTypes, ['choice', 'noul', 'score']);
  assert.equal(Object.isFrozen(describeModel('kev-0.6b-q4f16').backend.requiredLimits), true);
  const limits = describeModel('kev-0.6b-q4f16').features.textChoice.limits;
  assert.deepEqual(
    limits.find((limit) => limit.resource === 'questions'),
    {
      resource: 'questions',
      scope: 'request',
      min: 1
    }
  );
  assert.equal(limits.find((limit) => limit.resource === 'options').max, 255);
  assert.equal(
    limits.some((limit) => limit.resource === 'characters'),
    false
  );
});

test('the release registry rejects model IDs outside its scope', () => {
  assert.throws(() => describeModel('experimental-tinydecide-10m-q4'), {code: 'INVALID_INPUT'});
});
