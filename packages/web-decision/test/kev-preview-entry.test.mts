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
  const requiredLimits = describeModel('kev-0.6b-q4f16').backend.requiredLimits;
  assert.equal(Object.isFrozen(requiredLimits), true);
  assert.deepEqual(requiredLimits, {
    maxComputeWorkgroupStorageSize: 16384,
    maxStorageBuffersPerShaderStage: 8,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 32,
    maxComputeWorkgroupSizeY: 8,
    maxComputeWorkgroupSizeZ: 1,
    maxComputeWorkgroupsPerDimension: 65535
  });
  assert.equal(Object.hasOwn(requiredLimits, 'maxBufferSize'), false);
  assert.equal(Object.hasOwn(requiredLimits, 'maxStorageBufferBindingSize'), false);
  const limits = describeModel('kev-0.6b-q4f16').features.textChoice.limits;
  assert.deepEqual(
    limits.find((limit: {resource: string}) => limit.resource === 'questions'),
    {
      resource: 'questions',
      scope: 'request',
      min: 1
    }
  );
  const optionLimit = limits.find((limit: {resource: string}) => limit.resource === 'options');
  assert.ok(optionLimit);
  assert.equal(optionLimit.max, 255);
  assert.equal(
    limits.some((limit: {resource: string}) => limit.resource === 'characters'),
    false
  );
});

test('the release registry rejects model IDs outside its scope', () => {
  assert.throws(() => describeModel('experimental-tinydecide-10m-q4'), {code: 'INVALID_INPUT'});
});
