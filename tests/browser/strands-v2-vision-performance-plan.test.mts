import assert from 'node:assert/strict';
import {test} from 'node:test';
import {compareVisionOutputs, visionPerformancePlan} from './strands-v2-vision-performance-plan.mts';

test('warms each configuration and alternates measured order', () => {
  assert.deepEqual(visionPerformancePlan(2), [
    {interval: 2, warmup: true},
    {interval: 8, warmup: true},
    {interval: 2, warmup: false},
    {interval: 8, warmup: false},
    {interval: 8, warmup: false},
    {interval: 2, warmup: false}
  ]);
  for (const value of [0, -1, 1.5, NaN]) assert.throws(() => visionPerformancePlan(value), RangeError);
});

test('reports numeric error and rejects incompatible or non-finite results', () => {
  assert.deepEqual(compareVisionOutputs(Float32Array.of(1, 2), Float32Array.of(1, 4)), {
    count: 2,
    maxAbsoluteError: 2,
    meanAbsoluteError: 1
  });
  assert.throws(() => compareVisionOutputs(Float32Array.of(1), Float32Array.of(1, 2)), /shapes differ/);
  assert.throws(() => compareVisionOutputs(Float32Array.of(NaN), Float32Array.of(1)), /non-finite/);
});
