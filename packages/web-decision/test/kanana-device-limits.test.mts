import assert from 'node:assert/strict';
import test from 'node:test';
import {missingKananaWeightLimits} from '../src/kanana/device-limits.ts';

const largestPinnedWeightBytes = 656_670_720;

test('accepts WebGPU buffer limits that fit the largest pinned Kanana tensor', () => {
  assert.deepEqual(
    missingKananaWeightLimits(
      {
        maxBufferSize: largestPinnedWeightBytes,
        maxStorageBufferBindingSize: largestPinnedWeightBytes
      },
      largestPinnedWeightBytes
    ),
    []
  );
});

test('reports each WebGPU buffer limit below the largest pinned Kanana tensor', () => {
  assert.deepEqual(
    missingKananaWeightLimits(
      {
        maxBufferSize: largestPinnedWeightBytes - 1,
        maxStorageBufferBindingSize: largestPinnedWeightBytes
      },
      largestPinnedWeightBytes
    ),
    ['maxBufferSize']
  );
});

test('rejects invalid required buffer sizes', () => {
  assert.throws(() => missingKananaWeightLimits({maxBufferSize: 1, maxStorageBufferBindingSize: 1}, 0), {
    name: 'RangeError'
  });
});
