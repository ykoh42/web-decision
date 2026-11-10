import assert from 'node:assert/strict';
import test from 'node:test';
import {prepareKananaVisionGraphInputs} from '../src/kanana/vision-inputs.js';

test('matches the pinned Qwen3-VL patch order and bilinear lookup for a merged image grid', () => {
  const inputs = prepareKananaVisionGraphInputs([1, 4, 6], {numPositionEmbeddings: 9});

  assert.deepEqual(inputs.positionShape, [24, 2]);
  assert.deepEqual(inputs.interpolationShape, [24, 4]);
  assert.deepEqual(
    inputs.positionIds,
    new Int32Array([
      0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3, 0, 4, 0, 5, 1, 4, 1, 5, 2, 0, 2, 1, 3, 0, 3, 1, 2, 2, 2, 3, 3, 2,
      3, 3, 2, 4, 2, 5, 3, 4, 3, 5
    ])
  );
  assert.deepEqual(
    inputs.interpolationIndices.slice(0, 24),
    new Int32Array([0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4, 1, 2, 4, 5])
  );
  assert.deepEqual(inputs.interpolationWeights.slice(0, 4), new Float32Array([1, 0, 0, 0]));
  assert.deepEqual(inputs.interpolationWeights.slice(-4), new Float32Array([1, 0, 0, 0]));
  assert.deepEqual(inputs.mergedTokens, new Int32Array(6));
});

test('handles singleton grids and rejects invalid patch geometry', () => {
  const singleton = prepareKananaVisionGraphInputs([1, 2, 2], {numPositionEmbeddings: 1});
  assert.deepEqual(singleton.positionIds, new Int32Array([0, 0, 0, 1, 1, 0, 1, 1]));
  assert.deepEqual(singleton.interpolationIndices, new Int32Array(16));
  assert.deepEqual(singleton.interpolationWeights, new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]));
  assert.throws(() => prepareKananaVisionGraphInputs([1, 3, 4], {numPositionEmbeddings: 9}), /divisible/);
  assert.throws(() => prepareKananaVisionGraphInputs([1, 4, 4], {numPositionEmbeddings: 8}), /square/);
});
