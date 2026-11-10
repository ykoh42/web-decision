import assert from 'node:assert/strict';
import test from 'node:test';
import {prepareKananaPositionInputs} from '../src/kanana/mrope.js';
import {KANANA_IMAGE_TOKEN_ID} from '../src/kanana/contract.mjs';

test('matches pinned Kanana M-RoPE positions for mixed text and two images', () => {
  const inputIds = [
    50,
    51,
    128017,
    ...Array<number>(6).fill(KANANA_IMAGE_TOKEN_ID),
    128018,
    52,
    128017,
    ...Array<number>(4).fill(KANANA_IMAGE_TOKEN_ID),
    128018,
    53
  ];
  const result = prepareKananaPositionInputs(inputIds, [
    [1, 2, 3],
    [1, 2, 2]
  ]);
  const sequenceLength = inputIds.length;
  const axes = Array.from({length: 4}, (_, axis) =>
    Array.from(result.positionIds.slice(axis * sequenceLength, (axis + 1) * sequenceLength))
  );

  assert.deepEqual(result.positionShape, [4, 1, 18]);
  assert.deepEqual(result.imagePositions, Int32Array.of(3, 4, 5, 6, 7, 8, 12, 13, 14, 15));
  assert.deepEqual(
    axes[0],
    Array.from({length: 18}, (_, index) => index)
  );
  assert.deepEqual(axes.slice(1), [
    [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 9, 9, 9, 9, 11, 12],
    [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 9, 9, 10, 10, 11, 12],
    [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12]
  ]);
});

test('uses four identical position axes for a text-only request', () => {
  const result = prepareKananaPositionInputs([7, 9, 11], []);

  assert.deepEqual(result.positionIds, new Int32Array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]));
  assert.deepEqual(result.imagePositions, new Int32Array());
});

test('rejects image-token counts or grid counts that do not match the request', () => {
  assert.throws(() => prepareKananaPositionInputs([KANANA_IMAGE_TOKEN_ID], [[1, 2, 2]]), /expected 4/);
  assert.throws(() => prepareKananaPositionInputs([1], [[1, 2, 2]]), /grid count/);
});
