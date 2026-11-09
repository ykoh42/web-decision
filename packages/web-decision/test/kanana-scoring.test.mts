import assert from 'node:assert/strict';
import test from 'node:test';
import {scoring} from './kanana-core-modules.mjs';

const {
  KANANA_DECISION_TEMPERATURE,
  KANANA_DEFAULT_TEMPERATURE,
  dotKananaVectors,
  poolKananaHiddenStates,
  scaleKananaLogits
} = scoring;

test('pools the final hidden token for unpadded rows and returns normalized float32 vectors', () => {
  const hiddenStates = new Float32Array([
    1, 0, 100, 100, 0, 2, 100, 100, 3, 4, 100, 100, 2, 0, 100, 100, 0, 3, 100, 100, 0, -5, 100, 100
  ]);
  const pooled = poolKananaHiddenStates(hiddenStates, new Uint8Array(6).fill(1), [2, 3, 4], 2);

  assert.deepEqual(pooled.shape, [2, 2]);
  assert.ok(pooled.values instanceof Float32Array);
  assert.deepEqual(pooled.values, new Float32Array([0.6, 0.8, 0, -1]));
});

test('pools the last attended token for mixed left- and right-padded rows', () => {
  const hiddenStates = new Float32Array([
    9, 9, 90, 90, 3, 4, 90, 90, 7, 8, 90, 90, 100, 100, 90, 90, 9, 9, 90, 90, 7, 8, 90, 90, 5, 6, 90, 90, 0, -5, 90, 90
  ]);
  const attentionMask = new Uint8Array([1, 1, 0, 0, 0, 0, 1, 1]);
  const pooled = poolKananaHiddenStates(hiddenStates, attentionMask, [2, 4, 4], 2);

  assert.deepEqual(pooled.shape, [2, 2]);
  assert.deepEqual(pooled.values, new Float32Array([0.6, 0.8, 0, -1]));
});

test('L2 normalization preserves zero vectors and clamps tiny norms with the pinned epsilon', () => {
  const hiddenStates = new Float32Array([0, 0, 8, 1e-13, 0, 8]);
  const pooled = poolKananaHiddenStates(hiddenStates, new Uint8Array([1, 1]), [2, 1, 3], 2);

  assert.deepEqual(pooled.values, new Float32Array([0, 0, 0.1, 0]));
});

test('dots normalized FP32 vectors without changing the vector dimension', () => {
  assert.equal(dotKananaVectors(new Float32Array([0.25, -0.5, 0.75]), new Float32Array([0.5, 0.25, -1.25])), -0.9375);
  assert.throws(() => dotKananaVectors([1, 2], [1]), RangeError);
  assert.throws(() => dotKananaVectors([], []), RangeError);
});

test('uses the pinned decision temperature and upstream request default', () => {
  const similarities = [0.123456789, -0.5, 0];

  assert.equal(KANANA_DECISION_TEMPERATURE, 0.056139365295102696);
  assert.equal(KANANA_DEFAULT_TEMPERATURE, 1.0);
  assert.deepEqual(scaleKananaLogits(similarities), [2.1991126537323, -8.90640640258789, 0]);
});

test('applies decision scale and request temperature in upstream operation order', () => {
  const similarities = [0.123456789, -0.7654321, 0.25];
  const decisionTemperature = 0.056139365295102696;
  const temperature = 0.7;

  assert.deepEqual(
    scaleKananaLogits(similarities, decisionTemperature, temperature),
    [3.141589641571045, -19.477855682373047, 6.361719131469727]
  );
});

test('preserves score order and does not clip finite cosine values', () => {
  const similarities = new Float32Array([1.0000001, -1.0000001, 0]);
  const scaled = scaleKananaLogits(similarities, 1, 1);

  assert.deepEqual(scaled, [1.0000001192092896, -1.0000001192092896, 0]);
});

test('accepts the inclusive request-temperature upper boundary', () => {
  assert.deepEqual(scaleKananaLogits([1], 1, 100), [Math.fround(0.01)]);
});

test('rejects invalid temperatures', () => {
  for (const decisionTemperature of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(() => scaleKananaLogits([0.5], decisionTemperature, 1), RangeError);
  }

  for (const temperature of [0, -1, 100.0001, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(() => scaleKananaLogits([0.5], 1, temperature), RangeError);
  }
});

test('rejects non-array inputs and non-finite similarities', () => {
  assert.throws(() => scaleKananaLogits(null), TypeError);
  assert.throws(() => scaleKananaLogits([0.1, Number.NaN]), TypeError);
});
