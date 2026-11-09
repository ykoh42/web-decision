import assert from 'node:assert/strict';
import test from 'node:test';
import {scoring} from './kanana-core-modules.mjs';

const {KANANA_DECISION_TEMPERATURE, KANANA_DEFAULT_TEMPERATURE, scaleKananaLogits} = scoring;

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
