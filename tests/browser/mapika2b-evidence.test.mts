import assert from 'node:assert/strict';
import test from 'node:test';
import type {Mapika2bResult} from '../../packages/web-decision/src/mapika2b/readout.mts';
import {summarizeMapikaAnswerParity} from './mapika2b-evidence.mts';

test('answer parity reports changed scores and noul values', () => {
  const expected: Mapika2bResult = {
    model: 'decider-2b-v11',
    answers: {
      binary: {type: 'noul', noul: 0.25},
      rating: {type: 'score', score: 4.2, confidence: 0, x_p_max: 0, certainty: 0, probabilities: {}, legend: {}}
    },
    usage: {input_tokens: 1, output_tokens: 0}
  };
  const actual: Mapika2bResult = {
    ...expected,
    answers: {
      binary: {type: 'noul', noul: 0.26},
      rating: {type: 'score', score: 4.18, confidence: 0, x_p_max: 0, certainty: 0, probabilities: {}, legend: {}}
    }
  };
  assert.deepEqual(summarizeMapikaAnswerParity(actual, expected), {binary: false, rating: false});
  assert.deepEqual(summarizeMapikaAnswerParity(expected, expected), {binary: true, rating: true});
});
