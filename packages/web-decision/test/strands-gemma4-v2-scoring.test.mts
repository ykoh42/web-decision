import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {finishGemma4V2Answer, gemma4V2Probabilities} from '../src/strands/gemma4-v2-scoring.mjs';

type Sample = {
  question: {kind: 'choice' | 'score' | 'noul'; options: string[]; descriptions: string[]};
  logits: number[];
  probabilities: number[];
  answer: unknown;
};

test('v2 probabilities and typed answers match the pinned official readout', async () => {
  const samples = JSON.parse(
    await readFile(new URL('./fixtures/strands-gemma4-v2-scoring.json', import.meta.url), 'utf8')
  ) as Sample[];
  const config = {temperature: 1.5456, softcap: 30};
  for (const sample of samples) {
    const probabilities = gemma4V2Probabilities(sample.logits, sample.question.options.length, config);
    for (let i = 0; i < probabilities.length; i++)
      assert.ok(Math.abs(probabilities[i]! - sample.probabilities[i]!) < 0.000002);
    assert.deepEqual(finishGemma4V2Answer(sample.question, sample.logits, config), sample.answer);
  }
});

test('rejects invalid output, option counts, or calibration values', () => {
  const config = {temperature: 1.5456, softcap: 30};
  assert.throws(() => gemma4V2Probabilities([1, NaN], 2, config));
  assert.throws(() => gemma4V2Probabilities([1, 2], 1, config));
  assert.throws(() => gemma4V2Probabilities([1, 2], 2, {...config, temperature: 0}));
  assert.throws(() =>
    finishGemma4V2Answer({kind: 'noul', options: ['yes', 'no'], descriptions: ['', '']}, [0, 1], config)
  );
});
