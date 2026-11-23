import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {
  createGemma4V2ScoringKernel,
  finishGemma4V2Answer,
  GEMMA4_V2_SCORING_SHA256,
  gemma4V2Probabilities
} from '../src/strands/gemma4-v2-scoring.mjs';

const bytes = readFileSync(new URL('../assets/strands-gemma4-v2/scoring.bin', import.meta.url));
assert.equal(createHash('sha256').update(bytes).digest('hex'), GEMMA4_V2_SCORING_SHA256);
const kernel = createGemma4V2ScoringKernel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const cases = JSON.parse(
  readFileSync(new URL('./fixtures/strands-gemma4-v2-scoring-boundaries.json', import.meta.url), 'utf8')
) as Array<{
  name: string;
  kind: 'choice' | 'score';
  logits: number[];
  options: string[];
  smoothing: number;
  referenceProbabilities: number[];
  referenceAnswer: unknown;
}>;

test('fixed scorer reproduces the macOS NumPy reference at selection and rounding boundaries', () => {
  for (const row of cases) {
    const config = {temperature: 1.5456, softcap: 30, ordinalSmoothing: row.smoothing, kernel};
    assert.deepEqual(
      gemma4V2Probabilities(row.logits, row.logits.length, config),
      row.referenceProbabilities,
      row.name
    );
    assert.deepEqual(
      finishGemma4V2Answer(
        {kind: row.kind, options: row.options, descriptions: row.options.map((_, index) => `description ${index}`)},
        row.logits,
        config
      ),
      row.referenceAnswer,
      row.name
    );
  }
});

test('fixed scorer rejects nonfinite logits and mismatched checkpoint settings', () => {
  const config = {temperature: 1.5456, softcap: 30, kernel};
  assert.throws(() => gemma4V2Probabilities([0, NaN], 2, config));
  assert.throws(() => gemma4V2Probabilities([0, Infinity], 2, config));
  assert.throws(() => gemma4V2Probabilities([0, 1], 2, {...config, temperature: 2}));
  assert.throws(() => createGemma4V2ScoringKernel(new ArrayBuffer(8)));
});
