// Independent repro: `pnpm exec tsx scripts/strands/gemma4_v2/check_scoring_rounding.mts`.
// The fixture was generated with pinned Strands v2 and NumPy on Apple Silicon.
import {readFile} from 'node:fs/promises';
import {finishGemma4V2Answer} from '../../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';

const rows = JSON.parse(await readFile(new URL('./scoring-rounding-reference.json', import.meta.url), 'utf8'));
let failures = 0;
for (const row of rows) {
  const actual = finishGemma4V2Answer(row.question, row.logits, {
    temperature: 1.5456,
    softcap: 30,
    ordinalSmoothing: row.smoothing
  });
  const differences: string[] = [];
  for (const key of ['score', 'confidence'] as const) {
    if (actual[key] !== row.answer[key]) differences.push(`${key}: ${actual[key]} vs ${row.answer[key]}`);
  }
  if ('probabilities' in actual) {
    for (const [key, value] of Object.entries(actual.probabilities)) {
      if (value !== row.answer.probabilities[key])
        differences.push(`probabilities[${key}]: ${value} vs ${row.answer.probabilities[key]}`);
    }
  }
  if (differences.length) {
    failures++;
    console.error(`Case ${row.index}: ${differences.join(', ')}`);
  }
}
if (failures) throw new Error(`${failures} rounding cases differ from the pinned reference.`);
console.log(`${rows.length} rounding cases match the pinned reference.`);
