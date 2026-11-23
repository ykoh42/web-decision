import {readFile} from 'node:fs/promises';
import {
  finishGemma4V2Answer,
  gemma4V2Probabilities
} from '../../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';
const cases = JSON.parse(await readFile('../e2b-v2-scoring-fuzz.json', 'utf8'));
let mismatch = 0;
let maxError = 0;
for (let i = 0; i < cases.length; i++) {
  const row = cases[i];
  const config = {temperature: 1.5456, softcap: 30, ordinalSmoothing: row.smoothing};
  const probabilities = gemma4V2Probabilities(row.logits, row.logits.length, config);
  for (let j = 0; j < probabilities.length; j++)
    maxError = Math.max(maxError, Math.abs(probabilities[j]! - row.probabilities[j]));
  const answer = finishGemma4V2Answer(row.question, row.logits, config);
  if (JSON.stringify(answer) !== JSON.stringify(row.answer)) {
    if (mismatch < 5) console.log('mismatch', i, row.question.kind, JSON.stringify(answer), JSON.stringify(row.answer));
    mismatch++;
  }
}
console.log(JSON.stringify({count: cases.length, mismatch, maxError}));
