/** Synthetic parity diagnostics; a recorded discrepancy is not a desired behavior. */
import {readFile} from 'node:fs/promises';
import {
  finishGemma4V2Answer,
  gemma4V2Probabilities
} from '../../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';

type Case = {
  name: string;
  logits: number[];
  numpyChoice: string;
  numpyProbabilities: number[];
  observedJsChoice: string;
  observedJsProbabilities: number[];
};
const fixture = JSON.parse(await readFile(new URL('./scoring-choice-boundary.json', import.meta.url), 'utf8')) as {
  options: string[];
  configuration: {softcap: number; temperature: number};
  cases: Case[];
};
for (const row of fixture.cases) {
  const question = {kind: 'choice' as const, options: fixture.options, descriptions: ['', '']};
  const probabilities = gemma4V2Probabilities(row.logits, 2, fixture.configuration);
  const answer = finishGemma4V2Answer(question, row.logits, fixture.configuration);
  if (answer.type !== 'choice') throw new Error('Expected a choice response.');
  const rawLogitChoice = fixture.options[row.logits.indexOf(Math.max(...row.logits))];
  console.log(
    JSON.stringify({
      case: row.name,
      probabilities,
      choice: answer.choice,
      rawLogitChoice,
      sourceChoice: row.numpyChoice,
      matchesCaptured:
        answer.choice === row.observedJsChoice &&
        probabilities.every((value, index) => value === row.observedJsProbabilities[index])
    })
  );
}
