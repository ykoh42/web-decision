import {loadDecisionModel} from './index.js';

const model = await loadDecisionModel({modelId: 'strands-decider-e2b-gemma4-v2-2611', maxLength: 65536});
const result = await model.systemOne({
  state: {message: 'example'},
  questions: new Map([
    ['choice', {type: 'choice', instructions: 'Pick', criteria: {red: 'red', blue: null}}],
    ['noul', {type: 'noul', instructions: 'Is it true?'}]
  ])
});
const answer = result.answers.choice;
if (answer?.type === 'choice') answer.probabilities.red?.toFixed(4);
await model.dispose();
// @ts-expect-error Asset locations are internal to the candidate.
await loadDecisionModel({modelId: 'strands-decider-e2b-gemma4-v2-2611', assetsUrl: 'https://example.test'});
