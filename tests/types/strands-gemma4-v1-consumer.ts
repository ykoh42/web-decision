import {
  describeModel,
  loadDecisionModel,
  type StrandsGemma4V1DecisionModel,
  type StrandsGemma4V1ModelId
} from 'web-decision';

const modelId: StrandsGemma4V1ModelId = 'strands-decider-e2b-gemma4-v1-2610';
const info = describeModel(modelId);
const load = (): Promise<StrandsGemma4V1DecisionModel> =>
  loadDecisionModel({modelId, maxLength: 32768, maxBatch: 1, maxBatchTokens: 2048, usePrefixCache: false});
const answer = (model: StrandsGemma4V1DecisionModel) =>
  model.systemOne({
    state: 'An invoice needs correction.',
    questions: {route: {type: 'choice', instructions: 'Select the team', criteria: {billing: null, delivery: null}}}
  });
// @ts-expect-error The catalogue only accepts pinned asset locations.
const redirected = () => loadDecisionModel({modelId, assetBase: 'https://example.invalid/'});
export {answer, info, load, redirected};
