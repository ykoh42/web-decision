import {
  describeModel,
  isDecisionModel,
  listDecisionModels,
  loadDecisionModel,
  type LoadedKev06DecisionModel,
  type LoadedTamevNanoModel
} from 'web-decision';

const modelNames = listDecisionModels();
const hasNano = isDecisionModel('experimental-tamev-nano-tinybert-int8');
const nanoDescription = describeModel('experimental-tamev-nano-tinybert-int8');
const kev: Promise<LoadedKev06DecisionModel> = loadDecisionModel({assetsUrl: '/models/kev/'});
const nano: Promise<LoadedTamevNanoModel> = loadDecisionModel({
  model: 'experimental-tamev-nano-tinybert-int8',
  assetsUrl: '/models/tamev-nano/'
});

void modelNames;
void hasNano;
void nanoDescription.verification.currentBundleCpuWasm;
void kev.then((model) => model.systemOne('case', {route: {type: 'choice', criteria: {billing: null}}}));
void nano.then((model) =>
  model.systemOne('case', {
    route: {type: 'choice', criteria: {billing: null, fraud: 'review unauthorized use'}},
    urgent: {type: 'noul'},
    severity: {type: 'score', criteria: ['low', 'medium', 'high']}
  })
);
