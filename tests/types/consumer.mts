import {
  clearModelCache,
  describeModel,
  isDecisionModel,
  listDecisionModels,
  loadDecisionModel,
  WebDecisionError,
  type DecisionOptions,
  type DecisionRequest,
  type ErrorCode,
  type KevSystemOneQuestions,
  type LoadedKev06DecisionModel,
  type LoadedTamevMicroModel,
  type LoadedTamevNanoModel,
  type ModelDescriptionFor,
  type ModelName,
  type TamevNanoSystemOneQuestion
} from 'web-decision';

const assetsUrl = 'https://models.example.test/web-decision/';

async function checkConsumerTypes() {
  const modelNames: readonly ModelName[] = listDecisionModels();
  const candidate: unknown = 'experimental-tamev-micro-minilm-int8';
  const knownModel = isDecisionModel(candidate) ? candidate : undefined;
  const kevDescription: ModelDescriptionFor<'kev-0.6b-q4f16'> = describeModel('kev-0.6b-q4f16');
  const backend: 'tvm-webgpu' = kevDescription.backend.name;
  const request: DecisionRequest = {
    state: 'A customer reports a duplicate card charge.',
    questions: [{instructions: 'Which team should review this?', options: ['Billing', 'Fraud']}]
  };
  const options: DecisionOptions = {signal: new AbortController().signal};

  const kev: LoadedKev06DecisionModel = await loadDecisionModel({assetsUrl, model: 'kev-0.6b-q4f16'});
  const kevName: 'kev-0.6b-q4f16' = kev.description.model;
  const kevQuestions: KevSystemOneQuestions = {
    category: {type: 'choice', criteria: {billing: 'Billing', fraud: 'Fraud'}},
    urgent: {type: 'noul'},
    sentiment: {type: 'score', criteria: ['negative', 'neutral', 'positive']}
  };
  const decision = await kev.decide(request, options);
  const selected: string | undefined = decision.answers[0]?.selected;
  const legacy = await kev.systemOne({ticket: 'Duplicate charge.'}, kevQuestions, options);
  if (legacy.answers.category?.type === 'choice') {
    const confidence: number = legacy.answers.category.confidence;
    void confidence;
  }
  await kev.releaseCachedMemory();
  await kev.dispose();

  const nano: LoadedTamevNanoModel = await loadDecisionModel({
    assetsUrl,
    model: 'experimental-tamev-nano-tinybert-int8'
  });
  const tamevQuestions = {
    category: {type: 'choice', criteria: {billing: 'Billing', fraud: 'Fraud'}},
    urgent: {type: 'noul'},
    sentiment: {type: 'score', criteria: ['negative', 'neutral', 'positive']}
  } satisfies Readonly<Record<string, TamevNanoSystemOneQuestion>>;
  const nanoResult = await nano.systemOne({ticket: 'Duplicate charge.'}, tamevQuestions, options);
  const nanoName: 'experimental-tamev-nano-tinybert-int8' = nanoResult.model;
  await nano.decide(request, options);
  await nano.dispose();

  const micro: LoadedTamevMicroModel = await loadDecisionModel({
    assetsUrl,
    model: 'experimental-tamev-micro-minilm-int8'
  });
  const microResult = await micro.systemOne('Duplicate charge.', tamevQuestions, options);
  const microName: 'experimental-tamev-micro-minilm-int8' = microResult.model;
  await micro.dispose();

  const error: unknown = new WebDecisionError('INVALID_INPUT', 'Invalid request.', {cause: new Error('details')});
  if (error instanceof WebDecisionError) {
    const code: ErrorCode = error.code;
    void code;
  }
  const cacheCleared: boolean = await clearModelCache();

  // @ts-expect-error The stable package does not accept unreleased model identifiers.
  await loadDecisionModel({assetsUrl, model: 'kanana-2-v-decide-3b-2611'});
  // @ts-expect-error TAMEV load options reject arbitrary prefetch limits.
  await loadDecisionModel({assetsUrl, model: 'experimental-tamev-nano-tinybert-int8', maxPrefetchBytes: 1024});

  void [modelNames, knownModel, backend, kevName, selected, nanoName, microName, cacheCleared];
}

void checkConsumerTypes;
