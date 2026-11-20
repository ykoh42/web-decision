import {
  clearModelCache,
  describeModel,
  KananaFloat,
  kananaFloat,
  listModels,
  loadDecisionModel,
  WebDecisionError,
  type ModelId,
  type StrandsQwen35Question,
  type StrandsQwen35SystemOneRequest,
  type StrandsQwen35Value
} from 'web-decision';
// @ts-expect-error The private candidate entry exposes Qwen model types only.
export type {KananaDecisionModel} from 'web-decision';

const id: ModelId = 'strands-decider-2b-qwen3.5-v1-2610';
const models = listModels();
if (models.length !== 1 || models[0]?.modelId !== id || !(kananaFloat(1) instanceof KananaFloat))
  throw new Error('Installed candidate exports differ from the typed consumer.');
const questions = new Map<string, StrandsQwen35Question>([
  [
    '10',
    {
      type: 'choice',
      instructions: {cutoff: kananaFloat(1)},
      criteria: new Map([
        ['10', null],
        ['2', null]
      ])
    }
  ],
  ['2', {type: 'score', instructions: ['Rate.'], criteria: ['low', 'high']}],
  ['inferred', {instructions: 'True?', criteria: {false: null, true: {cutoff: new KananaFloat(1)}}}]
]);
const state = new Map<string, StrandsQwen35Value>([
  ['10', [1, kananaFloat(1), 9007199254740993n, null, true]],
  ['2', {threshold: -0.5}]
]);
const request: StrandsQwen35SystemOneRequest = {state, questions, images: ['base64-data']};

// Compiled against the installed tarball. The fixture never invokes the model.
export async function typecheckCandidate() {
  const model = await loadDecisionModel({
    modelId: id,
    cache: true,
    offline: true,
    onProgress: (progress) => {
      const phase: 'upstream-source' | 'model-assets' | 'vision-weights' = progress.phase;
      void phase;
    }
  });
  const metadataId: ModelId = describeModel(id).modelId;
  const loadedId: ModelId = model.description.modelId;
  const ids: ModelId[] = listModels().map((entry) => entry.modelId);
  const result = await model.systemOne(request, {maxLength: 1024, maxBatchTokens: null, strictWindow: true});
  await model.systemOne({state: '', questions: {q: {instructions: 'True?'}}}, null);
  const tokens: number = result.usage.input_tokens + result.usage.output_tokens;
  for (const answer of Object.values(result.answers)) {
    if (answer.type === 'choice') {
      const winner: string = answer.choice;
      const probability: number | undefined = answer.probabilities[winner];
      void probability;
    } else if (answer.type === 'score') {
      const score: number = answer.score;
      const legend: Readonly<Record<string, string>> = answer.legend;
      void [score, legend];
    } else {
      const probability: number = answer.noul;
      void probability;
    }
  }
  // @ts-expect-error Candidate accepts the curated Qwen ID only.
  describeModel('kanana-2-v-decide-3b-2611');
  // @ts-expect-error Caller-selected asset URLs do not exist on this model.
  await loadDecisionModel({modelId: id, assetsUrl: '/mirror'});
  // @ts-expect-error Source Content requires a string, dictionary, or list at its root.
  await model.systemOne({state: kananaFloat(1), questions});
  // @ts-expect-error Bare numbers also fail the top-level Content schema.
  await model.systemOne({state: 1, questions});
  // @ts-expect-error The bigint path is for nested values.
  await model.systemOne({state: 1n, questions});
  // @ts-expect-error The schema requires instructions.
  await model.systemOne({state: '', questions: {q: {type: 'noul'}}});
  // @ts-expect-error Instructions require Content at their root.
  await model.systemOne({state: '', questions: {q: {instructions: kananaFloat(1)}}});
  await model.systemOne({
    state: '',
    // @ts-expect-error Choice descriptions accept Content or null.
    questions: {q: {type: 'choice', instructions: 'Pick.', criteria: {a: 1, b: null}}}
  });
  // @ts-expect-error Score levels must be strings.
  await model.systemOne({state: '', questions: {q: {type: 'score', instructions: 'Rate.', criteria: [1, 2]}}});
  // @ts-expect-error Ordered question keys are strings.
  await model.systemOne({state: '', questions: new Map([[1, {instructions: 'True?'}]])});
  // @ts-expect-error strictWindow takes a boolean.
  await model.systemOne(request, {strictWindow: 'yes'});
  await model.releaseCachedMemory();
  await model.dispose();
  const cleared: boolean = await clearModelCache();
  const error = new WebDecisionError('INVALID_INPUT', 'Example');
  return {metadataId, loadedId, ids, tokens, cleared, code: error.code};
}
