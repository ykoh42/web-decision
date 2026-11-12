import {
  describeModel,
  kananaFloat,
  loadDecisionModel,
  type KananaAnswer,
  type KananaDecisionModel,
  type KananaDecisionRequest,
  type KananaDecisionResult,
  type KananaFloat,
  type KananaImageSource,
  type KananaQuestion,
  type LoadedDecisionModel,
  type ModelId
} from 'web-decision';

const modelId = 'kanana-2-v-decide-3b-2611' satisfies ModelId;
const description = describeModel(modelId);
const textChoiceStatus: 'implemented' = description.features.textChoice.status;
const imageChoiceStatus: 'implemented' | 'not-implemented' | undefined = description.features.imageChoice?.status;

const image: KananaImageSource = new Blob();
const imageChoiceRequest = {
  state: 'Choose the most prominent object.',
  images: [image],
  questions: {
    object: {
      type: 'choice',
      instructions: 'Which object is most prominent?',
      criteria: {rectangle: 'The gold rectangle.', circle: 'The teal circle.'}
    }
  }
} satisfies KananaDecisionRequest;

const typedFloatState: KananaDecisionRequest = {
  state: {count: 2, threshold: kananaFloat(2)},
  stateFormat: 'json',
  questions: {route: {type: 'choice', criteria: {yes: 'Yes', no: 'No'}}}
};
void typedFloatState;
// @ts-expect-error Only the branded helper can create a Kanana float.
const unbrandedFloat: KananaFloat = {value: 2};
void unbrandedFloat;

const messageImageRequest = {
  stateFormat: 'messages',
  state: [
    {
      role: 'user',
      content: [
        {type: 'text', text: 'What does this image show?'},
        {type: 'image_url', image_url: {url: new URL('https://example.test/image.png'), detail: 'auto'}}
      ]
    }
  ],
  questions: new Map<string, KananaQuestion>([
    [
      'object',
      {
        type: 'choice',
        criteria: new Map([
          ['cat', 'A cat.'],
          ['dog', 'A dog.']
        ])
      }
    ]
  ])
} satisfies KananaDecisionRequest;

const textQuestions = {
  action: {type: 'choice', criteria: {approve: 'Approve the request.', reject: 'Reject the request.'}},
  urgency: {type: 'noul', instructions: 'Is this urgent?'},
  quality: {type: 'score', criteria: ['poor', 'acceptable', 'excellent']}
} satisfies Readonly<Record<string, KananaQuestion>>;

async function checkKananaConsumerTypes(): Promise<void> {
  const model: KananaDecisionModel = await loadDecisionModel({modelId, cache: false});
  const textRequest = {state: 'A low-risk request.', questions: textQuestions} satisfies KananaDecisionRequest;
  const result: KananaDecisionResult = await model.systemOne(textRequest, {
    maxLength: 20480,
    truncateInputs: true,
    truncateQuestionCandidates: true,
    temperature: 1
  });
  const imageResult = await model.decide(imageChoiceRequest, {signal: new AbortController().signal});
  await model.systemOne(messageImageRequest);

  const answer: KananaAnswer | undefined = result.answers.action;
  const truncation = result.inputTruncation;
  if (truncation) {
    const maxLength: number = truncation.maxLength;
    const queryIds: readonly string[] = truncation.queryQuestionIds;
    const candidateKeys: readonly string[] | undefined = truncation.candidateDocuments[0]?.keys;
    void [maxLength, queryIds, candidateKeys];
  }
  if (answer?.type === 'choice') {
    const selected: string = answer.choice;
    const probability: number | undefined = answer.probabilities[selected];
    void probability;
  }
  const score = result.answers.quality;
  if (score?.type === 'score') {
    const value: number = score.score;
    const legendValue = score.legend['0'];
    void [value, legendValue];
  }
  const imageAnswer = imageResult.answers.object;
  if (imageAnswer?.type === 'choice') void imageAnswer.probabilityEntries;
  const noul = result.answers.urgency;
  if (noul?.type === 'noul') {
    const probability: number = noul.noul;
    void probability;
  }
  await model.dispose();
  void [textChoiceStatus, imageChoiceStatus];
}

void checkKananaConsumerTypes;

function checkLoadedModelType(model: LoadedDecisionModel): void {
  const kanana: KananaDecisionModel = model;
  void kanana.systemOne;
}
void checkLoadedModelType;

// @ts-expect-error Kanana-specific systemOne accepts one request object, not the legacy Kev signature.
void loadDecisionModel({modelId}).then((model) => model.systemOne('state', {urgent: {type: 'noul'}}));
// @ts-expect-error Kanana rejects the Kev-only model source option.
void loadDecisionModel({modelId, assetsUrl: 'https://example.test/'});
// @ts-expect-error maxInputTokens is a Kev load option; Kanana limits belong to per-request maxLength.
void loadDecisionModel({modelId, maxInputTokens: 128});
// @ts-expect-error maxInputCharacters is a Kev load option.
void loadDecisionModel({modelId, maxInputCharacters: 1024});
// @ts-expect-error maxPrefetchBytes is a Kev load option.
void loadDecisionModel({modelId, maxPrefetchBytes: 1024});
