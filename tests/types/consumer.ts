import {
  WebDecisionError,
  clearModelCache,
  describeModel,
  isDecisionModel,
  listDecisionModels,
  loadDecisionModel,
  type DecisionModel,
  type DecisionRequest,
  type DecisionResult,
  type ErrorCode,
  type LoadOptions,
  type LoadedDecisionModel,
  type LoadedDebertaDecisionModel,
  type LoadedKev06DecisionModel,
  type LoadedKevUpstreamDecisionModel,
  type LoadedKevReflexesDecisionModel,
  type LoadedKev08DecisionModel,
  type LoadedMapikaDecisionModel,
  type MapikaSystemOneResult,
  type DebertaSystemOneResult,
  type KevSystemOneResult,
  type KevSystemOneQuestion,
  type KevSystemOneJsonValue,
  type StrandsGemma4SystemOneQuestion,
  type Kev08LoadOptions,
  type KevUpstreamLoadOptions,
  type ModelName,
  type StandardLoadOptions,
  type OpenDeciderNanoLoadOptions,
  type OpenDeciderNanoSystemOneResult,
  type SafeDecideLoadOptions,
  type SafeDecideSystemOneResult,
  type SafeDecideDecisionResult,
  type DecimaSystemOneResult,
  type LoadedDecimaAgentModel,
  type DecimaAgentSystemOneResult,
  type LoadedTamevMicroModel,
  type LoadedTamevNanoModel,
  type LoadedTamevNanoFlashModel,
  type LoadedSkejjaInt8Model,
  type LoadedSkejjaInt8DynModel,
  type SkejjaInt8LoadOptions,
  type SkejjaInt8DynLoadOptions,
  type SkejjaSystemOneResult,
  type SkejjaQuestionCalibration,
  type EttinLoadOptions,
  type LoadedEttinModel,
  type LoadedLayaModel,
  type LayaLoadOptions,
  type LayaSystemOneResult,
  type LoadedLayaTypedDecisionsModel,
  type LayaTypedDecisionsLoadOptions,
  type LayaTypedDecisionsSystemOneResult,
  type XDecisionLoadOptions,
  type LoadedXDecisionModel,
  type XDecisionSystemOneResult,
  type LoadedGlinerMobileModel,
  type GlinerMobileLoadOptions,
  type GlinerMobileSystemOneResult,
  type D1OmniLoadOptions,
  type LoadedD1OmniModel,
  type D1OmniSystemOneResult,
  type VerdictLoadOptions,
  type LoadedVerdictModel,
  type VerdictSystemOneResult,
  type VerdictDecisionResult,
  type EttinSystemOneResult,
  type EttinSystemOneQuestion,
  type TamevMicroSystemOneResult,
  type TamevNanoSystemOneResult,
  type TamevNanoFlashSystemOneResult,
  type LoadedTinyDecideModel,
  type TinyDecideResult,
  type TinyDecideDecisionOptions,
  type TinyDecideLoadOptions,
  type TamevMicroLoadOptions,
  type TamevNanoLoadOptions,
  type TamevNanoFlashLoadOptions,
  type CrivoLoadOptions,
  type LoadedCrivoModel,
  type CrivoSystemOneResult
} from 'web-decision';
import type {LoadedKev06DecisionModel as KevHybridPreviewModel} from './kev-deberta-preview.js';

const request: DecisionRequest = {
  state: 'A billing ticket needs a team.',
  questions: [{instructions: 'Choose a team.', options: ['billing', 'delivery']}]
};
const options: StandardLoadOptions = {
  assetsUrl: '/models/kev-0.6b-q4f16/',
  model: 'kev-0.6b-q4f16',
  onProgress(progress) {
    const fraction: number = progress.fraction;
    void fraction;
  }
};
const kevLoadOptionsWithCaps: LoadOptions = {
  assetsUrl: '/models/kev-0.6b-q4f16/',
  maxInputTokens: 4096,
  maxInputCharacters: 65_536
};
const kevLoadOptionsWithoutCaps: LoadOptions = {assetsUrl: '/models/kev-0.6b-q4f16/'};
const kevReflexesLoadOptionsWithCaps: LoadOptions = {
  assetsUrl: '/models/kev-0.6b-reflexes-q4f16/',
  model: 'kev-0.6b-reflexes-q4f16',
  maxInputTokens: 4096,
  maxInputCharacters: 65_536
};
const kevUpstreamOptions: KevUpstreamLoadOptions = {
  model: 'experimental-kev-0.6b-q4f16-upstream',
  maxPrefetchBytes: 0,
  onProgress(progress) {
    const phase: 'bundled-assets' | 'upstream-source' | 'asset-verification' | undefined = progress.phase;
    void phase;
  }
};
const kevUpstream: Promise<LoadedKevUpstreamDecisionModel> = loadDecisionModel(kevUpstreamOptions);
void kevUpstream.then((model) => {
  const result: Promise<KevSystemOneResult<'experimental-kev-0.6b-q4f16-upstream'>> = model.systemOne(
    'A billing ticket.',
    {
      route: {type: 'choice', instructions: 'Choose a team.', criteria: {billing: null, delivery: null}}
    }
  );
  void result;
});
// @ts-expect-error The experimental source provider does not accept a separate asset directory.
loadDecisionModel({model: 'experimental-kev-0.6b-q4f16-upstream', assetsUrl: '/models/kev-upstream/'});
// @ts-expect-error The experimental source provider disables shard read-ahead to constrain extra memory.
loadDecisionModel({model: 'experimental-kev-0.6b-q4f16-upstream', maxPrefetchBytes: 1});
void [kevLoadOptionsWithCaps, kevLoadOptionsWithoutCaps, kevReflexesLoadOptionsWithCaps, kevUpstream];
const crivoOptions: CrivoLoadOptions = {
  model: 'experimental-crivo-32m-v2-int8',
  assetsUrl: '/models/crivo-32m-v2/',
  cache: false
};
const crivoModel: Promise<LoadedCrivoModel> = loadDecisionModel(crivoOptions);
void loadDecisionModel({model: 'experimental-crivo-32m-v2-int8'});
void loadDecisionModel({model: 'experimental-crivo-32m-v2-fp32'});
void crivoModel.then((model) => {
  const result: Promise<CrivoSystemOneResult> = model.systemOne(
    {ticket: {message: 'A customer was charged twice.'}},
    {
      route: {type: 'choice', instructions: 'Choose a team.', criteria: {billing: 'refund', support: null}},
      urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'high']},
      refund: {type: 'noul', instructions: 'Was a refund requested?'}
    }
  );
  void result;
});
// @ts-expect-error Crivo requires one of its two pinned graph variants.
loadDecisionModel({model: 'experimental-crivo-32m-v3', assetsUrl: '/models/crivo/'});
const mapikaOptions: LoadOptions = {assetsUrl: '/models/mapika-0.8b-package-f16-v2/', model: 'mapika-0.8b-f16'};
const mapikaQ4Options: LoadOptions = {assetsUrl: '/models/mapika-0.8b-q4f16/', model: 'mapika-0.8b-q4f16'};
const mapikaF16Typed = loadDecisionModel({assetsUrl: '/models/mapika-0.8b-f16/', model: 'mapika-0.8b-f16'});
const mapikaQ4Typed = loadDecisionModel({assetsUrl: '/models/mapika-0.8b-q4f16/', model: 'mapika-0.8b-q4f16'});
const mapikaConsumer: LoadedMapikaDecisionModel<'mapika-0.8b-q4f16'> = await mapikaQ4Typed;
const mapikaResult: Promise<MapikaSystemOneResult<'mapika-0.8b-q4f16'>> = mapikaConsumer.systemOne(
  {ticket: {message: 'refund requested'}, count: 2},
  {
    route: {type: 'choice', instructions: 'Which team?', criteria: {billing: 'refund', support: null}},
    urgency: {type: 'score', instructions: 'How urgent?', criteria: ['low', 'high']},
    refund: {type: 'noul', instructions: 'Is a refund requested?', options: {true: 'asks for money back'}}
  }
);
void [mapikaF16Typed, mapikaResult];
const kev08Options: LoadOptions = {assetsUrl: '/models/kev-0.8b-package-f16-v1/', model: 'kev-0.8b-f16'};
const strandsOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-strands-qwen-2b-fp16'};
const gemma4ChoiceDescriptions: StrandsGemma4SystemOneQuestion = {
  type: 'choice',
  instructions: 'Pick one.',
  criteria: {
    text: 'Plain text description',
    omitted: null,
    structured: {enabled: true, count: 2},
    listed: [false, 3]
  }
};
const gemma4NoulDescriptions: StrandsGemma4SystemOneQuestion = {
  type: 'noul',
  instructions: 'Is it true?',
  criteria: {true: {evidence: ['yes', 1]}, false: null}
};
const gemma4NumericChoiceDescription: StrandsGemma4SystemOneQuestion = {
  type: 'choice',
  instructions: 'Pick one.',
  // @ts-expect-error Pinned Strands descriptions do not allow a bare number.
  criteria: {
    first: null,
    second: 1
  }
};
const gemma4BooleanChoiceDescription: StrandsGemma4SystemOneQuestion = {
  type: 'choice',
  instructions: 'Pick one.',
  // @ts-expect-error Pinned Strands descriptions do not allow a bare boolean.
  criteria: {first: null, second: false}
};
const gemma4NumericNoulDescription: StrandsGemma4SystemOneQuestion = {
  type: 'noul',
  instructions: 'Is it true?',
  // @ts-expect-error Pinned Strands descriptions do not allow a bare number.
  criteria: {true: 1}
};
const gemma4BooleanNoulDescription: StrandsGemma4SystemOneQuestion = {
  type: 'noul',
  instructions: 'Is it true?',
  // @ts-expect-error Pinned Strands descriptions do not allow a bare boolean.
  criteria: {
    true: true
  }
};
void [
  gemma4ChoiceDescriptions,
  gemma4NoulDescriptions,
  gemma4NumericChoiceDescription,
  gemma4BooleanChoiceDescription,
  gemma4NumericNoulDescription,
  gemma4BooleanNoulDescription
];
const openDeciderOptions: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-opendecider-nano-fp16',
  cache: false,
  onProgress(progress) {
    const fraction: number = progress.fraction;
    void fraction;
  }
};
const safeDecideOptions: SafeDecideLoadOptions = {
  model: 'experimental-safedecide-140m-fp32',
  assetsUrl: '/safedecide-140m/',
  cache: false,
  onProgress(progress) {
    const totalBytes: number = progress.totalBytes;
    void totalBytes;
  }
};
const safeDecideModel = loadDecisionModel(safeDecideOptions);
void loadDecisionModel({model: 'experimental-safedecide-140m-fp32'});
void safeDecideModel.then((model) => {
  const typed: Promise<SafeDecideSystemOneResult> = model.systemOne(
    {ticket: {message: 'A customer was charged twice.'}},
    {
      route: {type: 'choice', instructions: 'Choose a team.', criteria: {billing: 'refunds', support: null}},
      urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'high']},
      fraud: {type: 'noul', instructions: 'Is fraud suspected?'}
    },
    {alpha: 0.2}
  );
  const generic: Promise<SafeDecideDecisionResult> = model.decide(request, {alpha: 0.05});
  void [typed, generic];
});
// @ts-expect-error SafeDecide does not expose prefetch tuning.
loadDecisionModel({assetsUrl: '/', model: 'experimental-safedecide-140m-fp32', maxPrefetchBytes: 1});
const decimaOptions: LoadOptions = {
  assetsUrl: '/decima-small/',
  model: 'experimental-decima-small-int8',
  cache: true,
  offline: true,
  onProgress(progress) {
    const fraction: number = progress.fraction;
    void fraction;
  }
};
loadDecisionModel({model: 'experimental-ettin-68m-int8'});
loadDecisionModel({model: 'experimental-decima-small-int8'});
loadDecisionModel({model: 'experimental-decima-base-clean-int8'});
loadDecisionModel({model: 'experimental-decima-agent-int8'});
const decimaTypedOptions = loadDecisionModel({
  assetsUrl: '/decima-small/',
  model: 'experimental-decima-small-int8',
  maxPrefetchBytes: 0
});
void decimaTypedOptions.then((model) =>
  model.decide(
    {state: 'ticket', questions: [{instructions: 'Choose a team.', options: ['billing', 'security']}]},
    {language: 'fa'}
  )
);
const decimaBaseCleanOptions: LoadOptions = {
  assetsUrl: '/decima-base-clean/',
  model: 'experimental-decima-base-clean-int8',
  cache: true,
  offline: true
};
const decimaBaseCleanTypedOptions = loadDecisionModel({
  assetsUrl: '/decima-base-clean/',
  model: 'experimental-decima-base-clean-int8',
  maxPrefetchBytes: 0
});
void decimaBaseCleanTypedOptions.then((model) =>
  model.systemOne('ticket', {route: {type: 'choice', criteria: ['billing', 'security']}})
);
const decimaAgentOptions: LoadOptions = {
  assetsUrl: '/decima-agent/',
  model: 'experimental-decima-agent-int8',
  cache: true,
  offline: true
};
const tamevOptions: TamevMicroLoadOptions = {
  assetsUrl: '/tamev-micro/',
  model: 'experimental-tamev-micro-minilm-int8',
  cache: true,
  offline: false
};
const tamevMicroDefaultOptions: TamevMicroLoadOptions = {
  model: 'experimental-tamev-micro-minilm-int8'
};
const tamevNanoDefaultOptions: TamevNanoLoadOptions = {
  model: 'experimental-tamev-nano-tinybert-int8'
};
const tamevMicroDefault = loadDecisionModel(tamevMicroDefaultOptions);
const tamevNanoDefault = loadDecisionModel(tamevNanoDefaultOptions);
loadDecisionModel({model: 'experimental-tamev-nano-flash-tinybert-int8'});
void [tamevMicroDefault, tamevNanoDefault];
const tamevTypedOptions = loadDecisionModel(tamevOptions);
void tamevTypedOptions.then((model) => {
  const tamev: LoadedTamevMicroModel = model;
  const result: Promise<TamevMicroSystemOneResult> = tamev.systemOne(
    {ticket: {detail: 'duplicate card charge'}},
    {
      route: {
        type: 'choice',
        instructions: 'Choose a queue',
        criteria: {billing: 'review duplicate charges', fraud: null}
      },
      urgent: {type: 'noul', instructions: 'Is the customer stranded?'},
      severity: {type: 'score', instructions: 'Rate the impact', criteria: ['low', 'high']}
    }
  );
  void result;
});
const tamevNanoOptions: TamevNanoLoadOptions = {
  assetsUrl: '/tamev-nano/',
  model: 'experimental-tamev-nano-tinybert-int8',
  cache: true
};
const tamevNanoTypedOptions = loadDecisionModel(tamevNanoOptions);
void tamevNanoTypedOptions.then((model) => {
  const tamev: LoadedTamevNanoModel = model;
  const result: Promise<TamevNanoSystemOneResult> = tamev.systemOne('state', {
    route: {type: 'choice', criteria: {billing: 'billing', fraud: 'fraud'}}
  });
  void result;
});
// @ts-expect-error TAMEV Nano does not implement prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-tamev-nano-tinybert-int8', maxPrefetchBytes: 1});
const tamevNanoFlashOptions: TamevNanoFlashLoadOptions = {
  assetsUrl: '/tamev-nano-flash/',
  model: 'experimental-tamev-nano-flash-tinybert-int8',
  cache: true
};
const tamevNanoFlashTypedOptions = loadDecisionModel(tamevNanoFlashOptions);
void tamevNanoFlashTypedOptions.then((model) => {
  const tamev: LoadedTamevNanoFlashModel = model;
  const result: Promise<TamevNanoFlashSystemOneResult> = tamev.systemOne('state', {
    route: {type: 'choice', criteria: {billing: 'billing', fraud: 'fraud'}}
  });
  void result;
});
// @ts-expect-error TAMEV Nano Flash does not implement prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-tamev-nano-flash-tinybert-int8', maxPrefetchBytes: 1});
const skejjaInt8Options: SkejjaInt8LoadOptions = {
  assetsUrl: '/skejja-int8/',
  model: 'experimental-skejja-edge32k-ft6-int8',
  cache: true
};
const skejjaInt8ModelPromise = loadDecisionModel(skejjaInt8Options);
void loadDecisionModel({model: 'experimental-skejja-edge32k-ft6-int8'});
void skejjaInt8ModelPromise.then((model) => {
  const skejja: LoadedSkejjaInt8Model = model;
  const result: Promise<SkejjaSystemOneResult<'experimental-skejja-edge32k-ft6-int8'>> = skejja.systemOne(
    'turn off lights',
    {
      route: {type: 'choice', criteria: {on: 'turn on lights', off: 'turn off lights'}},
      wantAlarm: {type: 'noul', instructions: 'Does the user want to set an alarm?'},
      urgency: {type: 'score', criteria: ['low', 'medium', 'high']}
    }
  );
  const calibration: Promise<SkejjaQuestionCalibration> = skejja.calibrateQuestion(
    {type: 'noul', instructions: 'Does the user want to set an alarm?'},
    Array.from({length: 8}, (_, index) => ({state: `example ${index}`, answer: index < 4}))
  );
  void result;
  void calibration;
});
// @ts-expect-error Skejja does not expose shard prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-skejja-edge32k-ft6-int8', maxPrefetchBytes: 1});
const skejjaInt8DynOptions: SkejjaInt8DynLoadOptions = {
  assetsUrl: '/skejja-int8dyn/',
  model: 'experimental-skejja-edge32k-ft6-int8dyn'
};
const skejjaInt8DynModelPromise = loadDecisionModel(skejjaInt8DynOptions);
void loadDecisionModel({model: 'experimental-skejja-edge32k-ft6-int8dyn'});
void skejjaInt8DynModelPromise.then((model) => {
  const skejja: LoadedSkejjaInt8DynModel = model;
  void skejja;
});
const ettinOptions: EttinLoadOptions = {
  assetsUrl: '/ettin/',
  model: 'experimental-ettin-68m-int8',
  cache: true
};
const ettinModelPromise = loadDecisionModel(ettinOptions);
void ettinModelPromise.then((model) => {
  const ettin: LoadedEttinModel = model;
  const result: Promise<EttinSystemOneResult> = ettin.systemOne('customer request', {
    route: {
      type: 'choice',
      instructions: 'Choose a queue.',
      criteria: {billing: 'review a charge', support: 'fix an app'}
    },
    truth: {type: 'noul', instructions: 'Was the customer charged twice?'},
    urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'medium', 'high']}
  });
  void result;
});
// @ts-expect-error Decider Ettin does not accept custom Noul criteria.
const customEttinNoul: EttinSystemOneQuestion = {
  type: 'noul',
  instructions: 'Is this true?',
  criteria: {true: 'custom yes'}
};
void customEttinNoul;
const layaOptions: LayaLoadOptions = {
  assetsUrl: '/laya/',
  model: 'experimental-laya-multilingual-fp16',
  cache: false
};
const layaModelPromise = loadDecisionModel(layaOptions);
void loadDecisionModel({model: 'experimental-laya-multilingual-fp16'});
void layaModelPromise.then((model) => {
  const laya: LoadedLayaModel = model;
  const result: Promise<LayaSystemOneResult> = laya.systemOne('customer request', {
    route: {type: 'choice', instructions: 'Choose a queue.', criteria: {billing: 'charges', support: 'product help'}},
    urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'medium', 'high']},
    truth: {type: 'noul', instructions: 'Is this true?', labels: {false: 'no', true: 'yes'}}
  });
  void result;
});
const xdecisionOptions: XDecisionLoadOptions = {
  assetsUrl: '/xdecision/',
  model: 'experimental-xdecision-fp16',
  cache: false
};
const xdecisionModelPromise = loadDecisionModel(xdecisionOptions);
void xdecisionModelPromise.then((model) => {
  const xdecision: LoadedXDecisionModel = model;
  const result: Promise<XDecisionSystemOneResult> = xdecision.systemOne('customer request', {
    route: {type: 'choice', instructions: 'Choose a queue.', criteria: ['billing', 'support']},
    urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'medium', 'high']},
    truth: {type: 'noul', instructions: 'Is this true?'}
  });
  void result;
});
const layaTypedOptions: LayaTypedDecisionsLoadOptions = {
  assetsUrl: '/laya-typed-decisions/',
  model: 'experimental-laya-typed-decisions-fp16',
  cache: false
};
const layaTypedModelPromise = loadDecisionModel(layaTypedOptions);
void loadDecisionModel({model: 'experimental-laya-typed-decisions-fp16'});
void layaTypedModelPromise.then((model) => {
  const layaTyped: LoadedLayaTypedDecisionsModel = model;
  const result: Promise<LayaTypedDecisionsSystemOneResult> = layaTyped.systemOne('A support request.', {
    team: {type: 'choice', instructions: 'Choose a team.', criteria: ['billing', 'support']}
  });
  void result;
});
// @ts-expect-error Laya does not implement read-ahead.
loadDecisionModel({assetsUrl: '/', model: 'experimental-laya-multilingual-fp16', maxPrefetchBytes: 1});
// @ts-expect-error Laya Typed-Decisions does not implement read-ahead.
loadDecisionModel({assetsUrl: '/', model: 'experimental-laya-typed-decisions-fp16', maxPrefetchBytes: 1});
// @ts-expect-error xDecision does not implement read-ahead.
loadDecisionModel({assetsUrl: '/', model: 'experimental-xdecision-fp16', maxPrefetchBytes: 1});
const glinerMobileOptions: GlinerMobileLoadOptions = {
  assetsUrl: '/gliner-mobile/',
  model: 'experimental-gliner2.5-decide-mobile-q4f16',
  cache: false
};
const glinerMobileModelPromise = loadDecisionModel(glinerMobileOptions);
void loadDecisionModel({model: 'experimental-gliner2.5-decide-mobile-q4f16'});
void glinerMobileModelPromise.then((model) => {
  const gliner: LoadedGlinerMobileModel = model;
  const result: Promise<GlinerMobileSystemOneResult> = gliner.systemOne('customer request', {
    route: {type: 'choice', instructions: 'Choose a queue.', criteria: ['billing', 'support']},
    urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'medium', 'high']},
    truth: {type: 'noul', instructions: 'Is this true?', criteria: {true: 'yes', false: 'no'}}
  });
  void result;
});
// @ts-expect-error GLiNER Mobile does not implement shard prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-gliner2.5-decide-mobile-q4f16', maxPrefetchBytes: 1});
const d1OmniOptions: D1OmniLoadOptions = {
  assetsUrl: '/d1-omni-600m-q8/',
  model: 'experimental-d1-omni-600m-q8',
  cache: false
};
const d1OmniModelPromise = loadDecisionModel(d1OmniOptions);
void loadDecisionModel({model: 'experimental-d1-omni-600m-q8'});
void d1OmniModelPromise.then((model) => {
  const d1: LoadedD1OmniModel = model;
  const result: Promise<D1OmniSystemOneResult> = d1.systemOne('A support request.', {
    route: {type: 'choice', instructions: 'Choose a queue.', criteria: {billing: 'charges', support: 'app issues'}},
    urgency: {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'medium', 'high']},
    truth: {type: 'noul', instructions: 'Is this true?', criteria: {true: 'yes', false: 'no'}}
  });
  void result;
});
// @ts-expect-error d1-omni does not implement shard prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-d1-omni-600m-q8', maxPrefetchBytes: 1});
// @ts-expect-error Decider Ettin does not expose shard prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-ettin-68m-int8', maxPrefetchBytes: 1});
const decimaAgentTypedOptions = loadDecisionModel(decimaAgentOptions);
void decimaAgentTypedOptions.then((model) => {
  const agent: LoadedDecimaAgentModel = model;
  const result: Promise<DecimaAgentSystemOneResult> = agent.systemOne('command', {
    gate: {type: 'choice', instructions: 'Should this run?', criteria: ['allow', 'ask', 'deny']},
    secret: {type: 'noul', instructions: 'Does this write a secret?'},
    risk: {type: 'score', instructions: 'How risky?', criteria: ['low', 'medium', 'high']}
  });
  // @ts-expect-error Decima-agent's upstream System One API does not support Rank.
  agent.systemOne('command', {priority: {type: 'rank', criteria: ['low', 'high']}});
  return result;
});
const tinyDecideOptions: TinyDecideLoadOptions = {
  assetsUrl: '/tinydecide/',
  model: 'experimental-tinydecide-10m-q4',
  cache: true,
  offline: true
};
loadDecisionModel({model: 'experimental-tinydecide-10m-q4'});
const tinyDecideTypedOptions = loadDecisionModel(tinyDecideOptions);
void tinyDecideTypedOptions.then((model) => {
  const typedModel: LoadedTinyDecideModel = model;
  const correction = typedModel.makeProtos('choice', [[{v: new Float32Array(128), z: new Float32Array(2)}], []]);
  const correctionOptions: TinyDecideDecisionOptions = {protos: [correction]};
  const raw: Promise<TinyDecideResult> = typedModel.answer(
    {
      state: 'Book a table for four at seven thirty.',
      questions: [
        {type: 'choice', text: 'Which app?', options: ['calendar', 'restaurants']},
        {type: 'span', text: 'Extract the time.'}
      ]
    },
    correctionOptions
  );
  const generic = typedModel.decide(
    {state: 'Book a table.', questions: [{instructions: 'Which app?', options: ['calendar', 'restaurants']}]},
    correctionOptions
  );
  const typed = typedModel.systemOne('Book a table for four at seven thirty.', {
    route: {type: 'choice', instructions: 'Which app?', options: ['calendar', 'restaurants']},
    time: {type: 'span', instructions: 'Extract the time.'}
  });
  return Promise.all([raw, generic, typed]);
});
// @ts-expect-error Decima does not prefetch ONNX model files.
const decimaPrefetch: LoadOptions = {assetsUrl: '/', model: 'experimental-decima-small-int8', maxPrefetchBytes: 1};
const decimaBaseCleanPrefetch: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-decima-base-clean-int8',
  // @ts-expect-error Decima does not prefetch ONNX model files.
  maxPrefetchBytes: 1
};
void [decimaOptions, decimaPrefetch, decimaBaseCleanOptions, decimaBaseCleanPrefetch];
const typedOpenDeciderOptions: OpenDeciderNanoLoadOptions = {
  assetsUrl: '/opendecider-nano/',
  model: 'experimental-opendecider-nano-fp16',
  cache: false
};
// @ts-expect-error OpenDecider-nano must not use a model cache.
const openDeciderCache: LoadOptions = {assetsUrl: '/', model: 'experimental-opendecider-nano-fp16', cache: true};
const openDeciderPrefetch: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-opendecider-nano-fp16',
  // @ts-expect-error OpenDecider-nano does not support prefetching.
  maxPrefetchBytes: 1
};
void [openDeciderOptions, typedOpenDeciderOptions, openDeciderCache, openDeciderPrefetch];
const nariOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-nari-decision-4b-q4f16', cache: false};
const nariMixedOptions: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-nari-decision-4b-mixed16-q4f16',
  cache: false
};
// @ts-expect-error Nari does not support cache or progress callbacks.
const nariCache: LoadOptions = {assetsUrl: '/', model: 'experimental-nari-decision-4b-q4f16', cache: true};
const h2oOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16', cache: false};
const h2oDlightOptions: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-h2o-lightning-4b-q4f16-dlight',
  cache: false,
  offline: false,
  maxPrefetchBytes: 0,
  onProgress: undefined
};
const h2oDualOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16-dual', cache: false};
const flashOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-pplx-decider-flash-v1.2-q4f16-dlight'};
const mapika4Options: LoadOptions = {assetsUrl: '/', model: 'experimental-mapika-4b-q4f16'};
const mapika4DlightOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-mapika-4b-q4f16-dlight'};
const mapika4AllOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-mapika-4b-q4f16-dlight-all'};
const elephantOptions: LoadOptions = {assetsUrl: '/', model: 'experimental-elephant-9b-q4f16-dlight'};
const elephant512Options: LoadOptions = {assetsUrl: '/', model: 'experimental-elephant-9b-q4f16-dlight-512'};
const clefOptions: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-clef-9b-q4f16-selected',
  cache: false,
  offline: false,
  maxPrefetchBytes: 0
};
// @ts-expect-error Clef does not support the model cache.
const clefCache: LoadOptions = {assetsUrl: '/', model: 'experimental-clef-9b-q4f16-selected', cache: true};
// @ts-expect-error Clef does not support progress callbacks.
const clefProgress: LoadOptions = {assetsUrl: '/', model: 'experimental-clef-9b-q4f16-selected', onProgress() {}};
// @ts-expect-error Clef does not support prefetching.
const clefPrefetch: LoadOptions = {assetsUrl: '/', model: 'experimental-clef-9b-q4f16-selected', maxPrefetchBytes: 1};
void [clefOptions, clefCache, clefProgress, clefPrefetch];
// @ts-expect-error Elephant does not support the model cache.
const elephantCache: LoadOptions = {assetsUrl: '/', model: 'experimental-elephant-9b-q4f16-dlight', cache: true};
// @ts-expect-error Elephant 512 does not support the model cache.
const elephant512Cache: LoadOptions = {assetsUrl: '/', model: 'experimental-elephant-9b-q4f16-dlight-512', cache: true};
void [elephantOptions, elephant512Options, elephantCache, elephant512Cache];
// @ts-expect-error Mapika 4B does not support the model cache.
const mapika4Cache: LoadOptions = {assetsUrl: '/', model: 'experimental-mapika-4b-q4f16', cache: true};
void [mapika4Options, mapika4DlightOptions, mapika4AllOptions, mapika4Cache];
const h2oUndefined: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-h2o-lightning-4b-q4f16',
  onProgress: undefined,
  debertaReadback: undefined
};
const strandsZero: LoadOptions = {assetsUrl: '/', model: 'experimental-strands-qwen-2b-fp16', maxPrefetchBytes: 0};
const strandsUndefined: LoadOptions = {
  assetsUrl: '/',
  model: 'experimental-strands-qwen-2b-fp16',
  maxPrefetchBytes: undefined
};
const defaultOptions: LoadOptions = {assetsUrl: '/'};
const defaultUndefinedReadback: LoadOptions = {assetsUrl: '/', debertaReadback: undefined};
const kevUndefinedReadback: LoadOptions = {assetsUrl: '/', model: 'kev-0.6b-q4f16', debertaReadback: undefined};
const debertaOptions: LoadOptions = {
  assetsUrl: '/',
  model: 'open-jev-deberta-v3-large-q4f16',
  debertaReadback: 'packed'
};
declare const dynamicModel: Exclude<
  NonNullable<LoadOptions['model']>,
  | 'experimental-tinydecide-10m-q4'
  | 'experimental-decima-small-int8'
  | 'experimental-decima-base-clean-int8'
  | 'experimental-decima-agent-int8'
  | 'experimental-tamev-micro-minilm-int8'
  | 'experimental-tamev-nano-tinybert-int8'
  | 'experimental-tamev-nano-flash-tinybert-int8'
  | 'experimental-skejja-edge32k-ft6-int8'
  | 'experimental-skejja-edge32k-ft6-int8dyn'
  | 'experimental-ettin-68m-int8'
  | 'experimental-laya-multilingual-fp16'
  | 'experimental-laya-typed-decisions-fp16'
  | 'experimental-xdecision-fp16'
  | 'experimental-gliner2.5-decide-mobile-q4f16'
  | 'experimental-safedecide-140m-fp32'
  | 'experimental-crivo-32m-v2-fp32'
  | 'experimental-crivo-32m-v2-int8'
  | 'experimental-d1-omni-600m-q8'
  | 'experimental-openjev-verdict-151m-fp16'
  | 'experimental-kev-0.6b-q4f16-upstream'
>;
declare const legacyModel: Exclude<
  NonNullable<LoadOptions['model']>,
  | 'experimental-strands-qwen-2b-fp16'
  | 'experimental-strands-gemma4-e2b-fp16'
  | 'experimental-nari-decision-4b-q4f16'
  | 'experimental-nari-decision-4b-mixed16-q4f16'
  | 'experimental-imajev-4b-candidate'
  | 'experimental-h2o-lightning-4b-q4f16'
  | 'experimental-h2o-lightning-4b-q4f16-dlight'
  | 'experimental-h2o-lightning-4b-q4f16-dual'
  | 'experimental-pplx-decider-flash-v1.2-q4f16-dlight'
  | 'experimental-mapika-4b-q4f16'
  | 'experimental-mapika-4b-q4f16-dlight'
  | 'experimental-mapika-4b-q4f16-dlight-all'
  | 'experimental-elephant-9b-q4f16-dlight'
  | 'experimental-elephant-9b-q4f16-dlight-512'
  | 'experimental-clef-9b-q4f16-selected'
  | 'experimental-opendecider-nano-fp16'
  | 'experimental-safedecide-140m-fp32'
  | 'experimental-crivo-32m-v2-fp32'
  | 'experimental-crivo-32m-v2-int8'
  | 'experimental-tinydecide-10m-q4'
  | 'experimental-decima-small-int8'
  | 'experimental-decima-base-clean-int8'
  | 'experimental-decima-agent-int8'
  | 'experimental-tamev-micro-minilm-int8'
  | 'experimental-tamev-nano-tinybert-int8'
  | 'experimental-tamev-nano-flash-tinybert-int8'
  | 'experimental-skejja-edge32k-ft6-int8'
  | 'experimental-skejja-edge32k-ft6-int8dyn'
  | 'experimental-ettin-68m-int8'
  | 'experimental-laya-multilingual-fp16'
  | 'experimental-laya-typed-decisions-fp16'
  | 'experimental-xdecision-fp16'
  | 'experimental-gliner2.5-decide-mobile-q4f16'
  | 'experimental-d1-omni-600m-q8'
  | 'experimental-openjev-verdict-151m-fp16'
  | 'experimental-kev-0.6b-q4f16-upstream'
>;
declare const dynamicBudget: number;
const dynamicOptions: LoadOptions = {assetsUrl: '/', model: dynamicModel};
const dynamicUndefinedReadback: LoadOptions = {assetsUrl: '/', model: dynamicModel, debertaReadback: undefined};
loadDecisionModel({assetsUrl: '/', model: dynamicModel});
loadDecisionModel({assetsUrl: '/', model: 'open-jev-deberta-v3-large-q4f16', debertaReadback: 'serial'});
loadDecisionModel({assetsUrl: '/', model: legacyModel, maxPrefetchBytes: 1024});
loadDecisionModel({assetsUrl: '/', model: dynamicModel, maxPrefetchBytes: 0});

const verdictOptions: VerdictLoadOptions = {
  model: 'experimental-openjev-verdict-151m-fp16',
  assetsUrl: '/models/openjev-verdict/'
};
const verdictModel: Promise<LoadedVerdictModel> = loadDecisionModel(verdictOptions);
void loadDecisionModel({model: 'experimental-openjev-verdict-151m-fp16'});
void verdictModel.then((model) => {
  const typed: Promise<VerdictSystemOneResult> = model.systemOne('A customer was charged twice.', {
    route: {
      type: 'choice',
      instructions: 'Choose a team.',
      criteria: {billing: 'charges and refunds', support: 'product issues'}
    },
    urgency: {
      type: 'score',
      instructions: 'Rate urgency.',
      criteria: [
        {id: 'low', description: 'low', value: 0},
        {id: 'high', description: 'high', value: 2}
      ]
    },
    refund: {
      type: 'noul',
      proposition: 'The customer asks for a refund.',
      semantics: 'conditional_on_sufficient_evidence_v2'
    }
  });
  const generic: Promise<VerdictDecisionResult> = model.decide({
    state: 'A customer was charged twice.',
    questions: [{instructions: 'Choose a team.', options: ['billing', 'support']}]
  });
  return Promise.all([typed, generic]);
});

if (dynamicModel === 'open-jev-deberta-v3-large-q4f16') {
  loadDecisionModel({assetsUrl: '/', model: dynamicModel, debertaReadback: 'packed'});
}

async function useInstalledPackage(): Promise<void> {
  const model: DecisionModel = await loadDecisionModel(options);
  const loaded: LoadedDecisionModel = await loadDecisionModel(options);
  const modelId: ModelName = loaded.description.model;
  const all: readonly ModelName[] = listDecisionModels();
  const description = describeModel('mapika-0.8b-q4f16');
  const temperature = describeModel('experimental-nari-decision-4b-q4f16').features.textChoice.score.temperature;
  if (typeof temperature === 'object' && 'byOptionCount' in temperature) {
    const minimum: number | undefined = temperature.byOptionCount[0]?.min;
    void minimum;
  }
  declareDynamicModel(description.model);
  // @ts-expect-error Descriptors are deeply readonly.
  description.features.textChoice.limits.push({resource: 'options', scope: 'question', min: 2, max: 3});
  // @ts-expect-error Image metadata is not an implemented operation.
  void loaded.decideImage;
  const result: DecisionResult = await model.decide(request, {signal: new AbortController().signal});
  const answer = result.answers[0];
  const firstProbability = answer?.probabilities[0];
  if (!answer || !firstProbability) throw new Error('Expected an answer and a probability.');
  const selected: string = answer.selected;
  const selectedIndex: number = answer.selectedIndex;
  const probability: number = firstProbability.probability;
  // @ts-expect-error Currently supported models do not expose abstention.
  void answer.abstained;
  // @ts-expect-error Currently supported models do not expose an unknown probability.
  void answer.unknownProbability;
  const cleared: boolean = await clearModelCache();
  await model.releaseCachedMemory();
  await model.dispose();
  void [selected, selectedIndex, probability, cleared, loaded, modelId, all];
}

async function useImajev(): Promise<void> {
  const model = await loadDecisionModel({model: 'experimental-imajev-4b-candidate', assetsUrl: '/assets/'});
  const result = await model.decide(request);
  const answer = result.answers[0];
  if (!answer) throw new Error('Expected an answer.');
  if (answer.abstained) {
    const selected: null = answer.selected;
    const known: string = answer.bestKnown.option;
    void [selected, known];
  } else {
    const selected: string = answer.selected;
    const index: number = answer.selectedIndex;
    void [selected, index];
  }
  const unknown: number = answer.unknownProbability;
  // @ts-expect-error Imajev cannot use the model cache.
  loadDecisionModel({model: 'experimental-imajev-4b-candidate', assetsUrl: '/', cache: true});
  void unknown;
}
void useImajev;

async function useOpenDeciderNano(): Promise<void> {
  const model = await loadDecisionModel({
    model: 'experimental-opendecider-nano-fp16',
    assetsUrl: '/opendecider-nano/',
    cache: false
  });
  const result: OpenDeciderNanoSystemOneResult = await model.systemOne(
    {ticket: {subject: 'Refund', attempts: 2}},
    {
      department: {
        type: 'choice',
        instructions: 'Which team should handle this?',
        criteria: {billing: 'charges, refunds', technical: 'bugs'}
      },
      urgency: {type: 'score', instructions: 'How urgent is this?', criteria: ['not urgent', 'soon', 'blocking']},
      churnRisk: {type: 'noul', instructions: 'Will the customer leave?'}
    }
  );
  const choice = result.answers.department;
  if (choice?.type === 'choice') {
    const selected: string = choice.choice;
    const probability: number | undefined = choice.probabilities[selected];
    void probability;
  }
  const score = result.answers.urgency;
  if (score?.type === 'score') {
    const expected: number = score.expected;
    const level: number = score.score;
    void [expected, level];
  }
  const noul = result.answers.churnRisk;
  if (noul?.type === 'noul') {
    const yesProbability: number = noul.noul;
    const noProbability: number = noul.probabilities.false;
    void [yesProbability, noProbability];
  }
  // @ts-expect-error Score levels must remain ordered strings.
  await model.systemOne('state', {urgency: {type: 'score', instructions: 'Rate.', criteria: {high: 'high'}}});
  // @ts-expect-error This is a yes probability, not a native abstention result.
  void noul?.abstained;
  await model.dispose();
}
void useOpenDeciderNano;

async function useDecima(): Promise<void> {
  const model = await decimaTypedOptions;
  const result: DecimaSystemOneResult = await model.systemOne(
    'A customer was charged twice.',
    {
      route: {type: 'choice', instructions: 'Which team?', criteria: {billing: 'charges', support: 'technical issues'}},
      urgency: {type: 'score', instructions: 'How urgent?', criteria: ['low', 'medium', 'high']},
      urgent: {type: 'noul', instructions: 'Does this need attention today?'},
      priority: {
        type: 'rank',
        instructions: 'Rank possible next steps.',
        criteria: ['refund', 'investigate', 'escalate']
      }
    },
    {language: 'en'}
  );
  const route = result.answers.route;
  if (route?.type === 'choice') {
    const selected: string = route.choice;
    void selected;
  }
  const rank = result.answers.priority;
  if (rank?.type === 'rank') {
    const probabilities: Readonly<Record<string, number>> = rank.probabilities;
    const top: string | undefined = rank.order[0];
    void [probabilities, top];
  }
  await model.dispose();
}
void useDecima;

async function useKev06SystemOne(): Promise<void> {
  const model: LoadedKev06DecisionModel = await loadDecisionModel({
    model: 'kev-0.6b-q4f16',
    assetsUrl: '/kev-0.6b/'
  });
  const result: KevSystemOneResult = await model.systemOne(
    {ticket: 'refund', attempts: 2},
    {
      department: {type: 'choice', instructions: 'Which team?', criteria: {billing: 'charges', other: null}},
      urgency: {type: 'score', instructions: 'How urgent?', criteria: ['later', 'soon', 'today']},
      refund: {type: 'noul', instructions: 'Requesting money back?', criteria: {true: 'yes', false: 'no'}}
    }
  );
  const choice = result.answers.department;
  if (choice?.type === 'choice') {
    const key: string = choice.choice;
    const confidence: number = choice.confidence;
    void [key, confidence];
  }
  const mapState = new Map<string, KevSystemOneJsonValue>([
    ['2', 'second'],
    ['1', new Map([['8', 9007199254740993123456789n]])]
  ]);
  const mapQuestions = new Map<string, KevSystemOneQuestion>();
  mapQuestions.set('10', {
    type: 'choice',
    instructions: new Map([['2', new Map([['1', 'context']])]]),
    criteria: new Map([
      ['2', new Map([['1', 'description']])],
      ['1', null]
    ]),
    applicationMetadata: {traceId: 'ignored'}
  });
  mapQuestions.set('2', {
    type: 'noul',
    instructions: null,
    criteria: new Map([
      ['true', 'yes'],
      ['false', 'no'],
      ['ignored', null]
    ])
  });
  const mapResult = await model.systemOne(mapState, mapQuestions);
  const [questionName, mapAnswer] = mapResult.answerEntries[0]!;
  if (mapAnswer.type === 'choice') {
    const [optionKey, probability] = mapAnswer.probabilityEntries[0]!;
    void [optionKey, probability];
  }
  void questionName;
  const score = result.answers.urgency;
  if (score?.type === 'score') {
    const expectedLevel: number = score.score;
    const levelName: string | undefined = score.legend['2'];
    void [expectedLevel, levelName];
  }
  const noul = result.answers.refund;
  if (noul?.type === 'noul') {
    const probabilityOfYes: number = noul.noul;
    // @ts-expect-error Kev Noul returns p(true), not an abstention or two-choice distribution.
    void noul.probabilities;
    void probabilityOfYes;
  }
  // @ts-expect-error Score criteria are ordered levels, not named options.
  await model.systemOne('state', {urgency: {type: 'score', instructions: 'Rate.', criteria: {high: 'high'}}});
  await model.systemOne('state', {
    department: {type: 'choice', criteria: {billing: 'charges', other: null}},
    urgency: {type: 'score', criteria: ['later', 'soon', 'today']},
    refund: {type: 'noul'}
  });
  await model.dispose();

  const defaultModel: LoadedKev06DecisionModel = await loadDecisionModel({assetsUrl: '/kev-default/'});
  await defaultModel.systemOne('state', {check: {type: 'noul', instructions: null}});
  await defaultModel.dispose();
}
void useKev06SystemOne;

async function useKev08SystemOne(): Promise<void> {
  const options: Kev08LoadOptions = {model: 'kev-0.8b-f16', assetsUrl: '/kev-0.8b/'};
  const model: LoadedKev08DecisionModel = await loadDecisionModel(options);
  const result: KevSystemOneResult<'kev-0.8b-f16'> = await model.systemOne(
    {ticket: {subject: 'Refund', attempts: 2}},
    {
      route: {type: 'choice', criteria: {billing: 'charges', support: null}},
      urgency: {type: 'score', criteria: ['low', 'medium', 'high']},
      refund: {type: 'noul', criteria: {true: 'asks for money back'}}
    }
  );
  const modelId: 'kev-0.8b-f16' = result.model;
  // @ts-expect-error Ordered entry helpers are specific to Kev 0.6 and Reflexes.
  void result.answerEntries;
  const choice = result.answers.route;
  if (choice?.type === 'choice') {
    const selected: string = choice.choice;
    const probability: number | undefined = choice.probabilities[selected];
    void probability;
  }
  // @ts-expect-error Kev 0.8 uses its separate JSON and record-only input adapter.
  await model.systemOne(new Map([['id', 1n]]), new Map());
  const score = result.answers.urgency;
  if (score?.type === 'score') {
    const expectedLevel: number = score.score;
    const label: string | undefined = score.legend['2'];
    void [expectedLevel, label];
  }
  const noul = result.answers.refund;
  if (noul?.type === 'noul') {
    const yesProbability: number = noul.noul;
    void yesProbability;
  }
  // @ts-expect-error Kev Choice criteria are named options, not a list.
  await model.systemOne('state', {bad: {type: 'choice', criteria: ['first', 'second']}});
  await model.dispose();
  void modelId;
}
void useKev08SystemOne;

async function useKevReflexesSystemOne(): Promise<void> {
  const model: LoadedKevReflexesDecisionModel = await loadDecisionModel({
    model: 'kev-0.6b-reflexes-q4f16',
    assetsUrl: '/kev-reflexes/'
  });
  const result = await model.systemOne('A customer asks for a refund.', {
    refund: {type: 'noul'},
    queue: {type: 'choice', criteria: {billing: 'charges', support: 'another issue'}},
    urgency: {type: 'score', criteria: ['low', 'high']}
  });
  const modelId: 'kev-0.6b-reflexes-q4f16' = result.model;
  const probabilityOfYes: number | undefined =
    result.answers.refund?.type === 'noul' ? result.answers.refund.noul : undefined;
  void [modelId, probabilityOfYes];
  // @ts-expect-error The base Kev route returns its own model identifier.
  const baseModelId: 'kev-0.6b-q4f16' = result.model;
  void baseModelId;
  await model.dispose();
}
void useKevReflexesSystemOne;

function checkKevHybridPreviewDeclaration(model: KevHybridPreviewModel): void {
  void model.systemOne('state', {
    choice: {type: 'choice', criteria: {first: 'one', second: 'two'}},
    score: {type: 'score', criteria: ['only level']},
    noul: {type: 'noul'}
  });
}
void checkKevHybridPreviewDeclaration;

async function useDebertaSystemOne(): Promise<void> {
  const model: LoadedDebertaDecisionModel = await loadDecisionModel({
    model: 'open-jev-deberta-v3-large-q4f16',
    assetsUrl: '/deberta/'
  });
  const result: DebertaSystemOneResult = await model.systemOne('A customer review.', {
    polarity: {type: 'choice', instructions: 'What is the sentiment?', options: ['positive', 'negative']},
    tone: {type: 'score', instructions: 'How positive?', options: ['negative', 'neutral', 'positive']},
    praise: {type: 'noul', instructions: 'Does the review express a positive opinion?'}
  });
  const choice = result.answers.polarity;
  if (choice?.type === 'choice') {
    const selected: string = choice.choice;
    const probability: number | undefined = choice.probabilities[selected];
    void probability;
  }
  const score = result.answers.tone;
  if (score?.type === 'score') {
    const expectedLevel: number = score.score;
    const label: string | undefined = score.legend['2'];
    void [expectedLevel, label];
  }
  const noul = result.answers.praise;
  if (noul?.type === 'noul') {
    const yesProbability: number = noul.noul;
    const falseProbability: number = noul.probabilities.false;
    void [yesProbability, falseProbability];
  }
  // @ts-expect-error DeBERTa's native typed protocol uses option lists.
  await model.systemOne('state', {tone: {type: 'score', instructions: 'Rate.', criteria: ['low', 'high']}});
  // @ts-expect-error DeBERTa Noul always uses the model's fixed no/yes options.
  await model.systemOne('state', {noul: {type: 'noul', instructions: 'True?', options: ['no', 'yes']}});
  await model.dispose();
}
void useDebertaSystemOne;

function declareDynamicModel(value: unknown): void {
  if (isDecisionModel(value)) {
    const model: ModelName = value;
    describeModel(model);
    if (model === 'experimental-kev-0.6b-q4f16-upstream') loadDecisionModel({model});
    else loadDecisionModel({assetsUrl: '/', model});
  }
}

const failure = new WebDecisionError('INVALID_INPUT', 'Invalid request.', {cause: new Error('cause')});
const errorCode: ErrorCode = failure.code;
void [
  useInstalledPackage,
  errorCode,
  mapikaOptions,
  mapikaQ4Options,
  kev08Options,
  useKev08SystemOne,
  strandsOptions,
  nariOptions,
  nariMixedOptions,
  nariCache,
  h2oOptions,
  h2oDlightOptions,
  h2oDualOptions,
  flashOptions,
  h2oUndefined,
  strandsZero,
  strandsUndefined,
  defaultOptions,
  defaultUndefinedReadback,
  kevUndefinedReadback,
  dynamicUndefinedReadback,
  debertaOptions,
  dynamicOptions
];

// @ts-expect-error The model assets URL is required.
loadDecisionModel({});
// @ts-expect-error Model identifiers are restricted to the supported previews.
loadDecisionModel({assetsUrl: '/', model: 'unknown-model'});
// @ts-expect-error An unregistered model cannot be described as available.
describeModel('imajev-4b');
// @ts-expect-error The default model does not support DeBERTa readback.
loadDecisionModel({assetsUrl: '/', debertaReadback: 'packed'});
// @ts-expect-error Kev does not support DeBERTa readback.
loadDecisionModel({assetsUrl: '/', model: 'kev-0.6b-q4f16', debertaReadback: 'serial'});
// @ts-expect-error Mapika does not support DeBERTa readback.
loadDecisionModel({assetsUrl: '/', model: 'mapika-0.8b-f16', debertaReadback: 'packed'});
// @ts-expect-error Mapika Q4 does not support DeBERTa readback.
loadDecisionModel({assetsUrl: '/', model: 'mapika-0.8b-q4f16', debertaReadback: 'packed'});
// @ts-expect-error Kev Reflexes does not support DeBERTa readback.
loadDecisionModel({assetsUrl: '/', model: 'kev-0.6b-reflexes-q4f16', debertaReadback: 'packed'});
// @ts-expect-error Kev 0.8B does not support DeBERTa readback.
loadDecisionModel({assetsUrl: '/', model: 'kev-0.8b-f16', debertaReadback: 'packed'});
// @ts-expect-error Strands does not support DeBERTa readback.
loadDecisionModel({assetsUrl: '/', model: 'experimental-strands-qwen-2b-fp16', debertaReadback: 'packed'});
// @ts-expect-error Strands does not implement prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-strands-qwen-2b-fp16', maxPrefetchBytes: 1});
// @ts-expect-error H2O does not implement caching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16', cache: true});
// @ts-expect-error H2O does not implement offline loading.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16', offline: true});
// @ts-expect-error H2O does not implement progress callbacks.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16', onProgress() {}});
// @ts-expect-error H2O does not implement prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16', maxPrefetchBytes: 1});
// @ts-expect-error H2O DLight does not implement caching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16-dlight', cache: true});
// @ts-expect-error H2O DLight does not implement offline loading.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16-dlight', offline: true});
// @ts-expect-error H2O DLight does not implement progress callbacks.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16-dlight', onProgress() {}});
// @ts-expect-error H2O DLight does not implement prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16-dlight', maxPrefetchBytes: 1});
// @ts-expect-error H2O dual does not implement caching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-h2o-lightning-4b-q4f16-dual', cache: true});
// @ts-expect-error Perplexity does not implement caching or prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-pplx-decider-flash-v1.2-q4f16-dlight', cache: true});
// @ts-expect-error Perplexity does not implement prefetching.
loadDecisionModel({assetsUrl: '/', model: 'experimental-pplx-decider-flash-v1.2-q4f16-dlight', maxPrefetchBytes: 1});
// @ts-expect-error A number that may be nonzero is not supported for Strands.
loadDecisionModel({assetsUrl: '/', model: 'experimental-strands-qwen-2b-fp16', maxPrefetchBytes: dynamicBudget});
// @ts-expect-error A dynamic model may be Strands and must be narrowed before prefetching.
loadDecisionModel({assetsUrl: '/', model: dynamicModel, maxPrefetchBytes: 1024});
// @ts-expect-error A dynamic model must be narrowed before setting DeBERTa readback.
loadDecisionModel({assetsUrl: '/', model: dynamicModel, debertaReadback: 'packed'});
// @ts-expect-error DeBERTa readback only accepts serial or packed.
loadDecisionModel({assetsUrl: '/', model: 'open-jev-deberta-v3-large-q4f16', debertaReadback: 'other'});
const invalidVariable = {assetsUrl: '/', model: 'kev-0.8b-f16' as const, debertaReadback: 'packed' as const};
// @ts-expect-error The restriction also applies to values stored in variables.
loadDecisionModel(invalidVariable);
// @ts-expect-error Choice options must be strings.
const invalidRequest: DecisionRequest = {state: '', questions: [{instructions: 'x', options: [1, 2]}]};
void invalidRequest;
