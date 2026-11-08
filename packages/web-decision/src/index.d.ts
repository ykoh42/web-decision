export type ErrorCode =
  | 'INVALID_INPUT'
  | 'INVALID_ASSET'
  | 'UNSUPPORTED_ENVIRONMENT'
  | 'DEVICE_RESOURCE_LIMIT'
  | 'ASSET_LOAD_FAILED'
  | 'MODEL_LOAD_FAILED'
  | 'INFERENCE_ERROR'
  | 'SESSION_CLOSED';

export class WebDecisionError extends Error {
  constructor(code: ErrorCode, message: string, options?: {cause?: unknown});
  readonly code: ErrorCode;
}

export interface DecisionQuestion {
  instructions: string;
  options: readonly string[];
}

export interface DecisionRequest {
  state: string;
  questions: readonly DecisionQuestion[];
}

export type DecisionJsonValue =
  | null
  | boolean
  | number
  | string
  | readonly DecisionJsonValue[]
  | Readonly<{[key: string]: DecisionJsonValue}>;

export interface OpenDeciderChoiceQuestion {
  readonly type: 'choice';
  readonly instructions: string;
  readonly criteria: readonly string[] | Readonly<Record<string, string | null>>;
}

export interface OpenDeciderScoreQuestion {
  readonly type: 'score';
  readonly instructions: string;
  /** Ordered from the lowest level (0) to the highest. */
  readonly criteria: readonly string[];
}

export interface OpenDeciderNoulQuestion {
  readonly type: 'noul';
  readonly instructions: string;
  /** Optional descriptions for the true and false options. */
  readonly criteria?: Readonly<{true?: string | null; false?: string | null}>;
}

export type OpenDeciderTypedQuestion = OpenDeciderChoiceQuestion | OpenDeciderScoreQuestion | OpenDeciderNoulQuestion;

export type OpenDeciderTypedAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      /** Normalized margin above a uniform distribution; not calibrated confidence. */
      confidence: number;
      truncated?: true;
    }>
  | Readonly<{
      type: 'score';
      /** The most likely ordered level, with 0 as the lowest. */
      score: number;
      /** The probability-weighted mean level. */
      expected: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
      truncated?: true;
    }>
  | Readonly<{
      type: 'noul';
      /** Probability that the answer is yes. This is not an abstention. */
      noul: number;
      probabilities: Readonly<{true: number; false: number}>;
      confidence: number;
      truncated?: true;
    }>;

export interface OpenDeciderNanoSystemOneResult {
  readonly model: 'experimental-opendecider-nano-fp16';
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, OpenDeciderTypedAnswer>>;
}

export type LayaSystemOneQuestion =
  | Readonly<{
      type: 'choice';
      instructions: string;
      criteria: readonly string[] | Readonly<Record<string, DecisionJsonValue>>;
    }>
  | Readonly<{type: 'score'; instructions: string; criteria: readonly DecisionJsonValue[]}>
  | Readonly<{
      type: 'noul';
      instructions: string;
      criteria?: Readonly<{true?: DecisionJsonValue; false?: DecisionJsonValue}>;
      labels?: Readonly<{true: string; false: string}>;
    }>;

type LayaAnswerFields = Readonly<{
  /** Entropy based for Choice and Score; max probability for Noul. Not calibrated. */
  confidence: number;
  /** Highest candidate probability, not calibrated. */
  answerConfidence: number;
  /** Diagnostic probability from the upstream action head; this head is not an abstention signal. */
  actionProbability: number;
  /** Raw two-class action-head logits for diagnostics only. */
  actionLogits: readonly [number, number];
}>;

export type LayaSystemOneAnswer =
  | (LayaAnswerFields &
      Readonly<{
        type: 'choice';
        choice: string;
        probabilities: Readonly<Record<string, number>>;
      }>)
  | (LayaAnswerFields &
      Readonly<{
        type: 'score';
        /** Probability weighted mean of ordered level indices starting at zero. */
        score: number;
        expected: number;
        legend: Readonly<Record<string, string>>;
        probabilities: Readonly<Record<string, number>>;
      }>)
  | (LayaAnswerFields &
      Readonly<{
        type: 'noul';
        /** Probability that the statement is true; this is not abstention. */
        noul: number;
        probabilities: Readonly<{false: number; true: number}>;
      }>);

export interface LayaSystemOneResult {
  readonly model: 'experimental-laya-multilingual-fp16';
  /** Sum of independently encoded question rows. */
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly truncatedQuestions?: readonly string[];
  readonly collapsedOptions?: Readonly<Record<string, Readonly<{total: number; distinct: number}>>>;
  readonly answers: Readonly<Record<string, LayaSystemOneAnswer>>;
}

export interface LayaTypedDecisionsSystemOneResult extends Omit<LayaSystemOneResult, 'model'> {
  readonly model: 'experimental-laya-typed-decisions-fp16';
}

export type XDecisionSystemOneQuestion = LayaSystemOneQuestion;
export type XDecisionSystemOneAnswer = LayaSystemOneAnswer;

export interface XDecisionSystemOneResult extends Omit<LayaSystemOneResult, 'model'> {
  readonly model: 'experimental-xdecision-fp16';
}

export type GlinerMobileSystemOneQuestion =
  | Readonly<{
      type: 'choice';
      instructions: string;
      criteria: readonly string[] | Readonly<Record<string, string | null>>;
    }>
  | Readonly<{type: 'score'; instructions: string; criteria: readonly string[]}>
  | Readonly<{
      type: 'noul';
      instructions: string;
      criteria?: Readonly<{true?: string | null; false?: string | null}> | null;
    }>;

export type GlinerMobileSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      confidence: number;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{
      type: 'score';
      /** Probability weighted mean of ordered level indices starting at zero. */
      score: number;
      expected: number;
      confidence: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{
      type: 'noul';
      /** Probability that the statement is true; this model has no abstention output. */
      noul: number;
      confidence: number;
      probabilities: Readonly<{false: number; true: number}>;
    }>;

export interface GlinerMobileSystemOneResult {
  readonly model: 'experimental-gliner2.5-decide-mobile-q4f16';
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, GlinerMobileSystemOneAnswer>>;
}

export type KevSystemOneJsonValue =
  | null
  | boolean
  | number
  | bigint
  | string
  | readonly KevSystemOneJsonValue[]
  | Readonly<{[key: string]: KevSystemOneJsonValue}>
  | Map<string, KevSystemOneJsonValue>;

export interface KevChoiceQuestion {
  readonly type: 'choice';
  readonly instructions?: KevSystemOneJsonValue;
  readonly criteria: Readonly<Record<string, KevSystemOneJsonValue>> | Map<string, KevSystemOneJsonValue>;
  readonly [key: string]: unknown;
}

export interface KevScoreQuestion {
  readonly type: 'score';
  readonly instructions?: KevSystemOneJsonValue;
  /** Ordered from the lowest level to the highest. */
  readonly criteria: readonly KevSystemOneJsonValue[];
  readonly [key: string]: unknown;
}

export interface KevNoulQuestion {
  readonly type: 'noul';
  readonly instructions?: KevSystemOneJsonValue;
  readonly criteria?: Readonly<Record<string, KevSystemOneJsonValue>> | Map<string, KevSystemOneJsonValue> | null;
  readonly [key: string]: unknown;
}

export type KevSystemOneQuestion = KevChoiceQuestion | KevScoreQuestion | KevNoulQuestion;
export type KevSystemOneQuestions = Readonly<Record<string, KevSystemOneQuestion>> | Map<string, KevSystemOneQuestion>;

export interface Kev08ChoiceQuestion {
  readonly type: 'choice';
  readonly instructions?: DecisionJsonValue;
  readonly criteria: Readonly<Record<string, DecisionJsonValue>>;
}

export interface Kev08ScoreQuestion {
  readonly type: 'score';
  readonly instructions?: DecisionJsonValue;
  readonly criteria: readonly DecisionJsonValue[];
}

export interface Kev08NoulQuestion {
  readonly type: 'noul';
  readonly instructions?: DecisionJsonValue;
  readonly criteria?: Readonly<{true?: DecisionJsonValue; false?: DecisionJsonValue}> | null;
}

export type Kev08SystemOneQuestion = Kev08ChoiceQuestion | Kev08ScoreQuestion | Kev08NoulQuestion;

export type MapikaCriteria = readonly DecisionJsonValue[] | Readonly<Record<string, DecisionJsonValue>>;

type MapikaQuestionText =
  | Readonly<{instructions: DecisionJsonValue; question?: DecisionJsonValue}>
  | Readonly<{instructions?: undefined; question: DecisionJsonValue}>;

type MapikaQuestionCriteria =
  | Readonly<{criteria: MapikaCriteria; options?: MapikaCriteria}>
  | Readonly<{criteria?: undefined; options: MapikaCriteria}>;

export type MapikaChoiceQuestion = Readonly<
  MapikaQuestionText & MapikaQuestionCriteria & {type?: 'choice'; isolated?: boolean}
>;

export type MapikaScoreQuestion = Readonly<
  MapikaQuestionText &
    (
      | Readonly<{
          criteria: readonly DecisionJsonValue[] | Readonly<Record<string, DecisionJsonValue>>;
          options?: never;
        }>
      | Readonly<{
          criteria?: undefined;
          options: readonly DecisionJsonValue[] | Readonly<Record<string, DecisionJsonValue>>;
        }>
    ) & {type: 'score'; isolated?: boolean}
>;

export type MapikaNoulQuestion = Readonly<{
  type: 'noul' | 'bool';
  instructions?: DecisionJsonValue;
  question?: DecisionJsonValue;
  criteria?: Readonly<{true?: DecisionJsonValue; false?: DecisionJsonValue}> | null;
  options?: Readonly<{true?: DecisionJsonValue; false?: DecisionJsonValue}> | null;
}>;

export type MapikaSystemOneQuestion = MapikaChoiceQuestion | MapikaScoreQuestion | MapikaNoulQuestion;

export type MapikaSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      /** TypeSafe normalized confidence: uniform choices map to 0 and a certain choice to 1. */
      confidence: number;
      /** Probability of the selected choice before confidence normalization. */
      x_p_max: number;
      certainty: number;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{
      type: 'score';
      /** Expected level, rounded to two decimal places. */
      score: number;
      /** TypeSafe confidence based on expected distance from the most likely level. */
      confidence: number;
      /** Probability of the most likely level. */
      x_p_max: number;
      certainty: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      /** Present when levels are evaluated independently, as in the upstream default. */
      levelFit?: Readonly<Record<string, number>>;
      fitMass?: number;
    }>
  | Readonly<{type: 'noul'; /** Probability that the answer is yes. */ noul: number}>;

export type MapikaSystemOneModelName = 'mapika-0.8b-f16' | 'mapika-0.8b-q4f16';

export interface MapikaSystemOneResult<Model extends MapikaSystemOneModelName = MapikaSystemOneModelName> {
  readonly model: Model;
  /** Shared context prefix plus every independently evaluated question branch. */
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, MapikaSystemOneAnswer>>;
}

export type KevSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      /** Probabilities in the exact option order supplied to Kev. */
      probabilityEntries: readonly (readonly [key: string, probability: number])[];
      confidence: number;
    }>
  | Readonly<{
      type: 'score';
      /** Probability-weighted mean level, starting at 0. */
      score: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      /** Probabilities in ascending score-level order. */
      probabilityEntries: readonly (readonly [level: string, probability: number])[];
      /** Concentration around the most likely level; not calibrated confidence. */
      confidence: number;
    }>
  | Readonly<{type: 'noul'; /** Probability of true (yes). */ noul: number}>;

export type Kev08SystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
    }>
  | Readonly<{
      type: 'score';
      score: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
    }>
  | Readonly<{type: 'noul'; noul: number}>;

export type KevSystemOneModelName =
  | 'kev-0.6b-q4f16'
  | 'experimental-kev-0.6b-q4f16-upstream'
  | 'kev-0.6b-reflexes-q4f16'
  | 'kev-0.8b-f16';

export interface Kev08SystemOneResult {
  readonly model: 'kev-0.8b-f16';
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, Kev08SystemOneAnswer>>;
}

export type KevSystemOneResult<Model extends KevSystemOneModelName = 'kev-0.6b-q4f16'> = Model extends 'kev-0.8b-f16'
  ? Kev08SystemOneResult
  : Readonly<{
      model: Model;
      tokenCount: number;
      answers: Readonly<Record<string, KevSystemOneAnswer>>;
      answerEntries: readonly (readonly [question: string, answer: KevSystemOneAnswer])[];
    }>;

export interface DebertaChoiceQuestion {
  readonly type: 'choice';
  readonly instructions: string;
  /** DeBERTa accepts 2 to 255 distinct options. */
  readonly options: readonly string[];
}

export interface DebertaScoreQuestion {
  readonly type: 'score';
  readonly instructions: string;
  /** Ordered low to high; DeBERTa accepts 2 to 10 levels. */
  readonly options: readonly string[];
}

export interface DebertaNoulQuestion {
  readonly type: 'noul';
  readonly instructions: string;
}

export type DebertaSystemOneQuestion = DebertaChoiceQuestion | DebertaScoreQuestion | DebertaNoulQuestion;

export type DebertaSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      /** Highest option probability, using the checkpoint's 1.05 temperature. */
      confidence: number;
    }>
  | Readonly<{
      type: 'score';
      /** Probability-weighted level index, with 0 as the lowest level. */
      score: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      /** Highest level probability, using the checkpoint's 1.05 temperature. */
      confidence: number;
    }>
  | Readonly<{
      type: 'noul';
      /** Probability of yes; this is not abstention. */
      noul: number;
      probabilities: Readonly<{false: number; true: number}>;
      confidence: number;
    }>;

export interface DebertaSystemOneResult {
  readonly model: 'open-jev-deberta-v3-large-q4f16';
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, DebertaSystemOneAnswer>>;
}

export interface OptionScore {
  option: string;
  /** Per-model softmax score; temperature may be applied and calibration is not guaranteed. */
  probability: number;
  logit: number;
}

export interface DecisionAnswer {
  /** Non-Imajev routes select the highest probability and resolve exact ties by option order. */
  selected: string;
  selectedIndex: number;
  probabilities: OptionScore[];
}

export interface DecisionResult {
  tokenCount: number;
  /** True when a model shortened the state to fit its input limit. */
  truncated?: boolean;
  answers: DecisionAnswer[];
}

export type TinyDecideQuestion =
  | Readonly<{type: 'choice' | 'score'; text: string; options: readonly string[]}>
  | Readonly<{type: 'noul' | 'span'; text: string; options?: readonly string[]}>;

export interface TinyDecideRequest {
  state: string;
  questions: readonly TinyDecideQuestion[];
}

export type TinyDecideAnswer =
  | Readonly<{
      type: 'choice' | 'score';
      probs: readonly number[];
      pick: number;
      confidence: number;
      score?: number;
      qvec: readonly number[];
      z0: readonly number[];
      proj: boolean;
    }>
  | Readonly<{
      type: 'noul';
      p: number;
      qvec: readonly number[];
      z0: readonly [number, number];
      proj: boolean;
    }>
  | Readonly<{
      type: 'span';
      text: string;
      p_present: number;
      p_span: number;
      tok: readonly [number, number];
      char: readonly [number, number] | null;
    }>;

export interface TinyDecideResult {
  answers: readonly TinyDecideAnswer[];
  tokens: Readonly<{state: number; questions: number; total: number}>;
  truncated: boolean;
  ms: number;
  ids: readonly number[];
}

/** Native correction vectors for one choice, score or Noul question. */
export interface TinyDecidePrototypes {
  vec: Float32Array;
  cnt: Int32Array;
  /** Center vector used by the prototype cosine; null uses the count-weighted prototype mean. */
  center: Float32Array | null;
  lam?: number;
}

export interface TinyDecideCorrectionExample {
  /** The `qvec` from a previous Choice, Score or Noul answer. */
  v: ArrayLike<number>;
  /** The `z0` from that same answer, used to fit the correction weight. */
  z: ArrayLike<number>;
}

export interface TinyDecideDecisionOptions extends DecisionOptions {
  /** Per-question prototypes in request order, prepared from earlier qvec outputs and corrections. */
  protos?: readonly (TinyDecidePrototypes | null | undefined)[];
}

export type TinyDecideSystemOneQuestion =
  | Readonly<{type: 'choice' | 'score'; instructions: string; options: readonly string[]}>
  | Readonly<{type: 'noul' | 'span'; instructions: string; options?: readonly string[]}>;

export interface TinyDecideSystemOneResult {
  readonly model: 'experimental-tinydecide-10m-q4';
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, TinyDecideAnswer>>;
}

/** Known-choice probabilities are conditional on the model not choosing unknown. */
export type ImajevDecisionAnswer = Readonly<{
  unknownProbability: number;
  probabilities: readonly Readonly<OptionScore>[];
}> &
  (
    | Readonly<{abstained: false; selected: string; selectedIndex: number}>
    | Readonly<{
        abstained: true;
        selected: null;
        selectedIndex: null;
        /** Highest-scoring known option; this is not a selected answer. */
        bestKnown: Readonly<{option: string; index: number}>;
      }>
  );

export interface ImajevDecisionResult {
  /** Maximum token count among independently processed questions. */
  readonly tokenCount: number;
  readonly answers: readonly ImajevDecisionAnswer[];
}

export interface DecisionOptions {
  /** Stop waiting and skip a queued request. Already-running GPU work continues; later requests and disposal wait for it. */
  signal?: AbortSignal;
}

export type SafeDecideAlpha = 0.05 | 0.1 | 0.2;
export type SafeDecideAction = 'automate' | 'escalate';
export type SafeDecideState = string | readonly DecisionJsonValue[] | Readonly<{[key: string]: DecisionJsonValue}>;

export type SafeDecideSystemOneQuestion =
  | Readonly<{
      type?: 'choice';
      instructions?: string | null;
      criteria: Readonly<Record<string, string | null | undefined>>;
    }>
  | Readonly<{
      type: 'score';
      instructions?: string | null;
      /** Ordered from the lowest level (0) to the highest. */
      criteria: readonly (string | null | undefined)[];
    }>
  | Readonly<{
      type: 'noul';
      instructions?: string | null;
      criteria?: Readonly<{true?: string | null; false?: string | null}> | null;
    }>;

export type SafeDecideSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
      action: SafeDecideAction;
    }>
  | Readonly<{
      type: 'score';
      /** Probability-weighted mean of ordered level indices starting at zero. */
      score: number;
      /** Most likely ordered level label. */
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
      action: SafeDecideAction;
    }>
  | Readonly<{
      type: 'noul';
      /** Probability that the statement is true; this is not abstention. */
      noul: number;
      probabilities: Readonly<{false: number; true: number}>;
      confidence: number;
      action: SafeDecideAction;
    }>;

export interface SafeDecideGuarantee {
  readonly alpha: SafeDecideAlpha;
  readonly delta: number;
  readonly statement: string;
}

export interface SafeDecideSystemOneResult {
  readonly model: 'experimental-safedecide-140m-fp32';
  readonly tokenCount: number;
  readonly outputTokenCount: 0;
  readonly truncated?: true;
  readonly truncatedQuestions?: readonly string[];
  readonly guarantee: SafeDecideGuarantee;
  readonly answers: Readonly<Record<string, SafeDecideSystemOneAnswer>>;
}

export interface SafeDecideDecisionOptions extends DecisionOptions {
  /** Select one of the checkpoint's certified selective-action rules. Defaults to 0.1. */
  alpha?: SafeDecideAlpha;
}

export interface SafeDecideDecisionAnswer extends DecisionAnswer {
  readonly confidence: number;
  readonly action: SafeDecideAction;
}

export interface SafeDecideDecisionResult {
  readonly tokenCount: number;
  readonly outputTokenCount: 0;
  readonly truncated?: true;
  readonly truncatedQuestions?: readonly string[];
  readonly guarantee: SafeDecideGuarantee;
  answers: SafeDecideDecisionAnswer[];
}

export interface ModelProgress {
  loadedBytes: number;
  totalBytes: number;
  fraction: number;
  phase?: 'bundled-assets' | 'upstream-source' | 'asset-verification';
}

export type ModelName =
  | 'experimental-laya-multilingual-fp16'
  | 'experimental-laya-typed-decisions-fp16'
  | 'experimental-xdecision-fp16'
  | 'experimental-gliner2.5-decide-mobile-q4f16'
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
  | 'kev-0.6b-q4f16'
  | 'experimental-kev-0.6b-q4f16-upstream'
  | 'kev-0.6b-reflexes-q4f16'
  | 'open-jev-deberta-v3-large-q4f16'
  | 'mapika-0.8b-f16'
  | 'mapika-0.8b-q4f16'
  | 'kev-0.8b-f16'
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
  | 'experimental-d1-omni-600m-q8'
  | 'experimental-openjev-verdict-151m-fp16';

type PrefetchModelName = Exclude<
  ModelName,
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
  | 'experimental-d1-omni-600m-q8'
  | 'experimental-openjev-verdict-151m-fp16'
  | 'experimental-kev-0.6b-q4f16-upstream'
  | 'experimental-laya-multilingual-fp16'
  | 'experimental-laya-typed-decisions-fp16'
  | 'experimental-xdecision-fp16'
  | 'experimental-gliner2.5-decide-mobile-q4f16'
>;

export type ModelLimit =
  | Readonly<{resource: 'tokens'; scope: 'request' | 'question' | 'state'; max: number; description: string}>
  | Readonly<{resource: 'characters'; scope: 'request'; max: number; description: string}>
  | Readonly<{resource: 'options'; scope: 'question' | 'request'; min: number; max?: number}>
  | Readonly<{resource: 'questions'; scope: 'request'; min: number; max?: number}>;

export interface TextChoiceCapability {
  /** The library implements this path; see verification for actual GPU evidence. */
  readonly status: 'implemented';
  /** Imajev can return an explicit abstention; other current routes cannot. */
  readonly abstention: 'implemented' | 'not-implemented';
  readonly limits: readonly ModelLimit[];
  /** A packed request can preserve cross-question context; independent rows are separate model calls. */
  readonly handling: 'packed-request' | 'independent-questions' | 'independent-options';
  /** Meaning of DecisionResult.tokenCount for this model. */
  readonly tokenCount:
    | 'packed-request'
    | 'sum-question-rows'
    | 'max-question-rows'
    | 'shared-prefix-and-branches'
    | 'sum-option-pairs';
  readonly score: Readonly<{
    basis: 'choices' | 'conditional-known-choices' | 'ordered-levels';
    temperature:
      | number
      | Readonly<{byOptionCount: readonly Readonly<{min: number; max: number; value: number}>[]}>
      | Readonly<{
          byType: Readonly<Record<'choice' | 'score' | 'noul', number>>;
          byQuestionTypeAndOptionCount: Readonly<Record<string, number>>;
        }>
      | Readonly<{
          /** Keys count substantive candidates plus the abstention candidate. */
          byCandidateCount: Readonly<Record<string, number>>;
          fallback: number;
        }>;
    temperatureAppliedBy: 'decoder' | 'model';
    calibration: 'not-guaranteed';
  }>;
}

export type AssetIntegrity = Readonly<{
  location: 'separate' | 'package-and-pinned-upstream';
  requiredFiles: readonly string[];
  additionalFiles: string;
}> &
  (
    | Readonly<{integrity: 'manifest-declared'}>
    | Readonly<{integrity: 'pinned-files'}>
    | Readonly<{integrity: 'pinned-manifest'; manifestSha256: string}>
    | Readonly<{
        integrity: 'candidate-pinned-manifest-and-shards';
        manifestSha256: string;
        sourceDataSha256: string;
        candidateWasmSha256: string;
      }>
  );

export interface SelectiveActionCapability {
  readonly status: 'implemented';
  readonly actions: readonly SafeDecideAction[];
  readonly calibration: 'exchangeable-data';
  readonly levels: readonly Readonly<{alpha: number; delta: number; threshold: number | null}>[];
}

/** Immutable package metadata. This is not a GPU compatibility or model-quality guarantee. */
export interface ModelDescription {
  readonly model: ModelName;
  readonly precision: 'q4' | 'q4f16' | 'f16' | 'f32' | 'mixed-f16-q4f16' | 'int8' | 'q8';
  /** Expected checkpoint provenance; see assets.integrity for what the loader actually pins. */
  readonly checkpoint: Readonly<{
    repository: string;
    revision: string;
    weightSha256?: string;
    graphSha256?: string;
    sourceRepository?: string;
    sourceRevision?: string;
  }>;
  readonly assets: AssetIntegrity;
  readonly licenses: Readonly<{
    project: 'Apache-2.0';
    modelTerms: string;
    modelCard: string;
    modelLicenseUrl?: string;
    sourceModelCard?: string;
  }>;
  readonly backend:
    | Readonly<{
        name: 'tvm-webgpu';
        requiredFeatures: readonly ['shader-f16'];
        requiredLimits: Readonly<Record<string, number>>;
        requiresNonFallbackAdapter: boolean;
      }>
    | Readonly<{
        name: 'onnxruntime-web-wasm';
        executionProvider: 'wasm';
        intraOpNumThreads: 1;
        requiredFeatures: readonly [];
        requiredLimits: Readonly<Record<string, number>>;
        requiresNonFallbackAdapter: false;
      }>
    | Readonly<{
        name: 'onnxruntime-web-webgpu';
        executionProvider: 'webgpu';
        /** ONNX Runtime may execute nodes unsupported by WebGPU on its CPU/WASM provider. */
        cpuFallback: 'runtime-managed';
        requiredFeatures: readonly [];
        requiredLimits: Readonly<Record<string, number>>;
        requiresNonFallbackAdapter: true;
      }>
    | Readonly<{name: 'javascript'}>;
  readonly features: Readonly<{
    textChoice: TextChoiceCapability;
    /** Native typed text outputs exposed by a model-specific method. */
    typedTextQuestionTypes?: readonly ('choice' | 'noul' | 'score' | 'span')[];
    /** Model-provided selective action metadata; it is not a general calibration guarantee. */
    selectiveAction?: SelectiveActionCapability;
    /** Metadata only: there is no image-choice method in the current package. */
    imageChoice: Readonly<{status: 'not-implemented'}>;
    /** Metadata only: there is no rating method in the current package. */
    rating: Readonly<{status: 'not-implemented'}>;
  }>;
  readonly verification: Readonly<{
    currentBundleGpu: 'not-applicable' | 'not-revalidated' | 'verified';
    currentBundleCpuWasm: 'not-applicable' | 'not-revalidated' | 'verified';
    currentBundleJavascript: 'not-applicable' | 'not-revalidated' | 'verified';
    priorEvidence: string;
  }>;
}

interface CommonLoadOptions {
  /** URL of a directory containing the selected model assets. */
  assetsUrl: string;
  /** Cache verified model files for this origin. Defaults to true. */
  cache?: boolean;
  /** Use only a previously cached manifest and verified assets; never fetch. Defaults to false. */
  offline?: boolean;
  /** A started cache write may finish after cancellation; without Web Locks a pending manifest write can delay cancellation. */
  signal?: AbortSignal;
  onProgress?: (progress: ModelProgress) => void;
}

/** Model family defaults to kev-0.6b-q4f16. H2O is an online-only experimental candidate. */
export type StandardLoadOptions =
  | (CommonLoadOptions &
      (
        | {
            model: 'open-jev-deberta-v3-large-q4f16';
            /** Experimental: declared bytes for one additional shard read ahead; defaults to 0. Not a total-memory limit. */
            maxPrefetchBytes?: number;
            /** Experimental GPU result readback; defaults to serial. Packed uses pinned TVM internals. */
            debertaReadback?: 'serial' | 'packed';
          }
        | {
            model: 'experimental-tinydecide-10m-q4';
            maxPrefetchBytes?: 0;
            debertaReadback?: undefined;
          }
        | {model?: PrefetchModelName; debertaReadback?: undefined; maxPrefetchBytes?: number}
        | {model: 'experimental-strands-qwen-2b-fp16'; debertaReadback?: undefined; maxPrefetchBytes?: 0 | undefined}
      ))
  | TamevMicroLoadOptions
  | SafeDecideLoadOptions
  | CrivoLoadOptions
  | LayaLoadOptions
  | LayaTypedDecisionsLoadOptions
  | XDecisionLoadOptions
  | GlinerMobileLoadOptions
  | D1OmniLoadOptions
  | VerdictLoadOptions
  | TamevNanoLoadOptions
  | TamevNanoFlashLoadOptions
  | SkejjaInt8LoadOptions
  | SkejjaInt8DynLoadOptions
  | EttinLoadOptions
  | {
      model:
        | 'experimental-nari-decision-4b-q4f16'
        | 'experimental-nari-decision-4b-mixed16-q4f16'
        | 'experimental-h2o-lightning-4b-q4f16'
        | 'experimental-h2o-lightning-4b-q4f16-dlight'
        | 'experimental-h2o-lightning-4b-q4f16-dual'
        | 'experimental-pplx-decider-flash-v1.2-q4f16-dlight'
        | 'experimental-mapika-4b-q4f16'
        | 'experimental-mapika-4b-q4f16-dlight'
        | 'experimental-mapika-4b-q4f16-dlight-all'
        | 'experimental-elephant-9b-q4f16-dlight'
        | 'experimental-elephant-9b-q4f16-dlight-512'
        | 'experimental-clef-9b-q4f16-selected';
      assetsUrl: string;
      signal?: AbortSignal | undefined;
      /** These experimental models do not cache, including when omitted. */
      cache?: false | undefined;
      offline?: false | undefined;
      maxPrefetchBytes?: 0 | undefined;
      onProgress?: undefined;
      debertaReadback?: undefined;
    }
  | {
      model: 'experimental-opendecider-nano-fp16';
      assetsUrl: string;
      signal?: AbortSignal | undefined;
      cache?: false | undefined;
      offline?: false | undefined;
      maxPrefetchBytes?: 0 | undefined;
      onProgress?: (progress: ModelProgress) => void;
      debertaReadback?: undefined;
    }
  | StrandsGemma4LoadOptions;

export interface StrandsGemma4LoadOptions {
  model: 'experimental-strands-gemma4-e2b-fp16';
  assetsUrl: string;
  signal?: AbortSignal;
  /** This experimental model reads each verified asset directly and does not use the browser cache. */
  cache?: false;
  offline?: false;
  maxPrefetchBytes?: 0;
  onProgress?: (progress: ModelProgress) => void;
  debertaReadback?: undefined;
}

export interface DecimaSmallLoadOptions {
  model: 'experimental-decima-small-int8';
  assetsUrl: string;
  cache?: boolean;
  offline?: boolean;
  signal?: AbortSignal;
  onProgress?: (progress: ModelProgress) => void;
  maxPrefetchBytes?: 0;
}

export interface DecimaBaseCleanLoadOptions extends Omit<DecimaSmallLoadOptions, 'model'> {
  model: 'experimental-decima-base-clean-int8';
}

export interface DecimaAgentLoadOptions extends Omit<DecimaSmallLoadOptions, 'model'> {
  model: 'experimental-decima-agent-int8';
}

export type DecimaLoadOptions = DecimaSmallLoadOptions | DecimaBaseCleanLoadOptions | DecimaAgentLoadOptions;

export interface TamevMicroLoadOptions extends Omit<CommonLoadOptions, 'assetsUrl'> {
  model: 'experimental-tamev-micro-minilm-int8';
  /** Optional trusted mirror; defaults to the model's pinned public revision. */
  assetsUrl?: string;
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface TamevNanoLoadOptions extends Omit<CommonLoadOptions, 'assetsUrl'> {
  model: 'experimental-tamev-nano-tinybert-int8';
  /** Optional trusted mirror; defaults to the model's pinned public revision. */
  assetsUrl?: string;
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface TamevNanoFlashLoadOptions extends CommonLoadOptions {
  model: 'experimental-tamev-nano-flash-tinybert-int8';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface SkejjaInt8LoadOptions extends CommonLoadOptions {
  model: 'experimental-skejja-edge32k-ft6-int8';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface SkejjaInt8DynLoadOptions extends CommonLoadOptions {
  model: 'experimental-skejja-edge32k-ft6-int8dyn';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface EttinLoadOptions extends CommonLoadOptions {
  model: 'experimental-ettin-68m-int8';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface LayaLoadOptions extends CommonLoadOptions {
  model: 'experimental-laya-multilingual-fp16';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface LayaTypedDecisionsLoadOptions extends CommonLoadOptions {
  model: 'experimental-laya-typed-decisions-fp16';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface XDecisionLoadOptions extends CommonLoadOptions {
  model: 'experimental-xdecision-fp16';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface GlinerMobileLoadOptions extends CommonLoadOptions {
  model: 'experimental-gliner2.5-decide-mobile-q4f16';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface D1OmniLoadOptions extends CommonLoadOptions {
  model: 'experimental-d1-omni-600m-q8';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export interface VerdictLoadOptions extends CommonLoadOptions {
  model: 'experimental-openjev-verdict-151m-fp16';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export type D1OmniSystemOneQuestion =
  | Readonly<{
      type: 'choice';
      instructions: string;
      criteria: readonly string[] | Readonly<Record<string, DecisionJsonValue>>;
    }>
  | Readonly<{type: 'score'; instructions: string; criteria: readonly DecisionJsonValue[]}>
  | Readonly<{
      type: 'noul';
      instructions: string;
      criteria?: Readonly<{
        true?: DecisionJsonValue;
        false?: DecisionJsonValue;
        yes?: DecisionJsonValue;
        no?: DecisionJsonValue;
      }>;
    }>;

export type D1OmniSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      confidence: number;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{
      type: 'score';
      /** d1-omni's score is the probability-weighted mean of ordered levels. */
      score: number;
      expected: number;
      mostLikelyLevel: number;
      confidence: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{
      type: 'noul';
      /** Probability that the statement is true; this is not an abstention. */
      noul: number;
      confidence: number;
      probabilities: Readonly<{true: number; false: number}>;
    }>;

export interface D1OmniSystemOneResult {
  readonly model: 'experimental-d1-omni-600m-q8';
  readonly tokenCount: number;
  readonly outputTokenCount: 0;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, D1OmniSystemOneAnswer>>;
}

export type VerdictSystemOneQuestion =
  | Readonly<{
      type: 'choice';
      instructions: string;
      /** Stable option IDs map to model-facing natural-language descriptions. */
      criteria: Readonly<Record<string, string>>;
    }>
  | Readonly<{
      type: 'score';
      instructions: string;
      /** Ordered from lowest to highest; values must be finite and strictly increasing. */
      criteria: readonly Readonly<{id: string; description: string; value: number}>[];
    }>
  | Readonly<{
      type: 'noul';
      proposition: string;
      semantics: 'conditional_on_sufficient_evidence_v2';
    }>;

export type VerdictCalibrationSource = 'global' | 'per-cardinality';

type VerdictAnswerFields = Readonly<{
  isAbstention: boolean;
  probabilities: Readonly<Record<string, number>>;
  logits: Readonly<Record<string, number>>;
  calibrationStatus: 'calibrated_for_scope';
  calibrationTemperature: number;
  calibrationSource: VerdictCalibrationSource;
}>;

export type VerdictSystemOneAnswer =
  | (VerdictAnswerFields &
      Readonly<{
        type: 'choice';
        /** The selected option ID, or `__insufficient_evidence__` when abstaining. */
        choice: string;
        selectedProbability: number;
        concentration: number;
      }>)
  | (VerdictAnswerFields &
      Readonly<{
        type: 'score';
        selectedLevelId: string;
        selectedValue: number | null;
        /** Expected rubric value conditional on not abstaining; null on abstention. */
        expectedScore: number | null;
        abstentionProbability: number;
      }>)
  | (VerdictAnswerFields &
      Readonly<{
        type: 'noul';
        selectedOutcome: 'true' | 'false' | '__insufficient_evidence__';
        /** P(true | sufficient evidence); null when the model abstains. */
        pTrueGivenSufficientEvidence: number | null;
        pInsufficientEvidence: number;
      }>);

export interface VerdictSystemOneResult {
  readonly model: 'experimental-openjev-verdict-151m-fp16';
  readonly tokenCount: number;
  readonly outputTokenCount: 0;
  readonly forwardPassCount: 1;
  readonly truncated?: true;
  readonly truncatedQuestions?: readonly string[];
  readonly answers: Readonly<Record<string, VerdictSystemOneAnswer>>;
}

export interface VerdictDecisionAnswer extends DecisionAnswer {
  /** True when `selected` is the reserved abstention ID and `selectedIndex` is -1. */
  readonly abstained: boolean;
  readonly abstentionProbability: number;
  readonly selectedProbability: number;
}

export interface VerdictDecisionResult extends Omit<DecisionResult, 'answers'> {
  readonly answers: VerdictDecisionAnswer[];
}

export interface TinyDecideLoadOptions extends CommonLoadOptions {
  model: 'experimental-tinydecide-10m-q4';
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}

export type OpenDeciderNanoLoadOptions = Extract<StandardLoadOptions, {model: 'experimental-opendecider-nano-fp16'}>;

/** Imajev's pinned experimental assets are currently online-only. */
export interface ImajevLoadOptions {
  model: 'experimental-imajev-4b-candidate';
  assetsUrl: string;
  signal?: AbortSignal | undefined;
  cache?: false | undefined;
  offline?: false | undefined;
  maxPrefetchBytes?: 0 | undefined;
  onProgress?: undefined;
  debertaReadback?: undefined;
}

export type MapikaLoadOptions<Model extends MapikaSystemOneModelName = MapikaSystemOneModelName> = CommonLoadOptions & {
  model: Model;
  maxPrefetchBytes?: number;
  debertaReadback?: undefined;
};

export type Kev08LoadOptions = CommonLoadOptions & {
  model: 'kev-0.8b-f16';
  maxPrefetchBytes?: number;
  debertaReadback?: undefined;
};

export type LoadOptions =
  | StandardLoadOptions
  | ImajevLoadOptions
  | MapikaLoadOptions
  | Kev06LoadOptions
  | KevUpstreamLoadOptions
  | KevReflexesLoadOptions
  | Kev08LoadOptions
  | DecimaLoadOptions
  | TinyDecideLoadOptions
  | TamevMicroLoadOptions
  | TamevNanoLoadOptions
  | TamevNanoFlashLoadOptions
  | SkejjaInt8LoadOptions
  | SkejjaInt8DynLoadOptions
  | EttinLoadOptions
  | SafeDecideLoadOptions
  | CrivoLoadOptions
  | LayaTypedDecisionsLoadOptions
  | GlinerMobileLoadOptions;

export type Kev06LoadOptions = CommonLoadOptions & {
  model?: 'kev-0.6b-q4f16';
  maxPrefetchBytes?: number;
  /** Optional cap on the complete packed token sequence; omitted means no aggregate token cap. */
  maxInputTokens?: number;
  /** Optional cap on rendered prompt text in UTF-16 code units; omitted means no character cap. */
  maxInputCharacters?: number;
  debertaReadback?: undefined;
};

export type KevUpstreamLoadOptions = Omit<CommonLoadOptions, 'assetsUrl'> & {
  model: 'experimental-kev-0.6b-q4f16-upstream';
  /** The upstream source is pinned in the package; this route does not accept an asset mirror. */
  assetsUrl?: never;
  maxPrefetchBytes?: 0;
  /** Optional cap on the complete packed token sequence; omitted means no aggregate token cap. */
  maxInputTokens?: number;
  /** Optional cap on rendered prompt text in UTF-16 code units; omitted means no character cap. */
  maxInputCharacters?: number;
  debertaReadback?: undefined;
};

export type KevReflexesLoadOptions = CommonLoadOptions & {
  model: 'kev-0.6b-reflexes-q4f16';
  maxPrefetchBytes?: number;
  /** Optional cap on the complete packed token sequence; omitted means no aggregate token cap. */
  maxInputTokens?: number;
  /** Optional cap on rendered prompt text in UTF-16 code units; omitted means no character cap. */
  maxInputCharacters?: number;
  debertaReadback?: undefined;
};

export type DebertaLoadOptions = CommonLoadOptions & {
  model: 'open-jev-deberta-v3-large-q4f16';
  maxPrefetchBytes?: number;
  debertaReadback?: 'serial' | 'packed';
};

export interface DecisionModel {
  /** Calls are queued; input is tokenized and snapshotted when submitted. */
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<DecisionResult>;
  /** Queue release of reusable runtime memory or encoded model caches; later calls may allocate them again. */
  releaseCachedMemory(): Promise<void>;
  /** Wait for submitted calls, then attempt to release WebGPU resources; uncertain cleanup can retain them. */
  dispose(): Promise<void>;
}

/** A loaded model exposes the same immutable descriptor used for routing. */
export interface LoadedDecisionModel extends DecisionModel {
  readonly description: ModelDescription;
}

export interface LoadedTinyDecideModel extends LoadedDecisionModel {
  /** TinyDecide's native choice, Noul, Score and Span output contract. */
  answer(request: TinyDecideRequest, options?: TinyDecideDecisionOptions): Promise<TinyDecideResult>;
  /** Generic choice decisions, optionally adjusted by native per-question prototypes. */
  decide(request: DecisionRequest, options?: TinyDecideDecisionOptions): Promise<DecisionResult>;
  /** Build native correction prototypes from labeled prior answers; returns null for Span or empty examples. */
  makeProtos(
    type: TinyDecideQuestion['type'],
    lists: readonly (readonly TinyDecideCorrectionExample[])[],
    center?: ArrayLike<number> | null
  ): TinyDecidePrototypes | null;
  /** Named native typed questions, including Span extraction. */
  systemOne(
    state: string,
    questions: Readonly<Record<string, TinyDecideSystemOneQuestion>>,
    options?: TinyDecideDecisionOptions
  ): Promise<TinyDecideSystemOneResult>;
}

export interface LoadedKev06DecisionModel extends LoadedDecisionModel {
  /** Kev's Choice, Noul, and ordered Score contract, evaluated in one packed request. */
  systemOne(
    state: KevSystemOneJsonValue,
    questions: KevSystemOneQuestions,
    options?: DecisionOptions
  ): Promise<KevSystemOneResult<'kev-0.6b-q4f16'>>;
}

export interface LoadedKevUpstreamDecisionModel extends LoadedDecisionModel {
  /** Experimental q2-k128 Kev route; uses the same Choice, Noul, and ordered Score protocol. */
  systemOne(
    state: KevSystemOneJsonValue,
    questions: KevSystemOneQuestions,
    options?: DecisionOptions
  ): Promise<KevSystemOneResult<'experimental-kev-0.6b-q4f16-upstream'>>;
}

export interface LoadedKevReflexesDecisionModel extends LoadedDecisionModel {
  /** The upstream Kev typed Choice, Noul, and ordered Score contract. */
  systemOne(
    state: KevSystemOneJsonValue,
    questions: KevSystemOneQuestions,
    options?: DecisionOptions
  ): Promise<KevSystemOneResult<'kev-0.6b-reflexes-q4f16'>>;
}

export interface LoadedKev08DecisionModel extends LoadedDecisionModel {
  /** Kev 0.8B's Choice, Score, and Noul schemas; each question runs as its own pointer-head row. */
  systemOne(
    state: DecisionJsonValue,
    questions: Readonly<Record<string, Kev08SystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<KevSystemOneResult<'kev-0.8b-f16'>>;
}

export interface LoadedDebertaDecisionModel extends LoadedDecisionModel {
  /** Open-Jev's Choice, Noul, and ordered Score contract in one packed request. */
  systemOne(
    state: string,
    questions: Readonly<Record<string, DebertaSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<DebertaSystemOneResult>;
}

export interface LoadedOpenDeciderNanoModel extends LoadedDecisionModel {
  /** OpenDecider's Choice, Score, and Noul schema, decoded from the model's option probabilities. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, OpenDeciderTypedQuestion>>,
    options?: DecisionOptions
  ): Promise<OpenDeciderNanoSystemOneResult>;
}

export type StrandsGemma4Content = string | readonly DecisionJsonValue[] | Readonly<{[key: string]: DecisionJsonValue}>;

/** Pinned Strands Content (string, object, or list), plus null for a bare option label. */
export type StrandsGemma4Description = StrandsGemma4Content | null;

export type StrandsGemma4SystemOneQuestion =
  | Readonly<{
      type: 'choice';
      instructions: StrandsGemma4Content;
      criteria: readonly string[] | Readonly<Record<string, StrandsGemma4Description>>;
    }>
  | Readonly<{type: 'score'; instructions: StrandsGemma4Content; criteria: readonly string[]}>
  | Readonly<{
      type: 'noul';
      instructions: StrandsGemma4Content;
      criteria?: Readonly<{true?: StrandsGemma4Description; false?: StrandsGemma4Description}> | null;
    }>;

export type StrandsGemma4SystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
    }>
  | Readonly<{
      type: 'score';
      /** Probability weighted mean level, rounded to the checkpoint's four decimal places. */
      score: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
    }>
  | Readonly<{type: 'noul'; noul: number}>;

export interface StrandsGemma4SystemOneResult {
  readonly model: 'experimental-strands-gemma4-e2b-fp16';
  /** Sum of independently tokenized state/question rows. */
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, StrandsGemma4SystemOneAnswer>>;
}

export interface LoadedStrandsGemma4Model extends LoadedDecisionModel {
  /** Native Strands choice, ordered score, and Noul protocol. Rows exceeding 4,096 tokens are rejected without truncation. */
  systemOne(
    state: StrandsGemma4Content,
    questions:
      | Readonly<Record<string, StrandsGemma4SystemOneQuestion>>
      | ReadonlyMap<string, StrandsGemma4SystemOneQuestion>,
    options?: DecisionOptions
  ): Promise<StrandsGemma4SystemOneResult>;
}

export interface SafeDecideLoadOptions {
  model: 'experimental-safedecide-140m-fp32';
  assetsUrl: string;
  /** Browser caching is opt-in because the FP32 graph is 563 MB. */
  cache?: boolean;
  /** Offline mode requires `cache: true`. */
  offline?: boolean;
  signal?: AbortSignal;
  onProgress?: (progress: ModelProgress) => void;
}

export interface LoadedSafeDecideModel extends LoadedDecisionModel {
  /** SafeDecide's native Choice, ordered Score, Noul, and selective-action contract. */
  systemOne(
    state: SafeDecideState,
    questions: Readonly<Record<string, SafeDecideSystemOneQuestion>>,
    options?: SafeDecideDecisionOptions
  ): Promise<SafeDecideSystemOneResult>;
  /** Generic option-choice API with SafeDecide's calibrated confidence and action fields. */
  decide(request: DecisionRequest, options?: SafeDecideDecisionOptions): Promise<SafeDecideDecisionResult>;
}

export type CrivoModelName = 'experimental-crivo-32m-v2-fp32' | 'experimental-crivo-32m-v2-int8';

export type CrivoSystemOneQuestion =
  | Readonly<{
      type: 'choice';
      instructions?: DecisionJsonValue;
      criteria: Readonly<Record<string, DecisionJsonValue>>;
    }>
  | Readonly<{
      type: 'score';
      instructions?: DecisionJsonValue;
      criteria: readonly DecisionJsonValue[] | Readonly<Record<string, DecisionJsonValue>>;
    }>
  | Readonly<{
      type: 'noul';
      instructions?: DecisionJsonValue;
      criteria?: Readonly<{true?: DecisionJsonValue; false?: DecisionJsonValue}>;
    }>;

export type CrivoSystemOneAnswer =
  | Readonly<{type: 'choice'; choice: string; probabilities: Readonly<Record<string, number>>}>
  | Readonly<{type: 'score'; score: number; probabilities: Readonly<Record<string, number>>}>
  | Readonly<{type: 'noul'; noul: number}>;

export interface CrivoSystemOneResult {
  readonly model: CrivoModelName;
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, CrivoSystemOneAnswer>>;
}

export interface CrivoLoadOptions {
  model: CrivoModelName;
  assetsUrl: string;
  /** Browser caching is enabled by default. */
  cache?: boolean;
  /** Offline mode requires a previously cached set of verified assets and cache: true. */
  offline?: boolean;
  signal?: AbortSignal;
  onProgress?: (progress: ModelProgress) => void;
}

export interface LoadedCrivoModel extends LoadedDecisionModel {
  /** Crivo's native Choice, ordered Score and binary Noul contract. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, CrivoSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<CrivoSystemOneResult>;
}

export interface DecimaSystemOneChoiceQuestion {
  readonly type: 'choice';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria: readonly DecisionJsonValue[] | Readonly<Record<string, DecisionJsonValue>>;
  readonly ordinal?: boolean;
  readonly label_blind?: boolean;
}

export interface DecimaSystemOneScoreQuestion {
  readonly type: 'score';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria: readonly DecisionJsonValue[];
}

export interface DecimaSystemOneNoulQuestion {
  readonly type: 'noul';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria?: Readonly<{true?: DecisionJsonValue; false?: DecisionJsonValue}> | null;
}

export interface DecimaSystemOneRankQuestion {
  readonly type: 'rank';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria: readonly DecisionJsonValue[] | Readonly<Record<string, DecisionJsonValue>>;
}

export type DecimaSystemOneQuestion =
  | DecimaSystemOneChoiceQuestion
  | DecimaSystemOneScoreQuestion
  | DecimaSystemOneNoulQuestion
  | DecimaSystemOneRankQuestion;

export type DecimaSystemOneAnswer =
  | Readonly<{type: 'choice'; choice: string; confidence: number; probabilities: Readonly<Record<string, number>>}>
  | Readonly<{
      type: 'score';
      score: number;
      confidence: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{type: 'noul'; noul: number}>
  | Readonly<{type: 'rank'; probabilities: Readonly<Record<string, number>>; order: readonly string[]}>;

export interface DecimaSystemOneResult {
  readonly model:
    | 'experimental-decima-small-int8'
    | 'experimental-decima-base-clean-int8'
    | 'experimental-decima-agent-int8';
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, DecimaSystemOneAnswer>>;
}

export interface DecimaDecisionOptions extends DecisionOptions {
  /** Language code used for Arabic and Persian text normalization; other codes pass text through. */
  readonly language?: string;
}

export interface LoadedDecimaSmallModel extends LoadedDecisionModel {
  /** Decima Choice, Noul, Score and independent-probability Rank routes. */
  decide(request: DecisionRequest, options?: DecimaDecisionOptions): Promise<DecisionResult>;
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, DecimaSystemOneQuestion>>,
    options?: DecimaDecisionOptions
  ): Promise<DecimaSystemOneResult>;
}

export type LoadedDecimaBaseCleanModel = LoadedDecimaSmallModel;

export type DecimaAgentSystemOneQuestion = Exclude<DecimaSystemOneQuestion, DecimaSystemOneRankQuestion>;
export type DecimaAgentSystemOneAnswer = Exclude<DecimaSystemOneAnswer, {type: 'rank'}>;

export interface DecimaAgentSystemOneResult {
  readonly model: 'experimental-decima-agent-int8';
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, DecimaAgentSystemOneAnswer>>;
}

export interface TamevMicroChoiceQuestion {
  readonly type: 'choice';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria: Readonly<Record<string, DecisionJsonValue>>;
}

export interface TamevMicroNoulQuestion {
  readonly type: 'noul';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria?: Readonly<{
    true?: DecisionJsonValue;
    false?: DecisionJsonValue;
    yes?: DecisionJsonValue;
    no?: DecisionJsonValue;
  }> | null;
}

export interface TamevMicroScoreQuestion {
  readonly type: 'score';
  readonly instructions?: DecisionJsonValue | null;
  /** Ordered from the lowest level to the highest. TAMEV returns the probability-weighted expected level. */
  readonly criteria: readonly DecisionJsonValue[];
}

export type TamevMicroSystemOneQuestion = TamevMicroChoiceQuestion | TamevMicroNoulQuestion | TamevMicroScoreQuestion;

export type TamevMicroSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      confidence: number;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{type: 'noul'; noul: number}>
  | Readonly<{
      type: 'score';
      /** Probability-weighted expected level, not the modal level. */
      score: number;
      confidence: number;
      legend: Readonly<Record<string, DecisionJsonValue>>;
      probabilities: Readonly<Record<string, number>>;
    }>;

export interface TamevSystemOneResult<Model extends TamevModelName> {
  readonly model: Model;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, TamevMicroSystemOneAnswer>>;
}

export type TamevModelName =
  | 'experimental-tamev-micro-minilm-int8'
  | 'experimental-tamev-nano-tinybert-int8'
  | 'experimental-tamev-nano-flash-tinybert-int8';
export type TamevMicroSystemOneResult = TamevSystemOneResult<'experimental-tamev-micro-minilm-int8'>;
export type TamevNanoSystemOneQuestion = TamevMicroSystemOneQuestion;
export type TamevNanoSystemOneAnswer = TamevMicroSystemOneAnswer;
export type TamevNanoSystemOneResult = TamevSystemOneResult<'experimental-tamev-nano-tinybert-int8'>;
export type TamevNanoFlashSystemOneQuestion = TamevMicroSystemOneQuestion;
export type TamevNanoFlashSystemOneAnswer = TamevMicroSystemOneAnswer;
export type TamevNanoFlashSystemOneResult = TamevSystemOneResult<'experimental-tamev-nano-flash-tinybert-int8'>;

export interface LoadedTamevModel<Model extends TamevModelName> extends LoadedDecisionModel {
  /** TAMEV's choice helper and native Choice, Noul and Score System One protocol. */
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<DecisionResult>;
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, TamevMicroSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<TamevSystemOneResult<Model>>;
}

export type LoadedTamevMicroModel = LoadedTamevModel<'experimental-tamev-micro-minilm-int8'>;
export type LoadedTamevNanoModel = LoadedTamevModel<'experimental-tamev-nano-tinybert-int8'>;
export type LoadedTamevNanoFlashModel = LoadedTamevModel<'experimental-tamev-nano-flash-tinybert-int8'>;

export type SkejjaModelName = 'experimental-skejja-edge32k-ft6-int8' | 'experimental-skejja-edge32k-ft6-int8dyn';
export interface SkejjaChoiceQuestion {
  readonly type: 'choice';
  readonly instructions?: DecisionJsonValue;
  readonly criteria: Readonly<Record<string, DecisionJsonValue>>;
  readonly [key: string]: unknown;
}

export interface SkejjaScoreQuestion {
  readonly type: 'score';
  readonly instructions?: DecisionJsonValue;
  readonly criteria: readonly DecisionJsonValue[];
  readonly [key: string]: unknown;
}

export interface SkejjaNoulQuestion {
  readonly type: 'noul';
  readonly instructions?: DecisionJsonValue;
  readonly criteria?: Readonly<Record<string, DecisionJsonValue>> | null;
  readonly [key: string]: unknown;
}

export type SkejjaSystemOneQuestion = SkejjaChoiceQuestion | SkejjaScoreQuestion | SkejjaNoulQuestion;
export type SkejjaSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      /** Normalized probability above uniform; it is not a calibrated confidence estimate. */
      confidence: number;
    }>
  | Readonly<{
      type: 'score';
      /** Probability-weighted expected ordinal level, starting at 0. */
      score: number;
      probabilities: Readonly<Record<string, number>>;
      legend: Readonly<Record<string, DecisionJsonValue>>;
      confidence: number;
    }>
  | Readonly<{type: 'noul'; /** Calibrated probability that the answer is true. */ noul: number}>;

export interface SkejjaSystemOneResult<Model extends SkejjaModelName> {
  readonly model: Model;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, SkejjaSystemOneAnswer>>;
}

export interface SkejjaCalibrationExample {
  readonly state: DecisionJsonValue;
  readonly answer: boolean | 0 | 1;
}

export interface SkejjaQuestionCalibration {
  readonly a: number;
  readonly b: number;
  readonly examples: number;
  readonly positives: number;
}

export interface LoadedSkejjaModel<Model extends SkejjaModelName> extends LoadedDecisionModel {
  /** Skejja's typed Choice, Noul, and ordinal Score outputs. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, SkejjaSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<SkejjaSystemOneResult<Model>>;
  /** Fit in-memory Platt calibration for one Noul question using 8 to 256 labeled states. */
  calibrateQuestion(
    question: SkejjaNoulQuestion,
    examples: readonly SkejjaCalibrationExample[],
    options?: DecisionOptions
  ): Promise<SkejjaQuestionCalibration>;
}

export type LoadedSkejjaInt8Model = LoadedSkejjaModel<'experimental-skejja-edge32k-ft6-int8'>;
export type LoadedSkejjaInt8DynModel = LoadedSkejjaModel<'experimental-skejja-edge32k-ft6-int8dyn'>;

export type EttinSystemOneQuestion =
  | OpenDeciderChoiceQuestion
  | OpenDeciderScoreQuestion
  | Omit<OpenDeciderNoulQuestion, 'criteria'>
  | Readonly<{
      type: 'noul';
      instructions: string;
      criteria?: Readonly<{true?: '' | null; false?: '' | null}> | null;
    }>;

export type EttinSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      confidence: number;
      probabilities: Readonly<Record<string, number>>;
      truncated?: true;
    }>
  | Readonly<{
      type: 'noul';
      /** Probability of true (yes); the model does not abstain. */
      noul: number;
      probabilities: Readonly<{false: number; true: number}>;
      truncated?: true;
    }>
  | Readonly<{
      type: 'score';
      /** Most likely ordered level, where zero is the lowest level. */
      score: number;
      /** Probability-weighted mean level. */
      expected: number;
      legend: Readonly<Record<string, string>>;
      confidence: number;
      probabilities: Readonly<Record<string, number>>;
      truncated?: true;
    }>;

export interface EttinSystemOneResult {
  readonly model: 'experimental-ettin-68m-int8';
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, EttinSystemOneAnswer>>;
}

export interface LoadedEttinModel extends LoadedDecisionModel {
  /** Decider Ettin's native Choice, Noul and ordered Score heads. Noul has no abstention path. */
  systemOne(
    state: string,
    questions: Readonly<Record<string, EttinSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<EttinSystemOneResult>;
}

export interface LoadedLayaModel extends LoadedDecisionModel {
  /** Laya's native Choice, ordered Score and Noul outputs. The diagnostic action head cannot abstain. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, LayaSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<LayaSystemOneResult>;
}

export interface LoadedLayaTypedDecisionsModel extends LoadedDecisionModel {
  /** Laya Typed-Decisions' native Choice, ordered Score and Noul outputs. Its action head is not abstention. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, LayaSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<LayaTypedDecisionsSystemOneResult>;
}

export interface LoadedXDecisionModel extends LoadedDecisionModel {
  /** xDecision's Choice, ordered Score and Noul outputs; action probability is not abstention. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, XDecisionSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<XDecisionSystemOneResult>;
}

export interface LoadedGlinerMobileModel extends LoadedDecisionModel {
  /** GLiNER's Choice, ordered Score and yes/no Noul outputs; its Noul head cannot abstain. */
  systemOne(
    state: string,
    questions: Readonly<Record<string, GlinerMobileSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<GlinerMobileSystemOneResult>;
}

export interface LoadedD1OmniModel extends LoadedDecisionModel {
  /** d1-omni's text-only Choice, ordered Score, and Noul outputs. Noul is a yes probability, not abstention. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, D1OmniSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<D1OmniSystemOneResult>;
}

export interface LoadedVerdictModel extends LoadedDecisionModel {
  /** Verdict's native Choice, ordered Score, and abstaining Noul contract. */
  systemOne(
    state: string,
    questions: Readonly<Record<string, VerdictSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<VerdictSystemOneResult>;
  /** Generic Choice requests use the same 24-option limit and expose abstention explicitly. */
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<VerdictDecisionResult>;
}

export interface LoadedDecimaAgentModel extends LoadedDecisionModel {
  /** Decima-agent 2.3 exposes the upstream Choice, Score and Noul question types. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, DecimaAgentSystemOneQuestion>>,
    options?: DecimaDecisionOptions
  ): Promise<DecimaAgentSystemOneResult>;
}

export interface LoadedMapikaDecisionModel<
  Model extends MapikaSystemOneModelName = MapikaSystemOneModelName
> extends LoadedDecisionModel {
  /** Mapika's Choice, isolated or listwise Score, and Noul protocol. Score levels default to independent yes/no rows. */
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, MapikaSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<MapikaSystemOneResult<Model>>;
}

export interface LoadedImajevDecisionModel {
  readonly description: ModelDescription;
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<ImajevDecisionResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

/** Implemented model routes; inclusion does not guarantee compatibility or a validated release. */
export function listDecisionModels(): readonly ModelName[];
/** Narrow an untrusted or dynamic ID before describing or loading it. */
export function isDecisionModel(value: unknown): value is ModelName;
/** Describe a route without loading assets or checking the current GPU. */
export function describeModel(model: ModelName): ModelDescription;
/** Load the selected decision model on its declared backend. */
export function loadDecisionModel(options: ImajevLoadOptions): Promise<LoadedImajevDecisionModel>;
export function loadDecisionModel(options: StrandsGemma4LoadOptions): Promise<LoadedStrandsGemma4Model>;
export function loadDecisionModel(options: TinyDecideLoadOptions): Promise<LoadedTinyDecideModel>;
export function loadDecisionModel(options: OpenDeciderNanoLoadOptions): Promise<LoadedOpenDeciderNanoModel>;
export function loadDecisionModel(options: SafeDecideLoadOptions): Promise<LoadedSafeDecideModel>;
export function loadDecisionModel(options: CrivoLoadOptions): Promise<LoadedCrivoModel>;
export function loadDecisionModel(options: DecimaSmallLoadOptions): Promise<LoadedDecimaSmallModel>;
export function loadDecisionModel(options: DecimaBaseCleanLoadOptions): Promise<LoadedDecimaBaseCleanModel>;
export function loadDecisionModel(options: DecimaAgentLoadOptions): Promise<LoadedDecimaAgentModel>;
export function loadDecisionModel(options: TamevMicroLoadOptions): Promise<LoadedTamevMicroModel>;
export function loadDecisionModel(options: TamevNanoLoadOptions): Promise<LoadedTamevNanoModel>;
export function loadDecisionModel(options: TamevNanoFlashLoadOptions): Promise<LoadedTamevNanoFlashModel>;
export function loadDecisionModel(options: SkejjaInt8LoadOptions): Promise<LoadedSkejjaInt8Model>;
export function loadDecisionModel(options: SkejjaInt8DynLoadOptions): Promise<LoadedSkejjaInt8DynModel>;
export function loadDecisionModel(options: EttinLoadOptions): Promise<LoadedEttinModel>;
export function loadDecisionModel(options: LayaLoadOptions): Promise<LoadedLayaModel>;
export function loadDecisionModel(options: LayaTypedDecisionsLoadOptions): Promise<LoadedLayaTypedDecisionsModel>;
export function loadDecisionModel(options: XDecisionLoadOptions): Promise<LoadedXDecisionModel>;
export function loadDecisionModel(options: GlinerMobileLoadOptions): Promise<LoadedGlinerMobileModel>;
export function loadDecisionModel(options: D1OmniLoadOptions): Promise<LoadedD1OmniModel>;
export function loadDecisionModel(options: VerdictLoadOptions): Promise<LoadedVerdictModel>;
export function loadDecisionModel(
  options: MapikaLoadOptions<'mapika-0.8b-f16'>
): Promise<LoadedMapikaDecisionModel<'mapika-0.8b-f16'>>;
export function loadDecisionModel(
  options: MapikaLoadOptions<'mapika-0.8b-q4f16'>
): Promise<LoadedMapikaDecisionModel<'mapika-0.8b-q4f16'>>;
export function loadDecisionModel(options: DebertaLoadOptions): Promise<LoadedDebertaDecisionModel>;
export function loadDecisionModel(options: KevReflexesLoadOptions): Promise<LoadedKevReflexesDecisionModel>;
export function loadDecisionModel(options: Kev06LoadOptions): Promise<LoadedKev06DecisionModel>;
export function loadDecisionModel(options: KevUpstreamLoadOptions): Promise<LoadedKevUpstreamDecisionModel>;
export function loadDecisionModel(options: Kev08LoadOptions): Promise<LoadedKev08DecisionModel>;
export function loadDecisionModel<
  Options extends Readonly<{
    assetsUrl: string;
    model: Exclude<ModelName, 'experimental-kev-0.6b-q4f16-upstream'>;
  }>
>(
  options: Options & Record<Exclude<keyof Options, 'assetsUrl' | 'model'>, never>
): Promise<LoadedDecisionModel | LoadedImajevDecisionModel>;
export function loadDecisionModel(options: StandardLoadOptions): Promise<LoadedDecisionModel>;
export function loadDecisionModel(options: LoadOptions): Promise<LoadedDecisionModel | LoadedImajevDecisionModel>;
/** Clear only this library's cached model files for the current origin. */
export function clearModelCache(): Promise<boolean>;
