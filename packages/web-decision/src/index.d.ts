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

export interface ModelProgress {
  loadedBytes: number;
  totalBytes: number;
  fraction: number;
  phase?: 'bundled-assets' | 'upstream-source' | 'asset-verification';
}

export type ModelName =
  | 'experimental-tinydecide-10m-q4'
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
  | 'experimental-opendecider-nano-fp16';

/** Stable public IDs available through the legacy loader. */
export type StableModelId = 'kev-0.6b-q4f16';

type PrefetchModelName = Exclude<
  ModelName,
  | 'experimental-tinydecide-10m-q4'
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
  | 'experimental-kev-0.6b-q4f16-upstream'
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
    | Readonly<{name: 'javascript'}>;
  readonly features: Readonly<{
    textChoice: TextChoiceCapability;
    /** Native typed text outputs exposed by a model-specific method. */
    typedTextQuestionTypes?: readonly ('choice' | 'noul' | 'score' | 'span')[];
    /** Model-provided selective action metadata; it is not a general calibration guarantee. */
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

/** Public metadata for a stable ID; internal route names are not exposed here. */
export interface StableModelDescription extends Omit<ModelDescription, 'model'> {
  readonly model: StableModelId;
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
  | TinyDecideLoadOptions
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

export type StableKev06LoadOptions = Omit<Kev06LoadOptions, 'model' | 'assetsUrl'> & {
  modelId: 'kev-0.6b-q4f16';
  model?: undefined;
};

export type StableModelIdLoadOptions = Readonly<{
  modelId: StableModelId;
  model?: undefined;
  cache?: boolean;
  offline?: boolean;
  signal?: AbortSignal;
  onProgress?: (progress: ModelProgress) => void;
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}>;

export type StableModelLoadOptions = StableKev06LoadOptions | StableModelIdLoadOptions;

export interface TinyDecideLoadOptions extends Omit<CommonLoadOptions, 'assetsUrl'> {
  model: 'experimental-tinydecide-10m-q4';
  /** Optional trusted mirror; defaults to the model's pinned public revision. */
  assetsUrl?: string;
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
  | TinyDecideLoadOptions;

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

export interface LoadedStableKev06DecisionModel extends Omit<LoadedKev06DecisionModel, 'description'> {
  readonly description: StableModelDescription;
}

export type LoadedStableDecisionModel = LoadedStableKev06DecisionModel;

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
/** List stable public IDs. */
export function listStableModelIds(): readonly StableModelId[];
/** Narrow a dynamic value to a stable public model ID. */
export function isStableModelId(value: unknown): value is StableModelId;
/** Describe a stable public ID without exposing its internal route ID. */
export function describeStableModel(model: StableModelId): StableModelDescription;
/** Load a stable public ID. Kev reconstructs its pinned TVM assets from upstream source and does not accept assetsUrl. */
export function loadDecisionModel(options: StableKev06LoadOptions): Promise<LoadedStableKev06DecisionModel>;
export function loadDecisionModel(options: StableModelLoadOptions): Promise<LoadedStableDecisionModel>;
/** Load a model through its existing route-specific options. */
export function loadDecisionModel(options: ImajevLoadOptions): Promise<LoadedImajevDecisionModel>;
export function loadDecisionModel(options: StrandsGemma4LoadOptions): Promise<LoadedStrandsGemma4Model>;
export function loadDecisionModel(options: TinyDecideLoadOptions): Promise<LoadedTinyDecideModel>;
export function loadDecisionModel(options: OpenDeciderNanoLoadOptions): Promise<LoadedOpenDeciderNanoModel>;
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
