export type SkejjaModelName = 'experimental-skejja-edge32k-ft6-int8' | 'experimental-skejja-edge32k-ft6-int8dyn';

export type DecisionJsonValue =
  | null
  | boolean
  | number
  | string
  | readonly DecisionJsonValue[]
  | Readonly<{[key: string]: DecisionJsonValue}>;

export interface DecisionQuestion {
  readonly instructions: string;
  readonly options: readonly string[];
}

export interface DecisionRequest {
  readonly state: string;
  readonly questions: readonly DecisionQuestion[];
}

export interface DecisionAnswer {
  readonly selected: string;
  readonly selectedIndex: number;
  readonly probabilities: readonly Readonly<{option: string; probability: number; logit: number}>[];
}

export interface DecisionResult {
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: readonly DecisionAnswer[];
}

export interface DecisionOptions {
  readonly signal?: AbortSignal;
}

export interface SkejjaChoiceQuestion {
  readonly type?: 'choice';
  readonly instructions?: DecisionJsonValue;
  readonly criteria: Readonly<Record<string, DecisionJsonValue>>;
}

export interface SkejjaScoreQuestion {
  readonly type: 'score';
  readonly instructions?: DecisionJsonValue;
  /** Ordered from the lowest level to the highest. */
  readonly criteria: readonly DecisionJsonValue[];
}

export interface SkejjaNoulQuestion {
  readonly type: 'noul';
  readonly instructions?: DecisionJsonValue;
  readonly criteria?: Readonly<{true?: DecisionJsonValue; false?: DecisionJsonValue}> | null;
}

export type SkejjaSystemOneQuestion = SkejjaChoiceQuestion | SkejjaScoreQuestion | SkejjaNoulQuestion;

export type SkejjaSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
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
  | Readonly<{
      type: 'noul';
      /** Probability that the answer is true. Skejja Noul is not abstention. */
      noul: number;
    }>;

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

export interface SkejjaModelDescription {
  readonly model: SkejjaModelName;
  readonly precision: 'int8';
  readonly checkpoint: Readonly<{repository: string; revision: string; graphSha256: string}>;
  readonly assets: Readonly<{
    integrity: 'pinned-files';
    location: 'separate';
    requiredFiles: readonly string[];
    additionalFiles: string;
  }>;
  readonly licenses: Readonly<{
    project: 'Apache-2.0';
    modelTerms: 'separate-review-required';
    modelCard: string;
  }>;
  readonly backend: Readonly<{
    name: 'onnxruntime-web-wasm';
    executionProvider: 'wasm';
    intraOpNumThreads: 1;
    requiredFeatures: readonly [];
    requiredLimits: Readonly<Record<string, number>>;
    requiresNonFallbackAdapter: false;
  }>;
  readonly features: Readonly<{
    textChoice: Readonly<{
      status: 'implemented';
      abstention: 'not-implemented';
      limits: readonly Readonly<Record<string, string | number>>[];
      handling: 'independent-questions';
      tokenCount: 'sum-question-rows';
      score: Readonly<{
        basis: 'choices';
        temperature: number;
        temperatureAppliedBy: 'decoder';
        calibration: 'not-guaranteed';
      }>;
    }>;
    typedTextQuestionTypes: readonly ['choice', 'noul', 'score'];
    imageChoice: Readonly<{status: 'not-implemented'}>;
    rating: Readonly<{status: 'not-implemented'}>;
  }>;
  readonly verification: Readonly<{
    currentBundleGpu: 'not-applicable';
    currentBundleCpuWasm: 'verified';
    currentBundleJavascript: 'not-applicable';
    priorEvidence: string;
  }>;
}

export interface ModelProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

export type SkejjaLoadOptions<Model extends SkejjaModelName> = Readonly<{
  model: Model;
  assetsUrl: string;
  cache?: boolean;
  offline?: boolean;
  signal?: AbortSignal;
  onProgress?: (progress: ModelProgress) => void;
  maxPrefetchBytes?: 0;
  debertaReadback?: undefined;
}>;

export interface SkejjaModel<Model extends SkejjaModelName> {
  readonly description: SkejjaModelDescription & Readonly<{model: Model}>;
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<DecisionResult>;
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, SkejjaSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<SkejjaSystemOneResult<Model>>;
  /** Fit an in-memory Platt calibration for one Noul question using 8 to 256 labeled states. */
  calibrateQuestion(
    question: SkejjaNoulQuestion,
    examples: readonly SkejjaCalibrationExample[],
    options?: DecisionOptions
  ): Promise<SkejjaQuestionCalibration>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

export type SkejjaInt8LoadOptions = SkejjaLoadOptions<'experimental-skejja-edge32k-ft6-int8'>;
export type SkejjaInt8DynLoadOptions = SkejjaLoadOptions<'experimental-skejja-edge32k-ft6-int8dyn'>;

export type ErrorCode =
  | 'INVALID_INPUT'
  | 'INVALID_ASSET'
  | 'UNSUPPORTED_ENVIRONMENT'
  | 'ASSET_LOAD_FAILED'
  | 'MODEL_LOAD_FAILED'
  | 'INFERENCE_ERROR'
  | 'SESSION_CLOSED';

export class WebDecisionError extends Error {
  constructor(code: ErrorCode, message: string, options?: {cause?: unknown});
  readonly code: ErrorCode;
}

/** List only the static and dynamic int8 Skejja routes in this package. */
export function listDecisionModels(): readonly SkejjaModelName[];
export function isDecisionModel(value: unknown): value is SkejjaModelName;
export function describeModel(model: SkejjaModelName): SkejjaModelDescription;
export function loadDecisionModel(
  options: SkejjaInt8LoadOptions
): Promise<SkejjaModel<'experimental-skejja-edge32k-ft6-int8'>>;
export function loadDecisionModel(
  options: SkejjaInt8DynLoadOptions
): Promise<SkejjaModel<'experimental-skejja-edge32k-ft6-int8dyn'>>;
export function clearModelCache(): Promise<boolean>;
