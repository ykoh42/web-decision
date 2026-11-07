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

export type DecisionJsonValue =
  | null
  | boolean
  | number
  | string
  | readonly DecisionJsonValue[]
  | Readonly<{[key: string]: DecisionJsonValue}>;

export interface DecisionQuestion {
  instructions: string;
  options: readonly string[];
}

export interface DecisionRequest {
  state: string;
  questions: readonly DecisionQuestion[];
}

export interface OptionScore {
  option: string;
  probability: number;
  logit: number;
}

export interface DecisionAnswer {
  selected: string;
  selectedIndex: number;
  probabilities: OptionScore[];
}

export interface DecisionResult {
  tokenCount: number;
  truncated?: boolean;
  answers: DecisionAnswer[];
}

export interface DecisionOptions {
  signal?: AbortSignal;
}

export interface ModelProgress {
  loadedBytes: number;
  totalBytes: number;
  fraction: number;
}

export type ModelName = 'kev-0.6b-q4f16' | 'experimental-tamev-nano-tinybert-int8';

export type ModelLimit =
  | Readonly<{resource: 'tokens'; scope: 'request' | 'question' | 'state'; max: number; description: string}>
  | Readonly<{resource: 'options'; scope: 'question'; min: number; max: number}>
  | Readonly<{resource: 'questions'; scope: 'request'; min: number; max?: number}>;

export interface Kev06ModelDescription {
  readonly model: 'kev-0.6b-q4f16';
  readonly precision: 'q4f16';
  readonly checkpoint: Readonly<{repository: string; revision: string}>;
  readonly assets: Readonly<{
    integrity: 'pinned-manifest';
    manifestSha256: string;
    location: 'separate';
    requiredFiles: readonly string[];
    additionalFiles: string;
  }>;
  readonly licenses: Readonly<{
    project: 'Apache-2.0';
    modelTerms: string;
    modelCard: string;
  }>;
  readonly backend: Readonly<{
    name: 'tvm-webgpu';
    requiredFeatures: readonly ['shader-f16'];
    requiredLimits: Readonly<Record<string, number>>;
    requiresNonFallbackAdapter: boolean;
  }>;
  readonly features: Readonly<{
    textChoice: Readonly<{
      status: 'implemented';
      abstention: 'not-implemented';
      limits: readonly ModelLimit[];
      handling: 'packed-request';
      tokenCount: 'packed-request';
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
    currentBundleGpu: 'not-revalidated' | 'verified';
    currentBundleCpuWasm: 'not-applicable';
    currentBundleJavascript: 'not-applicable';
    priorEvidence: string;
  }>;
}

export type TamevModelLimit =
  | Readonly<{resource: 'tokens'; scope: 'state' | 'question'; max: number; description: string}>
  | Readonly<{resource: 'options'; scope: 'question'; min: number; description: string}>
  | Readonly<{resource: 'questions'; scope: 'request'; min: number; description: string}>;

export interface TamevNanoModelDescription {
  readonly model: 'experimental-tamev-nano-tinybert-int8';
  readonly precision: 'int8';
  readonly checkpoint: Readonly<{repository: string; revision: string}>;
  readonly assets: Readonly<{
    integrity: 'pinned-per-file-sha256';
    location: 'separate';
    requiredFiles: readonly string[];
    totalBytes: number;
  }>;
  readonly licenses: Readonly<{
    project: 'Apache-2.0';
    modelTerms: 'Apache-2.0-declared-by-pinned-model-card';
    modelCard: string;
  }>;
  readonly backend: Readonly<{
    name: 'onnxruntime-web-wasm';
    executionProvider: 'wasm';
    intraOpNumThreads: 1;
  }>;
  readonly features: Readonly<{
    textChoice: Readonly<{
      status: 'implemented';
      abstention: 'not-implemented';
      limits: readonly TamevModelLimit[];
      handling: 'sequential-questions';
      tokenCount: 'sum-of-encoded-rows';
      score: Readonly<{
        basis: 'candidate-probabilities';
        temperature: number;
        temperatureAppliedBy: 'model-export';
        calibration: 'per-option-count-on-generic-decision';
      }>;
    }>;
    typedTextQuestionTypes: readonly ['choice', 'noul', 'score'];
    imageChoice: Readonly<{status: 'not-implemented'}>;
    rating: Readonly<{status: 'not-implemented'}>;
  }>;
  readonly verification: Readonly<{
    currentBundleGpu: 'not-applicable';
    currentBundleCpuWasm: 'not-revalidated' | 'verified';
    currentBundleJavascript: 'not-applicable';
    priorEvidence: string;
  }>;
}

export type ModelDescription = Kev06ModelDescription | TamevNanoModelDescription;
export type ModelDescriptionFor<Model extends ModelName> = Extract<ModelDescription, {model: Model}>;

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

export type KevSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      probabilities: Readonly<Record<string, number>>;
      probabilityEntries: readonly (readonly [key: string, probability: number])[];
      confidence: number;
    }>
  | Readonly<{
      type: 'score';
      score: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      probabilityEntries: readonly (readonly [level: string, probability: number])[];
      confidence: number;
    }>
  | Readonly<{type: 'noul'; noul: number}>;

export interface KevSystemOneResult {
  readonly model: 'kev-0.6b-q4f16';
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, KevSystemOneAnswer>>;
  readonly answerEntries: readonly (readonly [question: string, answer: KevSystemOneAnswer])[];
}

export interface Kev06LoadOptions {
  assetsUrl: string;
  model?: 'kev-0.6b-q4f16';
  cache?: boolean;
  offline?: boolean;
  signal?: AbortSignal;
  maxPrefetchBytes?: number;
  /** Optional cap on the complete packed token sequence; omitted means no aggregate token cap. */
  maxInputTokens?: number;
  /** Optional cap on rendered prompt text in UTF-16 code units; omitted means no character cap. */
  maxInputCharacters?: number;
  onProgress?: (progress: ModelProgress) => void;
  debertaReadback?: undefined;
}

export interface TamevNanoLoadOptions {
  assetsUrl: string;
  model: 'experimental-tamev-nano-tinybert-int8';
  cache?: boolean;
  offline?: boolean;
  signal?: AbortSignal;
  maxPrefetchBytes?: 0;
  onProgress?: (progress: ModelProgress) => void;
  debertaReadback?: undefined;
}

export type LoadOptions = Kev06LoadOptions | TamevNanoLoadOptions;

export interface DecisionModel {
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<DecisionResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

export interface LoadedDecisionModel<Model extends ModelName = ModelName> extends DecisionModel {
  readonly description: ModelDescriptionFor<Model>;
}

export interface LoadedKev06DecisionModel extends LoadedDecisionModel<'kev-0.6b-q4f16'> {
  systemOne(
    state: KevSystemOneJsonValue,
    questions: KevSystemOneQuestions,
    options?: DecisionOptions
  ): Promise<KevSystemOneResult>;
}

export interface TamevNanoChoiceQuestion {
  readonly type: 'choice';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria: Readonly<Record<string, DecisionJsonValue>>;
}

export interface TamevNanoNoulQuestion {
  readonly type: 'noul';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria?: Readonly<{
    true?: DecisionJsonValue;
    false?: DecisionJsonValue;
    yes?: DecisionJsonValue;
    no?: DecisionJsonValue;
  }> | null;
}

export interface TamevNanoScoreQuestion {
  readonly type: 'score';
  readonly instructions?: DecisionJsonValue | null;
  readonly criteria: readonly DecisionJsonValue[];
}

export type TamevNanoSystemOneQuestion =
  | TamevNanoChoiceQuestion
  | TamevNanoNoulQuestion
  | TamevNanoScoreQuestion;

export type TamevNanoSystemOneAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      confidence: number;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{type: 'noul'; noul: number}>
  | Readonly<{
      type: 'score';
      score: number;
      confidence: number;
      legend: Readonly<Record<string, DecisionJsonValue>>;
      probabilities: Readonly<Record<string, number>>;
    }>;

export interface TamevNanoSystemOneResult {
  readonly model: 'experimental-tamev-nano-tinybert-int8';
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, TamevNanoSystemOneAnswer>>;
}

export interface LoadedTamevNanoModel extends LoadedDecisionModel<'experimental-tamev-nano-tinybert-int8'> {
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<DecisionResult>;
  systemOne(
    state: DecisionJsonValue,
    questions: Readonly<Record<string, TamevNanoSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<TamevNanoSystemOneResult>;
}

export function listDecisionModels(): readonly ModelName[];
export function isDecisionModel(value: unknown): value is ModelName;
export function describeModel<Model extends ModelName>(model: Model): ModelDescriptionFor<Model>;
export function loadDecisionModel(options: Kev06LoadOptions): Promise<LoadedKev06DecisionModel>;
export function loadDecisionModel(options: TamevNanoLoadOptions): Promise<LoadedTamevNanoModel>;
export function clearModelCache(): Promise<boolean>;
