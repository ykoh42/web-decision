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

export type ModelName = 'open-jev-deberta-v3-large-q4f16';

export type ModelLimit =
  | Readonly<{resource: 'tokens'; scope: 'request' | 'state'; max: number; description: string}>
  | Readonly<{resource: 'options'; scope: 'question'; min: number; max: number}>
  | Readonly<{resource: 'questions'; scope: 'request'; min: number}>;

export interface ModelDescription {
  readonly model: ModelName;
  readonly precision: 'q4f16';
  readonly checkpoint: Readonly<{
    repository: string;
    revision: string;
    graphSha256?: string;
    weightSha256?: string;
  }>;
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
    name: 'tvm-webgpu';
    requiredFeatures: readonly ['shader-f16'];
    requiredLimits: Readonly<Record<string, number>>;
    requiresNonFallbackAdapter: false;
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

export interface DebertaChoiceQuestion {
  readonly type: 'choice';
  readonly instructions: string;
  readonly options: readonly string[];
}

export interface DebertaScoreQuestion {
  readonly type: 'score';
  readonly instructions: string;
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
      confidence: number;
    }>
  | Readonly<{
      type: 'score';
      score: number;
      legend: Readonly<Record<string, string>>;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
    }>
  | Readonly<{
      type: 'noul';
      noul: number;
      probabilities: Readonly<{false: number; true: number}>;
      confidence: number;
    }>;

export interface DebertaSystemOneResult {
  readonly model: ModelName;
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, DebertaSystemOneAnswer>>;
}

export interface DebertaLoadOptions {
  assetsUrl: string;
  model: ModelName;
  cache?: boolean;
  offline?: boolean;
  signal?: AbortSignal;
  maxPrefetchBytes?: number;
  onProgress?: (progress: ModelProgress) => void;
  debertaReadback?: 'serial' | 'packed';
}

export interface LoadedDebertaDecisionModel {
  readonly description: ModelDescription;
  decide(request: DecisionRequest, options?: DecisionOptions): Promise<DecisionResult>;
  systemOne(
    state: string,
    questions: Readonly<Record<string, DebertaSystemOneQuestion>>,
    options?: DecisionOptions
  ): Promise<DebertaSystemOneResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

export function listDecisionModels(): readonly [ModelName];
export function isDecisionModel(value: unknown): value is ModelName;
export function describeModel(model: ModelName): ModelDescription;
export function loadDecisionModel(options: DebertaLoadOptions): Promise<LoadedDebertaDecisionModel>;
export function clearModelCache(): Promise<boolean>;
