export type DecimaBaseCleanModelName = 'experimental-decima-base-clean-int8';

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

export interface DecimaDecisionOptions extends DecisionOptions {
  /** Language code for Arabic and Persian text normalization. */
  readonly language?: string;
}

export interface ModelProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

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
  readonly model: DecimaBaseCleanModelName;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, DecimaSystemOneAnswer>>;
}

export interface DecimaBaseCleanDescription {
  readonly model: DecimaBaseCleanModelName;
  readonly precision: 'int8';
  readonly checkpoint: Readonly<{repository: string; revision: string}>;
  readonly assets: Readonly<{
    integrity: 'pinned-files';
    location: 'separate';
    requiredFiles: readonly string[];
    additionalFiles: 'No additional files are required.';
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
    requiredLimits: Readonly<Record<string, never>>;
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
        temperatureAppliedBy: 'model';
        calibration: 'not-guaranteed';
      }>;
    }>;
    typedTextQuestionTypes: readonly ['choice', 'score', 'noul', 'rank'];
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

export interface DecimaBaseCleanLoadOptions {
  readonly model: DecimaBaseCleanModelName;
  readonly assetsUrl: string;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: ModelProgress) => void;
  readonly maxPrefetchBytes?: 0;
}

export interface LoadedDecimaBaseCleanModel {
  readonly description: DecimaBaseCleanDescription;
  decide(request: DecisionRequest, options?: DecimaDecisionOptions): Promise<DecisionResult>;
  systemOne(
    state: string | DecisionJsonValue,
    questions: Readonly<Record<string, DecimaSystemOneQuestion>>,
    options?: DecimaDecisionOptions
  ): Promise<DecimaSystemOneResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

/** List only the Decima base-clean int8 route in this package. */
export function listDecisionModels(): readonly DecimaBaseCleanModelName[];
export function isDecisionModel(value: unknown): value is DecimaBaseCleanModelName;
export function describeModel(model: DecimaBaseCleanModelName): DecimaBaseCleanDescription;
export function loadDecisionModel(options: DecimaBaseCleanLoadOptions): Promise<LoadedDecimaBaseCleanModel>;
export function clearModelCache(): Promise<boolean>;
