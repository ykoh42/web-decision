import type {ModelId} from '../catalog/types.generated.js';

export type {ModelId} from '../catalog/types.generated.js';

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

export interface ModelProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
  readonly phase?: 'bundled-assets' | 'upstream-source' | 'asset-verification';
}

export interface LoadOptions {
  readonly modelId: ModelId;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: ModelProgress) => void;
  readonly maxPrefetchBytes?: number;
  readonly maxInputTokens?: number;
  readonly maxInputCharacters?: number;
}

export interface DecisionQuestion {
  readonly instructions: string;
  readonly options: readonly string[];
}

export interface DecisionRequest {
  readonly state: string;
  readonly questions: readonly DecisionQuestion[];
}

export interface DecisionResult {
  readonly tokenCount: number;
  readonly answers: readonly Readonly<{
    selected: string;
    selectedIndex: number;
    probabilities: readonly Readonly<{option: string; probability: number; logit: number}>[];
  }>[];
}

export type KevValue =
  | null
  | boolean
  | number
  | bigint
  | string
  | readonly KevValue[]
  | Readonly<{[key: string]: KevValue}>
  | Map<string, KevValue>;

export type KevQuestion =
  | Readonly<{
      type: 'choice';
      instructions?: KevValue;
      criteria: Readonly<Record<string, KevValue>> | Map<string, KevValue>;
      [key: string]: unknown;
    }>
  | Readonly<{
      type: 'score';
      instructions?: KevValue;
      criteria: readonly KevValue[];
      [key: string]: unknown;
    }>
  | Readonly<{
      type: 'noul';
      instructions?: KevValue;
      criteria?: Readonly<Record<string, KevValue>> | Map<string, KevValue> | null;
      [key: string]: unknown;
    }>;

export type KevAnswer =
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

export interface KevResult {
  readonly model: ModelId;
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, KevAnswer>>;
  readonly answerEntries: readonly (readonly [question: string, answer: KevAnswer])[];
}

export interface ModelDescription {
  readonly modelId: ModelId;
  readonly precision: string;
  readonly backend: Readonly<{
    name: 'tvm-webgpu';
    requiredFeatures: readonly string[];
    requiredLimits: Readonly<Record<string, number>>;
    requiresNonFallbackAdapter: boolean;
  }>;
  readonly checkpoint: Readonly<{repository: string; revision: string; weightSha256?: string}>;
  readonly assets: Readonly<{
    integrity: 'pinned-manifest';
    manifestSha256: string;
    location: string;
    requiredFiles: readonly string[];
    additionalFiles: string;
  }>;
  readonly licenses: Readonly<Record<string, string>>;
  readonly features: Readonly<{
    textChoice: Readonly<{
      status: 'implemented';
      abstention: string;
      limits: readonly Readonly<{
        resource: 'tokens' | 'options' | 'questions';
        scope: 'state' | 'question' | 'request';
        min?: number;
        max?: number;
        description?: string;
      }>[];
      handling: string;
      tokenCount: string;
      score: Readonly<Record<string, unknown>>;
    }>;
    typedTextQuestionTypes?: readonly string[];
    imageChoice?: Readonly<{status: 'not-implemented'}>;
    rating?: Readonly<{status: 'not-implemented'}>;
  }>;
  readonly verification: Readonly<Record<string, string>>;
}

export interface DecisionModel {
  readonly description: ModelDescription;
  decide(request: DecisionRequest, options?: {signal?: AbortSignal}): Promise<DecisionResult>;
  systemOne(
    state: KevValue,
    questions: Readonly<Record<string, KevQuestion>> | Map<string, KevQuestion>,
    options?: {signal?: AbortSignal}
  ): Promise<KevResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

export function listModels(): readonly ModelDescription[];
export function describeModel(modelId: ModelId): ModelDescription;
export function loadDecisionModel(options: LoadOptions): Promise<DecisionModel>;
export function clearModelCache(): Promise<boolean>;
