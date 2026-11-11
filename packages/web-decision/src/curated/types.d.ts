import type {CatalogModelLimit, ModelId} from '../catalog/types.generated.js';

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
  readonly phase?:
    | 'bundled-assets'
    | 'model-assets'
    | 'upstream-source'
    | 'asset-verification'
    | 'vision-assets'
    | 'vision-weights';
}

export interface CommonLoadOptions {
  readonly modelId: ModelId;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: ModelProgress) => void;
}

export interface KevLoadOptions extends Omit<CommonLoadOptions, 'modelId'> {
  readonly modelId: 'kev-0.6b-q4f16';
  readonly maxPrefetchBytes?: number;
  readonly maxInputTokens?: number;
  readonly maxInputCharacters?: number;
}

export interface KananaLoadOptions extends Omit<CommonLoadOptions, 'modelId'> {
  readonly modelId: 'kanana-2-v-decide-3b-2611';
}

export type LoadOptions = KananaLoadOptions;

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
      limits: readonly CatalogModelLimit[];
      handling: string;
      tokenCount: string;
      score: Readonly<Record<string, unknown>>;
    }>;
    typedTextQuestionTypes?: readonly string[];
    imageChoice?: Readonly<{status: 'implemented' | 'not-implemented'; limits?: readonly CatalogModelLimit[]}>;
    rating?: Readonly<{status: 'not-implemented'}>;
  }>;
  readonly verification: Readonly<Record<string, string>>;
}

export interface DecisionModel {
  readonly modelId: 'kev-0.6b-q4f16';
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

export type KananaValue =
  | null
  | boolean
  | number
  | bigint
  | KananaFloat
  | string
  | readonly KananaValue[]
  | Readonly<{[key: string]: KananaValue}>
  | Map<string, KananaValue>;

export declare class KananaFloat {
  private readonly kananaFloatBrand;
  readonly value: number;
  constructor(value: number);
}

export declare function kananaFloat(value: number): KananaFloat;

export type KananaImageSource = string | URL | Blob | File | HTMLImageElement;

export type KananaMessagePart =
  | Readonly<{type: 'text' | 'input_text'; text: string}>
  | Readonly<{type: 'image'; image: KananaImageSource}>
  | Readonly<{type: 'image_url'; image_url: KananaImageSource | Readonly<{url: KananaImageSource; detail?: 'auto'}>}>
  | Readonly<Record<string, unknown>>;

export interface KananaMessage {
  readonly role: string;
  readonly content?: string | null | readonly KananaMessagePart[];
  readonly tool_calls?: unknown;
  readonly function_call?: unknown;
  readonly [key: string]: unknown;
}

export type KananaQuestion =
  | Readonly<{
      type: 'choice';
      instructions?: KananaValue;
      criteria: Readonly<Record<string, KananaValue>> | Map<string, KananaValue>;
    }>
  | Readonly<{type: 'score'; instructions?: KananaValue; criteria: readonly KananaValue[]}>
  | Readonly<{
      type: 'noul';
      instructions?: KananaValue;
      criteria?: Readonly<Record<'false' | 'true', KananaValue>> | Map<'false' | 'true', KananaValue> | null;
    }>;

export interface KananaDecisionRequest {
  readonly state?: KananaValue | readonly KananaMessage[];
  readonly stateFormat?: 'text' | 'json' | 'messages';
  readonly questions: Readonly<Record<string, KananaQuestion>> | Map<string, KananaQuestion>;
  readonly images?: readonly KananaImageSource[];
  readonly temperature?: number;
  readonly querySchemas?: Readonly<Record<string, KananaQuestion | null>> | Map<string, KananaQuestion | null>;
}

export interface KananaRequestOptions {
  readonly signal?: AbortSignal;
  readonly maxLength?: number;
  readonly truncateInputs?: boolean;
  readonly truncateQuestionCandidates?: boolean;
  readonly temperature?: number;
}

export type KananaAnswer =
  | Readonly<{
      type: 'choice';
      choice: string;
      confidence: number;
      probabilities: Readonly<Record<string, number>>;
      probabilityEntries: readonly (readonly [key: string, probability: number])[];
    }>
  | Readonly<{
      type: 'score';
      score: number;
      confidence: number;
      legend: Readonly<Record<string, KananaValue>>;
      probabilities: Readonly<Record<string, number>>;
      probabilityEntries: readonly (readonly [level: string, probability: number])[];
    }>
  | Readonly<{type: 'noul'; noul: number}>;

export interface KananaDecisionResult {
  readonly model: 'kanana-2-v-decide-3b-2611';
  readonly answers: Readonly<Record<string, KananaAnswer>>;
  readonly answerEntries: readonly (readonly [question: string, answer: KananaAnswer])[];
  readonly usage: Readonly<{billing_units: number; input_tokens: number; output_tokens: 0}>;
  readonly inputTruncation?: Readonly<{
    truncated: true;
    maxLength: number;
    queryQuestionIds: readonly string[];
    candidateDocuments: readonly Readonly<{questionId: string; keys: readonly string[]}>[];
  }>;
}

export interface KananaDecisionModel {
  readonly modelId: 'kanana-2-v-decide-3b-2611';
  readonly description: ModelDescription;
  decide(request: KananaDecisionRequest, options?: KananaRequestOptions): Promise<KananaDecisionResult>;
  systemOne(request: KananaDecisionRequest, options?: KananaRequestOptions): Promise<KananaDecisionResult>;
  dispose(): Promise<void>;
}

export type LoadedDecisionModel = KananaDecisionModel;

export function listModels(): readonly ModelDescription[];
export function describeModel(modelId: ModelId): ModelDescription;
export function loadDecisionModel(options: KananaLoadOptions): Promise<KananaDecisionModel>;
export function loadDecisionModel(options: LoadOptions): Promise<LoadedDecisionModel>;
export function loadDecisionModel(options: CommonLoadOptions): Promise<LoadedDecisionModel>;
export function clearModelCache(): Promise<boolean>;
