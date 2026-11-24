import type {Mapika2bDecisionModel, Mapika2bLoadOptions} from './mapika2b-types.js';
import type {PublicModelId} from './public-model-id.generated.js';
import type {
  StrandsQwen35CandidateLoadOptions,
  StrandsQwen35LoadedCandidateModel
} from './strands-qwen35-candidate-types.js';

export type {
  StrandsQwen35Content,
  StrandsQwen35Dictionary,
  StrandsQwen35ModelProgress,
  StrandsQwen35Object,
  StrandsQwen35Question,
  StrandsQwen35SystemOneAnswer,
  StrandsQwen35SystemOneOptions,
  StrandsQwen35SystemOneRequest,
  StrandsQwen35SystemOneResult,
  StrandsQwen35Value
} from './strands-qwen35-candidate-types.js';

export type ModelId = PublicModelId;

export interface CatalogModelLimit {
  readonly resource: 'tokens' | 'options' | 'questions' | 'images';
  readonly scope: 'state' | 'question' | 'request';
  readonly min?: number;
  readonly max?: number;
  /** Default configured maximum when the caller can choose a different limit. */
  readonly defaultMax?: number;
  readonly description?: string;
}

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

export interface KananaLoadOptions extends Omit<CommonLoadOptions, 'modelId'> {
  readonly modelId: Extract<ModelId, 'kanana-2-v-decide-3b-2611'>;
}

export type StrandsQwen35ModelId = Extract<ModelId, 'strands-decider-2b-qwen3.5-v1-2610'>;
export type StrandsQwen35LoadOptions = StrandsQwen35CandidateLoadOptions & Readonly<{modelId: StrandsQwen35ModelId}>;
export type Mapika2bModelId = Extract<ModelId, 'mapika-decider-2b-v12'>;
export type Mapika2bCuratedLoadOptions = Mapika2bLoadOptions & Readonly<{modelId: Mapika2bModelId}>;
export type StrandsGemma4V2ModelId = Extract<ModelId, 'strands-decider-e2b-gemma4-v2-2611'>;
export interface StrandsGemma4V2LoadOptions extends Omit<CommonLoadOptions, 'modelId'> {
  readonly modelId: StrandsGemma4V2ModelId;
  readonly maxLength?: number;
}
export type LoadOptions =
  | KananaLoadOptions
  | StrandsQwen35LoadOptions
  | Mapika2bCuratedLoadOptions
  | (StrandsGemma4V2ModelId extends never ? never : StrandsGemma4V2LoadOptions);
export type * from './mapika2b-types.js';

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
  readonly model: Extract<ModelId, 'kanana-2-v-decide-3b-2611'>;
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
  readonly modelId: Extract<ModelId, 'kanana-2-v-decide-3b-2611'>;
  readonly description: ModelDescription;
  decide(request: KananaDecisionRequest, options?: KananaRequestOptions): Promise<KananaDecisionResult>;
  systemOne(request: KananaDecisionRequest, options?: KananaRequestOptions): Promise<KananaDecisionResult>;
  dispose(): Promise<void>;
}

export type StrandsQwen35DecisionModel = StrandsQwen35LoadedCandidateModel<StrandsQwen35ModelId>;
export type StrandsGemma4V2Content =
  | string
  | Readonly<Record<string, unknown>>
  | ReadonlyMap<string, unknown>
  | readonly unknown[];
export type StrandsGemma4V2Dictionary<T> = Readonly<Record<string, T>> | ReadonlyMap<string, T>;
export interface StrandsGemma4V2Question {
  readonly type?: 'choice' | 'noul' | 'score';
  readonly instructions: StrandsGemma4V2Content;
  readonly criteria?: StrandsGemma4V2Dictionary<StrandsGemma4V2Content | null> | readonly string[] | null;
}

export interface StrandsGemma4V2SystemOneRequest {
  readonly model?: string;
  readonly state?: StrandsGemma4V2Content;
  readonly questions: StrandsGemma4V2Dictionary<StrandsGemma4V2Question>;
  readonly images?: readonly string[];
  readonly audio?: null;
}

export interface StrandsGemma4V2SystemOneOptions {
  readonly signal?: AbortSignal | null;
  readonly maxLength?: number;
}

export type StrandsGemma4V2SystemOneAnswer =
  | Readonly<{type: 'noul'; noul: number}>
  | Readonly<{type: 'choice'; choice: string; probabilities: Readonly<Record<string, number>>; confidence: number}>
  | Readonly<{
      type: 'score';
      score: number;
      legend: Readonly<Record<string, StrandsGemma4V2Content | null>>;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
    }>;

export interface StrandsGemma4V2SystemOneResult {
  readonly model: StrandsGemma4V2ModelId;
  readonly answers: Readonly<Record<string, StrandsGemma4V2SystemOneAnswer>>;
  readonly usage: Readonly<{input_tokens: number; output_tokens: number}>;
}

export interface StrandsGemma4V2DecisionModel {
  readonly modelId: StrandsGemma4V2ModelId;
  readonly description: ModelDescription;
  systemOne(
    request: StrandsGemma4V2SystemOneRequest,
    options?: StrandsGemma4V2SystemOneOptions | null
  ): Promise<StrandsGemma4V2SystemOneResult>;
  releaseCachedMemory(): Promise<void>;
  /** Verify all cached weights, fetching missing or corrupt ranges only when online. This can read about 9.6 GB. */
  prepareOffline(options?: {signal?: AbortSignal; onProgress?: (progress: ModelProgress) => void}): Promise<void>;
  /** Check pinned static assets and weight presence without rehashing the full weight cache. */
  isOfflineReady(signal?: AbortSignal | null): Promise<boolean>;
  dispose(): Promise<void>;
}

export type LoadedDecisionModel =
  | KananaDecisionModel
  | (StrandsQwen35ModelId extends never ? never : StrandsQwen35DecisionModel)
  | (Mapika2bModelId extends never ? never : Mapika2bDecisionModel<Mapika2bModelId>)
  | (StrandsGemma4V2ModelId extends never ? never : StrandsGemma4V2DecisionModel);

export function listModels(): readonly ModelDescription[];
export function describeModel(modelId: ModelId): ModelDescription;
export function loadDecisionModel(options: KananaLoadOptions): Promise<KananaDecisionModel>;
export function loadDecisionModel(options: StrandsQwen35LoadOptions): Promise<StrandsQwen35DecisionModel>;
export function loadDecisionModel(options: Mapika2bCuratedLoadOptions): Promise<Mapika2bDecisionModel<Mapika2bModelId>>;
export function loadDecisionModel(options: StrandsGemma4V2LoadOptions): Promise<StrandsGemma4V2DecisionModel>;
export function loadDecisionModel(options: LoadOptions): Promise<LoadedDecisionModel>;
export function loadDecisionModel(options: CommonLoadOptions): Promise<LoadedDecisionModel>;
export function clearModelCache(): Promise<boolean>;
