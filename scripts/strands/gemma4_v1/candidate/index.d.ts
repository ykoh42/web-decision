import type {
  CommonLoadOptions as PublicLoadOptions,
  LoadedDecisionModel as PublicDecisionModel,
  LoadOptions as PublicOptionUnion,
  ModelDescription as PublicDescription,
  ModelId as PublicModelId,
  loadDecisionModel as publicLoad
} from '../src/curated/public-index.js';

export * from '../src/curated/public-index.js';
export type Gemma4V1ModelId = 'strands-decider-e2b-gemma4-v1-2610';
export type JsonValue = string | number | boolean | null | readonly JsonValue[] | {readonly [key: string]: JsonValue};
export type Content = string | readonly JsonValue[] | {readonly [key: string]: JsonValue};
type Dict<T> = Readonly<Record<string, T>> | ReadonlyMap<string, T>;
export type Question =
  | {
      readonly type?: 'noul';
      readonly instructions: Content;
      readonly criteria?: Readonly<Partial<Record<'false' | 'true', Content | null>>> | null;
    }
  | {readonly type?: 'choice'; readonly instructions: Content; readonly criteria: Dict<Content | null>}
  | {readonly type?: 'score'; readonly instructions: Content; readonly criteria: readonly string[]};
export type Answer =
  | {readonly type: 'noul'; readonly noul: number}
  | {
      readonly type: 'choice';
      readonly choice: string;
      readonly probabilities: Readonly<Record<string, number>>;
      readonly confidence: number;
    }
  | {
      readonly type: 'score';
      readonly score: number;
      readonly legend: Readonly<Record<string, string>>;
      readonly probabilities: Readonly<Record<string, number>>;
      readonly confidence: number;
    };
export interface Request {
  readonly model?: string;
  readonly state: Content;
  readonly questions: Dict<Question>;
  readonly images?: readonly [];
}
export interface Result {
  readonly model: Gemma4V1ModelId;
  readonly answers: Readonly<Record<string, Answer>>;
  readonly usage: Readonly<{input_tokens: number; output_tokens: number}>;
}
export type ModelId = PublicModelId | Gemma4V1ModelId;
export type ModelDescription = Omit<PublicDescription, 'modelId'> & {readonly modelId: ModelId};
export interface Gemma4V1LoadOptions extends Omit<PublicLoadOptions, 'modelId'> {
  readonly modelId: Gemma4V1ModelId;
  readonly maxLength?: number;
  readonly maxBatch?: number;
  readonly maxBatchTokens?: number;
  readonly usePrefixCache?: boolean;
}
export interface Gemma4V1DecisionModel {
  readonly modelId: Gemma4V1ModelId;
  readonly description: ModelDescription;
  systemOne(request: Request, options?: {readonly signal?: AbortSignal}): Promise<Result>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
  prepareOffline(signal?: AbortSignal): Promise<void>;
  isOfflineReady(signal?: AbortSignal): Promise<boolean>;
}
export type LoadOptions = PublicOptionUnion | Gemma4V1LoadOptions;
export type LoadedDecisionModel = PublicDecisionModel | Gemma4V1DecisionModel;
export function listModels(): readonly ModelDescription[];
export function describeModel(modelId: ModelId): ModelDescription;
export const loadDecisionModel: ((options: Gemma4V1LoadOptions) => Promise<Gemma4V1DecisionModel>) &
  typeof publicLoad &
  ((options: LoadOptions) => Promise<LoadedDecisionModel>);
