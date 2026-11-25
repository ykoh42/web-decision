import type {CommonLoadOptions, ModelDescription, ModelId} from './public-types.js';

export type StrandsGemma4V1ModelId = Extract<ModelId, 'strands-decider-e2b-gemma4-v1-2610'>;
export type StrandsGemma4V1JsonValue =
  | string
  | number
  | boolean
  | null
  | readonly StrandsGemma4V1JsonValue[]
  | {readonly [key: string]: StrandsGemma4V1JsonValue};
export type StrandsGemma4V1Content =
  | string
  | readonly StrandsGemma4V1JsonValue[]
  | {readonly [key: string]: StrandsGemma4V1JsonValue};
type Dict<T> = Readonly<Record<string, T>> | ReadonlyMap<string, T>;
export type StrandsGemma4V1Question =
  | {
      readonly type?: 'noul';
      readonly instructions: StrandsGemma4V1Content;
      readonly criteria?: Readonly<Partial<Record<'false' | 'true', StrandsGemma4V1Content | null>>> | null;
    }
  | {
      readonly type?: 'choice';
      readonly instructions: StrandsGemma4V1Content;
      readonly criteria: Dict<StrandsGemma4V1Content | null>;
    }
  | {readonly type?: 'score'; readonly instructions: StrandsGemma4V1Content; readonly criteria: readonly string[]};
export type StrandsGemma4V1Answer =
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
export interface StrandsGemma4V1Request {
  readonly model?: string;
  readonly state: StrandsGemma4V1Content;
  readonly questions: Dict<StrandsGemma4V1Question>;
  readonly images?: readonly [];
}
export interface StrandsGemma4V1Result {
  readonly model: StrandsGemma4V1ModelId;
  readonly answers: Readonly<Record<string, StrandsGemma4V1Answer>>;
  readonly usage: Readonly<{input_tokens: number; output_tokens: number}>;
}
export interface StrandsGemma4V1LoadOptions extends Omit<CommonLoadOptions, 'modelId'> {
  readonly modelId: StrandsGemma4V1ModelId;
  readonly maxLength?: number;
  readonly maxBatch?: number;
  readonly maxBatchTokens?: number;
  readonly usePrefixCache?: boolean;
}
export interface StrandsGemma4V1DecisionModel {
  readonly modelId: StrandsGemma4V1ModelId;
  readonly description: ModelDescription;
  systemOne(request: StrandsGemma4V1Request, options?: {readonly signal?: AbortSignal}): Promise<StrandsGemma4V1Result>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
  prepareOffline(signal?: AbortSignal): Promise<void>;
  isOfflineReady(signal?: AbortSignal): Promise<boolean>;
}
