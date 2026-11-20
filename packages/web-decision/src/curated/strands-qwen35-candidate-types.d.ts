import type {KananaFloat} from '../kanana/json-float.js';
import type {ModelDescription, ModelProgress} from './public-types.js';

export interface StrandsQwen35ModelProgress extends ModelProgress {
  readonly phase: 'model-assets' | 'vision-weights' | 'upstream-source';
}

export interface StrandsQwen35CandidateLoadOptions {
  readonly assetsUrl?: never;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: StrandsQwen35ModelProgress) => void;
}

export type StrandsQwen35Value = null | boolean | number | bigint | KananaFloat | StrandsQwen35Content;

export interface StrandsQwen35Object {
  readonly [key: string]: StrandsQwen35Value;
}

export type StrandsQwen35Content =
  | string
  | readonly StrandsQwen35Value[]
  | StrandsQwen35Object
  | ReadonlyMap<string, StrandsQwen35Value>;

/** Native Maps preserve insertion order for integer-like keys in questions, criteria and nested content. */
export type StrandsQwen35Dictionary<Value> = Readonly<Record<string, Value>> | ReadonlyMap<string, Value>;

export type StrandsQwen35Question = Readonly<{
  instructions: StrandsQwen35Content;
  [field: string]: unknown;
}> &
  (
    | Readonly<{
        type?: 'noul';
        criteria?:
          | Readonly<Partial<Record<'false' | 'true', StrandsQwen35Content | null>>>
          | ReadonlyMap<'false' | 'true', StrandsQwen35Content | null>
          | null;
      }>
    | Readonly<{type?: 'choice'; criteria: StrandsQwen35Dictionary<StrandsQwen35Content | null>}>
    | Readonly<{type?: 'score'; criteria: readonly string[]}>
  );

export interface StrandsQwen35SystemOneRequest {
  readonly model?: string;
  readonly state: StrandsQwen35Content;
  readonly questions: StrandsQwen35Dictionary<StrandsQwen35Question>;
  readonly images?: readonly string[];
  readonly [field: string]: unknown;
}

export interface StrandsQwen35SystemOneOptions {
  readonly maxBatch?: number;
  readonly maxQuestionFraction?: number;
  readonly maxBatchTokens?: number | null;
  readonly maxImages?: number;
  readonly maxLength?: number;
  readonly strictWindow?: boolean;
  readonly imageLongSide?: number;
  readonly maxImagePixels?: number;
  readonly signal?: AbortSignal | null;
}

export type StrandsQwen35SystemOneAnswer =
  | Readonly<{type: 'noul'; noul: number}>
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
    }>;

export interface StrandsQwen35SystemOneResult {
  readonly model: string;
  readonly answers: Readonly<Record<string, StrandsQwen35SystemOneAnswer>>;
  readonly usage: Readonly<{input_tokens: number; output_tokens: number}>;
}

export interface StrandsQwen35CandidateModel {
  systemOne(
    request: StrandsQwen35SystemOneRequest,
    options?: StrandsQwen35SystemOneOptions | null
  ): Promise<StrandsQwen35SystemOneResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

export interface StrandsQwen35LoadedCandidateModel<Id extends string = string> extends StrandsQwen35CandidateModel {
  readonly modelId: Id;
  readonly description: Omit<ModelDescription, 'modelId'> & Readonly<{modelId: Id}>;
}
