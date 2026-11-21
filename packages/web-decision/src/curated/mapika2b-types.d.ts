import type {KananaFloat} from '../kanana/json-float.js';
import type {ModelDescription, ModelProgress} from './public-types.js';

export type Mapika2bValue =
  | null
  | boolean
  | number
  | bigint
  | string
  | KananaFloat
  | readonly Mapika2bValue[]
  | Readonly<{[key: string]: Mapika2bValue}>
  | ReadonlyMap<string, Mapika2bValue>;

export type Mapika2bDictionary<T> = Readonly<Record<string, T>> | ReadonlyMap<string, T>;
export type Mapika2bQuestion = Readonly<{
  type?: 'choice' | 'score' | 'noul' | 'bool';
  instructions?: Mapika2bValue;
  question?: Mapika2bValue;
  criteria?: Mapika2bValue;
  options?: Mapika2bValue;
  isolated?: boolean;
}>;

export interface Mapika2bSystemOneRequest {
  readonly state: Mapika2bValue;
  readonly questions: Mapika2bDictionary<Mapika2bQuestion>;
  readonly independent?: boolean;
  readonly assets?: Mapika2bDictionary<Readonly<{type?: 'image'; format: 'png' | 'jpg' | 'jpeg'; data_base64: string}>>;
}

export interface Mapika2bLoadOptions {
  readonly assetsUrl?: never;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: ModelProgress) => void;
  readonly temperature?: number;
  readonly temperatureByType?: Readonly<Partial<Record<'choice' | 'score' | 'noul', number>>>;
  readonly limits?: Readonly<{
    maxStateTokens?: number;
    maxRowTokens?: number;
    maxRequestTokens?: number;
    maxRows?: number;
    maxImages?: number;
  }>;
}

export type Mapika2bAnswer =
  | Readonly<{type: 'noul'; noul: number}>
  | Readonly<{
      type: 'choice';
      choice: string;
      confidence: number;
      x_p_max: number;
      certainty: number;
      probabilities: Readonly<Record<string, number>>;
    }>
  | Readonly<{
      type: 'score';
      score: number;
      confidence: number;
      x_p_max: number;
      certainty: number;
      probabilities: Readonly<Record<string, number>>;
      legend: Readonly<Record<string, string>>;
      level_fit?: Readonly<Record<string, number>>;
      fit_mass?: number;
    }>;

export interface Mapika2bSystemOneResult {
  readonly model: string;
  readonly answers: Readonly<Record<string, Mapika2bAnswer>>;
  readonly usage: Readonly<{input_tokens: number; output_tokens: 0}>;
}

export interface Mapika2bDecisionModel<Id extends string = string> {
  readonly modelId: Id;
  readonly description: Omit<ModelDescription, 'modelId'> & Readonly<{modelId: Id}>;
  systemOne(request: Mapika2bSystemOneRequest): Promise<Mapika2bSystemOneResult>;
  dispose(): Promise<void>;
}
