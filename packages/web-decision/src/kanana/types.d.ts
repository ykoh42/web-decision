import type {CommonLoadOptions, ModelDescription} from '../curated/types.js';

export interface KananaLoadOptions extends Omit<CommonLoadOptions, 'modelId'> {
  readonly modelId: 'kanana-2-v-decide-3b-2611';
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
