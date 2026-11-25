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
export function listModels(): readonly {readonly modelId: Gemma4V1ModelId; readonly status: 'private-test-candidate'}[];
export function loadDecisionModel(options: {
  readonly modelId: Gemma4V1ModelId;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly maxLength?: number;
  readonly maxBatch?: number;
  readonly maxBatchTokens?: number;
  readonly usePrefixCache?: boolean;
  readonly signal?: AbortSignal;
}): Promise<{
  systemOne(request: Request, options?: {readonly signal?: AbortSignal}): Promise<Result>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
  prepareOffline(signal?: AbortSignal): Promise<void>;
  isOfflineReady(signal?: AbortSignal): Promise<boolean>;
}>;
