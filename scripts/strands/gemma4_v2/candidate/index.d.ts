/** Private, unreleased test candidate. This declaration is not part of the public package. */
export type Content = string | Record<string, unknown> | ReadonlyMap<string, unknown> | readonly unknown[];
export type Dictionary<T> = Readonly<Record<string, T>> | ReadonlyMap<string, T>;
export type Question = Readonly<{
  type?: 'noul' | 'choice' | 'score';
  instructions: Content;
  criteria?: Dictionary<Content | null> | readonly string[] | null;
}>;
export interface SystemOneRequest {
  readonly model?: string;
  readonly state?: Content;
  readonly questions: Dictionary<Question>;
  readonly images?: readonly string[];
}
export type Answer =
  | Readonly<{type: 'noul'; noul: number}>
  | Readonly<{type: 'choice'; choice: string; probabilities: Readonly<Record<string, number>>; confidence: number}>
  | Readonly<{
      type: 'score';
      score: number;
      legend: Readonly<Record<string, Content | null>>;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
    }>;
export interface Model {
  readonly modelId: 'strands-decider-e2b-gemma4-v2-2611';
  readonly description: Readonly<{modelId: Model['modelId']; status: 'private-test-candidate'}>;
  systemOne(
    request: SystemOneRequest,
    options?: {signal?: AbortSignal | null; maxLength?: number} | null
  ): Promise<{
    model: string;
    answers: Readonly<Record<string, Answer>>;
    usage: {input_tokens: number; output_tokens: number};
  }>;
  releaseCachedMemory(): Promise<void>;
  prepareOffline(options?: {
    signal?: AbortSignal;
    onProgress?: (progress: {loadedBytes: number; totalBytes: number; fraction: number; phase?: string}) => void;
  }): Promise<void>;
  isOfflineReady(signal?: AbortSignal | null): Promise<boolean>;
  dispose(): Promise<void>;
}
export function loadDecisionModel(options: {
  modelId: Model['modelId'];
  cache?: boolean;
  offline?: boolean;
  maxLength?: number;
  signal?: AbortSignal;
  onProgress?: (progress: {loadedBytes: number; totalBytes: number; fraction: number; phase?: string}) => void;
}): Promise<Model>;
export function listModels(): readonly Model['description'][];
