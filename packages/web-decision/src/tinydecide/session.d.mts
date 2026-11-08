import type {TinyDecideEngineAnswer, TinyDecideMetadata, TinyDecidePrototype, TinyDecide, WordPiece} from './engine.js';

export interface TinyDecideSessionEngine {
  readonly meta: TinyDecideMetadata;
  readonly cfg: TinyDecide['cfg'];
  readonly tok: WordPiece;
  answer: TinyDecide['answer'];
}

export interface TinyDecideDecisionAnswer {
  readonly selected: string | undefined;
  readonly selectedIndex: number;
  readonly probabilities: readonly {
    readonly option: string | undefined;
    readonly probability: number;
    readonly logit: number | undefined;
  }[];
}

export interface TinyDecideDecisionResult {
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: readonly TinyDecideDecisionAnswer[];
}

export interface TinyDecideSystemOneResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, TinyDecideEngineAnswer['answers'][number]>>;
}

export interface TinyDecideSession {
  answer(request: unknown, options?: unknown): Promise<TinyDecideEngineAnswer>;
  makeProtos(type: string, lists: unknown, center?: unknown): TinyDecidePrototype | null;
  decide(request: unknown, options?: unknown): Promise<TinyDecideDecisionResult>;
  systemOne(state: unknown, questions: unknown, options?: unknown): Promise<TinyDecideSystemOneResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

export function createTinyDecideSession(options: {
  readonly engine: TinyDecideSessionEngine;
  readonly model: string;
}): TinyDecideSession;
