import type {CRIVO_MODEL_FP32, CRIVO_MODEL_INT8} from './contract.mjs';

export interface CrivoTokenizer {
  readonly special: {
    readonly cls: number;
    readonly sep: number;
    readonly pad: number;
    readonly opt: number;
  };
  encode(text: string): number[];
}

export interface CrivoPreparedRow {
  readonly inputIds: number[];
  readonly optionPositions: number[];
  readonly qtype: number;
}

export type CrivoSystemOneRow = CrivoPreparedRow &
  (
    | {readonly id: string; readonly type: 'choice'; readonly labels: string[]}
    | {readonly id: string; readonly type: 'noul'; readonly labels: null}
    | {readonly id: string; readonly type: 'score'; readonly labels: null}
  );

export interface PreparedCrivoSystemOne {
  readonly rows: CrivoSystemOneRow[];
  readonly tokenCount: number;
}

export interface PreparedCrivoDecision {
  readonly rows: (CrivoPreparedRow & {readonly labels: string[]})[];
  readonly tokenCount: number;
  readonly questions: {readonly id: string; readonly labels: string[]}[];
}

export interface CrivoAbortSignalLike {
  readonly aborted?: boolean;
  readonly reason?: unknown;
  throwIfAborted(): void;
  addEventListener: AbortSignal['addEventListener'];
  removeEventListener: AbortSignal['removeEventListener'];
}

export type CrivoModelName = typeof CRIVO_MODEL_FP32 | typeof CRIVO_MODEL_INT8;

export type CrivoSystemOneResult = {
  readonly model: CrivoModelName;
  readonly tokenCount: number;
  readonly answers: Readonly<
    Record<
      string,
      | {readonly type: 'choice'; readonly choice: string; readonly probabilities: Readonly<Record<string, number>>}
      | {readonly type: 'noul'; readonly noul: number}
      | {readonly type: 'score'; readonly score: number; readonly probabilities: Readonly<Record<string, number>>}
    >
  >;
};

export interface CrivoDecisionResult {
  readonly tokenCount: number;
  readonly answers: readonly {
    readonly selected: string;
    readonly selectedIndex: number;
    readonly probabilities: readonly {readonly option: string; readonly probability: number; readonly logit: number}[];
  }[];
}

export function prepareCrivoSystemOne(
  state: unknown,
  questions: unknown,
  tokenizer: CrivoTokenizer
): PreparedCrivoSystemOne;
export function prepareCrivoDecision(request: unknown, tokenizer: CrivoTokenizer): PreparedCrivoDecision;
export function finishCrivoSystemOne(
  prepared: PreparedCrivoSystemOne,
  logits: readonly Float32Array[],
  model: CrivoModelName
): CrivoSystemOneResult;
export function finishCrivoDecision(
  prepared: PreparedCrivoDecision,
  logits: readonly Float32Array[]
): CrivoDecisionResult;
export function validateCrivoCallOptions(options: unknown): CrivoAbortSignalLike | null | undefined;
