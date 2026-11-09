import type {DecisionRequest, DecisionResult} from '../curated/types.js';
import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {packMapikaRequest, unpackMapikaScores} from '../mapika.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {MAPIKA4_MAX_OPTIONS} from './contract.mjs';

interface MapikaTokenizer {
  encode(...args: unknown[]): unknown;
}

interface MapikaLabels {
  readonly names: readonly string[];
  readonly ids: readonly number[];
  readonly open: readonly number[];
}

interface MapikaLimits {
  readonly controlIds: readonly number[];
  readonly maxStateTokens: number;
  readonly maxInputTokens: number;
  readonly temperature: number;
}

interface PackedMapikaRequest {
  readonly ids: readonly number[];
  readonly groups: readonly {readonly slot: number; readonly options: readonly string[]}[];
}

interface MapikaRunner {
  run(packed: PackedMapikaRequest): MaybePromise<readonly Float32Array[]>;
  releaseCachedMemory(): MaybePromise<void>;
  dispose(): MaybePromise<unknown>;
}

interface CreateMapika4SessionOptions {
  readonly tokenizer: MapikaTokenizer | null | undefined;
  readonly labels: MapikaLabels | null | undefined;
  readonly limits: MapikaLimits | null | undefined;
  readonly runner: MapikaRunner | null | undefined;
}

interface DecisionOptions {
  readonly signal?: AbortSignal | null | undefined;
}

export interface Mapika4Session {
  decide(request: DecisionRequest, options?: DecisionOptions | null): Promise<DecisionResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

type MaybePromise<Value> = Value | PromiseLike<Value>;

export function createMapika4Session({tokenizer, labels, limits, runner}: CreateMapika4SessionOptions): Mapika4Session {
  if (
    typeof tokenizer?.encode !== 'function' ||
    !labels ||
    !limits ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function'
  )
    throw new TypeError('Mapika 4B tokenizer, labels, limits and runner are required.');
  let queue: Promise<unknown> = Promise.resolve(),
    closed = false,
    disposing: Promise<void> | undefined;
  const requireOpen = () => {
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
  };
  const enqueue = <Value,>(task: () => MaybePromise<Value>): Promise<Value> => {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  };
  return {
    async decide(request: DecisionRequest, options: DecisionOptions | null = {}): Promise<DecisionResult> {
      requireOpen();
      const signal = options?.signal;
      requireOpen();
      if (signal != null) validateH2oSignal(signal);
      signal?.throwIfAborted();
      validateDecisionFields(options);
      const checked = validateRequestFields(request, 255, Infinity, Infinity, MAPIKA4_MAX_OPTIONS);
      requireOpen();
      const packed: PackedMapikaRequest = packMapikaRequest(tokenizer, labels, checked, limits);
      requireOpen();
      const schedule = () => {
        requireOpen();
        return enqueue(async () => {
          signal?.throwIfAborted();
          const scores = await runner!.run(packed);
          signal?.throwIfAborted();
          return unpackMapikaScores(scores, packed, limits!.temperature);
        });
      };
      return signal ? waitForAbort(schedule, signal) : schedule();
    },
    async releaseCachedMemory(): Promise<void> {
      requireOpen();
      return enqueue(() => runner!.releaseCachedMemory());
    },
    dispose(): Promise<void> {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner!.dispose();
        } finally {
          tokenizer = labels = limits = runner = null;
        }
      });
      return disposing;
    }
  };
}
