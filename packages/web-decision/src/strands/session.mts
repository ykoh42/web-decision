import type {DecisionRequest, DecisionResult} from '../curated/types.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {packStrandsRequest, unpackStrandsScores, validateStrandsRow} from './choice.mjs';
import {assertUsable} from './scope.mjs';

interface SessionRunner {
  run(row: unknown): MaybePromise<unknown>;
  releaseCachedMemory(): MaybePromise<void>;
  dispose(): MaybePromise<unknown>;
}

type MaybePromise<Value> = Value | PromiseLike<Value>;

interface SessionOptions {
  readonly tokenizer: unknown;
  readonly runner: SessionRunner;
}

interface DecisionOptions {
  readonly signal?: AbortSignal | null | undefined;
}

interface StrandsRow {
  readonly inputIds: readonly number[];
  readonly answerIndex: number;
  readonly optionIndices: readonly number[];
  readonly options: readonly string[];
}

interface PackedRequest {
  readonly rows: readonly StrandsRow[];
  readonly tokenCount: number;
}

/** Independent experiment; the caller supplies an owned fused-graph runner. */
export class ExperimentalStrandsSession {
  declare tokenizer: unknown | null;
  declare runner: SessionRunner | null;
  declare queue: Promise<unknown>;
  declare closed: boolean;
  declare disposal: Promise<void> | null;

  constructor({tokenizer, runner}: SessionOptions) {
    if (
      !tokenizer ||
      typeof runner?.run !== 'function' ||
      typeof runner?.releaseCachedMemory !== 'function' ||
      typeof runner?.dispose !== 'function'
    )
      throw new TypeError('Tokenizer and owned runner are required');
    this.tokenizer = tokenizer;
    this.runner = runner;
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }

  async decide(request: DecisionRequest, options: DecisionOptions | null = {}): Promise<DecisionResult> {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this.runner);
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFields(options);
    request = validateRequestFields(request, Infinity, Infinity, Infinity, 255) as DecisionRequest;
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const packed: PackedRequest = packStrandsRequest(this.tokenizer!, request);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const schedule = () => {
      if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const task = this.queue.then(async (): Promise<DecisionResult> => {
        const scores: unknown[] = [];
        for (const row of packed.rows) {
          signal?.throwIfAborted();
          validateStrandsRow(row);
          // Once submitted, GPU work continues until synchronized even if the caller stops waiting.
          scores.push(await this.runner!.run(row));
        }
        signal?.throwIfAborted();
        return unpackStrandsScores(scores, packed);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  async releaseCachedMemory(): Promise<void> {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this.runner);
    const task = this.queue.then(() => this.runner!.releaseCachedMemory());
    this.queue = task.catch(() => {});
    return task;
  }

  async dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.closed = true;
    this.disposal = (async () => {
      await this.queue;
      try {
        await this.runner!.dispose();
      } finally {
        this.tokenizer = this.runner = null;
      }
    })();
    return this.disposal;
  }
}
