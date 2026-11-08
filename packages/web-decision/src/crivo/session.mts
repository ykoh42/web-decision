import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {CRIVO_MODEL_FP32, CRIVO_MODEL_INT8} from './contract.mjs';
import {
  finishCrivoDecision,
  finishCrivoSystemOne,
  prepareCrivoDecision,
  prepareCrivoSystemOne,
  validateCrivoCallOptions
} from './protocol.mjs';
import type {
  CrivoAbortSignalLike,
  CrivoDecisionResult,
  CrivoModelName,
  CrivoPreparedRow,
  CrivoSystemOneResult,
  CrivoTokenizer,
  PreparedCrivoDecision,
  PreparedCrivoSystemOne
} from './protocol.mjs';

interface CrivoRunner {
  run(rows: readonly CrivoPreparedRow[], signal?: CrivoAbortSignalLike | null): Promise<Float32Array[]>;
  dispose(): unknown;
}

interface CrivoSessionOptions {
  readonly runner: CrivoRunner;
  readonly tokenizer: CrivoTokenizer;
  readonly model: string;
}

const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Crivo model has been disposed.');

export function createCrivoSession({runner: initialRunner, tokenizer: initialTokenizer, model}: CrivoSessionOptions) {
  let runner: CrivoRunner | null = initialRunner;
  let tokenizer: CrivoTokenizer | null = initialTokenizer;
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    ![CRIVO_MODEL_FP32, CRIVO_MODEL_INT8].includes(model)
  )
    throw new TypeError('Crivo session components are required.');

  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  function infer<TPrepared extends {readonly rows: readonly CrivoPreparedRow[]}, TResult>(
    prepared: TPrepared,
    signal: CrivoAbortSignalLike | null | undefined,
    finish: (input: TPrepared, logits: readonly Float32Array[]) => TResult
  ): Promise<TResult> {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await (runner as CrivoRunner).run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finish(prepared, logits);
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal as AbortSignal) : task;
  }

  return {
    async decide(request: unknown, options: unknown = {}): Promise<CrivoDecisionResult> {
      if (closed) throw closedError();
      const signal = validateCrivoCallOptions(options);
      const prepared: PreparedCrivoDecision = prepareCrivoDecision(request, tokenizer as CrivoTokenizer);
      return infer(prepared, signal, finishCrivoDecision);
    },
    async systemOne(state: unknown, questions: unknown, options: unknown = {}): Promise<CrivoSystemOneResult> {
      if (closed) throw closedError();
      const signal = validateCrivoCallOptions(options);
      const prepared: PreparedCrivoSystemOne = prepareCrivoSystemOne(state, questions, tokenizer as CrivoTokenizer);
      return infer(prepared, signal, (input, logits) => finishCrivoSystemOne(input, logits, model as CrivoModelName));
    },
    async releaseCachedMemory() {
      if (closed) throw closedError();
      await queue;
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await (runner as CrivoRunner).dispose();
        } finally {
          tokenizer = null;
          runner = null;
        }
      });
      return disposing;
    }
  };
}
