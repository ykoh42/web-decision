import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {STRANDS_GEMMA4_E2B_DECISION_CONFIG} from './gemma4-e2b-contract.mjs';
import {
  STRANDS_GEMMA4_MAX_LENGTH,
  finishStrandsGemma4Answer,
  packPinnedStrandsGemma4V1Request,
  type Gemma4Answer,
  type Gemma4PackedRow
} from './gemma4-protocol.mjs';

export const STRANDS_GEMMA4_V1_E2B_MODEL = 'strands-decider-e2b-gemma4-v1-2610';

interface PointerQuestion {
  readonly ids: readonly number[];
  readonly optionIndices: readonly number[];
  readonly answerIndex: number;
}

export interface Gemma4V1OwnedRunner {
  run(
    state: readonly number[],
    questions: readonly PointerQuestion[],
    signal: AbortSignal,
    usePrefixCache?: boolean
  ): Promise<readonly ArrayLike<number>[]>;
  releaseCachedMemory(): Promise<void> | void;
  dispose(): Promise<void> | void;
}

interface CallOptions {
  readonly signal?: AbortSignal | null;
}

const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Strands Gemma 4 v1 model has been disposed.');

export interface Gemma4V1ServingOptions {
  readonly maxLength?: number;
  readonly maxBatch?: number;
  readonly maxBatchTokens?: number;
  readonly usePrefixCache?: boolean;
}

export function resolveGemma4V1ServingOptions(options: Gemma4V1ServingOptions = {}) {
  const {maxLength = STRANDS_GEMMA4_MAX_LENGTH, maxBatch = 32, maxBatchTokens = 0, usePrefixCache = true} = options;
  if (!Number.isSafeInteger(maxLength) || maxLength < 1)
    throw new WebDecisionError('INVALID_INPUT', 'maxLength must be a positive integer.');
  if (!Number.isSafeInteger(maxBatch) || maxBatch < 1)
    throw new WebDecisionError('INVALID_INPUT', 'maxBatch must be a positive integer.');
  if (!Number.isSafeInteger(maxBatchTokens) || maxBatchTokens < 0)
    throw new WebDecisionError('INVALID_INPUT', 'maxBatchTokens must be a non-negative integer.');
  if (typeof usePrefixCache !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'usePrefixCache must be a boolean.');
  return {maxLength, maxBatch, maxBatchTokens, usePrefixCache};
}

function* questionBatches(rows: readonly Gemma4PackedRow[], maxBatch: number, maxBatchTokens: number) {
  for (let first = 0; first < rows.length; first += maxBatch) {
    const group = rows.slice(first, first + maxBatch);
    let start = 0;
    let longest = 0;
    for (const [index, row] of group.entries()) {
      const length = row.inputIds.length;
      if (maxBatchTokens && index > start && (index - start + 1) * Math.max(longest, length) > maxBatchTokens) {
        yield group.slice(start, index);
        start = index;
        longest = 0;
      }
      longest = Math.max(longest, length);
    }
    yield group.slice(start);
  }
}

function questionSuffix(row: Gemma4PackedRow, prefix: number): PointerQuestion {
  return {
    ids: row.inputIds.slice(prefix),
    optionIndices: row.optionIndices.map((index) => index - prefix),
    answerIndex: row.answerIndex - prefix
  };
}

/** Run the checkpoint's typed System One contract over an owned causal decoder. */
export function createGemma4V1Session(
  tokenizer: unknown,
  runner: Gemma4V1OwnedRunner,
  policy = resolveGemma4V1ServingOptions()
) {
  const {maxLength, maxBatch, maxBatchTokens, usePrefixCache} = policy;
  const closed = new AbortController();
  let queue: Promise<unknown> = Promise.resolve();
  let disposal: Promise<void> | undefined;

  function enqueue<Result>(work: () => Result | Promise<Result>): Promise<Result> {
    const task = queue.then(work);
    queue = task.catch(() => {});
    return task;
  }

  async function systemOne(request: unknown, options?: CallOptions | null) {
    closed.signal.throwIfAborted();
    if (options != null && (typeof options !== 'object' || Array.isArray(options)))
      throw new WebDecisionError('INVALID_INPUT', 'Session options must be an object.');
    const caller = options?.signal;
    if (caller != null) validateH2oSignal(caller);
    const signal = caller ? AbortSignal.any([closed.signal, caller]) : closed.signal;
    signal.throwIfAborted();
    const {stateIds, rows} = packPinnedStrandsGemma4V1Request(tokenizer, request, maxLength);
    const task = enqueue(async () => {
      let inputTokens = 0;
      const answers: Array<[string, Gemma4Answer]> = [];
      for (const batch of questionBatches(rows, maxBatch, maxBatchTokens)) {
        signal.throwIfAborted();
        const questions = batch.map((row) => questionSuffix(row, stateIds.length));
        const logits = await runner.run(stateIds, questions, signal, usePrefixCache);
        signal.throwIfAborted();
        if (logits.length !== batch.length)
          throw new WebDecisionError('INFERENCE_ERROR', 'Strands Gemma 4 v1 returned the wrong number of answers.');
        const prefixes = usePrefixCache ? 1 : questions.length;
        inputTokens += questions.reduce((sum, question) => sum + question.ids.length, stateIds.length * prefixes);
        for (const [index, row] of batch.entries())
          answers.push([row.name, finishStrandsGemma4Answer(row, logits[index], STRANDS_GEMMA4_E2B_DECISION_CONFIG)]);
      }
      return {
        model: STRANDS_GEMMA4_V1_E2B_MODEL,
        usage: {input_tokens: inputTokens, output_tokens: answers.length},
        answers: Object.fromEntries(answers)
      };
    });
    return waitForAbort(() => task, signal);
  }

  return {
    systemOne,
    async releaseCachedMemory() {
      closed.signal.throwIfAborted();
      await enqueue(() => {
        closed.signal.throwIfAborted();
        return runner.releaseCachedMemory();
      });
    },
    dispose() {
      if (!disposal) {
        closed.abort(closedError());
        disposal = enqueue(() => runner.dispose());
      }
      return disposal;
    }
  };
}
