import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {packElephantRequest, unpackElephantScores} from './protocol.mjs';
import type {ElephantPackedRequest, ElephantRequest, ElephantSlotTokenizer} from './protocol.mjs';

const MAX_ELEPHANT_OPTIONS = 255;

interface ElephantRunner {
  run(packed: ElephantPackedRequest, options: {signal?: AbortSignal | undefined}): Promise<Float32Array[]>;
  dispose(): unknown | PromiseLike<unknown>;
  releaseCachedMemory(): unknown | PromiseLike<unknown>;
}

interface ElephantSessionOptions {
  readonly signal?: AbortSignal;
}

interface CreateElephantSessionArgs {
  tokenizer: unknown;
  addedTokens: unknown;
  runner: unknown;
  maxBranchTokens?: number;
}

export function createElephantSession({
  tokenizer,
  addedTokens,
  runner,
  maxBranchTokens = 128
}: CreateElephantSessionArgs) {
  if (
    typeof (tokenizer as ElephantSlotTokenizer | null)?.encode !== 'function' ||
    !Array.isArray(addedTokens) ||
    typeof (runner as ElephantRunner | null)?.run !== 'function' ||
    typeof (runner as ElephantRunner | null)?.dispose !== 'function' ||
    typeof (runner as ElephantRunner | null)?.releaseCachedMemory !== 'function' ||
    ![128, 512].includes(maxBranchTokens)
  )
    throw new TypeError('Elephant tokenizer, control tokens and runner are required.');
  let queue: Promise<unknown> = Promise.resolve(),
    closed = false,
    disposing: Promise<void> | undefined;
  const requireOpen = () => {
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
  };
  const enqueue = <T,>(task: () => T | PromiseLike<T>): Promise<T> => {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  };
  return {
    async decide(request: unknown, options: ElephantSessionOptions | null = {}) {
      requireOpen();
      const signal = options?.signal;
      if (signal != null) validateH2oSignal(signal);
      signal?.throwIfAborted();
      validateDecisionFields(options);
      const checked = validateRequestFields(
        request,
        Infinity,
        Infinity,
        Infinity,
        MAX_ELEPHANT_OPTIONS
      ) as ElephantRequest;
      let packed: ElephantPackedRequest;
      try {
        packed = packElephantRequest(tokenizer as ElephantSlotTokenizer, checked, {addedTokens, maxBranchTokens} as {
          addedTokens: string[];
          maxBranchTokens: number;
        });
      } catch (cause) {
        throw new WebDecisionError('INVALID_INPUT', 'Invalid Elephant request.', {cause});
      }
      requireOpen();
      const schedule = () => {
        requireOpen();
        return enqueue(async () => {
          signal?.throwIfAborted();
          const scores = await (runner as ElephantRunner).run(packed, {signal});
          signal?.throwIfAborted();
          try {
            return unpackElephantScores(scores, packed);
          } catch (cause) {
            throw new WebDecisionError('INFERENCE_ERROR', 'Invalid Elephant scores.', {cause});
          }
        });
      };
      return signal ? waitForAbort(schedule, signal) : schedule();
    },
    async releaseCachedMemory() {
      requireOpen();
      return enqueue(() => (runner as ElephantRunner).releaseCachedMemory());
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await (runner as ElephantRunner).dispose();
        } finally {
          tokenizer = addedTokens = runner = null;
        }
      });
      return disposing;
    }
  };
}
