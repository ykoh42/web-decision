import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {packElephantRequest, unpackElephantScores} from './protocol.mjs';

const MAX_ELEPHANT_OPTIONS = 255;

export function createElephantSession({tokenizer, addedTokens, runner, maxBranchTokens = 128}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    !Array.isArray(addedTokens) ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    ![128, 512].includes(maxBranchTokens)
  )
    throw new TypeError('Elephant tokenizer, control tokens and runner are required.');
  let queue = Promise.resolve(),
    closed = false,
    disposing;
  const requireOpen = () => {
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
  };
  const enqueue = (task) => {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  };
  return {
    async decide(request, options = {}) {
      requireOpen();
      const signal = options?.signal;
      if (signal != null) validateH2oSignal(signal);
      signal?.throwIfAborted();
      validateDecisionFields(options);
      const checked = validateRequestFields(request, Infinity, Infinity, Infinity, MAX_ELEPHANT_OPTIONS);
      let packed;
      try {
        packed = packElephantRequest(tokenizer, checked, {addedTokens, maxBranchTokens});
      } catch (cause) {
        throw new WebDecisionError('INVALID_INPUT', 'Invalid Elephant request.', {cause});
      }
      requireOpen();
      const schedule = () => {
        requireOpen();
        return enqueue(async () => {
          signal?.throwIfAborted();
          const scores = await runner.run(packed, {signal});
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
      return enqueue(() => runner.releaseCachedMemory());
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner.dispose();
        } finally {
          tokenizer = addedTokens = runner = null;
        }
      });
      return disposing;
    }
  };
}
