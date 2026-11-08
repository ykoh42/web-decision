import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {CLEF_MAX_OPTIONS, packChoiceRequest, unpackChoiceScores} from './protocol.mjs';
import {approximateSoftmax} from './softmax.mjs';

export function createClefSession({tokenizer, controlIds, runner}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    !Array.isArray(controlIds) ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function'
  )
    throw new TypeError('Clef dependencies are required.');
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
      const checked = validateRequestFields(request, 8, Infinity, Infinity, CLEF_MAX_OPTIONS, CLEF_MAX_OPTIONS);
      let packed;
      try {
        packed = packChoiceRequest(tokenizer, checked, controlIds, {maxInputTokens: 512});
      } catch (cause) {
        throw new WebDecisionError('INVALID_INPUT', 'Invalid Clef request.', {cause});
      }
      requireOpen();
      const schedule = () => {
        requireOpen();
        return enqueue(async () => {
          signal?.throwIfAborted();
          const rows = await runner.run(packed, {signal});
          signal?.throwIfAborted();
          try {
            const result = unpackChoiceScores(packed, rows, approximateSoftmax(rows));
            return {
              tokenCount: result.tokenCount,
              answers: result.answers.map(({selected, selectedIndex, probabilities}) => ({
                selected,
                selectedIndex,
                probabilities
              }))
            };
          } catch (cause) {
            throw new WebDecisionError('INFERENCE_ERROR', 'Invalid Clef scores.', {cause});
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
          tokenizer = controlIds = runner = null;
        }
      });
      return disposing;
    }
  };
}
