import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {packMapikaRequest, unpackMapikaScores} from '../mapika.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {MAPIKA4_MAX_OPTIONS} from './contract.mjs';

export function createMapika4Session({tokenizer, labels, limits, runner}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    !labels ||
    !limits ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function'
  )
    throw new TypeError('Mapika 4B tokenizer, labels, limits and runner are required.');
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
      requireOpen();
      if (signal != null) validateH2oSignal(signal);
      signal?.throwIfAborted();
      validateDecisionFields(options);
      const checked = validateRequestFields(request, 255, Infinity, Infinity, MAPIKA4_MAX_OPTIONS);
      requireOpen();
      const packed = packMapikaRequest(tokenizer, labels, checked, limits);
      requireOpen();
      const schedule = () => {
        requireOpen();
        return enqueue(async () => {
          signal?.throwIfAborted();
          const scores = await runner.run(packed);
          signal?.throwIfAborted();
          return unpackMapikaScores(scores, packed, limits.temperature);
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
          tokenizer = labels = limits = runner = null;
        }
      });
      return disposing;
    }
  };
}
