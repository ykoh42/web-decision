import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {FLASH_MAX_OPTIONS, packFlashRequest, unpackFlashScores} from './protocol.mjs';
import decision from './decision-config.json' with {type: 'json'};

const TEMPERATURE = decision.temperature;

export function createFlashSession({tokenizer, config, runner}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    !config ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function'
  )
    throw new TypeError('Perplexity tokenizer, configuration and runner are required.');
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
      const checked = validateRequestFields(request, 255, Infinity, Infinity, FLASH_MAX_OPTIONS);
      requireOpen();
      const packed = packFlashRequest(tokenizer, checked, config);
      requireOpen();
      const schedule = () => {
        requireOpen();
        return enqueue(async () => {
          signal?.throwIfAborted();
          const rows = await runner.run(packed);
          signal?.throwIfAborted();
          return unpackFlashScores(rows, packed, TEMPERATURE);
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
          tokenizer = config = runner = null;
        }
      });
      return disposing;
    }
  };
}
