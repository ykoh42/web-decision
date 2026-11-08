// Joint-choice session for the dynamic Nari graph; not yet a public loader.
import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {pack, unpack} from './adapter.mjs';
import {snapshotToqInput} from './runner.mjs';

const fail = (code, message) => new WebDecisionError(code, message);
const NARI_MAX_OPTIONS_PER_REQUEST = 64;

export function createNariSession({tokenizer, controlIds, runner}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    !Array.isArray(controlIds) ||
    typeof runner?.run !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    typeof runner?.dispose !== 'function'
  )
    throw new TypeError('Tokenizer, control IDs and runner are required.');
  const controls = [...controlIds];
  let queue = Promise.resolve(),
    closed = false,
    disposing;
  const requireOpen = () => {
    if (closed) throw fail('SESSION_CLOSED', 'The model has been disposed.');
  };
  const enqueue = (task) => {
    const result = queue.then(task);
    queue = result.catch(() => {});
    return result;
  };
  return {
    async decide(request, options = {}) {
      requireOpen();
      const signal = options?.signal;
      requireOpen();
      if (signal != null) validateH2oSignal(signal);
      signal?.throwIfAborted();
      validateDecisionFields(options);
      requireOpen();
      const snapshot = validateRequestFields(
        request,
        8,
        Infinity,
        Infinity,
        NARI_MAX_OPTIONS_PER_REQUEST,
        NARI_MAX_OPTIONS_PER_REQUEST
      );
      requireOpen();
      const input = snapshotToqInput(pack(tokenizer, snapshot, controls));
      requireOpen();
      const run = async () => {
        signal?.throwIfAborted();
        const scores = await runner.run(input);
        signal?.throwIfAborted();
        return unpack(input, scores);
      };
      const schedule = () => {
        requireOpen();
        return enqueue(run);
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
          tokenizer = runner = null;
        }
      });
      return disposing;
    }
  };
}
