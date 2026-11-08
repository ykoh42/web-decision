import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {finishOpenDeciderNanoRequest, packOpenDeciderNanoRequest} from './protocol.mjs';
import {finishOpenDeciderNanoSystemOne, prepareOpenDeciderNanoSystemOne} from './typed.mjs';

export function createOpenDeciderNanoSession({tokenizer, runner}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function'
  )
    throw new TypeError('OpenDecider-nano tokenizer and runner are required.');
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
  async function infer(request, options = {}) {
    requireOpen();
    const signal = options?.signal;
    requireOpen();
    if (signal != null) validateH2oSignal(signal);
    signal?.throwIfAborted();
    validateDecisionFields(options);
    const checked = validateRequestFields(request);
    requireOpen();
    const packed = packOpenDeciderNanoRequest(tokenizer, checked);
    const schedule = () => {
      requireOpen();
      return enqueue(async () => {
        const rows = [];
        for (const question of packed.questions) {
          signal?.throwIfAborted();
          rows.push(await runner.run(question.ids, question.markerPositions));
          signal?.throwIfAborted();
        }
        return finishOpenDeciderNanoRequest(packed, rows);
      });
    };
    const result = await (signal ? waitForAbort(schedule, signal) : schedule());
    return {result, packed};
  }

  return {
    async decide(request, options = {}) {
      return (await infer(request, options)).result;
    },
    async systemOne(state, questions, options = {}) {
      requireOpen();
      const prepared = prepareOpenDeciderNanoSystemOne(state, questions);
      const {result, packed} = await infer(prepared.request, options);
      return finishOpenDeciderNanoSystemOne(prepared, result, packed);
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
