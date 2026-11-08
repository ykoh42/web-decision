import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {ETTIN_MAX_OPTIONS, ETTIN_MODEL} from './contract.mjs';
import {finishEttinDecision, finishEttinSystemOne, prepareEttinDecision, prepareEttinSystemOne} from './protocol.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');

function readSignal(options) {
  if (options == null) return undefined;
  validateDecisionFields(options);
  const descriptor = Object.getOwnPropertyDescriptor(options, 'signal');
  if (descriptor && !Object.hasOwn(descriptor, 'value'))
    throw invalid('Decider Ettin does not accept signal accessors.');
  const signal = descriptor?.value;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid('Decider Ettin signal must be an AbortSignal.');
  return signal;
}

export function createEttinSession({runner, tokenizer, model = ETTIN_MODEL}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function'
  )
    throw new TypeError('Decider Ettin session components are required.');
  if (model !== ETTIN_MODEL) throw new TypeError('Decider Ettin model id does not match the pinned contract.');

  let queue = Promise.resolve();
  let closed = false;
  let disposing;

  const requireOpen = () => {
    if (closed) throw closedError();
  };

  function enqueue(task) {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  }

  async function infer(prepared, signal) {
    requireOpen();
    const schedule = () =>
      enqueue(async () => {
        requireOpen();
        const rows = [];
        for (const question of prepared.questions) {
          signal?.throwIfAborted();
          rows.push(await runner.run(question));
        }
        return rows;
      });
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  return {
    async decide(request, options = {}) {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const snapshot = validateRequestFields(request, Infinity, Infinity, Infinity, ETTIN_MAX_OPTIONS);
      const prepared = prepareEttinDecision(tokenizer, snapshot);
      const rows = await infer(prepared, signal);
      return finishEttinDecision(prepared, rows);
    },
    async systemOne(state, questions, options = {}) {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const prepared = prepareEttinSystemOne(state, questions, tokenizer);
      const rows = await infer(prepared, signal);
      return finishEttinSystemOne(prepared, rows, model);
    },
    releaseCachedMemory() {
      requireOpen();
      return enqueue(() => {});
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner.dispose();
        } finally {
          runner = tokenizer = null;
        }
      });
      return disposing;
    }
  };
}
