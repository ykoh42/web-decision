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

const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Crivo model has been disposed.');

export function createCrivoSession({runner, tokenizer, model}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    ![CRIVO_MODEL_FP32, CRIVO_MODEL_INT8].includes(model)
  )
    throw new TypeError('Crivo session components are required.');

  let queue = Promise.resolve();
  let closed = false;
  let disposing;

  function infer(prepared, signal, finish) {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finish(prepared, logits);
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  return {
    async decide(request, options = {}) {
      if (closed) throw closedError();
      const signal = validateCrivoCallOptions(options);
      const prepared = prepareCrivoDecision(request, tokenizer);
      return infer(prepared, signal, finishCrivoDecision);
    },
    async systemOne(state, questions, options = {}) {
      if (closed) throw closedError();
      const signal = validateCrivoCallOptions(options);
      const prepared = prepareCrivoSystemOne(state, questions, tokenizer);
      return infer(prepared, signal, (input, logits) => finishCrivoSystemOne(input, logits, model));
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
          await runner.dispose();
        } finally {
          runner = tokenizer = null;
        }
      });
      return disposing;
    }
  };
}
