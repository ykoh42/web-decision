import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {TAMEV_NANO_PROFILE} from './contract.mjs';
import {calibrateTamevProbabilities, finishTamevDecision, prepareTamevDecision} from './protocol.mjs';
import {finishTamevSystemOne, prepareTamevSystemOne} from './typed.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');

function readSignal(options, profile) {
  if (options == null) return undefined;
  validateDecisionFields(options);
  const descriptor = Object.getOwnPropertyDescriptor(options, 'signal');
  if (descriptor && !Object.hasOwn(descriptor, 'value'))
    throw invalid(`${profile.name} does not accept signal accessors.`);
  const signal = descriptor?.value;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid(`${profile.name} signal must be an AbortSignal.`);
  return signal;
}

export function createTamevSession({runner, tokenizer, config, profile = TAMEV_NANO_PROFILE, model = profile.model}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    typeof tokenizer?.encodeContext !== 'function' ||
    typeof tokenizer?.encodeOption !== 'function' ||
    !config?.calibration_by_k ||
    model !== profile.model
  )
    throw new TypeError(`${profile.name} session components are required.`);

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

  async function infer(prepared, signal, applyOptionCalibration) {
    requireOpen();
    const schedule = () =>
      enqueue(async () => {
        requireOpen();
        const rows = [];
        for (const question of prepared.questions) {
          signal?.throwIfAborted();
          const row = await runner.run(question.contextIds, question.optionRows);
          rows.push({
            ...row,
            probabilities: applyOptionCalibration
              ? calibrateTamevProbabilities(row.probabilities, question.optionRows.length, config)
              : row.probabilities
          });
        }
        return rows;
      });
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  return {
    async decide(request, options = {}) {
      requireOpen();
      const signal = readSignal(options, profile);
      signal?.throwIfAborted();
      const checked = validateRequestFields(request, Infinity, Infinity, Infinity, Infinity, Infinity, {
        dataPropertiesOnly: true,
        allowInheritedDataProperties: true
      });
      const prepared = prepareTamevDecision(tokenizer, checked, profile);
      const rows = await infer(prepared, signal, true);
      return finishTamevDecision(prepared, rows);
    },
    async systemOne(state, questions, options = {}) {
      requireOpen();
      const signal = readSignal(options, profile);
      signal?.throwIfAborted();
      const prepared = prepareTamevSystemOne(state, questions, tokenizer, profile);
      const rows = await infer(prepared, signal, false);
      return finishTamevSystemOne(prepared, rows, model);
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
          runner = tokenizer = config = null;
        }
      });
      return disposing;
    }
  };
}
