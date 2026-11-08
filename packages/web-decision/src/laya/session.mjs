import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {LAYA_MAX_OPTIONS, LAYA_MODEL} from './contract.mjs';
import {finishLayaSystemOne, prepareLayaSystemOne, validateLayaDecisionRequest} from './protocol.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Laya model has been disposed.');

function validateSignal(options) {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid('Laya signal must be an AbortSignal.');
  return signal;
}

export function createLayaSession({runner, tokenizer, temperatures, model = LAYA_MODEL}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    !Array.isArray(temperatures?.byType) ||
    temperatures.byType.length !== 3 ||
    !temperatures.byOptions ||
    typeof model !== 'string'
  )
    throw new TypeError('Laya session components are required.');

  let queue = Promise.resolve();
  let closed = false;
  let disposing;

  function infer(prepared, signal) {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const raw = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finishLayaSystemOne(prepared, raw, temperatures, model);
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(state, questions, options = {}) {
    if (closed) throw closedError();
    const signal = validateSignal(options);
    signal?.throwIfAborted();
    const prepared = prepareLayaSystemOne(state, questions, tokenizer);
    return infer(prepared, signal);
  }

  return {
    systemOne,
    async decide(request, options = {}) {
      if (closed) throw closedError();
      const checked = validateRequestFields(request, Infinity, Infinity, Infinity, LAYA_MAX_OPTIONS);
      validateLayaDecisionRequest(checked);
      const questions = Object.create(null);
      for (let index = 0; index < checked.questions.length; index++) {
        const question = checked.questions[index];
        questions[String(index)] = {type: 'choice', instructions: question.instructions, criteria: question.options};
      }
      const typed = await systemOne(checked.state, questions, options);
      return {
        tokenCount: typed.tokenCount,
        ...(typed.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const answer = typed.answers[String(index)];
          return {
            selected: answer.choice,
            selectedIndex: question.options.indexOf(answer.choice),
            probabilities: question.options.map((option) => ({
              option,
              probability: answer.probabilities[option]
            }))
          };
        })
      };
    },
    async releaseCachedMemory() {
      if (closed) throw closedError();
      await queue;
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(() => runner.dispose());
      return disposing;
    }
  };
}
