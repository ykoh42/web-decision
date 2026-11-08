import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {D1_OMNI_MODEL, D1_OMNI_TEMPERATURES} from './contract.mjs';
import {finishD1OmniSystemOne, prepareD1OmniSystemOne} from './protocol.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The d1-omni model has been disposed.');

function signalFrom(options) {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (signal != null) validateH2oSignal(signal);
  signal?.throwIfAborted();
  return signal;
}

export function createD1OmniSession({tokenizer, runner, temperatures = D1_OMNI_TEMPERATURES, model = D1_OMNI_MODEL}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    !temperatures ||
    typeof temperatures !== 'object' ||
    typeof model !== 'string'
  )
    throw new TypeError('d1-omni session components are required.');

  let queue = Promise.resolve();
  let closed = false;
  let disposing;

  function infer(prepared, signal) {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const rows = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finishD1OmniSystemOne(prepared, rows, temperatures);
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(state, questions, options = {}) {
    if (closed) throw closedError();
    const signal = signalFrom(options);
    const prepared = prepareD1OmniSystemOne(state, questions, tokenizer);
    return infer(prepared, signal);
  }

  return {
    systemOne,
    async decide(request, options = {}) {
      if (closed) throw closedError();
      const checked = validateRequestFields(request);
      const questions = Object.create(null);
      checked.questions.forEach((question, index) => {
        if (
          !Array.isArray(question.options) ||
          question.options.length < 2 ||
          question.options.some((option) => typeof option !== 'string')
        )
          throw invalid(`Question ${index} needs at least 2 string options.`);
        questions[String(index)] = {
          type: 'choice',
          instructions: question.instructions,
          criteria: question.options
        };
      });
      const typed = await systemOne(checked.state, questions, options);
      return {
        tokenCount: typed.tokenCount,
        ...(typed.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const answer = typed.answers[String(index)];
          return {
            selected: answer.choice,
            selectedIndex: question.options.indexOf(answer.choice),
            probabilities: question.options.map((option) => ({option, probability: answer.probabilities[option]}))
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
