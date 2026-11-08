import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {GLINER_MAX_OPTIONS, GLINER_MAX_TOKENS, GLINER_MOBILE_MODEL} from './contract.mjs';
import {finishGlinerMobileSystemOne, prepareGlinerMobileSystemOne} from './protocol.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The GLiNER model has been disposed.');

function validateSignal(options) {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid('GLiNER signal must be an AbortSignal.');
  return signal;
}

export function createGlinerMobileSession({runner, tokenizer, model = GLINER_MOBILE_MODEL}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function'
  )
    throw new TypeError('GLiNER session components are required.');

  let queue = Promise.resolve();
  let closed = false;
  let disposing;

  function runPrepared(prepared, signal) {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await runner.run(prepared.inputIds, prepared.markerPositions, signal);
      signal?.throwIfAborted();
      return {result: finishGlinerMobileSystemOne(prepared, logits, model), logits};
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(state, questions, options = {}) {
    if (closed) throw closedError();
    const signal = validateSignal(options);
    signal?.throwIfAborted();
    const prepared = prepareGlinerMobileSystemOne(state, questions, tokenizer);
    return (await runPrepared(prepared, signal)).result;
  }

  return {
    systemOne,
    async decide(request, options = {}) {
      if (closed) throw closedError();
      const checked = validateRequestFields(request, Infinity, Infinity, GLINER_MAX_TOKENS, GLINER_MAX_OPTIONS);
      if (typeof checked.state !== 'string') throw invalid('GLiNER state must be text.');
      const questions = Object.create(null);
      for (let index = 0; index < checked.questions.length; index++) {
        const question = checked.questions[index];
        questions[String(index)] = {type: 'choice', instructions: question.instructions, criteria: question.options};
      }
      const signal = validateSignal(options);
      signal?.throwIfAborted();
      const prepared = prepareGlinerMobileSystemOne(checked.state, questions, tokenizer);
      const {result, logits} = await runPrepared(prepared, signal);
      return {
        tokenCount: result.tokenCount,
        ...(result.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const id = String(index);
          const answer = result.answers[id];
          const preparedQuestion = prepared.questions[index];
          return {
            selected: answer.choice,
            selectedIndex: preparedQuestion.labels.indexOf(answer.choice),
            probabilities: preparedQuestion.labels.map((option, optionIndex) => ({
              option,
              probability: answer.probabilities[option],
              logit: logits[preparedQuestion.markerOffset + optionIndex]
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
