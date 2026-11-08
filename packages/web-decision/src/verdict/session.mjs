import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {VERDICT_ABSTENTION_ID, VERDICT_MAX_CANDIDATES, VERDICT_MODEL} from './contract.mjs';
import {finishVerdictSystemOne, prepareVerdictSystemOne} from './protocol.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Verdict model has been disposed.');

function signalFrom(options) {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (signal != null) validateH2oSignal(signal);
  signal?.throwIfAborted();
  return signal;
}

function validateGenericQuestion(question, index) {
  if (
    typeof question.instructions !== 'string' ||
    question.instructions.length === 0 ||
    !Array.isArray(question.options) ||
    question.options.length < 2 ||
    question.options.length > VERDICT_MAX_CANDIDATES ||
    question.options.some((option) => typeof option !== 'string' || option.length === 0) ||
    new Set(question.options).size !== question.options.length
  )
    throw invalid(
      `Verdict question ${index} needs non-empty instructions and 2 to ${VERDICT_MAX_CANDIDATES} unique string options.`
    );
}

export function createVerdictSession({runner, tokenizer, calibrator, model = VERDICT_MODEL}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    typeof model !== 'string'
  )
    throw new TypeError('Verdict session components are required.');

  let queue = Promise.resolve();
  let closed = false;
  let disposing;

  function infer(prepared, signal) {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finishVerdictSystemOne(prepared, logits, calibrator);
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  function systemOne(state, questions, options = {}) {
    if (closed) throw closedError();
    const signal = signalFrom(options);
    const prepared = prepareVerdictSystemOne(state, questions, tokenizer);
    return infer(prepared, signal);
  }

  return {
    systemOne,
    async decide(request, options = {}) {
      if (closed) throw closedError();
      const checked = validateRequestFields(request, Infinity, Infinity, Infinity, VERDICT_MAX_CANDIDATES);
      const questions = Object.create(null);
      checked.questions.forEach((question, index) => {
        validateGenericQuestion(question, index);
        const criteria = Object.create(null);
        for (const option of question.options) criteria[option] = option;
        questions[String(index)] = {type: 'choice', instructions: question.instructions, criteria};
      });
      const typed = await systemOne(checked.state, questions, options);
      return {
        tokenCount: typed.tokenCount,
        ...(typed.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const answer = typed.answers[String(index)];
          const abstained = answer.choice === VERDICT_ABSTENTION_ID;
          return {
            selected: answer.choice,
            selectedIndex: abstained ? -1 : question.options.indexOf(answer.choice),
            selectedProbability: answer.selectedProbability,
            abstained,
            abstentionProbability: answer.probabilities[VERDICT_ABSTENTION_ID],
            probabilities: [...question.options, VERDICT_ABSTENTION_ID].map((option) => ({
              option,
              probability: answer.probabilities[option],
              logit: answer.logits[option]
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
