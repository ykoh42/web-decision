import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {assertUsable} from './scope.mjs';
import {packStrandsGemma4Request, unpackStrandsGemma4Scores} from './gemma4-protocol.mjs';
import {
  STRANDS_GEMMA4_E2B_CONTEXT,
  STRANDS_GEMMA4_E2B_DECISION_CONFIG,
  STRANDS_GEMMA4_E2B_MODEL
} from './gemma4-e2b-contract.mjs';

function closedError() {
  return new WebDecisionError('SESSION_CLOSED', 'The Strands Gemma 4 model has been disposed.');
}

function requestSignal(options) {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (signal != null) validateH2oSignal(signal);
  signal?.throwIfAborted();
  return signal;
}

function genericQuestions(checked) {
  if (typeof checked.state !== 'string')
    throw new WebDecisionError('INVALID_INPUT', 'state must be a string for decide().');
  const entries = checked.questions.map((question, index) => {
    if (
      typeof question.instructions !== 'string' ||
      !Array.isArray(question.options) ||
      question.options.some((option) => typeof option !== 'string')
    )
      throw new WebDecisionError('INVALID_INPUT', `Question ${index} needs text instructions and string options.`);
    return [String(index), {type: 'choice', instructions: question.instructions, criteria: question.options}];
  });
  return Object.fromEntries(entries);
}

export function createGemma4Session({tokenizer, runner}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    typeof runner?.run !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    typeof runner?.dispose !== 'function'
  )
    throw new TypeError('Strands Gemma 4 needs a tokenizer and owned runner.');
  let queue = Promise.resolve();
  let closed = false;
  let disposal;

  function requireOpen() {
    if (closed) throw closedError();
    assertUsable(runner);
  }

  function infer(packed, signal) {
    requireOpen();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      requireOpen();
      const scoreRows = [];
      for (const row of packed.rows) {
        signal?.throwIfAborted();
        scoreRows.push(await runner.run(row));
        signal?.throwIfAborted();
      }
      return {
        result: unpackStrandsGemma4Scores(
          packed,
          scoreRows,
          STRANDS_GEMMA4_E2B_DECISION_CONFIG,
          STRANDS_GEMMA4_E2B_MODEL
        ),
        scoreRows
      };
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  return {
    async systemOne(state, questions, options = {}) {
      requireOpen();
      const signal = requestSignal(options);
      const packed = packStrandsGemma4Request(tokenizer, state, questions, STRANDS_GEMMA4_E2B_CONTEXT);
      signal?.throwIfAborted();
      return (await infer(packed, signal)).result;
    },

    async decide(request, options = {}) {
      requireOpen();
      const signal = requestSignal(options);
      const checked = validateRequestFields(request);
      const questions = genericQuestions(checked);
      const packed = packStrandsGemma4Request(tokenizer, checked.state, questions, STRANDS_GEMMA4_E2B_CONTEXT);
      signal?.throwIfAborted();
      const {result, scoreRows} = await infer(packed, signal);
      return {
        tokenCount: result.tokenCount,
        answers: checked.questions.map((question, index) => {
          const answer = result.answers[String(index)];
          return {
            selected: answer.choice,
            selectedIndex: question.options.indexOf(answer.choice),
            probabilities: question.options.map((option, optionIndex) => ({
              option,
              probability: answer.probabilities[option],
              logit: scoreRows[index][optionIndex]
            }))
          };
        })
      };
    },

    async releaseCachedMemory() {
      requireOpen();
      const task = queue.then(() => runner.releaseCachedMemory());
      queue = task.catch(() => {});
      return task;
    },

    dispose() {
      if (disposal) return disposal;
      closed = true;
      disposal = queue.then(() => runner.dispose());
      return disposal.finally(() => {
        queue = Promise.resolve();
        tokenizer = runner = null;
      });
    }
  };
}
