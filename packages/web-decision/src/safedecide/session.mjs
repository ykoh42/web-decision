import {waitForAbort} from '../assets.js';
import {validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_MODEL} from './contract.mjs';
import {finishSafeDecideSystemOne, prepareSafeDecideSystemOne} from './protocol.mjs';

const invalid = (message) => {
  throw new WebDecisionError('INVALID_INPUT', message);
};
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The SafeDecide model has been disposed.');
const ALLOWED_ALPHA = Object.freeze([0.05, 0.1, 0.2]);

function validateCallOptions(options) {
  if (options == null) return {signal: undefined, alpha: undefined};
  if (typeof options !== 'object' || Array.isArray(options)) invalid('SafeDecide call options must be an object.');
  const values = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (
      typeof key !== 'string' ||
      !['signal', 'alpha'].includes(key) ||
      !descriptor ||
      !Object.hasOwn(descriptor, 'value')
    )
      invalid('SafeDecide call options contain an unsupported field.');
    values[key] = descriptor.value;
  }
  const signal = values.signal;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    invalid('SafeDecide signal must be an AbortSignal.');
  const alpha = values.alpha == null ? undefined : values.alpha;
  if (alpha !== undefined && !ALLOWED_ALPHA.includes(alpha))
    invalid(`SafeDecide alpha must be one of ${ALLOWED_ALPHA.join(', ')}.`);
  return {signal, alpha};
}

function decisionQuestions(questions) {
  return Object.fromEntries(
    questions.map((question, index) => [
      String(index),
      {
        type: 'choice',
        instructions: question.instructions,
        criteria: Object.fromEntries(question.options.map((name) => [name, null]))
      }
    ])
  );
}

function finishDecisionResult(checked, typed, logits) {
  return {
    tokenCount: typed.tokenCount,
    outputTokenCount: 0,
    ...(typed.truncated ? {truncated: true, truncatedQuestions: typed.truncatedQuestions} : {}),
    guarantee: typed.guarantee,
    answers: checked.questions.map((question, index) => {
      const answer = typed.answers[String(index)];
      return {
        selected: answer.choice,
        selectedIndex: question.options.indexOf(answer.choice),
        probabilities: question.options.map((option, optionIndex) => ({
          option,
          probability: answer.probabilities[option],
          logit: logits[checked.offsets[index] + optionIndex]
        })),
        confidence: answer.confidence,
        action: answer.action
      };
    })
  };
}

export function createSafeDecideSession({runner, tokenizer, config, model = SAFEDECIDE_MODEL}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encodePair !== 'function' ||
    !config?.temperatures ||
    !Array.isArray(config.selective_rules) ||
    !config.selective_rule ||
    typeof model !== 'string'
  )
    throw new TypeError('SafeDecide session components are required.');

  let queue = Promise.resolve();
  let closed = false;
  let disposing;

  function prepare(state, questions) {
    const prepared = prepareSafeDecideSystemOne(state, questions, tokenizer);
    return prepared;
  }

  function infer(prepared, signal, alpha) {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return {
        logits,
        result: finishSafeDecideSystemOne(prepared, logits, config, model, alpha)
      };
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(state, questions, options = {}) {
    if (closed) throw closedError();
    const {signal, alpha} = validateCallOptions(options);
    signal?.throwIfAborted();
    return (await infer(prepare(state, questions), signal, alpha)).result;
  }

  return {
    systemOne,
    async decide(request, options = {}) {
      if (closed) throw closedError();
      const {signal, alpha} = validateCallOptions(options);
      signal?.throwIfAborted();
      const snapshot = validateRequestFields(request);
      if (typeof snapshot.state !== 'string' || snapshot.questions.length < 1)
        invalid('SafeDecide decision requests need a string state and at least one question.');
      for (const [index, question] of snapshot.questions.entries()) {
        if (
          typeof question.instructions !== 'string' ||
          !Array.isArray(question.options) ||
          question.options.length < 2 ||
          question.options.some((option) => typeof option !== 'string') ||
          new Set(question.options).size !== question.options.length
        )
          invalid(`SafeDecide decision question ${index} needs at least 2 distinct string options.`);
      }
      const typed = await infer(prepare(snapshot.state, decisionQuestions(snapshot.questions)), signal, alpha);
      const offsets = [];
      let offset = 0;
      for (const question of snapshot.questions) {
        offsets.push(offset);
        offset += question.options.length;
      }
      return finishDecisionResult({...snapshot, offsets}, typed.result, typed.logits);
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
