import {validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {TINYDECIDE_MAX_CHOICES, TINYDECIDE_QUESTIONS_PER_CHUNK} from './contract.mjs';
import {makeProtos as buildProtos} from './corrections.js';
import {
  validateTinyDecideDecisionOptions,
  validateTinyDecideRequest,
  validateTinyDecideSystemOne,
  validateTinyDecideRows
} from './inputs.mjs';

const closed = () => new WebDecisionError('SESSION_CLOSED', 'TinyDecide model session is closed.');
const inference = (cause) =>
  cause instanceof WebDecisionError || cause?.name === 'AbortError'
    ? cause
    : new WebDecisionError('INFERENCE_ERROR', 'TinyDecide inference failed.', {cause});

function answerInChunks(engine, request, protos, signal) {
  const answers = [];
  const ids = [];
  let stateTokens = 0;
  let questionTokens = 0;
  let truncated = false;
  let ms = 0;

  for (let start = 0; start < request.questions.length; start += TINYDECIDE_QUESTIONS_PER_CHUNK) {
    signal?.throwIfAborted();
    const end = Math.min(start + TINYDECIDE_QUESTIONS_PER_CHUNK, request.questions.length);
    const chunk = {state: request.state, questions: request.questions.slice(start, end)};
    const chunkProtos = protos?.slice(start, end);
    const result = engine.answer(chunk, chunkProtos);
    answers.push(...result.answers);
    questionTokens += result.tokens.questions;
    truncated ||= result.truncated;
    ms += result.ms;
    if (start === 0) {
      stateTokens = result.tokens.state;
      ids.push(...result.ids);
    } else {
      ids.push(...result.ids.slice(result.tokens.state));
    }
  }

  return {
    answers,
    tokens: {state: stateTokens, questions: questionTokens, total: stateTokens + questionTokens},
    truncated,
    ms,
    ids
  };
}

function readCorrectionVector(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length)
    throw new WebDecisionError('INVALID_INPUT', `TinyDecide correction ${label} must have ${length} values.`);
  const vector = Array.from(value);
  if (vector.some((item) => !Number.isFinite(item)))
    throw new WebDecisionError('INVALID_INPUT', `TinyDecide correction ${label} must contain finite numbers.`);
  return vector;
}

function validateCorrectionExample(value, optionIndex, exampleIndex, count, dimensions) {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is invalid.`);
  const fields = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if ((key !== 'v' && key !== 'z') || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is invalid.`);
    fields[key] = descriptor.value;
  }
  if (!Object.hasOwn(fields, 'v') || !Object.hasOwn(fields, 'z'))
    throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is incomplete.`);
  return {
    v: readCorrectionVector(fields.v, dimensions, `option ${optionIndex + 1} example ${exampleIndex + 1} vector`),
    z: readCorrectionVector(fields.z, count, `option ${optionIndex + 1} example ${exampleIndex + 1} logits`)
  };
}

export function createTinyDecideSession({engine, model}) {
  let queue = Promise.resolve();
  let disposed;
  let isClosed = false;

  function submit(request, signal, protos) {
    signal?.throwIfAborted();
    const result = queue.then(() => {
      signal?.throwIfAborted();
      validateTinyDecideRows(engine, request);
      signal?.throwIfAborted();
      try {
        return answerInChunks(engine, request, protos, signal);
      } catch (cause) {
        throw inference(cause);
      }
    });
    queue = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  function requireOpen() {
    if (isClosed) throw closed();
  }

  function makeProtos(type, lists, center) {
    requireOpen();
    if (!['choice', 'noul', 'score', 'span'].includes(type) || !Array.isArray(lists))
      throw new WebDecisionError('INVALID_INPUT', 'TinyDecide correction examples are invalid.');
    if (type === 'span') return buildProtos(type, lists, engine.meta.beta, center);
    const count = type === 'noul' ? 2 : lists.length;
    if (count < 2 || count > 32 || (type === 'noul' && lists.length !== 2))
      throw new WebDecisionError('INVALID_INPUT', 'TinyDecide correction lists do not match the question.');
    const dimensions = engine.meta.cfg.dh_head;
    const checkedLists = lists.map((list, optionIndex) => {
      if (!Array.isArray(list))
        throw new WebDecisionError('INVALID_INPUT', `Correction list ${optionIndex + 1} is invalid.`);
      return Array.from({length: list.length}, (_, exampleIndex) => {
        const descriptor = Object.getOwnPropertyDescriptor(list, String(exampleIndex));
        if (!descriptor || !Object.hasOwn(descriptor, 'value'))
          throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is missing.`);
        const example = descriptor.value;
        const fields = validateCorrectionExample(example, optionIndex, exampleIndex, count, dimensions);
        return {v: fields.v, z: fields.z};
      });
    });
    const checkedCenter = center == null ? null : readCorrectionVector(center, dimensions, 'center');
    return buildProtos(type, checkedLists, engine.meta.beta, checkedCenter);
  }

  return {
    async answer(request, options = {}) {
      requireOpen();
      const checkedRequest = validateTinyDecideRequest(request);
      const checkedOptions = validateTinyDecideDecisionOptions(options, checkedRequest, engine.meta.cfg.dh_head);
      return submit(checkedRequest, checkedOptions?.signal, checkedOptions?.protos);
    },
    makeProtos,
    async decide(request, options = {}) {
      requireOpen();
      const checked = validateRequestFields(request, Infinity, Infinity, Infinity, TINYDECIDE_MAX_CHOICES);
      if (typeof checked.state !== 'string' || checked.questions.length === 0)
        throw new WebDecisionError('INVALID_INPUT', 'TinyDecide needs a string state and at least one question.');
      const nativeRequest = validateTinyDecideRequest({
        state: checked.state,
        questions: checked.questions.map(({instructions, options: choices}) => ({
          type: 'choice',
          text: instructions,
          options: choices
        }))
      });
      const checkedOptions = validateTinyDecideDecisionOptions(options, nativeRequest, engine.meta.cfg.dh_head);
      const result = await submit(nativeRequest, checkedOptions?.signal, checkedOptions?.protos);
      return {
        tokenCount: result.tokens.total,
        ...(result.truncated ? {truncated: true} : {}),
        answers: result.answers.map((answer, index) => ({
          selected: nativeRequest.questions[index].options[answer.pick],
          selectedIndex: answer.pick,
          probabilities: answer.probs.map((probability, optionIndex) => ({
            option: nativeRequest.questions[index].options[optionIndex],
            probability,
            logit: answer.z0[optionIndex]
          }))
        }))
      };
    },
    async systemOne(state, questions, options = {}) {
      requireOpen();
      const checked = validateTinyDecideSystemOne(state, questions);
      const checkedOptions = validateTinyDecideDecisionOptions(options, checked.request, engine.meta.cfg.dh_head);
      const result = await submit(checked.request, checkedOptions?.signal, checkedOptions?.protos);
      return {
        model,
        tokenCount: result.tokens.total,
        ...(result.truncated ? {truncated: true} : {}),
        answers: Object.fromEntries(checked.names.map((name, index) => [name, result.answers[index]]))
      };
    },
    async releaseCachedMemory() {
      requireOpen();
    },
    dispose() {
      if (disposed) return disposed;
      isClosed = true;
      disposed = queue.then(() => {
        engine = null;
      });
      return disposed;
    }
  };
}
