import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateRequestFields} from '../request-fields.mjs';
import {DECIMA_SMALL_MODEL} from './contract.mjs';
import {decodeDecimaScores, finishDecimaChoice, validateDecimaRequest} from './protocol.mjs';
import {prepareDecimaQuestion} from './tokenizer.mjs';
import {finishDecimaSystemOne, prepareDecimaSystemOne} from './typed.mjs';

const OPTION_CACHE_LIMIT = 16 * 1024 * 1024;
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);

export function createDecimaSession({runner, tokenizer, config, model = DECIMA_SMALL_MODEL}) {
  if (
    typeof runner?.encode !== 'function' ||
    typeof runner?.scoreEncoded !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    !config?.ordinal ||
    typeof model !== 'string'
  )
    throw new TypeError('Decima session components are required.');
  let queue = Promise.resolve();
  let closed = false;
  let disposing;
  let choiceCacheBytes = 0;
  const choiceCache = new Map();

  const requireOpen = () => {
    if (closed) throw closedError();
  };

  function enqueue(task) {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  }

  function choiceKey(prepared) {
    return JSON.stringify([prepared.choiceIds, prepared.choices]);
  }

  function inferenceOptions(options) {
    if (options == null) return {};
    if (typeof options !== 'object' || Array.isArray(options))
      throw invalid('Decima inference options must be an object.');
    const allowed = new Set(['signal', 'language']);
    const values = Object.create(null);
    const ownKeys = new Set(Reflect.ownKeys(options));
    for (const key of Reflect.ownKeys(options)) {
      const descriptor = Object.getOwnPropertyDescriptor(options, key);
      if (typeof key !== 'string' || !allowed.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
        throw invalid('Unsupported Decima inference option.');
      values[key] = descriptor.value;
    }
    for (const key in options)
      if (!ownKeys.has(key)) throw invalid('Inherited Decima inference options are unsupported.');
    const signal = values.signal;
    const language = values.language;
    if (
      signal != null &&
      (typeof signal.throwIfAborted !== 'function' ||
        typeof signal.addEventListener !== 'function' ||
        typeof signal.removeEventListener !== 'function')
    )
      throw invalid('Decima signal must be an AbortSignal.');
    if (language !== undefined && (typeof language !== 'string' || !language.trim()))
      throw invalid('Decima language must be a non-empty string.');
    return {signal, language: language ?? 'en'};
  }

  async function cachedChoices(prepared) {
    const key = choiceKey(prepared);
    const cached = choiceCache.get(key);
    if (cached) {
      choiceCache.delete(key);
      choiceCache.set(key, cached);
      return cached.rows;
    }
    const encoded = await runner.encode(prepared.choiceIds);
    const bytes = encoded.states.data.byteLength + encoded.mask.data.byteLength;
    if (bytes > OPTION_CACHE_LIMIT) return encoded;
    choiceCache.set(key, {rows: encoded, bytes});
    choiceCacheBytes += bytes;
    while (choiceCacheBytes > OPTION_CACHE_LIMIT) {
      const [oldest, value] = choiceCache.entries().next().value;
      choiceCache.delete(oldest);
      choiceCacheBytes -= value.bytes;
    }
    return encoded;
  }

  async function infer(prepared, kind, signal) {
    requireOpen();
    const schedule = () =>
      enqueue(async () => {
        requireOpen();
        signal?.throwIfAborted();
        const choices = await cachedChoices(prepared);
        const state = await runner.encode([prepared.stateIds]);
        const raw = await runner.scoreEncoded(state, choices, prepared.choiceIds.length);
        const probabilities = decodeDecimaScores(raw, config, kind);
        return {
          probabilities,
          scores: raw.scores,
          logits: raw.scores.map((score) => score / config.temperature),
          tokenCount: prepared.tokenCount,
          truncated: prepared.truncated
        };
      });
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  function pack(state, instructions, choices, language = 'en', allowDuplicates = false) {
    return prepareDecimaQuestion(state, instructions, choices, config, tokenizer, language, allowDuplicates);
  }

  return {
    async decide(request, options = {}) {
      requireOpen();
      const {signal, language} = inferenceOptions(options);
      const checked = validateRequestFields(request);
      validateDecimaRequest(checked);
      const prepared = checked.questions.map((question) =>
        pack(checked.state, question.instructions, question.options, language)
      );
      const answers = [];
      for (const item of prepared) {
        const result = await infer(item, 'choice', signal);
        const answer = finishDecimaChoice(item, result.probabilities, result.logits);
        answers.push(answer);
      }
      return {
        tokenCount: prepared.reduce((sum, item) => sum + item.tokenCount, 0),
        ...(prepared.some((item) => item.truncated) ? {truncated: true} : {}),
        answers
      };
    },
    async systemOne(state, questions, options = {}) {
      requireOpen();
      const {signal, language} = inferenceOptions(options);
      const prepared = prepareDecimaSystemOne(state, questions, language, config.supported_question_types);
      const answers = [];
      for (const question of prepared.questions) {
        const input = pack(
          prepared.state,
          question.instructions,
          question.choices,
          language,
          question.outputType === 'score'
        );
        const result = await infer(input, question.type, signal);
        answers.push({...result, tokenCount: input.tokenCount});
      }
      return finishDecimaSystemOne(prepared, answers, model);
    },
    async releaseCachedMemory() {
      requireOpen();
      return enqueue(() => {
        choiceCache.clear();
        choiceCacheBytes = 0;
      });
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        choiceCache.clear();
        choiceCacheBytes = 0;
        await runner.dispose();
        runner = tokenizer = config = null;
      });
      return disposing;
    }
  };
}
