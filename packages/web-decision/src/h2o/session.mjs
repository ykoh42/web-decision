// Experimental H2O choice session used by the preview loader.
import {WebDecisionError} from '../kev.js';
import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {MAX_H2O_OPTIONS, packH2oRequest, unpackH2oScores} from './protocol.mjs';
import {validateH2oSignal} from './signal.mjs';

const LABEL_HASH = '3bc0a3b69d62425be332c93e83bce1585fbeb3f4fdd8e0221cf22d0b73f61283';
const fail = (code, message) => new WebDecisionError(code, message);
const digest = async (bytes) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((x) => x.toString(16).padStart(2, '0'))
    .join('');

function graphInputs(ids) {
  const n = ids.length;
  return [
    {dtype: 'int32', shape: [1, n], data: Int32Array.from(ids)},
    {dtype: 'int32', shape: [1, n], data: Int32Array.from({length: n}, (_, i) => i)},
    {dtype: 'float16', shape: [1, 1, n, n], data: new Uint16Array(n * n)},
    {dtype: 'float16', shape: [1, n], data: new Uint16Array(n).fill(0x3c00)},
    {dtype: 'int32', shape: [1], data: Int32Array.of(n - 1)}
  ];
}

export async function createH2oSession({tokenizer, plan, addedTokens, runner}) {
  if (
    !tokenizer ||
    typeof tokenizer.encode !== 'function' ||
    !plan ||
    !Array.isArray(plan.labels) ||
    !Array.isArray(plan.labelIds) ||
    !Array.isArray(addedTokens) ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function'
  )
    throw new TypeError('Tokenizer, plan and runner required');
  const labels = [...plan.labels],
    labelIds = [...plan.labelIds];
  if ((await digest(new TextEncoder().encode(JSON.stringify([labels, labelIds])))) !== LABEL_HASH)
    throw fail('INVALID_ASSET', 'Label rows do not match the pinned head.');
  const config = {labels, labelIds, addedTokens: [...addedTokens], maxInputTokens: 512};
  let queue = Promise.resolve(),
    closed = false,
    disposing;
  const requireOpen = () => {
    if (closed) throw fail('SESSION_CLOSED', 'The model has been disposed.');
  };
  const enqueue = (task) => {
    const p = queue.then(task);
    queue = p.catch(() => {});
    return p;
  };
  return {
    async decide(request, options = {}) {
      requireOpen();
      const signal = options?.signal;
      requireOpen();
      if (signal != null) validateH2oSignal(signal);
      requireOpen();
      signal?.throwIfAborted();
      requireOpen();
      validateDecisionFields(options);
      requireOpen();
      request = validateRequestFields(request, Infinity, Infinity, Infinity, MAX_H2O_OPTIONS);
      requireOpen();
      const packed = packH2oRequest(tokenizer, request, config);
      requireOpen();
      if (
        packed.questions.some(
          ({ids}) =>
            !Array.isArray(ids) ||
            ids.length < 2 ||
            ids.length > 512 ||
            Array.from({length: ids.length}, (_, i) => i).some(
              (i) => !Object.hasOwn(ids, i) || !Number.isSafeInteger(ids[i]) || ids[i] < 0 || ids[i] >= 248320
            )
        )
      )
        throw fail('INVALID_INPUT', 'Invalid H2O graph input tokens.');
      const run = async () => {
        const rows = [];
        for (const question of packed.questions) {
          signal?.throwIfAborted();
          let scores;
          try {
            scores = await runner.run(graphInputs(question.ids));
          } catch (cause) {
            throw new WebDecisionError('INFERENCE_ERROR', 'H2O candidate inference failed.', {cause});
          }
          if (!(scores instanceof Float32Array) || scores.length !== MAX_H2O_OPTIONS || !scores.every(Number.isFinite))
            throw fail('INFERENCE_ERROR', 'The model returned invalid scores.');
          rows.push(Array.from(scores));
        }
        signal?.throwIfAborted();
        return unpackH2oScores(rows, packed, 0.75);
      };
      // Install the abort listener before enqueueing so a failed listener
      // setup cannot leave an unobserved inference running in the background.
      const schedule = () => {
        requireOpen();
        return enqueue(run);
      };
      return signal ? waitForAbort(schedule, signal) : schedule();
    },
    async releaseCachedMemory() {
      if (closed) throw fail('SESSION_CLOSED', 'The model has been disposed.');
      return enqueue(() => runner.releaseCachedMemory());
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner.dispose();
        } finally {
          // Closed sessions may remain reachable in application state.
          tokenizer = runner = null;
        }
      });
      return disposing;
    }
  };
}
