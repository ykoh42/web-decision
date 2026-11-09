import {waitForAbort} from '../assets.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {finishOpenDeciderNanoRequest, packOpenDeciderNanoRequest} from './protocol.mjs';
import {finishOpenDeciderNanoSystemOne, prepareOpenDeciderNanoSystemOne} from './typed.mjs';

interface NanoTokenizer {
  encode(text: string): unknown;
}
interface NanoRunner {
  run(ids: number[], markerPositions: number[]): Promise<Float32Array>;
  dispose(): unknown;
  releaseCachedMemory(): unknown;
}
interface NanoOptions {
  signal?: AbortSignal | null;
}

export function createOpenDeciderNanoSession({
  tokenizer,
  runner
}: {
  tokenizer: NanoTokenizer | null | undefined;
  runner: NanoRunner | null | undefined;
}) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function'
  )
    throw new TypeError('OpenDecider-nano tokenizer and runner are required.');
  let queue: Promise<unknown> = Promise.resolve(),
    closed = false,
    disposing: Promise<void> | undefined;
  const requireOpen = () => {
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
  };
  const enqueue = <T,>(task: () => T | PromiseLike<T>): Promise<T> => {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  };
  async function infer(request: unknown, options: NanoOptions | null = {}) {
    requireOpen();
    const signal = options?.signal;
    requireOpen();
    if (signal != null) validateH2oSignal(signal);
    signal?.throwIfAborted();
    validateDecisionFields(options);
    const checked = validateRequestFields(request);
    requireOpen();
    const packed = packOpenDeciderNanoRequest(tokenizer!, checked);
    const schedule = () => {
      requireOpen();
      return enqueue(async () => {
        const rows = [];
        for (const question of packed.questions) {
          signal?.throwIfAborted();
          rows.push(await runner!.run(question.ids, question.markerPositions));
          signal?.throwIfAborted();
        }
        return finishOpenDeciderNanoRequest(packed, rows);
      });
    };
    const result = await (signal ? waitForAbort(schedule, signal) : schedule());
    return {result, packed};
  }

  return {
    async decide(request: unknown, options: NanoOptions | null = {}) {
      return (await infer(request, options)).result;
    },
    async systemOne(state: unknown, questions: unknown, options: NanoOptions | null = {}) {
      requireOpen();
      const prepared = prepareOpenDeciderNanoSystemOne(state, questions);
      const {result, packed} = await infer(prepared.request, options);
      return finishOpenDeciderNanoSystemOne(prepared, result, packed);
    },
    async releaseCachedMemory() {
      requireOpen();
      return enqueue(() => runner!.releaseCachedMemory());
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner!.dispose();
        } finally {
          tokenizer = runner = null;
        }
      });
      return disposing;
    }
  };
}
