import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {ETTIN_MAX_OPTIONS, ETTIN_MODEL} from './contract.mjs';
import {finishEttinDecision, finishEttinSystemOne, prepareEttinDecision, prepareEttinSystemOne} from './protocol.mjs';
import type {EttinRunnerResult, EttinTokenizer, PackedEttinQuestion, PreparedEttinRequest} from './protocol.mjs';

interface EttinRunner {
  run(question: PackedEttinQuestion): Promise<EttinRunnerResult>;
  dispose(): Promise<void>;
}

interface EttinSessionOptions {
  readonly runner: EttinRunner | null;
  readonly tokenizer: EttinTokenizer | null;
  readonly model?: string;
}

interface EttinRequestSnapshot {
  readonly state: string;
  readonly questions: readonly {readonly instructions: unknown; readonly options: unknown}[];
}

const invalid = (message: string) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');

function readSignal(options: unknown): AbortSignal | null | undefined {
  if (options == null) return undefined;
  validateDecisionFields(options);
  const descriptor = Object.getOwnPropertyDescriptor(options as object, 'signal');
  if (descriptor && !Object.hasOwn(descriptor, 'value'))
    throw invalid('Decider Ettin does not accept signal accessors.');
  const signal = descriptor?.value as AbortSignal | null | undefined;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid('Decider Ettin signal must be an AbortSignal.');
  return signal;
}

export function createEttinSession({runner, tokenizer, model = ETTIN_MODEL}: EttinSessionOptions) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function'
  )
    throw new TypeError('Decider Ettin session components are required.');
  if (model !== ETTIN_MODEL) throw new TypeError('Decider Ettin model id does not match the pinned contract.');

  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  const requireOpen = () => {
    if (closed) throw closedError();
  };

  function enqueue<Result>(task: () => Result | Promise<Result>): Promise<Result> {
    const current = queue.then(task);
    queue = current.catch((): void => {});
    return current;
  }

  async function infer(
    prepared: PreparedEttinRequest,
    signal: AbortSignal | null | undefined
  ): Promise<EttinRunnerResult[]> {
    requireOpen();
    const schedule = (): Promise<EttinRunnerResult[]> =>
      enqueue(async () => {
        requireOpen();
        const rows = [];
        for (const question of prepared.questions) {
          signal?.throwIfAborted();
          rows.push(await runner!.run(question));
        }
        return rows;
      });
    return (signal ? waitForAbort(schedule, signal) : schedule()) as Promise<EttinRunnerResult[]>;
  }

  return {
    async decide(request: unknown, options: unknown = {}) {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const snapshot = validateRequestFields(
        request,
        Infinity,
        Infinity,
        Infinity,
        ETTIN_MAX_OPTIONS
      ) as EttinRequestSnapshot;
      const prepared = prepareEttinDecision(tokenizer!, snapshot);
      const rows = await infer(prepared, signal);
      return finishEttinDecision(prepared, rows);
    },
    async systemOne(state: unknown, questions: unknown, options: unknown = {}) {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const prepared = prepareEttinSystemOne(state, questions, tokenizer!);
      const rows = await infer(prepared, signal);
      return finishEttinSystemOne(prepared, rows, model);
    },
    releaseCachedMemory() {
      requireOpen();
      return enqueue(() => {});
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner!.dispose();
        } finally {
          runner = tokenizer = null;
        }
      });
      return disposing;
    }
  };
}
