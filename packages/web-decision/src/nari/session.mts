// Joint-choice session for the dynamic Nari graph; not yet a public loader.
import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {pack, unpack} from './adapter.mjs';
import {snapshotToqInput} from './runner.mjs';
import type {NariChoiceResult} from './protocol.mjs';

interface NariSessionOptions {
  readonly tokenizer: unknown;
  readonly controlIds: unknown;
  readonly runner: unknown;
}

interface NariDecisionOptions {
  readonly signal?: unknown;
  readonly [field: string]: unknown;
}

interface NariRunnerCandidate {
  run?: unknown;
  releaseCachedMemory?: unknown;
  dispose?: unknown;
}

interface NariRunner {
  run(input: unknown): Promise<unknown>;
  releaseCachedMemory(): unknown;
  dispose(): unknown;
}

export interface NariDecisionResult {
  readonly tokenCount: number;
  readonly answers: readonly NariChoiceResult[];
}

export interface NariSession {
  decide(request: unknown, options?: NariDecisionOptions): Promise<NariDecisionResult>;
  releaseCachedMemory(): Promise<unknown>;
  dispose(): Promise<void>;
}

const fail = (code: 'SESSION_CLOSED', message: string): WebDecisionError => new WebDecisionError(code, message);
const NARI_MAX_OPTIONS_PER_REQUEST = 64;

export function createNariSession({tokenizer, controlIds, runner}: NariSessionOptions): NariSession {
  if (
    typeof (tokenizer as {encode?: unknown} | null | undefined)?.encode !== 'function' ||
    !Array.isArray(controlIds) ||
    typeof (runner as NariRunnerCandidate | null | undefined)?.run !== 'function' ||
    typeof (runner as NariRunnerCandidate | null | undefined)?.releaseCachedMemory !== 'function' ||
    typeof (runner as NariRunnerCandidate | null | undefined)?.dispose !== 'function'
  )
    throw new TypeError('Tokenizer, control IDs and runner are required.');
  const controls = [...(controlIds as number[])];
  let queue: Promise<unknown> = Promise.resolve(),
    closed = false,
    disposing: Promise<void> | undefined;
  const requireOpen = (): void => {
    if (closed) throw fail('SESSION_CLOSED', 'The model has been disposed.');
  };
  const enqueue = <T,>(task: () => T | PromiseLike<T>): Promise<T> => {
    const result = queue.then(task);
    queue = result.catch(() => {});
    return result;
  };
  return {
    async decide(request: unknown, options: NariDecisionOptions = {}): Promise<NariDecisionResult> {
      requireOpen();
      const signal = (options as NariDecisionOptions | null | undefined)?.signal as AbortSignal | null | undefined;
      requireOpen();
      if (signal != null) validateH2oSignal(signal);
      signal?.throwIfAborted();
      validateDecisionFields(options);
      requireOpen();
      const snapshot = validateRequestFields(
        request,
        8,
        Infinity,
        Infinity,
        NARI_MAX_OPTIONS_PER_REQUEST,
        NARI_MAX_OPTIONS_PER_REQUEST
      );
      requireOpen();
      const input = snapshotToqInput(pack(tokenizer, snapshot, controls));
      requireOpen();
      const run = async (): Promise<NariDecisionResult> => {
        signal?.throwIfAborted();
        const scores = await (runner as NariRunner).run(input);
        signal?.throwIfAborted();
        return unpack(input, scores);
      };
      const schedule = (): Promise<NariDecisionResult> => {
        requireOpen();
        return enqueue(run);
      };
      return signal ? waitForAbort(schedule, signal) : schedule();
    },
    async releaseCachedMemory(): Promise<unknown> {
      requireOpen();
      return enqueue(() => (runner as NariRunner).releaseCachedMemory());
    },
    dispose(): Promise<void> {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await (runner as NariRunner).dispose();
        } finally {
          tokenizer = runner = null;
        }
      });
      return disposing;
    }
  };
}
