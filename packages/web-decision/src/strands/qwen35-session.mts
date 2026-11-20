import {waitForAbort} from '../abort.js';
import {WebDecisionError} from '../kev.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import type {Qwen35ImageInputOptions} from './qwen35-image-input.mts';
import {packQwen35RequestSnapshot, snapshotQwen35Request, type Qwen35RequestOptions} from './qwen35-protocol.mts';
import type {Qwen35GraphInputOptions} from './qwen35-graph-input.mts';
import {
  evaluateQwen35PackedRequest,
  type Qwen35GraphRunner,
  type Qwen35SessionAdapterOptions,
  type Qwen35SystemOneResponse
} from './qwen35-session-adapter.mts';

type MaybePromise<Value> = Value | PromiseLike<Value>;

export interface Qwen35OwnedRunner extends Qwen35GraphRunner {
  releaseCachedMemory(): MaybePromise<void>;
  dispose(): MaybePromise<void>;
}

export type Qwen35SystemOneOptions = Qwen35RequestOptions &
  Pick<Qwen35ImageInputOptions, 'imageLongSide' | 'maxImagePixels'> &
  Pick<Qwen35GraphInputOptions, 'strictWindow'> &
  Readonly<{signal?: AbortSignal | null}>;

export interface Qwen35SessionOptions {
  readonly tokenizer: unknown;
  readonly runner: Qwen35OwnedRunner;
  readonly modelName: string;
  readonly graphOptions?: Qwen35GraphInputOptions;
}

interface Qwen35CallOptions {
  readonly signal: AbortSignal | null | undefined;
  readonly graph: Qwen35GraphInputOptions;
}

function closedError(): WebDecisionError {
  return new WebDecisionError('SESSION_CLOSED', 'The Strands Qwen3.5 model has been disposed.');
}

function callOptions(value: Qwen35SystemOneOptions | null | undefined): Qwen35CallOptions {
  if (value == null) return {signal: undefined, graph: {}};
  if (typeof value !== 'object' || Array.isArray(value))
    throw new WebDecisionError('INVALID_INPUT', 'Qwen systemOne options must be an object.');
  if (value.strictWindow !== undefined && typeof value.strictWindow !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'strictWindow must be a boolean.');
  const signal = value.signal;
  if (signal != null) validateH2oSignal(signal);
  signal?.throwIfAborted();
  const {signal: _signal, ...graph} = value;
  return {signal, graph};
}

function ownedRunner(value: unknown): value is Qwen35OwnedRunner {
  const runner = value as Partial<Qwen35OwnedRunner> | null | undefined;
  return (
    typeof runner?.run === 'function' &&
    typeof runner.releaseCachedMemory === 'function' &&
    typeof runner.dispose === 'function'
  );
}

function encodeMethod(value: unknown): boolean {
  return typeof (value as {encode?: unknown} | null | undefined)?.encode === 'function';
}

function abortableRunner(runner: Qwen35OwnedRunner, signal: AbortSignal | null | undefined): Qwen35GraphRunner {
  if (!signal) return runner;
  return {
    async run(chunk, input, requestSignal) {
      const activeSignal = requestSignal ?? signal;
      activeSignal.throwIfAborted();
      const result = await runner.run(chunk, input, activeSignal);
      activeSignal.throwIfAborted();
      return result;
    }
  };
}

/** Native Strands System One session around an owned TVM graph runner. */
export function createQwen35Session(options: Qwen35SessionOptions) {
  if (
    !options ||
    !encodeMethod(options?.tokenizer) ||
    !ownedRunner(options?.runner) ||
    typeof options.modelName !== 'string' ||
    options.modelName.length === 0
  )
    throw new TypeError('Qwen3.5 needs a tokenizer, model name, and owned graph runner.');

  let tokenizer: unknown | null = options.tokenizer;
  let runner: Qwen35OwnedRunner | null = options.runner;
  const sessionOptions: Qwen35GraphInputOptions & Readonly<{modelName: string}> = {
    ...options.graphOptions,
    modelName: options.modelName
  };
  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposal: Promise<void> | undefined;

  function requireOpen(): Readonly<{runner: Qwen35OwnedRunner; tokenizer: unknown}> {
    if (closed || !runner || tokenizer === null) throw closedError();
    return {runner, tokenizer};
  }

  async function systemOne(
    request: unknown,
    rawOptions?: Qwen35SystemOneOptions | null
  ): Promise<Qwen35SystemOneResponse> {
    const current = requireOpen();
    const {signal, graph} = callOptions(rawOptions);
    const callGraphOptions = {...sessionOptions, ...graph, modelName: sessionOptions.modelName};
    const requestSnapshot = snapshotQwen35Request(request, callGraphOptions);
    const graphOptions: Qwen35SessionAdapterOptions = requestSnapshot.images.length
      ? {...callGraphOptions, maxBatchTokens: null}
      : callGraphOptions;
    const packed = packQwen35RequestSnapshot(current.tokenizer, requestSnapshot, graphOptions);
    const task = queue.then(async () => {
      const active = requireOpen();
      signal?.throwIfAborted();
      const activeRunner = abortableRunner(active.runner, signal);
      return evaluateQwen35PackedRequest(active.tokenizer, activeRunner, packed, graphOptions, signal);
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function releaseCachedMemory(): Promise<void> {
    const active = requireOpen();
    const task = queue.then(() => active.runner.releaseCachedMemory());
    queue = task.catch(() => {});
    return task;
  }

  function dispose(): Promise<void> {
    if (disposal) return disposal;
    closed = true;
    const activeRunner = runner;
    disposal = queue.then(async () => {
      try {
        await activeRunner?.dispose();
      } finally {
        tokenizer = null;
        runner = null;
      }
    });
    return disposal;
  }

  return Object.freeze({systemOne, releaseCachedMemory, dispose});
}
