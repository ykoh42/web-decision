import {waitForAbort} from '../abort.js';
import type {ModelProgress} from '../curated/types.js';
import {WebDecisionError} from '../kev.js';

type PrepareOptions = Readonly<{signal?: AbortSignal; onProgress?: (progress: ModelProgress) => void}>;
type Listener = (progress: ModelProgress) => void;

interface Preparation {
  readonly controller: AbortController;
  readonly listeners: Set<Listener>;
  readonly promise: Promise<void>;
}

const closedSession = () => new WebDecisionError('SESSION_CLOSED', 'The Mapika session is closed.');

export function createOfflinePreparation(
  initiallyComplete: boolean,
  prepare: (signal: AbortSignal, onProgress: Listener) => Promise<void>
) {
  let complete = initiallyComplete;
  let closed = false;
  let active: Preparation | undefined;

  function start(): Preparation {
    const previous = active;
    const controller = new AbortController();
    const listeners = new Set<Listener>();
    const promise = Promise.resolve()
      .then(async () => {
        await previous?.promise.catch(() => {});
        controller.signal.throwIfAborted();
        complete = false;
        await prepare(controller.signal, (progress) => {
          for (const listener of listeners) listener(progress);
        });
        controller.signal.throwIfAborted();
        complete = true;
      })
      .finally(() => {
        if (active?.controller === controller) active = undefined;
      });
    return (active = {controller, listeners, promise});
  }

  return {
    get complete() {
      return complete;
    },
    async prepareOffline(options: PrepareOptions = {}) {
      if (closed) throw closedSession();
      if (!options || typeof options !== 'object')
        throw new WebDecisionError('INVALID_INPUT', 'Mapika offline options must be an object.');
      const {signal, onProgress} = options;
      if (onProgress !== undefined && typeof onProgress !== 'function')
        throw new WebDecisionError('INVALID_INPUT', 'onProgress must be a function.');
      signal?.throwIfAborted();
      const job = active && !active.controller.signal.aborted ? active : start();
      const observer = new AbortController();
      const listener: Listener = (progress) => {
        try {
          onProgress?.(progress);
        } catch (cause) {
          observer.abort(cause);
        }
      };
      job.listeners.add(listener);
      try {
        await waitForAbort(() => waitForAbort(() => job.promise, observer.signal), signal);
      } finally {
        job.listeners.delete(listener);
        if (!job.listeners.size && active === job) job.controller.abort();
      }
    },
    dispose() {
      closed = true;
      active?.controller.abort(closedSession());
    }
  };
}
