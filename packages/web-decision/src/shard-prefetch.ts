import {waitForAbort} from './assets.js';
import {WebDecisionError} from './kev.js';

type ReadOutcome<Value> = {readonly ok: true; readonly value: Value} | {readonly ok: false; readonly error: unknown};

/** Validate a budget for the declared bytes of one additional model shard. */
export function validatePrefetchBytes(value: unknown): number {
  if (value === undefined) return 0;
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new WebDecisionError('INVALID_INPUT', 'maxPrefetchBytes must be a nonnegative safe integer.');
  }
  return value;
}

/** Read at most one additional shard while consuming shards in manifest order. */
export async function processPrefetchedShards<Shard extends {readonly nbytes: number}, Value>(
  records: readonly Shard[],
  read: (shard: Shard, signal: AbortSignal, index: number) => Value | PromiseLike<Value>,
  consume: (shard: Shard, value: Awaited<Value>, index: number) => unknown | PromiseLike<unknown>,
  signal?: AbortSignal | null,
  maxPrefetchBytes?: number
): Promise<void> {
  signal?.throwIfAborted();
  const budget = validatePrefetchBytes(maxPrefetchBytes);
  if (!Array.isArray(records) || records.some((record) => !Number.isSafeInteger(record?.nbytes) || record.nbytes < 0)) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid shard sizes.');
  }
  const controller = new AbortController();
  const noReadFailure = Symbol('no read failure');
  let firstReadFailure: unknown = noReadFailure;
  const forwardAbort = () => controller.abort(signal?.reason);
  signal?.addEventListener('abort', forwardAbort, {once: true});
  if (signal?.aborted) forwardAbort();
  // Every read has a rejection handler immediately. A later failure aborts the
  // current read too, so the loader need not wait on an unresponsive sibling.
  const failRead = (error: unknown): ReadOutcome<Awaited<Value>> => {
    if (!controller.signal.aborted) {
      firstReadFailure = error;
      controller.abort(error);
    }
    return {ok: false, error};
  };
  const start = (index: number): Promise<ReadOutcome<Awaited<Value>>> | null =>
    index < records.length
      ? Promise.resolve().then(() => {
          try {
            controller.signal.throwIfAborted();
            // Attach before adopting the read promise so its rejection is
            // observed in order with the active consume promise.
            return Promise.resolve(read(records[index]!, controller.signal, index)).then(
              (value) => ({ok: true as const, value}),
              failRead
            );
          } catch (error) {
            return failRead(error);
          }
        })
      : null;
  const mayPrefetch = (index: number) => budget > 0 && index < records.length && records[index]!.nbytes <= budget;
  let current = start(0),
    next = mayPrefetch(1) ? start(1) : null;
  try {
    for (let i = 0; i < records.length; i++) {
      const active = current;
      const outcome = await waitForAbort(() => active!, controller.signal);
      controller.signal.throwIfAborted();
      if (!outcome.ok) throw outcome.error;
      // GPU synchronization is deliberately awaited; it cannot be safely
      // abandoned while its resources may still be in use.
      await consume(records[i]!, outcome.value, i);
      controller.signal.throwIfAborted();
      current = next || start(i + 1);
      next = mayPrefetch(i + 2) ? start(i + 2) : null;
    }
  } catch (cause) {
    signal?.throwIfAborted();
    throw firstReadFailure !== noReadFailure ? firstReadFailure : cause;
  } finally {
    controller.abort();
    signal?.removeEventListener('abort', forwardAbort);
  }
}
