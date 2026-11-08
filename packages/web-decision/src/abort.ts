/** Stop waiting on a non-abortable operation without abandoning its rejection. */
export function waitForAbort<T>(
  operation: () => T | PromiseLike<T>,
  signal?: AbortSignal | null,
  onLateValue?: (value: Awaited<T>) => void
): Promise<Awaited<T>> {
  if (!signal) return Promise.resolve().then(operation) as Promise<Awaited<T>>;
  signal.throwIfAborted();
  return new Promise<Awaited<T>>((resolve, reject) => {
    let settled = false;
    const finishResolve = (value: Awaited<T>) => {
      if (settled) {
        try {
          onLateValue?.(value);
        } catch {}
        return;
      }
      settled = true;
      signal.removeEventListener('abort', onAbort);
      resolve(value);
    };
    const finishReject = (cause: unknown) => {
      if (settled) return;
      settled = true;
      signal.removeEventListener('abort', onAbort);
      reject(cause);
    };
    const onAbort = () => finishReject(signal.reason);
    signal.addEventListener('abort', onAbort, {once: true});
    if (signal.aborted) {
      onAbort();
      return;
    }
    try {
      // An in-flight Cache API operation can still resolve or reject after cancellation.
      Promise.resolve(operation()).then(finishResolve, finishReject);
    } catch (cause) {
      finishReject(cause);
    }
  });
}
