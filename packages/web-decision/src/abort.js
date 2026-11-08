/** Stop waiting on a non-abortable operation without abandoning its rejection. */
export function waitForAbort(operation, signal, onLateValue) {
  if (!signal) return Promise.resolve().then(operation);
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (callback, value) => {
      if (settled) {
        if (callback === resolve) {
          try {
            onLateValue?.(value);
          } catch {}
        }
        return;
      }
      settled = true;
      signal.removeEventListener('abort', onAbort);
      callback(value);
    };
    const onAbort = () => finish(reject, signal.reason);
    signal.addEventListener('abort', onAbort, {once: true});
    if (signal.aborted) {
      onAbort();
      return;
    }
    try {
      // An in-flight Cache API operation can still resolve or reject after cancellation.
      Promise.resolve(operation()).then(
        (value) => finish(resolve, value),
        (cause) => finish(reject, cause)
      );
    } catch (cause) {
      finish(reject, cause);
    }
  });
}
