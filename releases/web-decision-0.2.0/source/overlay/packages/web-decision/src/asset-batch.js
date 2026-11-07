/** Read the initial model files together, cancelling siblings on first failure. */
export async function readAssetBatch(names, read, signal) {
  signal?.throwIfAborted();
  const controller = new AbortController();
  const forwardAbort = () => controller.abort(signal.reason);
  signal?.addEventListener('abort', forwardAbort, {once: true});
  if (signal?.aborted) forwardAbort();
  let failed = false,
    firstFailure;
  try {
    const values = await Promise.all(
      names.map((name) =>
        Promise.resolve()
          .then(() => {
            controller.signal.throwIfAborted();
            return read(name, controller.signal);
          })
          .catch((cause) => {
            if (!failed) {
              failed = true;
              firstFailure = cause;
            }
            if (!controller.signal.aborted) controller.abort(cause);
            throw cause;
          })
      )
    );
    signal?.throwIfAborted();
    return values;
  } catch (cause) {
    signal?.throwIfAborted();
    throw failed ? firstFailure : cause;
  } finally {
    signal?.removeEventListener('abort', forwardAbort);
  }
}
