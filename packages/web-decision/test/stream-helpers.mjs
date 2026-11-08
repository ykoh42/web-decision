/** A controllable Response-like body for cancellation and stream failure tests. */
export function responseFromReader(read, {cancel = () => {}, ok = true, status = 200} = {}) {
  return {
    ok,
    status,
    body: {
      getReader: () => ({read, cancel, releaseLock() {}}),
      cancel
    }
  };
}
