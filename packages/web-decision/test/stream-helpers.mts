/** A controllable Response-like body for cancellation and stream failure tests. */
export function responseFromReader(
  read: () => unknown,
  {
    cancel = () => {},
    ok = true,
    status = 200
  }: {cancel?: (reason?: unknown) => unknown; ok?: boolean; status?: number} = {}
) {
  return {
    ok,
    status,
    body: {
      getReader: () => ({read, cancel, releaseLock() {}}),
      cancel
    }
  };
}
