/** Run cleanup on success or failure without replacing an earlier error. */
export async function withGemma4V2Cleanup<T>(work: () => Promise<T>, cleanup: () => void | Promise<void>): Promise<T> {
  let output: T | undefined;
  let failure: unknown;
  let failed = false;
  try {
    output = await work();
  } catch (error) {
    failed = true;
    failure = error;
  }
  try {
    await cleanup();
  } catch (error) {
    if (!failed) {
      failed = true;
      failure = error;
    }
  }
  if (failed) throw failure;
  return output as T;
}
