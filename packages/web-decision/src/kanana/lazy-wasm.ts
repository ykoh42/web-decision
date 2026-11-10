/** Fetch a packaged Kanana graph only when it is first needed. */
export function createLazyKananaWasmLoader(read: (signal?: AbortSignal) => Promise<ArrayBuffer>) {
  let loaded: ArrayBuffer | undefined;

  return async (signal?: AbortSignal): Promise<ArrayBuffer> => {
    signal?.throwIfAborted();
    if (loaded) return loaded;

    const bytes = await read(signal);
    signal?.throwIfAborted();
    loaded = bytes;
    return bytes;
  };
}
