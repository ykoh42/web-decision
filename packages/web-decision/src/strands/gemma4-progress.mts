type Progress = {loadedBytes: number; totalBytes: number; fraction: number};

export function createGemma4Progress(onProgress: ((event: Progress) => void) | undefined, totalBytes: number) {
  if (
    (onProgress !== undefined && typeof onProgress !== 'function') ||
    !Number.isSafeInteger(totalBytes) ||
    totalBytes < 1
  )
    throw new TypeError('Invalid Gemma 4 progress callback or byte total.');
  let loadedBytes = 0;
  return (bytes: number) => {
    if (!Number.isSafeInteger(bytes) || bytes < 1 || bytes > totalBytes - loadedBytes)
      throw new RangeError('Gemma 4 progress bytes exceed the pinned asset total.');
    loadedBytes += bytes;
    onProgress?.({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes});
  };
}
