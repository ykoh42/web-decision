export function createGemma4Progress(onProgress, totalBytes) {
  if (
    (onProgress !== undefined && typeof onProgress !== 'function') ||
    !Number.isSafeInteger(totalBytes) ||
    totalBytes < 1
  )
    throw new TypeError('Invalid Gemma 4 progress callback or byte total.');
  let loadedBytes = 0;
  return (bytes) => {
    if (!Number.isSafeInteger(bytes) || bytes < 1 || bytes > totalBytes - loadedBytes)
      throw new RangeError('Gemma 4 progress bytes exceed the pinned asset total.');
    loadedBytes += bytes;
    onProgress?.({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes});
  };
}
