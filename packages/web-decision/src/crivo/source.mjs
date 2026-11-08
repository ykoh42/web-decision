import {openAssetCache, readVerifiedAsset} from '../assets.js';

export async function openCrivoSource({base, signal, cacheEnabled, offline, onProgress}, spec) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name) {
      const buffer = await readVerifiedAsset(base, {files: spec.files}, name, {
        cache,
        signal,
        offline,
        onBytes(bytes) {
          loadedBytes += bytes;
          onProgress?.({loadedBytes, totalBytes: spec.totalBytes, fraction: loadedBytes / spec.totalBytes});
        }
      });
      return buffer;
    }
  };
}
