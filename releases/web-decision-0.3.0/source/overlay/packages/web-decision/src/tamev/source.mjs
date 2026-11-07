import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {TAMEV_NANO_PROFILE} from './contract.mjs';

export async function openTamevSource(
  {base, signal, cacheEnabled, offline, onProgress},
  profile = TAMEV_NANO_PROFILE
) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name) {
      const buffer = await readVerifiedAsset(base, {files: profile.files}, name, {
        cache,
        signal,
        offline,
        onBytes(bytes) {
          loadedBytes += bytes;
          onProgress?.({
            loadedBytes,
            totalBytes: profile.totalBytes,
            fraction: loadedBytes / profile.totalBytes
          });
        }
      });
      return buffer;
    }
  };
}
