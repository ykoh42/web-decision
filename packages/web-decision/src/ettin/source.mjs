import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {ETTIN_PROFILE} from './contract.mjs';

export async function openEttinSource({base, signal, cacheEnabled, offline, onProgress}, profile = ETTIN_PROFILE) {
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
          onProgress?.({loadedBytes, totalBytes: profile.totalBytes, fraction: loadedBytes / profile.totalBytes});
        }
      });
      return buffer;
    }
  };
}
