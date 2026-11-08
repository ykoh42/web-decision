import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {DECIMA_SMALL_SPEC} from './contract.mjs';

export async function openDecimaSource({base, signal, cacheEnabled, offline, onProgress, spec = DECIMA_SMALL_SPEC}) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name) {
      const buffer = await readVerifiedAsset(base, spec.manifest, name, {
        cache,
        signal,
        offline,
        maxAssetBytes: spec.maxAssetBytes,
        onBytes(bytes) {
          loadedBytes += bytes;
          onProgress?.({
            loadedBytes,
            totalBytes: spec.totalBytes,
            fraction: loadedBytes / spec.totalBytes
          });
        }
      });
      return buffer;
    }
  };
}
