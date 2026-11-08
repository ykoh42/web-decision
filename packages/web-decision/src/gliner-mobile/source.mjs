import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {GLINER_MOBILE_SPEC} from './contract.mjs';

export async function openGlinerMobileSource({base, signal, cacheEnabled, offline, onProgress}) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name) {
      const bytes = await readVerifiedAsset(base, GLINER_MOBILE_SPEC.manifest, name, {
        cache,
        signal,
        offline,
        maxAssetBytes: GLINER_MOBILE_SPEC.maxAssetBytes,
        onBytes(count) {
          loadedBytes += count;
          onProgress?.({
            loadedBytes,
            totalBytes: GLINER_MOBILE_SPEC.totalBytes,
            fraction: loadedBytes / GLINER_MOBILE_SPEC.totalBytes
          });
        }
      });
      return bytes;
    }
  };
}
