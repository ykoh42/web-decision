import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {D1_OMNI_SPEC} from './contract.mjs';

export async function openD1OmniSource({base, signal, cacheEnabled, offline, onProgress, spec = D1_OMNI_SPEC}) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;

  return {
    async read(name) {
      const bytes = await readVerifiedAsset(base, spec.manifest, name, {
        cache,
        signal,
        offline,
        maxAssetBytes: spec.maxAssetBytes,
        onBytes(count) {
          loadedBytes += count;
          onProgress?.({loadedBytes, totalBytes: spec.totalBytes, fraction: loadedBytes / spec.totalBytes});
        }
      });
      return bytes;
    }
  };
}
