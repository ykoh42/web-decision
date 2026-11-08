import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {TINYDECIDE_SPEC} from './contract.mjs';

export async function openTinyDecideSource({base, signal, cacheEnabled, offline, onProgress, spec = TINYDECIDE_SPEC}) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name) {
      const buffer = await readVerifiedAsset(base, spec.manifest, name, {
        cache,
        signal,
        offline,
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
